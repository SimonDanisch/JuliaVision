"""
Export TRELLIS.2's **dense** stages to DNNKernels' graph JSON + weights.

    ~/tools/TRELLIS.2/.venv/bin/python tools/export_trellis2.py --part cond
    ~/tools/TRELLIS.2/.venv/bin/python tools/export_trellis2.py --part ss
    ~/tools/TRELLIS.2/.venv/bin/python tools/export_trellis2.py --part ssdec

**Run it with TRELLIS.2's own interpreter, not the repo's.** That venv has the
CUDA extensions the package imports at load (`flex_gemm`, `o_voxel`) built
against its own torch; `tools/` only needs `export_graphs`, which is pure torch,
so it is put on `sys.path` instead of the other way round.

## Why only three of the eight models

`Trellis2ImageTo3DPipeline` runs, in order: a DINOv3 image encoder; a flow model
over a dense 16^3 latent; a conv3d decoder to a 64^3 occupancy grid; then
`torch.argwhere` on that grid, and everything after it — the four SLAT flow
models and two SLAT decoders, 10.6 GB of the 17.4 — operates on a **sparse** set
of active voxels whose *count depends on the image*.

That `argwhere` is the exact boundary. Before it every shape is static and every
op is one DNNKernels already runs. After it the tensors are ragged: attention
goes through `flash_attn_varlen_*` over `cu_seqlens`, convolution through
`flex_gemm`, and `trellis2/modules/sparse/` is 2394 lines of a tensor type that
has no equivalent in a runtime built on static shapes and a planned allocation
slab. That is a separate piece of work and not a matter of missing ops.

So this file exports the part that can be exported today, which is image -> a
64^3 occupancy voxel grid.

## What the three parts are

| part | model | params | shape |
|---|---|---|---|
| `cond` | DINOv3 ViT-L/16 | 303.1M | (1,3,512,512) -> (1,1029,1024) |
| `ss` | `SparseStructureFlowModel` | 1.292B | (1,8,16,16,16) + t + cond -> same |
| `ssdec` | `SparseStructureDecoder` | 73.7M | (1,8,16,16,16) -> (1,1,64,64,64) |

1029 tokens is 32x32 patches at 512/16, plus a class token and DINOv3's four
register tokens.

## Where it stands on Lava

`julia --project=. tools/verify_graph.jl`, mean |delta| as a percentage of the
reference's own range, RTX 4000 Ada:

| export | mean | corr |
|---|---|---|
| `trellis2-cond` | 0.000003% | 1.000000000 |
| `trellis2-ss-b3-fp32` (3 blocks) | 0.000008% | 1.000000000 |
| `trellis2-ss-b3` (3 blocks, fp16) | 0.004375% | 0.999999883 |
| `trellis2-ss` (30 blocks, fp16) | 0.026424% | 0.999992443 |
| `trellis2-ssdec` | 0.015571% | 0.999999009 |

The fp32 torso is at machine precision, so what is left in the fp16 rows is fp16
accumulation order and nothing else. **This is after the `gelu` fix** (`atenarg`
in `graph.jl`): before it the 30-block torso read 0.114% and the 3-block one
0.0125%, in fp32 and fp16 alike, because the model asks for the tanh
approximation as a *keyword* and the runtime only ever looked for it by position.

`trellis2-ss-fp32` at 30 blocks does not fit: 5.2 GB of weights and a ~9 GB slab
against a 20 GB card, and it OOMs cleanly. The 3-block slice answers the same
question.

**fp16, where upstream is bf16.** `SparseStructureFlowModel` is mixed by design:
`convert_to` puts only `self.blocks` in the checkpoint's `bfloat16` and leaves the
input layer, the timestep embedder, the output layer and the complex RoPE table in
fp32, with `manual_cast` at the boundary. DNNKernels has no `bfloat16` — its
`DTYPE_NAMES` does not carry one — so the torso is exported at fp16 instead.
Measured on one forward against the bf16 reference: **0.275% of range, correlation
0.9991, and no overflow** (bf16's advantage is exponent range, and nothing here
needs it; fp16 carries three more mantissa bits, so this is a different rounding
rather than a worse one). Do not "fix" this by casting the whole model — `.to()`
on the module silently drops the imaginary part of `rope_phases` and torch only
warns.

**`ATTN_BACKEND=sdpa`, set here before `trellis2` is imported.** The package reads
it at module scope. Its default is `flash_attn` and the pipeline normally runs
`xformers`; neither decomposes into ATen, while `sdpa` is
`F.scaled_dot_product_attention` and exports as the same
`_scaled_dot_product_flash_attention` every other model here carries.

Upstream: https://github.com/microsoft/TRELLIS (TRELLIS.2)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Before `trellis2` is imported anywhere: `trellis2.modules.attention.config`
# reads ATTN_BACKEND at module scope and there is no second chance.
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_graphs as EG

from common import find_root

ROOT = find_root()
GEN = ROOT / "gen"
CHECKOUT = Path("/home/simon/tools/TRELLIS.2")
REPO = "microsoft/TRELLIS.2-4B"

# `sparse_structure_decoder` is not in the TRELLIS.2 repo — `pipeline.json`
# points at TRELLIS 1's, and it is the only model the two share.
SS_FLOW = f"{REPO}/ckpts/ss_flow_img_dit_1_3B_64_bf16"
SS_DEC = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# The graph's input ids, which `torch.export` takes from the forward's parameter
# NAMES. `ss` and `ssdec` are exported unwrapped, so these are upstream's own
# names and not a choice — `SparseStructureDecoder.forward(self, x)` makes the id
# `x`, and calling it `z` here files the reference under a key no loader looks
# for. Checked against `graph.inputs` rather than read off the source.
ARGNAMES = {"trellis2_ss": ("x", "t", "cond"),
            "trellis2_ssdec": ("x",),
            "trellis2_cond": ("image",)}

# `DinoV3FeatureExtractor.transform`'s statistics, which the graph deliberately
# does not contain — the same split `export_hunyuan3d.py` makes, and the only
# place the Julia side can read them from.
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
DINO_RES = 512


def bootstrap():
    if not (CHECKOUT / "trellis2").is_dir():
        raise SystemExit(f"no trellis2 package at {CHECKOUT}")
    if str(CHECKOUT) not in sys.path:
        sys.path.insert(0, str(CHECKOUT))
    # `from_pretrained` resolves relative ckpt paths against the cwd.
    os.chdir(CHECKOUT)


def dinov3():
    """The image encoder, with the `extract_features` newer transformers needs.

    `transformers` is unpinned in TRELLIS.2's `setup.sh`, and from 5.15 the
    DINOv3 blocks sit at `.model.layer` rather than `.layer` while `embeddings`
    and `rope_embeddings` stay put. Upstream walks the old path only. Looking in
    both is what CrawCity's `gen_trellis2.py` does too, and it beats pinning the
    library back.
    """
    from trellis2.modules import image_feature_extractor as ife

    def extract_features(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        h = self.model.embeddings(image, bool_masked_pos=None)
        pe = self.model.rope_embeddings(image)
        layers = getattr(self.model, "layer", None)
        if layers is None:
            layers = self.model.model.layer
        for layer_module in layers:
            h = layer_module(h, position_embeddings=pe)
        return F.layer_norm(h, h.shape[-1:])

    ife.DinoV3FeatureExtractor.extract_features = extract_features
    return ife.DinoV3FeatureExtractor(DINO, image_size=DINO_RES)


class Conditioner(torch.nn.Module):
    """DINOv3's transformer, with the host-side normalisation left out.

    Takes the already-normalised 512x512 tensor; `MEAN`/`STD` above are the only
    place those statistics live, and they go to `preprocess.json` beside the
    graph.
    """

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.model = extractor.model

    # The parameter name is the graph's input id — `torch.export` takes it from
    # the placeholder — so it has to match `ARGNAMES`.
    def forward(self, image):
        return self.extractor.extract_features(image)


def write(out: Path, name: str, ep, model, args, y):
    """Graph JSON, weights, op histogram and reference, as every export here."""
    g = EG.convert(ep, tuple({} for _ in args), name)    # {} = all dims static
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()),
                            **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    ref = {n: a.detach().contiguous().cpu() for n, a in zip(ARGNAMES[name], args)}
    ref["out"] = y.detach().contiguous().cpu()
    save_file(ref, str(out / "reference.safetensors"))

    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"  {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights -> {out}")
    print(f"  reference {tuple(args[0].shape)} -> {tuple(y.shape)}, "
          f"absmax {y.float().abs().max().item():.4f}")
    print("  ops:", json.dumps(hist, sort_keys=True))
    return g


def export_ss(out: Path, dev: str, precision: str, blocks: int | None = None):
    """The dense flow model over the 16^3 latent."""
    from trellis2.models import from_pretrained
    m = from_pretrained(SS_FLOW).eval().to(dev)
    if blocks is not None and blocks < len(m.blocks):
        # Safe to truncate, unlike Hunyuan3D's denoiser: `forward` is a plain
        # `for block in self.blocks` with no U-Net skip stack to keep balanced,
        # so a slice is the same model with fewer layers. For bisecting whether
        # a disagreement grows with depth (accumulated rounding) or is already
        # there at block 3 (a wrong kernel).
        m.blocks = m.blocks[:blocks]
    print(f"trellis2_ss: {type(m).__name__}, resolution {m.resolution}, "
          f"{m.in_channels} channels, "
          f"{sum(p.numel() for p in m.parameters()) / 1e9:.3f}B parameters")
    # `convert_to`, NOT `.to()` — the latter casts `rope_phases` and torch only
    # warns that it "discarded the imaginary part".
    #
    # fp32 has to be converted too, and is not "leave it alone": the checkpoint
    # loads with a bf16 torso, and DNNKernels has no bfloat16, so skipping the
    # call exports buffers it cannot read. The dtype below is read back off the
    # model rather than echoed from the flag, because an `if` that quietly did
    # nothing is exactly how this was wrong the first time.
    m.convert_to({"fp16": torch.float16, "fp32": torch.float32}[precision])
    # Every dtype in the torso and how many parameters carry it, not
    # `next(m.blocks.parameters()).dtype`. `convert_module_to` deliberately
    # leaves some tensors alone, so the first parameter is not representative
    # and reported fp32 for a torso that was fp16 — the second time a print in
    # this function said something that was not true of the export beside it.
    from collections import Counter
    dts = Counter(str(p.dtype).removeprefix("torch.") for p in m.blocks.parameters())
    print(f"  torso {dict(dts)}; shell stays {m.input_layer.weight.dtype}, "
          f"rope {m.rope_phases.dtype}")

    torch.manual_seed(0)
    # fp32 inputs: the model's shell is fp32 and `manual_cast` narrows on the
    # way into the blocks. Handing it a narrow `x` fails in the input layer.
    args = (torch.randn(1, m.in_channels, *[m.resolution] * 3, device=dev),
            torch.tensor([500.0], device=dev),
            torch.randn(1, 1029, m.cond_channels if hasattr(m, "cond_channels") else 1024,
                        device=dev))
    with torch.no_grad():
        ep = torch.export.export(m, args, strict=False).run_decompositions()
        y = m(*args)
    return write(out, "trellis2_ss", ep, m, args, y)


def export_ssdec(out: Path, dev: str):
    """The conv3d decoder: 16^3 latent -> 64^3 occupancy logits."""
    from trellis2.models import from_pretrained
    m = from_pretrained(SS_DEC).eval().to(dev)
    print(f"trellis2_ssdec: {type(m).__name__}, "
          f"{sum(p.numel() for p in m.parameters()) / 1e6:.1f}M parameters, "
          f"{next(m.parameters()).dtype}")
    torch.manual_seed(0)
    args = (torch.randn(1, 8, 16, 16, 16, device=dev, dtype=next(m.parameters()).dtype),)
    with torch.no_grad():
        ep = torch.export.export(m, args, strict=False).run_decompositions()
        y = m(*args)
    return write(out, "trellis2_ssdec", ep, m, args, y)


def export_cond(out: Path, dev: str):
    """DINOv3 ViT-L/16 -> the 1029x1024 condition the flow model cross-attends to."""
    model = Conditioner(dinov3()).to(dev).eval()
    print(f"trellis2_cond: DINOv3 ViT-L/16 at {DINO_RES}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    torch.manual_seed(0)
    args = (torch.randn(1, 3, DINO_RES, DINO_RES, device=dev),)
    with torch.no_grad():
        ep = torch.export.export(model, args, strict=False).run_decompositions()
        y = model(*args)
    g = write(out, "trellis2_cond", ep, model, args, y)
    (out / "preprocess.json").write_text(json.dumps(
        {"mean": DINO_MEAN, "std": DINO_STD, "size": DINO_RES}, indent=1))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--part", default="ss", choices=["ss", "ssdec", "cond"])
    p.add_argument("--blocks", type=int, default=None,
                   help="--part ss only: export just the first N of the 30 blocks, "
                        "for bisecting a numerical disagreement by depth")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                   help="--part ss only: the torso's dtype. fp32 keeps upstream's "
                        "numbers at 5.2 GB and is for bisecting a disagreement, "
                        "not for shipping")
    a = p.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA. Every attention here goes through "
            "`F.scaled_dot_product_attention`, and on CPU there is no fused "
            "kernel to dispatch to, so `run_decompositions()` replaces it with "
            "an explicit softmax over a materialised matrix.")
    # As `export_hunyuan3d.py` does: TF32's 10-bit mantissa reads as ~2e-4
    # relative error, which is the same order as a real bug, and the reference
    # written here would be a second approximation rather than a reference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    bootstrap()
    default = {"ss": "trellis2-ss", "ssdec": "trellis2-ssdec", "cond": "trellis2-cond"}[a.part]
    outdir = a.out or (GEN / "graphs" / default)
    if a.part == "ss":
        export_ss(outdir, a.device, a.precision, a.blocks)
    elif a.part == "ssdec":
        export_ssdec(outdir, a.device)
    else:
        export_cond(outdir, a.device)
