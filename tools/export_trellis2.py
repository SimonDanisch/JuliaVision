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

**No complex numbers in the graphs.** Upstream's rotary multiplies complex
tensors (`view_as_complex`, complex `mul`, `view_as_real`), which DNNKernels
cannot run. `real_rotary` is the same product on cos and sin, exported as the
pair rotation `fusepairrope` turns into one `fused.pairrope`, and the table is a
real (2, L, 64) one: `ss`'s `rope_phases` buffer is swapped for it after the
complex forward is taken as the reference, and the SLat models take it as their
`rope` input.

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
# The sparse models' attention reads SPARSE_ATTN_BACKEND first, falls back to
# ATTN_BACKEND, and accepts only `xformers` and the two `flash_attn`s, so `sdpa`
# leaves it at `flash_attn` — whose extension in TRELLIS.2's venv is built
# against a different torch and fails to import. Only the dense wrapper is ever
# exported; the sparse model runs here as the thing it is checked against.
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

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
            "trellis2_cond": ("image",),
            "trellis2_shape512": ("x", "t", "cond", "rope"),
            "trellis2_shape1024": ("x", "t", "cond", "rope"),
            "trellis2_tex512": ("x", "t", "cond", "rope", "concat_cond"),
            "trellis2_tex1024": ("x", "t", "cond", "rope", "concat_cond")}

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


def patch_dinov3():
    """Give `DinoV3FeatureExtractor` the `extract_features` newer transformers needs.

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
    return ife


def real_rotary(x, phases):
    """`RotaryPositionEmbedder.apply_rotary_embedding` without complex numbers.

    Upstream views each head as P = 64 complex numbers and multiplies them by
    the unit phases, which exports as `view_as_complex`, a complex `mul` and
    `view_as_real`; DNNKernels has no complex dtype to run them on. This is the
    same product on the (cos, sin) pair, in the spelling `fusepairrope` matches
    (Qwen-Image 2.1's `real_rotary`), so each one becomes a `fused.pairrope`.

    `x` is (B, L, H, 2P) and `phases` is (2, L, P) fp32, cos over sin: one
    table, so that torch's `phases[0]` / `phases[1]` are contiguous runs of it
    in Julia's reversed layout and cost no copy.
    """
    cos, sin = phases[0][None, :, None], phases[1][None, :, None]
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out_real = real * cos - imag * sin
    out_imag = imag * cos + real * sin
    return torch.stack((out_real, out_imag), dim=-1).flatten(3).to(x.dtype)


def patch_rotary():
    """Route the dense attention through `real_rotary`. Only the dense
    `RotaryPositionEmbedder`: the sparse models checked against keep theirs."""
    from trellis2.modules.attention.rope import RotaryPositionEmbedder
    RotaryPositionEmbedder.apply_rotary_embedding = staticmethod(real_rotary)


def realphases(phases):
    """A complex (..., L, P) phase table as `real_rotary`'s (2, ..., L, P) one."""
    return torch.stack([phases.real, phases.imag]).contiguous()


def dinov3():
    """The image encoder, with `patch_dinov3` applied."""
    return patch_dinov3().DinoV3FeatureExtractor(DINO, image_size=DINO_RES)


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


def write(out: Path, name: str, ep, model, args, y, specs=None):
    """Graph JSON, weights, op histogram and reference, as every export here.

    `specs` names the symbolic dims per input, as `EG.convert` takes them;
    `None` is every dim static."""
    g = EG.convert(ep, specs or tuple({} for _ in args), name)
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
        want = m(*args)
    # The complex table becomes `real_rotary`'s real one, in place: `forward`
    # hands `self.rope_phases` to the blocks untouched, so nothing else changes.
    patch_rotary()
    m.rope_phases = realphases(m.rope_phases)
    with torch.no_grad():
        y = m(*args)
    mx, mean, c = agreement(y, want)
    print(f"  real vs complex rotary: max {mx:.6f}%  mean {mean:.6f}%  corr {c:.9f}")
    if c < 0.9999:
        raise SystemExit("`real_rotary` does not compute upstream's rotary; not exporting")
    with torch.no_grad():
        ep = torch.export.export(m, args, strict=False).run_decompositions()
    g = write(out, "trellis2_ss", ep, m, args, y)
    write_sampling(out, "ss")
    return g


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


def export_cond(out: Path, dev: str, res: int):
    """DINOv3 ViT-L/16 -> the condition the flow models cross-attend to.

    The pipeline runs it twice: at 512 px (1029 tokens) for the sparse structure
    and the 512 SLat models, at 1024 px (4101 tokens) for the 1024 ones. The
    patch grid is static in the graph, so each resolution is its own export.
    """
    ex = dinov3()
    ex.image_size = res
    model = Conditioner(ex).to(dev).eval()
    print(f"trellis2_cond: DINOv3 ViT-L/16 at {res}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    torch.manual_seed(0)
    args = (torch.randn(1, 3, res, res, device=dev),)
    with torch.no_grad():
        ep = torch.export.export(model, args, strict=False).run_decompositions()
        y = model(*args)
    g = write(out, "trellis2_cond", ep, model, args, y)
    (out / "preprocess.json").write_text(json.dumps(
        {"mean": DINO_MEAN, "std": DINO_STD, "size": res}, indent=1))
    return g


# The four SLat flow models, as `pipeline.json` names them. The 512 pair runs on
# a 32^3 token grid and the 1024 pair on 64^3; each is conditioned on DINOv3 at
# its own resolution, which fixes the condition's token count per graph.
SLAT = {"shape512": ("slat_flow_img2shape_dit_1_3B_512_bf16", 1029),
        "shape1024": ("slat_flow_img2shape_dit_1_3B_1024_bf16", 4101),
        "tex512": ("slat_flow_imgshape2tex_dit_1_3B_512_bf16", 1029),
        "tex1024": ("slat_flow_imgshape2tex_dit_1_3B_1024_bf16", 4101)}


# Each flow part's sampler in upstream's `pipeline.json`, the normalisation its
# output is undone with, and for the texture models the one their `concat_cond`
# (the shape latent) is normalised with on the way in.
STAGES = {"ss": ("sparse_structure_sampler", None, None),
          "shape512": ("shape_slat_sampler", "shape_slat_normalization", None),
          "shape1024": ("shape_slat_sampler", "shape_slat_normalization", None),
          "tex512": ("tex_slat_sampler", "tex_slat_normalization", "shape_slat_normalization"),
          "tex1024": ("tex_slat_sampler", "tex_slat_normalization", "shape_slat_normalization")}


def write_sampling(out: Path, part: str):
    """`sampling.json` beside a flow graph: the checkpoint's sampler settings and
    latent statistics, which the runner cannot derive and reads from here only,
    as `preprocess.json` is for the conditioner."""
    from huggingface_hub import hf_hub_download
    args = json.loads(Path(hf_hub_download(REPO, "pipeline.json")).read_text())["args"]
    sampler, norm, concat = STAGES[part]
    cfg = {"sampler": {**args[sampler]["params"], **args[sampler]["args"]}}
    if norm is not None:
        cfg["normalization"] = args[norm]
    if concat is not None:
        cfg["concat_normalization"] = args[concat]
    (out / "sampling.json").write_text(json.dumps(cfg, indent=1))
    print(f"  sampling.json: {cfg['sampler']}")


class DenseSLatFlow(torch.nn.Module):
    """`SLatFlowModel` over one sample's N voxels as a dense (1, N, C) sequence.

    The sparse model's blocks are `ModulatedSparseTransformerCrossBlock`, and
    with one sample and `attn_mode="full"` every op in them is the dense
    `ModulatedTransformerCrossBlock`'s on a sequence of length N: the norms and
    linears act on `feats` row-wise, the modulation broadcasts over the rows of
    the one batch entry, the attention is a single unmasked sequence, and the
    rotary phases are per row. So the checkpoint's block weights load into dense
    blocks with the same parameter names, and `export_slat` checks the two
    forwards against each other before anything is exported.

    **The rotary table is an input**, `rope` (2, N, 64) fp32, the cos and sin of
    the phases (see `real_rotary`). It depends on the voxel coordinates, which
    only the runner knows, and computing it host-side keeps `cos`/`sin` of
    angles up to ~63 rad out of the graph. `rope_table` below is how it is built.

    The shell (input layer, timestep embedder, shared modulation, output layer)
    stays fp32 and the torso takes `convert_module_to`'s dtype, which is exactly
    `SLatFlowModel.convert_to`'s split.
    """

    def __init__(self, sparse):
        super().__init__()
        from functools import partial
        from trellis2.modules.transformer import ModulatedTransformerCrossBlock
        from trellis2.modules.utils import convert_module_to
        s = sparse
        assert s.pe_mode == "rope" and s.share_mod, "only the configuration TRELLIS.2 ships"
        self.dtype = s.dtype
        self.t_embedder = s.t_embedder
        self.adaLN_modulation = s.adaLN_modulation
        self.input_layer = torch.nn.Linear(s.input_layer.in_features, s.input_layer.out_features)
        self.input_layer.load_state_dict(s.input_layer.state_dict())
        self.blocks = torch.nn.ModuleList([
            ModulatedTransformerCrossBlock(
                s.model_channels, s.cond_channels, num_heads=s.num_heads,
                mlp_ratio=s.mlp_ratio, attn_mode="full", use_rope=True,
                share_mod=s.share_mod, qk_rms_norm=s.qk_rms_norm,
                qk_rms_norm_cross=s.qk_rms_norm_cross)
            for _ in s.blocks])
        # Strict: a parameter the dense block names differently fails here
        # rather than staying at its random init.
        self.blocks.load_state_dict(s.blocks.state_dict())
        self.blocks.apply(partial(convert_module_to, dtype=s.dtype))
        self.out_layer = torch.nn.Linear(s.out_layer.in_features, s.out_layer.out_features)
        self.out_layer.load_state_dict(s.out_layer.state_dict())

    def body(self, x, t, cond, rope):
        """`SLatFlowModel.forward` after `sparse_cat`, on dense tensors."""
        from trellis2.modules.utils import manual_cast
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond, rope)
        h = manual_cast(h, x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        return self.out_layer(h)

    # The parameter names are the graph's input ids, so they match `ARGNAMES`.
    def forward(self, x, t, cond, rope):
        return self.body(x, t, cond, rope)


class DenseSLatFlowConcat(DenseSLatFlow):
    """The texture models: the shape latent is concatenated onto the noise
    channels, `sparse_cat([x, concat_cond], dim=-1)` upstream."""

    def forward(self, x, t, cond, rope, concat_cond):
        return self.body(torch.cat([x, concat_cond], dim=-1), t, cond, rope)


def rope_table(model, coords):
    """The rotary phases the sparse attention builds for `coords`, as (2, N, 64)
    fp32 cos over sin.

    `SparseRotaryPositionEmbedder.forward`'s own computation, through its own
    `_get_phases`: `outer(coords, freqs)` with the three axes' 21 frequencies
    each laid out x-major per voxel, then one unit phase to pad 63 to
    `head_dim / 2 = 64`.
    """
    rope = model.blocks[0].self_attn.rope
    phases = rope._get_phases(coords.reshape(-1)).reshape(*coords.shape[:-1], -1)
    padn = rope.head_dim // 2 - phases.shape[-1]
    if padn > 0:
        phases = torch.cat([phases, torch.polar(
            torch.ones(*phases.shape[:-1], padn, device=phases.device),
            torch.zeros(*phases.shape[:-1], padn, device=phases.device))], dim=-1)
    return realphases(phases)


def sparse_forward(model, coords, x, t, cond, concat_cond=None):
    """The real sparse model on the same inputs, for the check."""
    from trellis2.modules.sparse import SparseTensor
    c = torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1).int()
    kw = {} if concat_cond is None else {"concat_cond": SparseTensor(feats=concat_cond[0], coords=c)}
    return model(SparseTensor(feats=x[0], coords=c), t, cond, **kw).feats[None]


def agreement(got, want):
    """Max and mean |delta| as a percentage of `want`'s range, and the correlation."""
    got, want = got.double().flatten(), want.double().flatten()
    span = (want.max() - want.min()).item()
    d = (got - want).abs()
    c = torch.corrcoef(torch.stack([got, want]))[0, 1].item()
    return 100 * d.max().item() / span, 100 * d.mean().item() / span, c


def export_slat(out: Path, dev: str, which: str, precision: str, n: int,
                blocks: int | None = None):
    """One SLat flow model, over a symbolic voxel count `n`."""
    from trellis2.models import from_pretrained
    ckpt, ntok = SLAT[which]
    m = from_pretrained(f"{REPO}/ckpts/{ckpt}").eval().to(dev)
    if blocks is not None and blocks < len(m.blocks):
        m.blocks = m.blocks[:blocks]     # a plain loop, as in `export_ss`
    # `convert_to`, for the reason `export_ss` gives: the checkpoint's torso is
    # bf16, which DNNKernels cannot read.
    m.convert_to({"fp16": torch.float16, "fp32": torch.float32}[precision])
    concat = which.startswith("tex")
    dense = (DenseSLatFlowConcat if concat else DenseSLatFlow)(m).to(dev).eval()
    from collections import Counter
    dts = Counter(str(p.dtype).removeprefix("torch.") for p in dense.blocks.parameters())
    print(f"trellis2_{which}: {type(m).__name__}, resolution {m.resolution}, "
          f"{m.in_channels} -> {m.out_channels} channels, {len(m.blocks)} blocks, "
          f"{sum(p.numel() for p in dense.parameters()) / 1e9:.3f}B parameters; "
          f"torso {dict(dts)}")

    # `n` distinct voxels of the model's grid, the shape `argwhere` produces.
    g = torch.Generator().manual_seed(0)
    flat = torch.randperm(m.resolution ** 3, generator=g)[:n].sort().values
    r = m.resolution
    coords = torch.stack([flat // (r * r), flat // r % r, flat % r], dim=1).int().to(dev)
    torch.manual_seed(0)
    xch = m.in_channels - (m.out_channels if concat else 0)
    args = (torch.randn(1, n, xch, device=dev),
            torch.tensor([750.0], device=dev),
            torch.randn(1, ntok, m.cond_channels, device=dev),
            rope_table(m, coords))
    if concat:
        args += (torch.randn(1, n, m.out_channels, device=dev),)

    patch_rotary()
    with torch.no_grad():
        y = dense(*args)
        ys = sparse_forward(m, coords, args[0], args[1], args[2],
                            args[4] if concat else None)
    mx, mean, c = agreement(y, ys)
    print(f"  dense vs sparse ({precision}): max {mx:.6f}%  mean {mean:.6f}%  corr {c:.9f}")
    if c < 0.9999:
        raise SystemExit("the dense wrapper does not compute the sparse model; not exporting")
    del m, ys
    torch.cuda.empty_cache()

    from torch.export import Dim
    nd = Dim("n", min=2, max=64 ** 3)
    dynamic = ({1: nd}, {}, {}, {1: nd}) + (({1: nd},) if concat else ())
    specs = ({1: "n"}, {}, {}, {1: "n"}) + (({1: "n"},) if concat else ())
    with torch.no_grad():
        ep = torch.export.export(dense, args, dynamic_shapes=dynamic,
                                 strict=False).run_decompositions()
    g = write(out, f"trellis2_{which}", ep, dense, args, y, specs)
    write_sampling(out, which)
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Resolved at parse time: `bootstrap` chdirs into the checkout.
    p.add_argument("--out", type=lambda s: Path(s).resolve(), default=None)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--part", default="ss", choices=["ss", "ssdec", "cond", *SLAT, "sampling"],
                   help="`sampling` writes only `sampling.json` into every flow part's "
                        "existing export")
    p.add_argument("--res", type=int, default=DINO_RES, choices=[512, 1024],
                   help="--part cond only: DINOv3's input resolution")
    p.add_argument("--blocks", type=int, default=None,
                   help="--part ss and the SLat parts: export just the first N of "
                        "the 30 blocks, for bisecting a numerical disagreement by depth")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"],
                   help="--part ss and the SLat parts: the torso's dtype. fp32 keeps "
                        "upstream's numbers at 5.2 GB and is for bisecting a "
                        "disagreement, not for shipping")
    p.add_argument("--n", type=int, default=2048,
                   help="SLat parts: voxels in the tracing example and reference; "
                        "the graph's `n` is symbolic either way")
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

    if a.part == "sampling":
        for part in STAGES:
            write_sampling(GEN / "graphs" / f"trellis2-{part}", part)
        raise SystemExit(0)
    bootstrap()
    if a.part == "cond":
        default = "trellis2-cond" if a.res == DINO_RES else f"trellis2-cond-{a.res}"
    else:
        default = f"trellis2-{a.part}"
    outdir = a.out or (GEN / "graphs" / default)
    if a.part == "ss":
        export_ss(outdir, a.device, a.precision, a.blocks)
    elif a.part == "ssdec":
        export_ssdec(outdir, a.device)
    elif a.part == "cond":
        export_cond(outdir, a.device, a.res)
    else:
        export_slat(outdir, a.device, a.part, a.precision, a.n, a.blocks)
