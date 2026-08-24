"""
Export the Hunyuan3D-2.1 shape pipeline to DNNKernels' graph JSON + weights.

    uv run tools/export_hunyuan3d.py                  # the 21-block denoiser
    uv run tools/export_hunyuan3d.py --part cond      # DINOv2-large
    uv run tools/export_hunyuan3d.py --part vae       # post_kl + decoder
    uv run tools/export_hunyuan3d.py --part geo       # one occupancy chunk
    uv run tools/export_hunyuan3d.py --part moe       # one MoEBlock, CPU-safe
    uv run tools/export_hunyuan3d.py --blocks 3       # a bring-up slice
    uv run tools/export_hunyuan3d.py --precision fp32 # for a numerics bisect

Same shape as `export_wandit.py`: reach into an upstream checkout, bind the
non-tensor arguments on a wrapper module, export from CUDA, hand the
`ExportedProgram` to `export_graphs.convert`.

**One checkpoint, three models.** `model.fp16.ckpt` is 7.37 GB and is a dict of
three state dicts: `model` (the DiT, 3.051B), `vae` (0.328B) and `conditioner`
(DINOv2-large, 0.304B). They are four separate graphs rather than one because
they run at wildly different rates: the conditioner once per generation, the
denoiser fifty times, the VAE once, and the geometry decoder some seven thousand
times over the grid chunks.

**A wrapper's parameter names are the graph's input ids.** `torch.export` takes
them from the placeholders, so `forward(self, image)` and `ARGNAMES` have to
agree or the Julia side asks for a buffer that is not there. That is the one
thing to check when adding a part.

**What the config actually says**, which is not what the architecture defaults
say: `use_attention_pooling` is **False**, so there is no `AttentionPool` and no
`F.multi_head_attention_forward` in the graph; `use_pos_emb` is False, so the
4096 latents are an unordered *set*; `with_decoupled_ca` is False; and
`qkv_bias` is False. MoE is on blocks 15-20 (`depth - layer <= num_moe_layers`)
and the U-Net skip connections on blocks 11-20.

**The MoE is rewritten to a dense form, and that is the one deviation.**
`MoEBlock.moe_infer` cannot be exported and it is not close: it calls
`.cpu().numpy()` on the expert histogram, loops over experts in Python with
data-dependent bounds, `continue`s on empty ones, and gathers a dynamically-sized
slice per expert. None of that is a graph. `dense_moe_forward` below computes the
same function

    y[t] = sum_j  w[t,j] * E_{idx[t,j]}(x[t])         (upstream, top-2 of 8)
    y[t] = sum_e  m[t,e] * E_e(x[t])                  (here, all 8)

with `m` the top-k weights scattered into a (tokens, num_experts) matrix, so the
six non-selected experts contribute exactly `0 * E_e(x)`. Equal in exact
arithmetic; equal in floating point too unless a non-selected expert overflows,
which `--check-moe` tests against the real `moe_infer` before exporting anything.

It costs 4x the routed FLOPs on 6 of 21 blocks. That is the price of a correct
static graph, and replacing it with capacity-based routing is the optimisation
that comes after the port is numerically right, not before.

**fp16, not fp32.** The checkpoint is fp16 and the pipeline runs fp16, so an
fp32 export would be a reference for a model nobody runs, and 3.051B parameters
in fp32 is 12.2 GB before activations on a 20 GB card. `--precision fp32` exists
for bisecting a numerical disagreement on a `--blocks` slice, not for the
shipping export.

Upstream: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
License: **TENCENT HUNYUAN NON-COMMERCIAL** — like MatAnyone, this one cannot
ship in a commercial build.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

import export_graphs as EG

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
GEN = ROOT / "gen"

# The upstream checkout and the HuggingFace-side cache the official downloader
# fills. Neither is fetched by `tools/models.py` yet — the checkpoint is behind a
# gated repo, so it arrives through Tencent's own downloader.
CHECKOUT = Path("/home/simon/tools/Hunyuan3D-2.1/hy3dshape")
CKPT_DIR = Path("/home/simon/.cache/hy3dgen/tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1")


# --------------------------------------------------------------- upstream import
#
# `hy3dshape/__init__.py` imports the pipelines, which import `trimesh`; and
# `hy3dshape/models/__init__.py` reaches `surface_extractors`, which imports
# `skimage`. Neither is on the path this export needs — the denoiser is a plain
# `nn.Module` — so the parent packages are registered as bare module objects
# carrying only `__path__`. Submodule imports resolve through that and the
# `__init__` bodies never run, while relative imports inside the modules
# (`from ...utils import logger`) still work because `hy3dshape.utils` is a real
# subpackage that imports cleanly.
#
# `timm` is a genuine dependency of this import chain and has to be installed
# (`uv pip install --no-deps timm`). `moe_layers.py` imports three names from it
# and uses none of them, so a stub module looked like the smaller change; it is
# not, because `diffusers` feature-detects timm through `importlib.util.find_spec`
# and that rejects a module with no `__spec__`.

def _shell(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    m.__package__ = name
    sys.modules[name] = m
    return m


def _bootstrap():
    pkg = CHECKOUT / "hy3dshape"
    if not pkg.is_dir():
        raise SystemExit(
            f"no hy3dshape package at {pkg}\n"
            "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1")
    _shell("hy3dshape", pkg)
    _shell("hy3dshape.models", pkg / "models")
    _shell("hy3dshape.models.denoisers", pkg / "models" / "denoisers")
    _shell("hy3dshape.models.autoencoders", pkg / "models" / "autoencoders")
    return pkg


def upstream():
    """`(hunyuandit, moe_layers)` from the checkout, no package `__init__` run."""
    _bootstrap()
    import hy3dshape.models.denoisers.hunyuandit as dit
    import hy3dshape.models.denoisers.moe_layers as moe
    return dit, moe


def upstream_vae():
    """`hy3dshape.models.autoencoders.model`, which owns `ShapeVAE`.

    Reaches `surface_extractors` and through it `skimage`, so unlike the denoiser
    this one has a real third-party dependency (`uv pip install scikit-image`).
    Shelling that module out instead would break `model.py`'s
    `from .surface_extractors import MCSurfaceExtractor` — a shell has a search
    path and no attributes — and marching cubes is wanted here anyway, as the
    reference the Julia side gets compared against.
    """
    _bootstrap()
    import hy3dshape.models.autoencoders.model as vae
    return vae


def upstream_conditioner():
    """`hy3dshape.models.conditioner`, which owns `SingleImageEncoder`."""
    _bootstrap()
    import hy3dshape.models.conditioner as cond
    return cond


# -------------------------------------------------------------------- dense MoE

def dense_moe_forward(self, hidden_states):
    """`MoEBlock.forward` with `moe_infer` replaced by an all-experts sum.

    Upstream routes each token to its top-2 of 8 experts by sorting the flat
    expert assignment, slicing a variable-length run per expert and scattering
    the results back. This runs every expert on every token and weights by a
    scattered top-k mask, which is the same function with static shapes.

    The gate is untouched: `self.gate` still produces `topk_idx`/`topk_weight`
    exactly as upstream, so any disagreement is in the combination, not the
    routing.
    """
    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, _ = self.gate(hidden_states)

    flat = hidden_states.view(-1, orig_shape[-1])            # (T, D)
    n_experts = len(self.experts)

    # (T, E): the routing weight where the expert was selected, 0 elsewhere.
    # `scatter`, not `one_hot @ w` — the top-k indices are distinct, so this is
    # exactly the mask upstream applies through `flat_expert_weights[idxs[...]]`.
    mask = torch.zeros(flat.shape[0], n_experts,
                       dtype=topk_weight.dtype, device=flat.device)
    mask = mask.scatter(1, topk_idx, topk_weight)

    y = torch.zeros_like(flat)
    for e, expert in enumerate(self.experts):
        y = y + mask[:, e:e + 1] * expert(flat)

    y = y.view(*orig_shape)
    return y + self.shared_experts(identity)


MOE_REFUSAL = (
    "the dense MoE rewrite disagrees with upstream's `moe_infer` beyond "
    "tolerance — see `dense_moe_forward`. Export refused: a graph that computes "
    "a different function than the model is worse than no graph.")


def check_moe_module(moe, moe_mod, device, dtype, tokens=64, tol=1e-3):
    """Run one `MoEBlock` both ways on the same input and report the gap.

    The dense rewrite is exact in real arithmetic; what this checks is the
    floating-point claim, that `0 * E_e(x)` is 0 for the six experts a token did
    not select. It is not if any of them overflows fp16, and an 8192-wide GELU
    MLP on an out-of-distribution input is where that would happen.

    Reads `moe_mod.MoEBlock.forward` live, so it must be called BEFORE the
    monkeypatch — otherwise it compares the rewrite against itself and passes
    for free.
    """
    upstream_forward = moe_mod.MoEBlock.forward
    if upstream_forward is dense_moe_forward:
        raise RuntimeError("check_moe_module called after the monkeypatch — "
                           "it would be comparing the rewrite against itself")
    torch.manual_seed(0)
    x = torch.randn(1, tokens, moe.gate.gating_dim, device=device, dtype=dtype)
    with torch.no_grad():
        ref = upstream_forward(moe, x)
        got = dense_moe_forward(moe, x)
    d = (ref.float() - got.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    rel = d / max(scale, 1e-12)
    ok = rel <= tol
    print(f"  check-moe: absmax {d:.3e}  rel {rel:.3e}  (|ref| {scale:.3f})"
          f"{'' if ok else '   <-- OVER TOLERANCE'}")
    return ok


# ------------------------------------------------------------------ the wrapper

class Denoiser(nn.Module):
    """`HunYuanDiTPlain.forward` with the condition dict flattened to a tensor.

    Upstream takes `contexts` as a `{'main': tensor}` dict and `**kwargs`.
    torch.export can carry a dict, but every consumer downstream then has to know
    the key; three positional tensors is the interface the graph should have.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, cond):
        return self.net(x, t, {"main": cond})


class MoEOnly(nn.Module):
    """One `MoEBlock`, lifted out of the denoiser it belongs to.

    Its own export target for two reasons. The op set the mixture adds — the
    top-k, the scatter, the gate softmax — is the part of this model DNNKernels
    has never seen, and it is worth knowing separately from the 21-block graph
    that takes 8 GB to trace. And a `MoEBlock` contains no attention at all, so
    unlike the full denoiser it exports correctly from the CPU: there is no
    `scaled_dot_product_attention` to decompose.
    """

    def __init__(self, moe):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        return self.moe(x)


class Conditioner(nn.Module):
    """DINOv2-large's transformer, with the host-side preprocessing left out.

    Upstream's `ImageEncoder.forward` rescales from `(-1, 1)` to `(0, 1)`, then
    runs a torchvision `Resize`/`CenterCrop`/`Normalize` before the model sees
    anything. All three are host work on a fixed square, and baking them into the
    graph would fix one source resolution — the same call `export_depthanything.py`
    makes and for the same reason. What this graph takes is the normalised
    518x518 tensor; `MEAN`/`STD` below are the only place the statistics live.

    Returns `last_hidden_state` rather than the `ModelOutput` wrapper: 1370
    tokens of 1024, which is 37x37 patches plus the class token.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    # The parameter name is the graph's input id — `torch.export` takes it from
    # the placeholder — so it has to match the key `ARGNAMES` writes the
    # reference under, or the loader asks for a buffer the graph does not have.
    def forward(self, image):
        return self.enc.model(image).last_hidden_state


class ShapeLatents(nn.Module):
    """`ShapeVAE.forward` — `post_kl` then the 16-layer decoder transformer.

    Holds the two submodules rather than the whole VAE so the weights file is
    what the graph reads. The encoder is 8 more layers that only ever run when
    fitting a latent to an existing mesh, which is not this pipeline.
    """

    def __init__(self, vae):
        super().__init__()
        self.post_kl = vae.post_kl
        self.transformer = vae.transformer

    def forward(self, latents):
        return self.transformer(self.post_kl(latents))


class GeoDecoder(nn.Module):
    """The occupancy field: query points cross-attending to the shape latents.

    This is where the generation's time actually goes. The pipeline evaluates it
    over a dense `(octree_resolution + 1)^3` grid — 57.1M points at the default
    384 — in chunks, so the exported graph is ONE chunk and the caller loops. The
    chunk size is a static shape and therefore an export parameter, not a runtime
    one.

    `fourier_embedder` is shared with the VAE's encoder upstream; taking it
    through `geo_decoder` keeps its frequency buffer in this graph's weights
    under the name the decoder reaches it by.
    """

    def __init__(self, vae):
        super().__init__()
        self.geo_decoder = vae.geo_decoder

    def forward(self, queries, latents):
        return self.geo_decoder(queries=queries, latents=latents)


def build(blocks: int | None, dtype):
    """The denoiser on the CPU, weights loaded. The caller moves what it needs."""
    dit_mod, moe_mod = upstream()
    params = load_config()["model"]["params"]

    net = dit_mod.HunYuanDiTPlain(**params)
    # strict: a silent mismatch here is a wrong model that still runs.
    net.load_state_dict(load_ckpt()["model"])

    if blocks is not None and blocks < len(net.blocks):
        # Keep the first `blocks` and tell the forward about it, so the
        # push/pop of the U-Net skip list stays balanced. Blocks past the
        # halfway point of the *slice* will pop a skip value that their module
        # has no `skip_linear` for; the forward discards it, which is why this
        # truncates without restructuring anything.
        net.blocks = net.blocks[:blocks]
        net.depth = blocks

    return net.to(dtype).eval(), params, moe_mod


def inputs(params, batch, device, dtype):
    """One classifier-free-guidance pair, at the shapes the pipeline calls with.

    Row 0 is the image condition and row 1 the unconditional one.
    `DinoImageEncoder.unconditional_embedding` makes that row exactly zero rather
    than a learned null embedding, so the reference is built that way too.
    """
    torch.manual_seed(0)
    x = torch.randn(batch, params["input_size"], params["in_channels"],
                    device=device, dtype=dtype)
    # The pipeline hands the model `t / num_train_timesteps`, i.e. a 0..1 scalar
    # broadcast over the batch.
    t = torch.full((batch,), 0.5, device=device, dtype=dtype)
    cond = torch.randn(batch, params["text_len"], params["context_dim"],
                       device=device, dtype=dtype)
    if batch > 1:
        cond[1] = 0
    return x, t, cond


def write(out: Path, name: str, ep, model, args, y):
    """Graph JSON, weights, op histogram and reference, the way every export here
    writes them."""
    g = EG.convert(ep, tuple({} for _ in args), name)   # {} = all dims static
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


ARGNAMES = {"hunyuan3d_dit": ("x", "t", "cond"), "hunyuan3d_moe": ("x",),
            "hunyuan3d_cond": ("image",), "hunyuan3d_vae": ("latents",),
            "hunyuan3d_geo": ("queries", "latents")}

# From upstream's `DinoImageEncoder.mean`/`.std` — the ImageNet statistics its
# `transform` applies before the model sees a frame. Restated here because the
# export deliberately leaves the preprocessing out (see `Conditioner`), so this
# is the only place the Julia side can read them from, and getting them wrong
# produces a plausible mesh rather than an error.
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def load_ckpt():
    """The one file that holds `model`, `vae` and `conditioner`.

    `mmap=True`, and it is not a micro-optimisation: read into RAM the checkpoint
    is 7.37 GB and the denoiser it fills is another 6.1, and holding both at once
    is what the host OOM killer took this process out for on a 31 GB box with one
    other job running. Mapped, the pages are file-backed and evictable, so the
    peak is the model alone.
    """
    p = CKPT_DIR / "model.fp16.ckpt"
    if not p.is_file():
        raise SystemExit(
            f"no {p} — accept the licence, then "
            "`huggingface-cli download tencent/Hunyuan3D-2.1`")
    return torch.load(p, map_location="cpu", weights_only=True, mmap=True)


def load_config():
    import yaml
    return yaml.safe_load((CKPT_DIR / "config.yaml").read_text())


def export_conditioner(out: Path, precision: str, dev: str):
    """DINOv2-large -> the 1370x1024 condition the denoiser cross-attends to."""
    cond_mod = upstream_conditioner()
    config = load_config()
    params = config["conditioner"]["params"]

    dtype = torch.float16 if precision == "fp16" else torch.float32
    enc = cond_mod.SingleImageEncoder(**params)
    enc.load_state_dict(load_ckpt()["conditioner"])
    model = Conditioner(enc.main_image_encoder).to(device=dev, dtype=dtype).eval()

    res = params["main_image_encoder"]["kwargs"]["image_size"]
    patch = params["main_image_encoder"]["kwargs"]["config"]["patch_size"]
    print(f"hunyuan3d_cond: DINOv2-large, {res}x{res} = "
          f"{(res//patch)**2}+1 tokens, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, {precision}")

    torch.manual_seed(0)
    args = (torch.randn(1, 3, res, res, device=dev, dtype=dtype),)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False).run_decompositions()
    with torch.no_grad():
        y = model(*args)
    g = write(out, "hunyuan3d_cond", ep, model, args, y)
    (out / "preprocess.json").write_text(json.dumps(
        {"mean": DINO_MEAN, "std": DINO_STD, "size": res,
         "value_range": [-1, 1]}, indent=1))
    return g


def build_vae(dtype):
    vae_mod = upstream_vae()
    params = load_config()["vae"]["params"]
    vae = vae_mod.ShapeVAE(**params)
    # `strict=False`, as `pipelines.py:from_single_file` does. The checkpoint
    # carries the encoder's own extras and the module tree does not have all of
    # them; loading strictly here fails on a model the reference loads fine.
    missing, unexpected = vae.load_state_dict(load_ckpt()["vae"], strict=False)
    if missing:
        print(f"  vae: {len(missing)} missing keys, e.g. {missing[:3]}")
    if unexpected:
        print(f"  vae: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return vae.to(dtype).eval(), params


def export_vae(out: Path, precision: str, dev: str):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    vae, params = build_vae(dtype)
    model = ShapeLatents(vae).to(device=dev).eval()
    print(f"hunyuan3d_vae: {params['num_decoder_layers']} layers, "
          f"width {params['width']}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, {precision}")
    print(f"  scale_factor = {params['scale_factor']}  "
          "(the pipeline divides the sampled latents by this before decoding)")

    torch.manual_seed(0)
    args = (torch.randn(1, params["num_latents"], params["embed_dim"],
                        device=dev, dtype=dtype),)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False).run_decompositions()
    with torch.no_grad():
        y = model(*args)
    g = write(out, "hunyuan3d_vae", ep, model, args, y)
    (out / "scale_factor.json").write_text(
        json.dumps({"scale_factor": params["scale_factor"]}, indent=1))
    return g


def export_geo(out: Path, precision: str, dev: str, chunk: int):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    vae, params = build_vae(dtype)
    model = GeoDecoder(vae).to(device=dev).eval()
    print(f"hunyuan3d_geo: one chunk of {chunk} query points against "
          f"{params['num_latents']} latents, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, {precision}")

    torch.manual_seed(0)
    # Query points in the box the pipeline samples, `box_v = 1.01`.
    args = (torch.rand(1, chunk, 3, device=dev, dtype=dtype) * 2.02 - 1.01,
            torch.randn(1, params["num_latents"], params["width"],
                        device=dev, dtype=dtype))
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False).run_decompositions()
    with torch.no_grad():
        y = model(*args)
    return write(out, "hunyuan3d_geo", ep, model, args, y)


def export_moe(out: Path, precision: str, dev: str, tokens: int, do_check: bool):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    net, params, moe_mod = build(None, dtype)

    src = [b for b in net.blocks if b.use_moe]
    if not src:
        raise SystemExit("no MoE blocks in this model")
    # The last block's mixture. Which one is arbitrary — all six are the same
    # module with different weights — but naming it beats "some MoE block".
    model = MoEOnly(src[-1].moe).to(device=dev, dtype=dtype).eval()
    print(f"hunyuan3d_moe: block {len(net.blocks)-1}, "
          f"{len(model.moe.experts)} experts + 1 shared, top-{model.moe.moe_top_k}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters, {precision}")
    del net

    if do_check and not check_moe_module(model.moe, moe_mod, dev, dtype, tokens):
        raise SystemExit(MOE_REFUSAL)
    moe_mod.MoEBlock.forward = dense_moe_forward

    torch.manual_seed(0)
    args = (torch.randn(1, tokens, params["hidden_size"], device=dev, dtype=dtype),)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False).run_decompositions()
    with torch.no_grad():
        y = model(*args)
    return write(out, "hunyuan3d_moe", ep, model, args, y)


def main(out: Path, precision: str, dev: str, blocks: int | None, batch: int,
         do_check: bool):
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA. Every attention in this model goes through "
            "`F.scaled_dot_product_attention`, and on CPU there is no fused "
            "kernel to dispatch to, so `run_decompositions()` replaces it with "
            "an explicit softmax over a materialised (4097 x 4097) matrix. Pass "
            "--device cpu only to inspect that decomposition.")

    # TF32 off, as `dump_sam2_refs.py` does: its 10-bit mantissa reads as ~2e-4
    # relative error, the same order as a real bug, and the reference written
    # below would be a second approximation rather than a reference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    dtype = torch.float16 if precision == "fp16" else torch.float32
    net, params, moe_mod = build(blocks, dtype)
    n_moe = sum(1 for b in net.blocks if b.use_moe)
    print(f"hunyuan3d_dit: {len(net.blocks)} blocks ({n_moe} MoE), "
          f"{sum(p.numel() for p in net.parameters())/1e9:.3f}B parameters, {precision}")

    model = Denoiser(net).to(device=dev).eval()

    if do_check:
        ok = all(check_moe_module(b.moe, moe_mod, dev, dtype, 64)
                 for b in model.net.blocks if b.use_moe)
        if not ok:
            raise SystemExit(MOE_REFUSAL)
    # Installed on the CLASS, after the check, so the check compares against the
    # real implementation.
    moe_mod.MoEBlock.forward = dense_moe_forward

    args = inputs(params, batch, dev, dtype)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context
    with torch.no_grad():
        y = model(*args)
    return write(out, "hunyuan3d_dit", ep, model, args, y)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--blocks", type=int, default=None,
                   help="export only the first N of the 21 blocks (bring-up)")
    p.add_argument("--batch", type=int, default=2,
                   help="2 is the classifier-free-guidance pair the pipeline uses")
    p.add_argument("--no-check-moe", action="store_true",
                   help="skip the dense-vs-upstream MoE comparison")
    p.add_argument("--part", default="dit",
                   choices=["dit", "moe", "cond", "vae", "geo"],
                   help="dit: the 21-block denoiser (default). moe: one MoEBlock "
                        "on its own — it has no attention in it, so unlike the "
                        "rest this one exports correctly from the CPU. cond: the "
                        "DINOv2-large image encoder. vae: post_kl + the decoder "
                        "transformer. geo: one chunk of the occupancy field.")
    p.add_argument("--tokens", type=int, default=256,
                   help="--part moe sequence length; the op set does not depend on it")
    p.add_argument("--chunk", type=int, default=8000,
                   help="--part geo query points per call, the pipeline's num_chunks. "
                        "A static shape, so it is an export parameter")
    a = p.parse_args()
    default = {"dit": "hunyuan3d" if a.blocks is None else f"hunyuan3d-b{a.blocks}",
               "moe": "hunyuan3d-moe", "cond": "hunyuan3d-cond",
               "vae": "hunyuan3d-vae", "geo": "hunyuan3d-geo"}[a.part]
    outdir = a.out or (GEN / "graphs" / default)
    if a.part == "moe":
        export_moe(outdir, a.precision, a.device, a.tokens, not a.no_check_moe)
    elif a.part == "cond":
        export_conditioner(outdir, a.precision, a.device)
    elif a.part == "vae":
        export_vae(outdir, a.precision, a.device)
    elif a.part == "geo":
        export_geo(outdir, a.precision, a.device, a.chunk)
    else:
        main(outdir, a.precision, a.device, a.blocks, a.batch, not a.no_check_moe)
