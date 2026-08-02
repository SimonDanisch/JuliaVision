"""
Export one Wan 2.2 `WanAttentionBlock` to DNNKernels' graph JSON + weights.

The block, not the whole model, and that is deliberate. Exporting `WanModel`
fails with `GuardOnDataDependentSymNode` — its forward pads a list of
variable-length latents up to `seq_len`, which export cannot trace. The block is
163.7M parameters and the model stacks 30 of them, so it is essentially the whole
compute; getting it running proves the op coverage, and the outer padding can
then be replaced with a static wrapper.

Two things about the model code, both of which fail confusingly:

  * `attention.py` dispatches to flash-attn even when flash-attn is absent, and
    flash-attn asserts `q.device.type == 'cuda'`. The replacement has to be
    installed on the *attention module* before `model.py` imports the names from
    it — patching only `model.attention` leaves `flash_attention` reachable.
  * `WanAttentionBlock.__init__` takes `(dim, ffn_dim, num_heads, ...)`; the
    model's own construction passes a `cross_attn_type` string first, which the
    block does not accept.

    uv run tools/export_wandit.py --seq 192 --ctx 512
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import export_graphs as EG

ROOT = Path(__file__).resolve().parent.parent
MODS = ROOT / "dev" / "Wan2.2" / "wan" / "modules"
CFG = ROOT / "gen" / "wan22" / "config.json"


def sdpa_attention(q, k, v, **kw):
    """Wan's attention contract on `scaled_dot_product_attention`.

    q/k/v arrive as `[B, L, Nh, C]` (flash-attn's layout); sdpa wants the heads
    on axis 1.
    """
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2))
    return o.transpose(1, 2).contiguous()


def load_modules():
    pkg = types.ModuleType("wanmods")
    pkg.__path__ = [str(MODS)]
    sys.modules["wanmods"] = pkg

    def lp(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        m.__package__ = "wanmods"
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    att = lp("wanmods.attention", MODS / "attention.py")
    att.attention = sdpa_attention          # before model.py binds the names
    att.flash_attention = sdpa_attention
    mdl = lp("wanmods.model", MODS / "model.py")
    mdl.attention = sdpa_attention
    if hasattr(mdl, "flash_attention"):
        mdl.flash_attention = sdpa_attention
    return mdl


class Block(torch.nn.Module):
    """The block with its non-tensor arguments bound, so export sees three inputs."""

    def __init__(self, blk, seq_lens, grid, freqs, ctx_lens):
        super().__init__()
        self.blk = blk
        # Plain attributes, NOT `register_buffer`: registering makes export trace
        # them, and the block unpacks `grid` into loop bounds — which then reads
        # as data-dependent control flow and fails with
        # `GuardOnDataDependentSymNode`. As constants they are folded away.
        self.seq_lens = seq_lens
        self.grid = grid
        self.freqs = freqs
        self.ctx_lens = ctx_lens

    def forward(self, x, e, context):
        return self.blk(x, e, self.seq_lens, self.grid, self.freqs, context, self.ctx_lens)


def build(seq: int, ctxlen: int, grid=(3, 8, 8)):
    mdl = load_modules()
    cfg = json.loads(CFG.read_text())
    dim, nh, ffn = cfg["dim"], cfg["num_heads"], cfg["ffn_dim"]
    torch.manual_seed(0)
    blk = mdl.WanAttentionBlock(dim, ffn, nh, (-1, -1), True, True, cfg["eps"]).eval()
    freqs = torch.view_as_complex(torch.randn(1024, dim // nh // 2, 2))
    w = Block(blk, torch.tensor([seq]), torch.tensor([list(grid)]), freqs,
              torch.tensor([ctxlen])).eval()
    args = (torch.randn(1, seq, dim), torch.randn(1, 6, dim), torch.randn(1, ctxlen, dim))
    return w, args


def main(seq: int, ctxlen: int, out: Path, precision: str):
    w, args = build(seq, ctxlen)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(w, args, strict=False).run_decompositions()

    g = EG.convert(ep, ({}, {}, {}), "wandit_block")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wandit_block.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(w.named_parameters()), **dict(w.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    with torch.no_grad():
        y = w(*args)
    # export lifts `freqs` and `grid` to graph inputs, so the reference has to
    # carry them too. safetensors has no complex dtype — store the real view and
    # let the reader reinterpret, which is the same pairing `view_as_complex`
    # uses.
    save_file({"x": args[0].contiguous(), "e": args[1].contiguous(),
               "context": args[2].contiguous(), "out": y.contiguous(),
               "freqs_real": torch.view_as_real(w.freqs).contiguous(),
               "grid": w.grid.contiguous()},
              str(out / "refs.safetensors"))

    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"wandit_block: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights")
    print(f"  reference {tuple(args[0].shape)} -> {tuple(y.shape)}")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out) if a.out else ROOT / "gen" / "graphs" / f"wandit-{a.precision}"
    main(a.seq, a.ctx, outdir, a.precision)


SHARDS = ROOT / "gen" / "wan22"


def load_trained(model, layers: int, dtype=None):
    """Fill `model` from the shipped checkpoint, keeping only the first `layers`
    blocks.

    The shards are fp32 and total 18.6 GB, so they are read tensor by tensor
    through safetensors' mmap and cast on the way in: with `dtype=torch.float16`
    peak memory is the *model* (9.8 GB at 30 layers), not the checkpoint.

    Returns the number of tensors that were loaded — 0 means the shards are not
    on this machine, which is a fair thing to run without but not a fair thing to
    call "Wan": a randomly-initialised transformer is numerically checkable and
    produces nothing meaningful.
    """
    from safetensors import safe_open

    index = SHARDS / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        return 0
    wmap = json.loads(index.read_text())["weight_map"]
    want = dict(model.state_dict())
    got, byfile = 0, {}
    for k in want:
        f = wmap.get(k)
        f is None or byfile.setdefault(f, []).append(k)
    with torch.no_grad():
        for f, keys in byfile.items():
            with safe_open(str(SHARDS / f), framework="pt") as h:
                for k in keys:
                    t = h.get_tensor(k)
                    want[k].copy_(t if dtype is None else t.to(dtype))
                    got += 1
    missing = [k for k in want if k not in wmap]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters are not in the checkpoint, "
                           f"e.g. {missing[:3]} — the config does not match the shards")
    return got


def build_full(frames=3, size=16, ctxlen=512, layers=2, trained=True):
    """The whole model through `StaticWan`, with `layers` blocks.

    Fewer than the shipped 30 by default: the op set and the per-op correctness
    are identical (the blocks are the same module repeated), and 30 blocks of a
    5B model does not fit alongside a PyTorch reference on this machine.

    `trained` loads the real weights for the blocks that are kept. It changes
    nothing about the *graph* — but a graph run with random weights only proves
    the arithmetic, so it is the default.
    """
    from wan_static import StaticWan
    mdl = load_modules()
    cfg = json.loads(CFG.read_text())
    kw = {k: v for k, v in cfg.items() if not k.startswith("_")}
    kw["num_layers"] = layers
    torch.manual_seed(0)
    m = mdl.WanModel(**kw).eval()
    n = load_trained(m, layers) if trained else 0
    print(f"  weights: {'checkpoint, ' + str(n) + ' tensors' if n else 'RANDOM (no checkpoint)'}")
    if not n:
        # `WanModel.init_weights` zero-inits the output projection, as diffusion
        # models do — so a randomly-initialised model emits exactly zeros and any
        # comparison against it passes vacuously. Give the head real weights.
        with torch.no_grad():
            m.head.head.weight.normal_(0, 0.02)
            m.head.head.bias.normal_(0, 0.02)
    p = m.patch_size
    grid = (frames // p[0], size // p[1], size // p[2])
    seq = grid[0] * grid[1] * grid[2]
    sw = StaticWan(m, grid, seq).eval()
    args = (torch.randn(kw["in_dim"], frames, size, size),
            torch.tensor([500.0]), torch.randn(ctxlen, kw.get("text_dim", 4096)))
    return sw, args


def main_full(out: Path, precision="fp32", **kw):
    sw, args = build_full(**kw)
    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(sw, args, strict=False).run_decompositions()
    g = EG.convert(ep, ({}, {}, {}), "wan_dit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wan_dit.json").write_text(json.dumps(g, indent=1))
    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(sw.named_parameters()), **dict(sw.named_buffers())}.items()}
    # `freqs` is complex and safetensors has no complex dtype
    tensors = {k: (torch.view_as_real(v) if v.is_complex() else v) for k, v in tensors.items()}
    # `_grid_sizes`/`_seq_lens` are plain attributes (deliberately — see
    # wan_static) so they are not in `named_buffers`, but export lifts them to
    # graph inputs. Save them under the names the converter gives them.
    tensors["_grid_sizes"] = sw._grid_sizes.contiguous()
    tensors["_seq_lens"] = sw._seq_lens.contiguous()
    # `WanModel.freqs` is a plain attribute too, and it is complex
    tensors["m_freqs"] = torch.view_as_real(sw.m.freqs).contiguous()
    save_file(tensors, str(out / "weights.safetensors"))
    with torch.no_grad():
        y = sw(*args)
    save_file({"x": args[0].contiguous(), "t": args[1].contiguous(),
               "context": args[2].contiguous(), "out": y.contiguous()},
              str(out / "refs.safetensors"))
    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"wan_dit: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights, "
          f"out {tuple(y.shape)}")
    return g
