"""
Export RIFE 4.26's IFNet to DNNKernels' graph JSON + weights.

Same shape as `export_whisper.py`: load the module, name the inputs, export from
CUDA, hand the `ExportedProgram` to `export_graphs.convert`.

    uv run tools/export_rife.py                 # 1080p
    uv run tools/export_rife.py --height 2160 --width 3840

**One graph, both frames in and one frame out.** `RIFE_HDv3.Model.inference`
concatenates the two frames on the channel axis and returns `merged[-1]`; the
flow pyramid and the mask are intermediates the editor never sees, and returning
them would make the graph's output a nested pytree instead of one tensor. The
wrapper below is that call and nothing else.

**`timestep` is an input, not a constant.** Baking 0.5 would give exactly 2x
interpolation and nothing else; passing it as a `(1, 1, 1, 1)` tensor costs one
buffer and is what turns the port into retiming — any `t` in `[0, 1]`, which is
the feature `models-to-port.md` actually asks for ("slow motion, framerate
conversion and retime smoothing"). `IFNet.forward` already branches on
`torch.is_tensor(timestep)` and broadcasts it, so this is upstream's own path.

**Resolution is baked, and has to be.** The graph is static, so one export serves
one frame size. Upstream pads to a multiple of `max(128, 128/scale)` — 128 at the
default scale — so 1080p is exported at 1920x1152 and the caller pads. The scale
pyramid `[16, 8, 4, 2, 1]` is `inference`'s own default.

**The warp is `grid_sampler_2d`, which the runtime already has** — that is why
this model was picked as cheap. `model/warplayer.py` computes the sampling grid
from the flow with `align_corners=True` and `padding_mode='border'`, and it is
imported from upstream rather than reproduced here: the normalisation it applies
(`flow_x / ((W-1)/2)`) is easy to write down slightly wrong and impossible to
notice afterwards, since a half-pixel error looks like a marginally softer frame.

**Two checkouts, for one file.** `gen/rife/train_log` carries the architecture
next to the weights, but `IFNet_HDv3.py` still does `from model.warplayer import
warp`, so upstream has to be on the path as well. `models-to-port.md` says this
model "needs no separate upstream checkout"; that is not quite true, and the
clone is two files' worth of MIT code.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

import export_graphs as EG

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
GEN = ROOT / "gen"
WEIGHTS = GEN / "rife"                       # train_log/ lives here
CHECKOUT = ROOT / "dev" / "Practical-RIFE"   # for model/warplayer.py

# `inference`'s own list at scale=1.0. The loop in `IFNet.forward` runs five
# blocks and indexes this per block, so it has five entries, not four — the
# signature default of `[8, 4, 2, 1]` would raise on the fifth.
SCALE_LIST = [16, 8, 4, 2, 1]

# `tmp = max(128, int(128 / scale))` in `inference_video.py`, with scale = 1.
PAD_TO = 128


def padded(n: int) -> int:
    """Upstream's `((n - 1) // tmp + 1) * tmp`."""
    return ((n - 1) // PAD_TO + 1) * PAD_TO


class Interp(nn.Module):
    """`Model.inference` as a plain callable: two frames and a `t`, one frame out.

    `imgs` is the channel-wise concatenation the flownet expects, `(1, 6, H, W)`
    in 0..1 — the caller does the `cat` so the graph has one image input rather
    than two that it would immediately join.
    """

    def __init__(self, flownet):
        super().__init__()
        self.flownet = flownet

    def forward(self, imgs, timestep):
        _flow, _mask, merged = self.flownet(imgs, timestep, SCALE_LIST)
        return merged[-1]


def load_flownet():
    """`IFNet` with `flownet.pkl` loaded, from the weights dir plus the checkout."""
    if not (WEIGHTS / "train_log" / "flownet.pkl").is_file():
        raise SystemExit(f"no train_log/flownet.pkl under {WEIGHTS} — "
                         "`uv run tools/models.py fetch rife`")
    if not (CHECKOUT / "model" / "warplayer.py").is_file():
        raise SystemExit(
            f"no checkout at {CHECKOUT}\n"
            "  git clone --depth 1 https://github.com/hzwer/Practical-RIFE dev/Practical-RIFE\n"
            "The weights archive carries IFNet_HDv3.py but not the `model.warplayer` "
            "it imports.")

    # `train_log.IFNet_HDv3` and `model.warplayer` are both plain top-level
    # imports, so both roots go on the path. The weights dir first: it is the one
    # carrying the 4.26 architecture, and the checkout has its own older
    # `train_log` that must not win.
    sys.path.insert(0, str(CHECKOUT))
    sys.path.insert(0, str(WEIGHTS))

    from train_log.IFNet_HDv3 import IFNet   # noqa: E402  (needs sys.path first)

    net = IFNet()
    raw = torch.load(WEIGHTS / "train_log" / "flownet.pkl",
                     map_location="cpu", weights_only=True)
    # Trained under DDP, so every key is prefixed. `strict=False` because the
    # checkpoint also carries `teacher` and `caltime`, which `IFNet.__init__`
    # comments out as "not used during inference".
    sd = {k.replace("module.", ""): v for k, v in raw.items() if "module." in k}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"checkpoint is missing {len(missing)} tensors the model "
                         f"needs, first few: {missing[:5]}")
    return net.eval(), len(unexpected)


def main(out: Path, precision: str, dev: str, height: int, width: int):
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — see export_whisper.py's docstring; a CPU "
                         "export is not what the other models here are checked "
                         "against. Pass --device cpu to force it.")

    # TF32 off, for the same reason `dump_sam2_refs.py` turns it off: its 10-bit
    # mantissa reads as ~2e-4 relative error, which is the same order as a real
    # bug in a kernel. `EG.precision_ctx` does not cover this — it only chooses
    # autocast — so it has to be set here or the reference this file writes is
    # not a reference, it is a second approximation.
    #
    # Measured, not assumed: with TF32 left on, the Julia side matched this graph
    # to a mean of 1.1e-5 but a *max* of 7.9e-3 over 0.05% of samples, which reads
    # exactly like a broken warp at a few hundred pixels. It was the reference.
    # 63 convolutions deep, Ampere's tensor cores are the difference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    net, n_unused = load_flownet()
    ph, pw = padded(height), padded(width)
    model = Interp(net).to(dev).eval()

    imgs = torch.rand(1, 6, ph, pw, device=dev)
    timestep = torch.full((1, 1, 1, 1), 0.5, device=dev)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, (imgs, timestep), strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context

    g = EG.convert(ep, ({}, {}), "rife")   # {} = all dims static, one per input
    out.mkdir(parents=True, exist_ok=True)
    (out / "rife.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()),
                            **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    # The reference the Julia side diffs against, from the same process that
    # exported so a mismatch cannot be a different checkpoint. Two structured
    # frames rather than noise: RIFE estimates motion, and between two random
    # fields there is none to estimate, so a broken warp would still score well.
    # `model/warplayer.py` memoises the sampling grid in a module-level dict keyed
    # by `(device, flow.size())` — and tracing populated that key with a
    # FakeTensor. The real forward below hits the same key, reuses the fake grid
    # and hands back a fake output, which then fails at `save_file` with
    # "Cannot access data pointer of Tensor" a hundred lines from the cause.
    # Clearing it is the whole fix. (GUARDRAILS §8 is this bug in our own code: a
    # cache keyed on too little, returning an object that does not belong to the
    # caller. It is worth knowing that upstream ships one too.)
    import model.warplayer as WP   # noqa: E402  (sys.path set in load_flownet)
    WP.backwarp_tenGrid.clear()

    # Fresh tensors, not the ones handed to `torch.export`: those come back
    # functionalized, and `save_file` refuses a tensor whose storage it cannot
    # reach.
    with torch.no_grad():
        ref_t = torch.full((1, 1, 1, 1), 0.5, device=dev)
        yy, xx = torch.meshgrid(torch.arange(ph, device=dev, dtype=torch.float32),
                                torch.arange(pw, device=dev, dtype=torch.float32),
                                indexing="ij")
        def frame(dx, dy):
            r = ((xx + dx) % 64) / 64.0
            gch = ((yy + dy) % 48) / 48.0
            b = (((xx + dx) + (yy + dy)) % 96) / 96.0
            return torch.stack([r, gch, b]).unsqueeze(0)
        ref_in = torch.cat([frame(0.0, 0.0), frame(7.0, 3.0)], 1).contiguous()
        ref_out = model(ref_in, ref_t)
    save_file({"imgs": ref_in.detach().clone().cpu(),
               "timestep": ref_t.detach().clone().cpu(),
               "out": ref_out.detach().clone().cpu()},
              str(out / "reference.safetensors"))

    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    nparam = sum(v.numel() for v in dict(model.named_parameters()).values())
    print(f"rife: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights")
    print(f"  {nparam/1e6:.2f}M parameters, {n_unused} unused tensors in the checkpoint "
          f"(teacher/caltime)")
    print(f"  input {height}x{width} padded to {ph}x{pw}, scales {SCALE_LIST}")
    print("  ops:", json.dumps(hist, sort_keys=True))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "rife")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--width", type=int, default=1920)
    a = p.parse_args()
    main(a.out, a.precision, a.device, a.height, a.width)
