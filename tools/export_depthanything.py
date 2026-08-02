"""
Export Depth Anything V2 Small to DNNKernels' graph JSON + weights.

Same shape as `export_whisper.py`: load the module, name the inputs, export from
CUDA, hand the `ExportedProgram` to `export_graphs.convert`.

    uv run tools/export_depthanything.py
    uv run tools/export_depthanything.py --size top    # 518 -> a bigger square

**Small specifically.** The repo is Apache-2.0 and so is the Small checkpoint,
but Base and Large are CC-BY-NC-4.0 — checked on the HuggingFace API rather than
assumed (`models-to-port.md`). Large is better and can be added later as its own
non-commercial package; this one can ship.

**Export from CUDA.** The DINOv2 encoder is twelve blocks of attention, and
`export_whisper.py`'s finding applies directly: on CPU there is no fused kernel
to dispatch to, so `run_decompositions()` decomposes attention into an explicit
softmax over a materialised matrix. From CUDA it stays
`_scaled_dot_product_*`, which DNNKernels already implements.

**The input side is not in the graph, and that is a decision.** Upstream's
`image2tensor` resizes the long side to 518 keeping aspect, rounds to a multiple
of 14 (the ViT patch size), and normalises by the ImageNet statistics — all with
OpenCV on the host. Baking a resize into the graph would fix one frame size; the
normalisation is folded into the caller instead, where `GPUFiltering` already
does the resize for every other model here. What the graph takes is the
normalised square tensor and what it returns is the depth map at 1/14 scale
upsampled by the head, which the caller resamples to the frame.

**Square, not aspect-preserving.** Upstream keeps aspect and pads to a multiple
of 14, so the tensor shape depends on the clip. A static graph cannot, so this
exports the square case (`518x518`, `37x37` patches) — the editor scales the
frame into it and the depth back out. Non-square support is a second export, not
a runtime switch.
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
WEIGHTS = GEN / "depthanything"
CHECKOUT = ROOT / "dev" / "Depth-Anything-V2"

PATCH = 14   # DINOv2's patch size; every input side must be a multiple of it

# Only the Apache-2.0 checkpoint. `vitb`/`vitl` are CC-BY-NC-4.0 and are
# deliberately not reachable from here — adding them is a licensing decision, not
# a flag.
CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}

# ImageNet statistics, from upstream's `NormalizeImage` in `image2tensor`. Kept
# next to the export so the Julia side has one place to read them from.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Depth(nn.Module):
    """`DepthAnythingV2.forward` with the batch axis kept.

    Upstream ends in `depth.squeeze(1)`, which makes the output `(1, H, W)` — a
    rank-3 tensor whose leading axis is the batch, easy to mistake for a channel
    on the Julia side. Keeping `(1, 1, H, W)` makes the layout the same as every
    other image tensor in this runtime.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x).unsqueeze(1)


def load_model():
    if not CHECKOUT.is_dir():
        raise SystemExit(
            f"no checkout at {CHECKOUT}\n"
            "  git clone https://github.com/DepthAnything/Depth-Anything-V2 "
            "dev/Depth-Anything-V2")
    ckpt = WEIGHTS / "depth_anything_v2_vits.pth"
    if not ckpt.is_file():
        raise SystemExit(f"no {ckpt} — `uv run tools/models.py fetch depthanything`")

    sys.path.insert(0, str(CHECKOUT))
    from depth_anything_v2.dpt import DepthAnythingV2   # noqa: E402

    net = DepthAnythingV2(**CONFIG)
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return net.eval()


def main(out: Path, precision: str, dev: str, size: int):
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — see the module docstring; a CPU export "
                         "decomposes attention. Pass --device cpu to force it.")
    if size % PATCH:
        raise SystemExit(f"--size must be a multiple of {PATCH} (DINOv2's patch), got {size}")

    # TF32 off, as `dump_sam2_refs.py` does: its 10-bit mantissa reads as ~2e-4
    # relative error, the same order as a real bug. `EG.precision_ctx` only
    # chooses autocast and does not cover this, so the reference written below
    # would otherwise be a second approximation rather than a reference.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = Depth(load_model()).to(dev).eval()
    args = (torch.randn(1, 3, size, size, device=dev),)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context

    g = EG.convert(ep, ({},), "depthanything")   # {} = all dims static
    out.mkdir(parents=True, exist_ok=True)
    (out / "depthanything.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()),
                            **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    # The reference the Julia side diffs against, from the same process that
    # exported. A structured input rather than noise: depth is a smooth function
    # of the scene, and on white noise the model has nothing to key on, so a
    # broken block would still land in the same narrow range as a working one.
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.linspace(0, 1, size, device=dev),
                                torch.linspace(0, 1, size, device=dev),
                                indexing="ij")
        # Something with near and far structure: a ramp, a disc and a corner.
        disc = ((xx - 0.35) ** 2 + (yy - 0.6) ** 2 < 0.04).float()
        raw = torch.stack([yy, 0.5 * xx + 0.5 * disc, (xx + yy) / 2]).unsqueeze(0)
        mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
        ref_in = ((raw - mean) / std).contiguous()
        ref_out = model(ref_in)
    save_file({"input": ref_in.detach().clone().cpu(),
               "depth": ref_out.detach().clone().cpu()},
              str(out / "reference.safetensors"))

    nparam = sum(v.numel() for v in dict(model.named_parameters()).values())
    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"depthanything: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights")
    print(f"  vits, {nparam/1e6:.1f}M parameters, input (1, 3, {size}, {size}) "
          f"= {size//PATCH}x{size//PATCH} patches")
    print(f"  depth range on the reference: {ref_out.min():.3f} .. {ref_out.max():.3f}")
    print("  ops:", json.dumps(hist, sort_keys=True))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "depthanything")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--size", type=int, default=518,
                   help=f"square input side, a multiple of {PATCH}")
    a = p.parse_args()
    main(a.out, a.precision, a.device, a.size)
