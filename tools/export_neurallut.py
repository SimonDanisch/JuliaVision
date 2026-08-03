"""
Export Image-Adaptive 3D LUT's predictor to DNNKernels' graph JSON + weights.

Same shape as `export_whisper.py`: load the module, name the inputs, export from
CUDA, hand the `ExportedProgram` to `export_graphs.convert`.

    uv run tools/export_neurallut.py

**What is in the graph and what is not.** Upstream's `generator()` is three
steps:

    pred = classifier(img).squeeze()                       # 3 blend weights
    LUT  = pred[0]*LUT0 + pred[1]*LUT1 + pred[2]*LUT2      # (3, 33, 33, 33)
    out  = trilinear(LUT, img)                             # a CUDA extension

The first two are the graph — a small CNN and a weighted sum of three basis
tables. **The third is deliberately not**, and `models-to-port.md` says why: the
apply is a trilinear fetch into a 33³ table and belongs in `GPUFiltering`, beside
the other per-pixel image kernels, not as a graph op. Keeping it out is also what
makes the graph resolution-independent — the tensor that leaves it is the LUT,
which is 3x33x33x33 whatever the frame size is.

**The classifier's `Upsample` is stripped for the same reason.** `model[0]` is
`nn.Upsample(size=(256,256), mode='bilinear')`, so the CNN only ever sees 256x256
no matter what it is handed. Exporting from a 256x256 input drops one op and
keeps the graph free of any full-resolution shape; the editor does the 4K -> 256
reduction with `GPUFiltering`'s resize, which it needs anyway for the thumbnail.
The consequence for parity is stated rather than hidden: the Julia and PyTorch
sides are compared on the *same* 256x256 classifier input, and the resize is
checked separately.

**Export from CUDA.** Same rule as every other model here; see
`export_whisper.py`. This graph has no attention, so the decomposition
difference that bites Whisper does not apply — but the exporter refuses a silent
CPU export and this one does not need to be the exception.
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
CHECKOUT = ROOT / "dev" / "Image-Adaptive-3DLUT"

# 33 is the paper's table size and what every pretrained checkpoint here carries.
DIM = 33
RES = 256   # what the classifier's own Upsample resizes to


class LUTPredictor(nn.Module):
    """Image -> the blended 3D LUT, as one graph.

    `luts` is registered as a buffer rather than kept as three modules: the
    basis tables are constants at inference and stacking them turns the blend
    into one broadcast multiply and one reduction, which is three ops instead of
    five and is the same arithmetic.
    """

    def __init__(self, classifier, luts):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("luts", luts)   # (3 basis, 3 ch, D, D, D)

    def forward(self, img):
        # (1, 3, 1, 1) -> (3,). `reshape` rather than `squeeze` so the graph does
        # not carry a shape-dependent squeeze.
        pred = self.classifier(img).reshape(3)
        return (pred.reshape(3, 1, 1, 1, 1) * self.luts).sum(0)


def load_upstream(color_space: str, variant: str):
    """The upstream `Classifier` and the three basis LUTs from the checkout."""
    if not CHECKOUT.is_dir():
        raise SystemExit(
            f"no checkout at {CHECKOUT}\n"
            "  git clone https://github.com/HuiZeng/Image-Adaptive-3DLUT dev/Image-Adaptive-3DLUT\n"
            "This is the one model whose weights are not at a fetchable URL; "
            "`tools/models.py fetch` does not cover it (models-to-port.md).")
    sys.path.insert(0, str(CHECKOUT))

    # `models_x` does `import trilinear` at module scope — upstream's compiled
    # CUDA extension, built by `trilinear_cpp/setup.py`. It is only ever touched
    # inside `TrilinearInterpolationFunction`, which is exactly the piece that
    # does not go in the graph, so a stub gets `Classifier` importable without
    # an nvcc build. If anything ever calls through it, it raises rather than
    # returning something wrong.
    if "trilinear" not in sys.modules:
        stub = types.ModuleType("trilinear")

        def _refuse(*_a, **_k):
            raise RuntimeError(
                "upstream's `trilinear` extension is stubbed out: the LUT apply "
                "is a GPUFiltering kernel, not a graph op (see the module "
                "docstring). Nothing in the export path should reach this.")

        stub.forward = _refuse
        stub.backward = _refuse
        sys.modules["trilinear"] = stub

    import models_x as M   # noqa: E402  (needs the checkout on sys.path first)

    d = CHECKOUT / "pretrained_models" / color_space
    suffix = "_unpaired" if variant == "unpaired" else ""
    cls_ckpt = d / f"classifier{suffix}.pth"
    lut_ckpt = d / f"LUTs{suffix}.pth"
    for p in (cls_ckpt, lut_ckpt):
        if not p.is_file():
            raise SystemExit(f"missing {p}")

    classifier = (M.Classifier_unpaired() if variant == "unpaired" else M.Classifier())
    classifier.load_state_dict(torch.load(cls_ckpt, map_location="cpu", weights_only=True))

    # The checkpoint is {"0": {"LUT": ...}, "1": ..., "2": ...}; the keys are
    # strings and are not necessarily in order in the file.
    raw = torch.load(lut_ckpt, map_location="cpu", weights_only=True)
    luts = torch.stack([raw[str(i)]["LUT"] for i in range(len(raw))])   # (n, 3, D, D, D)
    if luts.shape[1:] != (3, DIM, DIM, DIM):
        raise SystemExit(f"unexpected LUT shape {tuple(luts.shape)}")
    return classifier, luts


def main(out: Path, precision: str, dev: str, color_space: str, variant: str):
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — see export_whisper.py's docstring; a CPU "
                         "export is not what the other models here are checked "
                         "against. Pass --device cpu to force it.")

    # TF32 off, as `dump_sam2_refs.py` does: its 10-bit mantissa reads as ~2e-4
    # relative error, the same order as a real bug. `EG.precision_ctx` only
    # chooses autocast and does not cover this, so the reference written below
    # would otherwise be a second approximation rather than a reference. Worth
    # 4x here (1.3e-5 -> 3.4e-6 on the LUT) and worth 24x on RIFE, which is 63
    # convolutions deep instead of 6.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    classifier, luts = load_upstream(color_space, variant)
    if luts.shape[0] != 3:
        raise SystemExit(f"{luts.shape[0]} basis LUTs; the blend below assumes 3")

    # Drop the leading Upsample — see the module docstring. Asserted rather than
    # sliced blind, so a checkpoint whose Sequential is shaped differently fails
    # here instead of silently exporting a resize into the graph.
    head = classifier.model[0]
    if not isinstance(head, nn.Upsample):
        raise SystemExit(f"expected model[0] to be nn.Upsample, got {type(head).__name__}")
    classifier.model = nn.Sequential(*list(classifier.model)[1:])

    model = LUTPredictor(classifier, luts).eval().to(dev)
    args = (torch.rand(1, 3, RES, RES, device=dev),)   # images are 0..1

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model, args, strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context

    g = EG.convert(ep, ({},), "neurallut")   # {} = all dims static
    out.mkdir(parents=True, exist_ok=True)
    (out / "neurallut.json").write_text(json.dumps(g, indent=1))

    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()),
                            **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))

    # The reference the Julia side diffs against: one fixed input, its predicted
    # weights and its blended LUT. Saved from the same process that exported, so
    # a mismatch cannot be a different checkpoint.
    with torch.no_grad():
        torch.manual_seed(0)
        ref_in = torch.rand(1, 3, RES, RES, device=dev)
        ref_lut = model(ref_in)
        ref_pred = model.classifier(ref_in).reshape(3)
    save_file({"input": ref_in.cpu(), "lut": ref_lut.cpu(), "pred": ref_pred.cpu()},
              str(out / "reference.safetensors"))

    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"neurallut: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights")
    print(f"  {color_space}/{variant}, {luts.shape[0]} basis LUTs at {DIM}^3, "
          f"classifier input {RES}x{RES}")
    print(f"  predicted weights on the reference input: {ref_pred.tolist()}")
    print("  ops:", json.dumps(hist, sort_keys=True))
    return g


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=GEN / "graphs" / "neurallut")
    p.add_argument("--precision", default="fp32")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--color-space", default="sRGB", choices=["sRGB", "XYZ"])
    p.add_argument("--variant", default="paired", choices=["paired", "unpaired"])
    a = p.parse_args()
    main(a.out, a.precision, a.device, a.color_space, a.variant)
