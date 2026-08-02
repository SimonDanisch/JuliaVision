"""
Export BasicVSR++ to DNNKernels' graph JSON + weights.

The MatAnyone exporter (`export_graphs.py`) is a MatAnyone driver wrapped around
one generic core: `convert(ep, specs, name)`, which turns an `ExportedProgram`
into the graph JSON `DNNKernels.loadgraph` reads. That core is model-agnostic, so a
second model needs only its own front end — load the module, name the inputs,
export, hand the result to `convert`.

Keeping this as a separate file rather than generalising `export_graphs.py`
in place is deliberate for now: MatAnyone's exporter carries `patches.apply()`
(bit-exactness fixes) and a `trace.json` type-parameter table that mean nothing
here, and merging them before a second model exists would have been guesswork.
With two real models the shared shape is visible and can be factored properly.

    uv run tools/export_basicvsrpp.py --frames 5 --size 64
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

import export_graphs as EG
from basicvsrpp import load_basicvsrpp

GEN = Path(__file__).resolve().parent.parent / "gen"
CKPT = GEN / "basicvsrpp" / "basicvsr_pp_reds4.pth"


def main(frames: int, size: int, out: Path, precision: str):
    model = load_basicvsrpp(CKPT)
    # (N, T, C, H, W): a clip of low-res frames. Static shapes for the first
    # export — dynamic T and H/W come after the graph runs at all, since a
    # dynamic dim that turns out to be specialised is a silent constraint
    # violation rather than an error.
    args = (torch.randn(1, frames, 3, size, size),)

    with torch.no_grad(), EG.precision_ctx(precision):
        ep = torch.export.export(model.eval(), args, strict=False)
        ep = ep.run_decompositions()   # must stay inside the precision context

    g = EG.convert(ep, ({},), "basicvsrpp")   # {} = all dims static
    out.mkdir(parents=True, exist_ok=True)
    (out / "basicvsrpp.json").write_text(json.dumps(g, indent=1))

    # Weights keyed the way the graph's `:weight` buffers name them.
    tensors = {k: v.detach().contiguous().cpu()
               for k, v in {**dict(model.named_parameters()),
                            **dict(model.named_buffers())}.items()}
    save_file(tensors, str(out / "weights.safetensors"))

    hist = {}
    for o in g["ops"]:
        hist[o["aten"]] = hist.get(o["aten"], 0) + 1
    (out / "op_histogram.json").write_text(json.dumps(hist, indent=1, sort_keys=True))
    nw = sum(1 for b in g["buffers"] if b["kind"] == "weight")
    print(f"basicvsrpp: {len(g['ops'])} ops, {len(g['buffers'])} buffers, {nw} weights, "
          f"{len(tensors)} tensors saved")
    print(f"{len(hist)} distinct ATen ops:")
    for op, n in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {n:5}  {op}")
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--precision", default="fp32", choices=["autocast", "fp32"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out) if a.out else GEN / "graphs" / f"basicvsrpp-{a.precision}"
    main(a.frames, a.size, outdir, a.precision)
