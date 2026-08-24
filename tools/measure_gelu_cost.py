"""What running the wrong `gelu` cost TRELLIS.2's sparse-structure model.

DNNKernels read the approximation flag as `arg1` and the exporter files it under
`approximate`, so every tanh gelu in the tree ran torch's *exact* formulation
instead. This measures the size of that substitution on the real weights and the
real inputs, in PyTorch, on the CPU — the same quantity DNNKernels' divergence
against the reference was made of, with nothing else changed.

    ~/tools/TRELLIS.2/.venv/bin/python tools/measure_gelu_cost.py --blocks 3

`x`, `t` and `cond` come out of the export's own `reference.safetensors`, so this
is the same forward pass the Julia side is checked against.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The checkpoint id, the checkout and the sys.path/cwd dance all come from the
# exporter rather than being restated here: a second copy of a model id is a
# second thing to get wrong, and getting it wrong is a 404 rather than a
# mismatch. Importing it also sets ATTN_BACKEND before `trellis2` loads.
from export_trellis2 import GEN, SS_FLOW, bootstrap

import torch
import torch.nn as nn
from safetensors.torch import load_file

GRAPHS = GEN / "graphs"


def report(label, a, b):
    """Both "% of range" figures, because the phrase names two of them.

    `verify_depthanything.jl` prints max|Δ| and mean|Δ| against the same span and
    calls both "of range", so a single number here would be comparable to the
    Julia-side measurements only by luck.
    """
    d = (a - b).abs()
    span = (b.max() - b.min()).item()
    corr = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
    print(f"{label:34s} max {100 * d.max().item() / span:.6f}% of range, "
          f"mean {100 * d.mean().item() / span:.6f}%, "
          f"rel l2 {(a - b).norm().item() / b.norm().item():.3e}, "
          f"corr {corr:.9f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--refdir", default=None,
                   help="defaults to the trellis2-ss[-bN]-fp32 export")
    a = p.parse_args()

    bootstrap()
    from trellis2.models import from_pretrained

    ref = Path(a.refdir) if a.refdir else GRAPHS / (
        "trellis2-ss-fp32" if a.blocks >= 30 else f"trellis2-ss-b{a.blocks}-fp32")
    t = load_file(ref / "reference.safetensors")
    print(f"reference: {ref.name}")

    m = from_pretrained(SS_FLOW).eval()
    m.blocks = m.blocks[:a.blocks]
    m.convert_to(torch.float32)

    args = (t["x"].float(), t["t"].float(), t["cond"].float())
    with torch.no_grad():
        y_tanh = m(*args)

    # The reference this file exists to explain. Reproducing it first is what
    # says the CPU forward is the same forward the export ran on the GPU.
    report("tanh gelu vs the reference:", y_tanh, t["out"])

    # Exactly the substitution the runtime was making: same weights, same
    # inputs, torch's default gelu in place of the tanh one.
    n = 0
    for mod in m.modules():
        if isinstance(mod, nn.GELU) and mod.approximate == "tanh":
            mod.approximate = "none"
            n += 1
    print(f"switched {n} GELU modules to the exact form")
    with torch.no_grad():
        y_exact = m(*args)

    report("exact gelu vs tanh gelu:", y_exact, y_tanh)


if __name__ == "__main__":
    main()
