"""
PyTorch's own time for one Depth Anything V2 Small forward, on this machine.

    uv run tools/baseline_depthanything.py

The other half of `tools/bench_depthanything.jl`. `models-to-port.md` sets this
model's target at "≥ PyTorch", and a target like that is only meaningful against
a number measured the same way on the same card — GUARDRAILS §5, do not rank
against an unmeasured denominator.

Measured to match the Julia side as closely as the two runtimes allow:

  * the same input, read from `reference.safetensors`, not fresh noise;
  * the same precision policy — fp32 with **TF32 off**. Leaving TF32 on would
    hand PyTorch Ampere's tensor cores for every convolution and matmul while
    Lava runs fp32, which measures a dtype choice and calls it an engine gap;
  * `torch.cuda.synchronize` around a batch of iterations rather than each one,
    so the sync is not what is being timed;
  * median of the same number of samples.

**No `torch.compile`.** The graph Lava runs came out of `torch.export`, which is
eager PyTorch's own decomposition — comparing against a fused Inductor build
would be comparing against a different program. Worth measuring separately if the
question is "what could this card do", which is not the question this file asks.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
GEN = ROOT / "gen"


def main(size: int, n: int, reps: int):
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA — there is nothing to compare against here")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    sys.path.insert(0, str(Path(__file__).parent))
    from export_depthanything import Depth, load_model

    d = GEN / "graphs" / "depthanything"
    ref = load_file(str(d / "reference.safetensors"))
    x = ref["input"].cuda()

    model = Depth(load_model()).cuda().eval()

    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()

        ts = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3 / reps)

    t = statistics.median(ts)
    # `torch.cuda.clock_rate` needs nvidia-ml-py, which is not a dependency here.
    # The clock is worth printing (GUARDRAILS §6) but not worth a hard failure.
    try:
        clk = torch.cuda.clock_rate()
    except Exception:
        clk = None
    print()
    print(f"PyTorch eager, fp32, TF32 off — THIS MACHINE ({torch.cuda.get_device_name(0)})")
    print(f"  one depth map            {t:8.2f} ms")
    print(f"  implied rate             {1000 / t:8.2f} fps")
    print(f"  input                    {tuple(x.shape)}")
    if clk:
        print(f"  SM clock                 {clk} MHz")
    print("compare against `julia --project=. tools/bench_depthanything.jl`")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=518)
    p.add_argument("-n", type=int, default=9, help="samples")
    p.add_argument("--reps", type=int, default=3, help="iterations per sample")
    a = p.parse_args()
    main(a.size, a.n, a.reps)
