"""Where PyTorch's 87.6 ms of SAM 2.1 encode actually goes, per kernel.

    uv run tools/sam2_pytorch_kernels.py --size large

`sam2_pytorch_baseline.py` answers "how long does encode take". This answers
"doing what", which is the only form of the number that says what to work on.
Written because the alternative — guessing which of our kernels is behind, then
tuning it and measuring whether the total moved — costs a full run per guess and
attributes nothing.

Pair it with `DNNKernels.OPTIMES`, which produces the same table for our side, and
`tools/sam2_kernel_table.jl`, which joins the two.

The aten names are torch's own, so they line up with the op names in the
exported graphs and therefore with `OPTIMES` keys — a `runop!` method is
dispatched on exactly that string. Where torch fuses several aten ops into one
CUDA kernel (an epilogue into a GEMM, say) the *kernel* time lands on whichever
aten op the profiler attributes it to; that is a real difference in what the two
runtimes do and worth seeing rather than hiding.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import export_graphs as EG
import export_sam2 as ES

ROOT = Path(__file__).resolve().parent.parent


def smclock():
    """SM clock in MHz — part of the result, not context. See the baseline."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def kerneltimes(fn, iters):
    """`{aten op: (calls, total device ms)}` over `iters` calls of `fn`.

    Device time, not wall: CUDA is asynchronous and the CPU-side duration of a
    launch says nothing about the kernel. `key_averages` sums the ranges the
    profiler recorded on the device, which is the number that compares with a
    serialising `OPTIMES` run.
    """
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()

    # Two levels come back and they are not additive: an `aten::addmm` range and
    # the `ampere_fp16_s1688gemm…` kernel it launched both carry the same device
    # time, so summing everything double-counts. Split them: `aten` is what
    # joins against our `OPTIMES` keys, `kernel` is what actually ran and what
    # sums to the wall time.
    aten, kernel = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0.0)
        dev = float(dev)
        if dev <= 0:
            continue
        name = e.key
        # A CUPTI buffer-overflow marker, not work. Left in, it was 26% of the
        # table and the totals did not reconcile with the baseline.
        if name.startswith("Command Buffer"):
            continue
        if name.startswith("aten::"):
            # `aten::addmm` -> `addmm.default`, matching the exported graph's op
            # names and therefore `DNNKernels.OPTIMES`.
            rec = aten[name[len("aten::"):] + ".default"]
        else:
            rec = kernel[name]
        rec[0] += e.count
        rec[1] += dev / 1e3 / iters      # us total over iters -> ms per call
    fix = lambda d: {k: (v[0] // iters, v[1]) for k, v in d.items()}
    return fix(aten), fix(kernel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"])
    a = ap.parse_args()

    torch.backends.cudnn.allow_tf32 = a.tf32
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.set_float32_matmul_precision("high" if a.tf32 else "highest")

    dev = "cuda"
    model = ES.build(a.size).to(dev)
    res = model.image_size
    enc = ES.Encoder(model).to(dev).eval()
    dec = ES.Decoder(model).to(dev).eval()

    image = torch.rand(1, 3, res, res, device=dev)
    with torch.no_grad(), EG.precision_ctx(a.precision):
        feats = enc(image)
        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        point = torch.full((1, ES.MAXPOINTS, 2), res / 2.0, device=dev)
        label = torch.full((1, ES.MAXPOINTS), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1

        # Same long warm-up as the baseline: this card idles at 495 MHz of 3105
        # and a short run measures the clock ramp.
        for _ in range(a.warmup):
            enc(image)
        torch.cuda.synchronize()

        e_aten, e_kern = kerneltimes(lambda: enc(image), a.iters)
        d_aten, d_kern = kerneltimes(lambda: dec(f0, f1, f2, point, label), a.iters)

    out = {"size": a.size, "res": res, "precision": a.precision, "tf32": a.tf32,
           "sm_clock_mhz": smclock(), "device": torch.cuda.get_device_name(0),
           "encode_kernels_ms": e_aten, "decode_kernels_ms": d_aten,
           "encode_cuda_ms": e_kern, "decode_cuda_ms": d_kern}
    dest = ROOT / "gen" / "graphs" / f"sam2-{a.size}" / "pytorch_kernels.json"
    dest.write_text(json.dumps(out, indent=1))

    def table(title, d, width=44):
        tot = sum(v[1] for v in d.values())
        print(f"\n{title}: {tot:.1f} ms across {len(d)} entries "
              f"@ {out['sm_clock_mhz']} MHz\n")
        print(f"{'':<{width}}{'calls':>7}{'ms':>9}{'share':>8}")
        for k, (n, ms) in sorted(d.items(), key=lambda kv: -kv[1][1])[:18]:
            print(f"{k[:width - 1]:<{width}}{n:>7}{ms:>9.2f}{100 * ms / tot:>7.1f}%")

    # The aten table is what joins with ours; the CUDA table is what actually
    # ran, and it is the one whose total should reconcile with the baseline.
    table("encode, by aten op", e_aten)
    table("encode, by CUDA kernel", e_kern, width=64)
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()
