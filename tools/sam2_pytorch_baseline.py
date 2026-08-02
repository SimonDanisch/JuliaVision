"""What SAM 2.1 costs in PyTorch on this card, split the way the editor uses it.

    uv run tools/sam2_pytorch_baseline.py --size large

Two numbers, because the editor pays them at different rates: `encode` runs once
when marking starts, `decode` runs on every click. A single "SAM takes 350 ms"
would hide that the interactive half is 40x cheaper than the half it depends on.

Measured on the exported wrappers rather than through `SAM2ImagePredictor`, so
this is the same computation DNNKernels runs — the predictor also resizes,
normalises and post-processes, and timing those here would flatter or punish the
comparison depending on which side does them.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import export_graphs as EG
import export_sam2 as ES

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()


def smclock():
    """SM clock in MHz, read the way `nvidia-smi` reports it.

    Recorded with the result because it is part of it: this card idles at 495 MHz
    of a 3105 MHz maximum, so a benchmark that does not say what clock it ran at
    is not comparable with one that ran at a different clock.
    """
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"min": ts[0], "p50": ts[len(ts) // 2], "mean": sum(ts) / len(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="large", choices=sorted(ES.CKPTS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 (default off, to match the fp32 reference)")
    ap.add_argument("--precision", default="autocast", choices=["autocast", "fp32"],
                    help="must match what the graphs were exported under, or the "
                         "comparison is between two different computations")
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
    with torch.no_grad():
        with EG.precision_ctx(a.precision):
            feats = enc(image)
        nfpn = len(feats) // 2
        f0, f1, f2 = feats[0], feats[1], feats[nfpn - 1]
        point = torch.full((1, ES.MAXPOINTS, 2), res / 2.0, device=dev)
        label = torch.full((1, ES.MAXPOINTS), -1, dtype=torch.int32, device=dev)
        label[0, 0] = 1

        # Same precision policy the graphs carry, and a warm-up long enough to
        # bring the SM clock up: this card idles at 495 MHz of 3105 and a short
        # benchmark measures the ramp rather than the kernel.
        with torch.no_grad(), EG.precision_ctx(a.precision):
            for _ in range(200):
                enc(image)
            torch.cuda.synchronize()
            e = bench(lambda: enc(image), a.warmup, a.iters)
            d = bench(lambda: dec(f0, f1, f2, point, label), a.warmup, a.iters)

    out = {"size": a.size, "res": res, "tf32": a.tf32, "precision": a.precision,
           "sm_clock_mhz": smclock(),
           "device": torch.cuda.get_device_name(0),
           "encode_ms": e, "decode_ms": d}
    print(json.dumps(out, indent=1))
    (ROOT / "gen" / "graphs" / f"sam2-{a.size}" / "pytorch_baseline.json").write_text(
        json.dumps(out, indent=1))
    print(f"\nencode {e['p50']:.1f} ms   decode {d['p50']:.2f} ms   (p50, TF32 "
          f"{'on' if a.tf32 else 'off'})")


if __name__ == "__main__":
    main()
