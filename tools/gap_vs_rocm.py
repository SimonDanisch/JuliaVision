"""
The vendor half of `gap_vs_rocm.jl`: times the shapes that file states, on the
same GPU, and prints the ratio.

    julia --project=. tools/gap_vs_rocm.jl
    tmp/baseline/preload.sh tools/gap_vs_rocm.py

`preload.sh` is not optional here. The PyTorch ROCm wheels bundle a ROCm
runtime that segfaults at the first kernel launch on gfx1151 while the system
one is fine; see `plans/2026-09-21-rocm-baseline.md`.

Shapes come from the JSON so there is one place they are written down. A row
this file cannot run is reported as such rather than skipped, because a missing
row reads like a passing one.
"""
import json, os, sys, time
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = os.path.join(HERE, "..", "tmp", "baseline", "ours.json")
dev = "cuda"


def best(fn, n=15, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = float("inf")
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t = min(t, (time.perf_counter() - t0) * 1e3)
    return t


def run(row):
    p, kind = row["params"], row["kind"]
    if kind == "attention":
        E, Lq, Lk, H = p["E"], p["Lq"], p["Lk"], p["H"]
        q = torch.randn(1, H, Lq, E, device=dev, dtype=torch.float16) * 0.3
        k = torch.randn(1, H, Lk, E, device=dev, dtype=torch.float16) * 0.3
        v = torch.randn(1, H, Lk, E, device=dev, dtype=torch.float16) * 0.3
        return best(lambda: F.scaled_dot_product_attention(q, k, v))
    if kind == "conv3x3":
        ci, co, h, w = p["Cin"], p["Cout"], p["H"], p["W"]
        x = torch.randn(1, ci, h, w, device=dev, dtype=torch.float16) * 0.3
        kk = torch.randn(co, ci, 3, 3, device=dev, dtype=torch.float16) * 0.05
        return best(lambda: F.conv2d(x, kk, padding=1), n=8, warm=3)
    if kind == "gemm-fp16":
        m, n, kd = p["M"], p["N"], p["K"]
        a = torch.randn(m, kd, device=dev, dtype=torch.float16) * 0.3
        b = torch.randn(kd, n, device=dev, dtype=torch.float16) * 0.3
        return best(lambda: torch.mm(a, b), n=8, warm=3)
    if kind == "copy":
        dt = torch.float16 if p["eltype"] == "Float16" else torch.float32
        n = p["bytes"] // dt.itemsize
        a = torch.empty(n, device=dev, dtype=dt)
        b = torch.empty(n, device=dev, dtype=dt)
        return best(lambda: b.copy_(a))
    return None


def main():
    if not os.path.exists(OURS):
        sys.exit(f"no {OURS}; run tools/gap_vs_rocm.jl first")
    rows = json.load(open(OURS))["rows"]
    torch.backends.cudnn.benchmark = True
    print(f"{'kind':10} {'name':22} {'ours ms':>9} {'torch ms':>9} {'gap':>7} {'ours TF':>8} {'torch TF':>9}")
    worst = []
    for r in rows:
        t = run(r)
        if t is None:
            print(f"{r['kind']:10} {r['name']:22} {'':>9} {'unsupported here':>9}")
            continue
        gap = r["ms"] / t
        fl = r["flops"]
        otf = f"{fl / (r['ms'] * 1e-3) / 1e12:.2f}" if fl else "-"
        ttf = f"{fl / (t * 1e-3) / 1e12:.2f}" if fl else "-"
        print(f"{r['kind']:10} {r['name']:22} {r['ms']:9.2f} {t:9.2f} {gap:6.2f}x {otf:>8} {ttf:>9}")
        worst.append((gap, r["kind"], r["name"]))
        torch.cuda.empty_cache()
    worst.sort(reverse=True)
    print("\nwidest gaps:")
    for g, k, n in worst[:5]:
        print(f"  {g:5.2f}x  {k} {n}")


main()
