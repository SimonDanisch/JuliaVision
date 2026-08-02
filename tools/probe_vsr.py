"""How many frames of 864x1056 fit through BasicVSR++, and does fp16 hold up?

Isolated from the demo because the demo's OOM had two candidate causes tangled
together (traceback-pinned activations, and a deform-conv buffer that does not
scale with T). Here each is measured on its own.
"""
import sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from basicvsrpp import load_basicvsrpp

H, W = 1056, 864
dev = "cuda"


def build(dtype):
    m = load_basicvsrpp(ROOT / "gen" / "basicvsrpp" / "basicvsr_pp_reds4.pth")
    return m.to(dev).to(dtype).eval()


def try_T(model, T, dtype):
    """Peak VRAM for a T-frame forward, or None on OOM. Frees properly."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(1, T, 3, H, W, device=dev, dtype=dtype)
    oom = False
    try:
        with torch.no_grad():
            y = model(x)
        out = (float(y.float().mean()), bool(torch.isnan(y).any()))
    except torch.OutOfMemoryError as e:
        # THE fix: the traceback holds every frame's locals, so the failed
        # attempt's activations stay alive until it is cleared. Calling
        # empty_cache() before this does nothing.
        traceback.clear_frames(e.__traceback__)
        oom = True
    peak = torch.cuda.max_memory_allocated() / 1e9
    if oom:
        del x
        torch.cuda.empty_cache()
        return None, peak
    del x, y
    torch.cuda.empty_cache()
    return out, peak


if __name__ == "__main__":
    free0 = torch.cuda.mem_get_info()[0] / 1e9
    print(f"free at start: {free0:.1f} GB\n")

    sweep = "--agree-only" not in sys.argv
    for dtype in (torch.float16, torch.float32) if sweep else ():
        print(f"=== {dtype}")
        model = build(dtype)
        for T in (4, 8, 12, 16, 24, 32):
            t0 = time.perf_counter()
            out, peak = try_T(model, T, dtype)
            dt = time.perf_counter() - t0
            if out is None:
                print(f"  T={T:<3} OOM (peak {peak:.1f} GB)")
                break
            mean, nan = out
            print(f"  T={T:<3} peak {peak:5.1f} GB  {dt/T:5.2f}s/frame  "
                  f"mean {mean:.4f}{'  *** NaN ***' if nan else ''}")
        del model
        torch.cuda.empty_cache()
        print()

    # Does fp16 cost anything? Must be at a size where fp32 also fits — it does
    # not at 864x1056, which is the finding above. Half res, and on real frames
    # rather than noise, since fp16 range problems show up on real statistics.
    print("=== fp16 vs fp32 agreement, half res, real frames")
    import subprocess
    h2, w2 = H // 2, W // 2
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", "11", "-t", "0.1",
         "-i", str(ROOT / "media" / "birds_export.mp4"), "-vf", f"scale={w2}:{h2}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8).reshape(-1, h2, w2, 3)[:4]
    x = (torch.from_numpy(a.copy()).permute(0, 3, 1, 2).float() / 255)[None].to(dev)
    with torch.no_grad():
        m = build(torch.float32)
        p = m(x).float().cpu()
        del m
        torch.cuda.empty_cache()
        m = build(torch.float16)
        q = m(x.half()).float().cpu()
    d = (p - q).abs()
    print(f"  mean abs diff {d.mean().item()*255:.4f}/255, "
          f"max {d.max().item()*255:.2f}/255")
