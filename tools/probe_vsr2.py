"""Max sequence length and speed for the two compression-aware models.

Same purpose as probe_vsr.py, for the models added after it: the NTIRE
decompression net (c128n25, 1x) and RealBasicVSR (4x). Their memory profiles are
nothing like BasicVSR++/REDS4's, so the chunk size has to be measured, not
assumed:

  decompress   propagates at H/4 (is_low_res_input=False) -> cheap per frame
  RealBasicVSR cleans at FULL res across all T at once    -> expensive per frame
"""
import sys, time, traceback
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from basicvsrpp import load_basicvsrpp, load_realbasicvsr

H, W, dev = 1056, 864, "cuda"
CK = ROOT / "gen" / "basicvsrpp"


def probe(name, make, Ts, dtype=torch.float16):
    print(f"=== {name} ({dtype})")
    model = make().to(dev).to(dtype).eval()
    for T in Ts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.rand(1, T, 3, H, W, device=dev, dtype=dtype)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                y = model(x)
            dt = time.perf_counter() - t0
            bad = bool(torch.isnan(y).any() or torch.isinf(y).any())
            print(f"  T={T:<3} peak {torch.cuda.max_memory_allocated()/1e9:5.1f} GB  "
                  f"{dt/T:5.2f}s/frame  out {tuple(y.shape[-2:])}"
                  f"{'  *** NaN/Inf ***' if bad else ''}")
            del y
        except torch.OutOfMemoryError as e:
            traceback.clear_frames(e.__traceback__)
            print(f"  T={T:<3} OOM")
            del x
            torch.cuda.empty_cache()
            break
        del x
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    print(f"free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB\n")
    probe("NTIRE decompress c128n25 (1x)",
          lambda: load_basicvsrpp(CK / "decompress_track1.pth", 128, 25,
                                  is_low_res_input=False),
          [4, 8, 16, 24, 32])
    probe("RealBasicVSR (4x)",
          lambda: load_realbasicvsr(CK / "realbasicvsr.pth"),
          [2, 4, 6, 8, 12])
