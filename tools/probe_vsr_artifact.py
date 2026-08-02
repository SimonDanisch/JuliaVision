"""Is the dotted texture in the 4x output my fp16 conversion, or the model?

Distinguishable: run the SAME native-resolution pixels through fp32 and fp16.
fp32 does not fit on a full 864x1056 frame, but it fits on a spatial crop — and
a crop, unlike a downscale, preserves the exact pixel statistics (H.264 blocking,
sensor noise, lens softness) that the full run saw.

    fp16 only  -> my bug
    both       -> the model on real compressed footage
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from basicvsrpp import load_basicvsrpp

CLIP = ROOT / "media" / "birds_export.mp4"
X, Y, CW, CH, T = 200, 500, 432, 528, 8
dev = "cuda"


def native_crop():
    """T frames at native resolution, cropped — never resized."""
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", "12.0", "-i", str(CLIP),
         "-frames:v", str(T), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8).reshape(-1, 1056, 864, 3)[:T]
    return a[:, Y:Y + CH, X:X + CW].copy()


def run(frames, dtype):
    m = load_basicvsrpp(ROOT / "gen" / "basicvsrpp" / "basicvsr_pp_reds4.pth")
    m = m.to(dev).to(dtype).eval()
    x = torch.from_numpy(frames).permute(0, 3, 1, 2)
    x = (x.float() / 255.0)[None].to(dev).to(dtype)
    with torch.no_grad():
        y = m(x)[0].clamp_(0, 1).float().cpu()
    del m, x
    torch.cuda.empty_cache()
    return (y.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def grid_energy(img, flat_box):
    """Power at the 1/2 and 1/4 pixel-shuffle frequencies in a FLAT patch.

    A pixelshuffle checkerboard is periodic with the upsample factor, so it
    shows as isolated spikes at the Nyquist/2 and Nyquist/4 bins. Real detail
    does not concentrate there.
    """
    x0, y0, s = flat_box
    p = np.asarray(Image.fromarray(img[y0:y0 + s, x0:x0 + s]).convert("L"), np.float32)
    p = p - p.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(p)))
    c = s // 2
    tot = F.sum() + 1e-9
    # the four (+-s/4, 0), (0, +-s/4) bins = period-4 pattern; likewise period-2
    p4 = F[c, c + s // 4] + F[c, c - s // 4] + F[c + s // 4, c] + F[c - s // 4, c]
    p2 = F[c, c + s // 2 - 1] + F[c + s // 2 - 1, c]
    return float(p4 / tot * 1e4), float(p2 / tot * 1e4)


if __name__ == "__main__":
    src = native_crop()
    print(f"input {src.shape} native pixels, no resize")
    bic = np.stack([np.asarray(Image.fromarray(f).resize((CW * 4, CH * 4), Image.BICUBIC))
                    for f in src])
    o32 = run(src, torch.float32)
    o16 = run(src, torch.float16)

    d = np.abs(o32.astype(np.float32) - o16.astype(np.float32))
    print(f"\nfp32 vs fp16 on native pixels: mean {d.mean():.3f}/255  max {d.max():.0f}/255  "
          f"p99 {np.percentile(d, 99):.2f}/255")

    # A flat patch of wood, in 4x output coords, away from the birds.
    flat = (120, 1500, 256)
    print(f"\n{'':10} {'period-4':>10} {'period-2':>10}   (grid energy x1e4)")
    for name, im in (("bicubic", bic), ("fp32", o32), ("fp16", o16)):
        a, b = grid_energy(im[T // 2], flat)
        print(f"{name:10} {a:10.2f} {b:10.2f}")

    k = T // 2
    x0, y0, s = flat
    Image.fromarray(np.concatenate(
        [im[k][y0:y0 + s, x0:x0 + s] for im in (bic, o32, o16)], axis=1)
    ).resize((s * 3 * 2, s * 2), Image.NEAREST).save("/tmp/vsr_flat.png")
    Image.fromarray(np.concatenate([bic[k], o32[k], o16[k]], axis=1)).save("/tmp/vsr_3way.png")
    print("\nwrote /tmp/vsr_flat.png (flat patch at 2x nearest) and /tmp/vsr_3way.png")
