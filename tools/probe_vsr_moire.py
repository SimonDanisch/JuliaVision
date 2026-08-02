"""Is the cross-grain flickering pattern spatial, or is it temporal misalignment?

Simon's observation: the artifact is higher-frequency than the wood grain, runs
AGAINST it, and flickers. Flicker rules out recovered structure. The suspect is
optical flow on a near-parallel-line surface — the aperture problem: motion along
the grain is unconstrained, so the flow slides along it and the aligned
neighbours beat against each other across it.

Controls, identical except for what varies between the input frames:

  A  real consecutive frames    -> flow is estimated, alignment can be wrong
  B  the SAME frame repeated T  -> true flow is exactly zero
  C  single frame, T=2 repeat   -> as B, minimal propagation depth

If the pattern is in A and not in B, it is the temporal alignment, not the
spatial path — and no amount of retraining the upsampler would fix it.
"""
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from common import find_root  # tools/ is symlinked; see find_root
ROOT = find_root()
sys.path.insert(0, str(ROOT / "tools"))
from basicvsrpp import load_basicvsrpp

CLIP = ROOT / "media" / "birds_export.mp4"
# The beam surface, in source pixels: fine near-parallel grain at a shallow angle.
# Fed as a 384x384 crop rather than the whole frame — the artifact is local, and
# this card is shared, so the test is sized to run beside whatever else is on it.
# 384 is a multiple of 32, which SPyNet's pyramid wants.
CX, CY, CS = 200, 560, 384
T, dev = 6, "cuda"


def frames(start, n):
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(CLIP),
         "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.uint8).reshape(-1, 1056, 864, 3)[:n]
    return a[:, CY:CY + CS, CX:CX + CS].copy()


def run(model, seq):
    x = torch.from_numpy(np.stack(seq)).permute(0, 3, 1, 2)
    x = (x.float() / 255.0)[None].to(dev).half()
    with torch.no_grad():
        y = model(x)[0].clamp_(0, 1)
    out = (y[len(seq) // 2].permute(1, 2, 0).float() * 255).round().byte().cpu().numpy()
    del x, y
    torch.cuda.empty_cache()
    return out


def cross_grain(img):
    """Energy off the wood's own orientation.

    The grain is near-horizontal, so its spectrum concentrates on the vertical
    frequency axis. An artifact running across it puts energy on the horizontal
    axis instead. The ratio separates the two without needing to know the exact
    angle.
    """
    p = np.asarray(Image.fromarray(img).convert("L"), np.float32)
    s = min(p.shape[0], p.shape[1], 512)
    q = p[:s, :s] - p[:s, :s].mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(q)))
    c = s // 2
    hi = slice(c + s // 8, c + s // 2)      # high frequencies only
    along = F[hi, c - 3:c + 4].sum()        # vertical axis  = horizontal stripes
    across = F[c - 3:c + 4, hi].sum()       # horizontal axis = vertical stripes
    return float(across / (along + 1e-9)), float(q.std())


if __name__ == "__main__":
    seq = frames(11.5, T)
    model = load_basicvsrpp(ROOT / "gen" / "basicvsrpp" / "basicvsr_pp_reds4.pth")
    model = model.to(dev).half().eval()

    cases = {
        "A real frames": list(seq),
        "B same frame x%d" % T: [seq[T // 2]] * T,
        "C same frame x2": [seq[T // 2]] * 2,
    }
    outs = {}
    print(f"{'case':18} {'cross/along':>12} {'std':>8}")
    for k, s in cases.items():
        outs[k] = run(model, s)
        r, sd = cross_grain(outs[k])
        print(f"{k:18} {r:12.3f} {sd:8.2f}")

    bic = np.asarray(Image.fromarray(seq[T // 2]).resize((CS * 4, CS * 4), Image.BICUBIC))
    r, sd = cross_grain(bic)
    print(f"{'bicubic':18} {r:12.3f} {sd:8.2f}")

    tiles = [bic] + [outs[k] for k in cases]
    Image.fromarray(np.concatenate(tiles, axis=1)).save("/tmp/moire.png")
    # A minus B: what the temporal path added, on its own.
    d = np.abs(outs["A real frames"].astype(np.float32)
               - outs["B same frame x%d" % T].astype(np.float32))
    print(f"\nA - B (what temporal alignment added): mean {d.mean():.2f}/255 "
          f"max {d.max():.0f}/255")
    Image.fromarray(np.clip(d * 6, 0, 255).astype(np.uint8)).save("/tmp/moire_diff.png")
    print("wrote /tmp/moire.png (bicubic | A | B | C) and /tmp/moire_diff.png (6x gain)")
