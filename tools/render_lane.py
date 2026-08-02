"""Render one upscaling lane at FULL frame, 864x1056 -> 3456x4224.

`upscale_compare` judged the lanes on a crop and discarded the full frames,
which was the wrong trade — the crop answers "which is better", but you still
want the finished shot for the one that wins. This renders a chosen lane whole,
streaming to the encoder so nothing has to be held.

    uv run tools/render_lane.py decompress_sr     # NTIRE decompress 1x -> REDS4 4x
    uv run tools/render_lane.py realbasicvsr      # RealBasicVSR 4x
    uv run tools/render_lane.py basicvsrpp        # REDS4 4x (already in basicvsrpp_4k/)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from basicvsrpp import load_basicvsrpp, load_realbasicvsr
from demos import _chunked, _ffmpeg_writer, _raw_frames

CLIP = ROOT / "media" / "birds_export.mp4"
CK = ROOT / "gen" / "basicvsrpp"
OUT = ROOT / "media" / "model-demos" / "upscale_compare"
START, SECS, FPS = 9.5, 5.0, 60
dev = "cuda"


def sr_model():
    return load_basicvsrpp(CK / "basicvsr_pp_reds4.pth").to(dev).half().eval()


if __name__ == "__main__":
    lane = sys.argv[1] if len(sys.argv) > 1 else "decompress_sr"
    OUT.mkdir(parents=True, exist_ok=True)
    src = _raw_frames(CLIP, START, SECS)
    n, (h, w) = len(src), src[0].shape[:2]
    print(f"{lane}: {n} frames {w}x{h} -> {w*4}x{h*4}", flush=True)
    t0 = time.perf_counter()

    if lane == "decompress_sr":
        # Pass 1 is 1x, so its output becomes the SR model's input. It propagates
        # at H/4, hence the much longer window than the 4x nets can take.
        dec = load_basicvsrpp(CK / "decompress_track1.pth", 128, 25,
                              is_low_res_input=False).to(dev).half().eval()
        clean = [None] * n
        _chunked(dec, src, 16, 4, 1, dev, lambda i, f: clean.__setitem__(i, f),
                 "decompress")
        del dec, src
        torch.cuda.empty_cache()
        print(f"  decompress pass done at {time.perf_counter()-t0:.0f}s", flush=True)
        model, source, chunk = sr_model(), clean, 6
    elif lane == "realbasicvsr":
        model, source, chunk = (load_realbasicvsr(CK / "realbasicvsr.pth")
                                .to(dev).half().eval(), src, 6)
    else:
        model, source, chunk = sr_model(), src, 6

    path = OUT / f"full_{lane}_4k.mp4"
    wr = _ffmpeg_writer(path, w * 4, h * 4, FPS, preset="fast", crf=20)
    mid = {}
    def emit(i, f):
        wr.stdin.write(np.ascontiguousarray(f).tobytes())
        if i == n // 2:
            mid["f"] = f.copy()
    _chunked(model, source, chunk, 2, 4, dev, emit, lane)
    wr.stdin.close()
    wr.wait()
    if "f" in mid:
        from PIL import Image
        Image.fromarray(mid["f"]).save(OUT / f"full_{lane}.png")
    print(f"done in {time.perf_counter()-t0:.0f}s -> {path.name}", flush=True)
