"""Shared paths and setup for the MatAnyone2 -> DNNKernels export tools.

Dev-only. Nothing here ships. See lava-dnn.md.

MatAnyone2 is used straight from its checkout in dev/ rather than installed as a
package: its declared dependencies pull in gradio, PySide6, pycocotools and a
git-sourced thinplate, none of which the inference path imports. Putting the
checkout on sys.path keeps the venv to what is actually needed and keeps a
`uv sync` from ever fighting an out-of-tree install.
"""

import os
import sys
from pathlib import Path


def find_root():
    """The workspace root — the directory that owns `gen/` and `dev/`.

    Not `tools/..`, and the reason is worth stating because it broke every script
    at once. `tools/` lives in the JuliaVision repo but is reachable through a
    symlink from the workspace root, so `Path(__file__).resolve()` follows the
    symlink and `parent.parent` lands on `dev/JuliaVision` — where `gen/` does not
    exist. Dropping `.resolve()` is not a fix either: it breaks the other
    invocation, `dev/JuliaVision/tools/x.py` run directly.

    And on a standalone clone of JuliaVision — which is what the laptops have —
    there is no workspace root at all, so the repo itself is the answer and
    `gen/` gets created inside it by the fetcher.

    Order: an explicit `VIDEOEDIT_ROOT`; then the nearest ancestor that already
    has `gen/` or `dev/JuliaVision/`; then the repo root. The walk is bounded so a
    stray `~/gen` cannot capture it.

    The marker is `dev/JuliaVision`, not `dev` — plain `dev` matches **`/dev`** on
    any Linux box, so the walk ran to the filesystem root and returned `/`.
    """
    env = os.environ.get("VIDEOEDIT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent
    for cand in [here, *list(here.parents)[:4]]:
        if (cand / "gen").is_dir() or (cand / "dev" / "JuliaVision").is_dir():
            return cand
    return here.parent


ROOT = find_root()
UPSTREAM = ROOT / "dev" / "MatAnyone2"
GEN = ROOT / "gen"
CKPT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
CKPT = UPSTREAM / "pretrained_models" / "matanyone2.pth"


def bootstrap():
    """Make `matanyone2` importable from the dev checkout."""
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    return UPSTREAM


def checkpoint():
    """Path to matanyone2.pth, downloading it on first use."""
    bootstrap()
    from matanyone2.utils.download_util import load_file_from_url

    return Path(load_file_from_url(CKPT_URL, str(CKPT.parent)))


IMAGE_EXT = (".jpg", ".jpeg", ".png")
VIDEO_EXT = (".mp4", ".mov", ".avi")


def read_frames(path):
    """Frames as a float TCHW tensor in [0,255], plus fps and name.

    Same contract as matanyone2.utils.inference_utils.read_frame_from_videos
    (inference_utils.py:12-29), reimplemented because that one calls
    torchvision.io.read_video, which 0.26 removed. Upstream's uv.lock pins
    torchvision 0.25; we would rather keep the newer torch for torch.export
    than pin the whole stack to a test-data loader.
    """
    import cv2
    import numpy as np
    import torch

    path = Path(path)
    if path.suffix.lower() in VIDEO_EXT:
        import imageio.v3 as iio

        frames = np.stack(list(iio.imiter(path, plugin="FFMPEG")))  # THWC, RGB, uint8
        meta = iio.immeta(path, plugin="FFMPEG")
        fps = meta.get("fps", 24)
        name = path.stem
    else:
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXT)
        frames = np.stack([cv2.imread(str(f))[..., ::-1] for f in files])  # BGR -> RGB
        fps, name = 24, path.name

    frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    return frames, fps, frames.shape[0], name


def load_model(device=None):
    """The upstream nn.Module, weights loaded, in eval mode."""
    bootstrap()
    import torch
    from matanyone2.utils.device import get_default_device
    from matanyone2.utils.get_default_model import get_matanyone2_model

    if device is None:
        device = get_default_device()
    model = get_matanyone2_model(str(checkpoint()), device)
    model.eval()
    return model, torch.device(device)
