"""Shared paths and setup for the MatAnyone2 -> DNNKernels export tools.

Dev-only. Nothing here ships. See lava-dnn.md.

MatAnyone2 is used straight from its checkout in dev/ rather than installed as a
package: its declared dependencies pull in gradio, PySide6, pycocotools and a
git-sourced thinplate, none of which the inference path imports. Putting the
checkout on sys.path keeps the venv to what is actually needed and keeps a
`uv sync` from ever fighting an out-of-tree install.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
