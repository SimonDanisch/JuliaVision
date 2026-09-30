"""
PyTorch reference for BasicVSRRunner's parity test.

Runs upstream BasicVSR++ (through `basicvsrpp.py`, so the upstream model files
unchanged) on a synthetic clip, with the weights from the `basicvsrpp` artifact
itself rather than the checkpoint, so the test compares the port and nothing
else. Writes `BasicVSRRunner/test/fixtures/parity.txt`: 512 sampled output
values, 1-based Julia indices into the `(256, 256, 3, 5, 1)` result.

The clip is built from integer arithmetic only (`clip` below, and `parityclip`
in the test), so both languages construct the same Float32 values bit for bit
without the input having to be stored.

    VIDEOEDIT_ROOT=<workspace> .venv/bin/python tools/verify_basicvsrpp.py \\
        ~/.julia/artifacts/<basicvsrpp tree>/weights.safetensors

Needs the upstream checkout at `<workspace>/dev/BasicVSR_PlusPlus`
(github.com/ckkelvinchan/BasicVSR_PlusPlus).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

import basicvsrpp as B

W = H = 64
T = 5
FIXTURE = Path(__file__).resolve().parent.parent / "BasicVSRRunner" / "test" / "fixtures" / "parity.txt"


def tri(x):
    """Triangle wave, 0..128, period 64."""
    return np.abs(np.mod(x, 64) - 32) * 4


def clip():
    """(1, T, 3, H, W) in [0, 1]: two triangle waves, moving 1.5 px a frame in x."""
    t, c, j, i = np.meshgrid(np.arange(T), np.arange(3), np.arange(1, H + 1),
                             np.arange(1, W + 1), indexing="ij")
    v = np.minimum(tri(2 * i + 3 * t + 11 * c) + tri(3 * j + 7 * c), 255)
    return (v.astype(np.float32) / np.float32(255))[None]


def main(weights):
    pp = B._load_upstream()
    model = pp.BasicVSRPlusPlus(mid_channels=64, num_blocks=7, is_low_res_input=True,
                                spynet_pretrained=None)
    model.load_state_dict(load_file(weights), strict=True)
    model.eval()
    with torch.no_grad():
        y = model(torch.from_numpy(clip())).numpy()[0]          # (T, 3, 4H, 4W)
    rng = np.random.default_rng(0)
    n = 512
    tt = rng.integers(0, T, n); cc = rng.integers(0, 3, n)
    yy = rng.integers(0, 4 * H, n); xx = rng.integers(0, 4 * W, n)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as io:
        io.write("# x y c t value: BasicVSR++ (PyTorch, fp32) on the test's parity clip,\n")
        io.write("# written by tools/verify_basicvsrpp.py. Indices are 1-based.\n")
        for x, yv, c, t in zip(xx, yy, cc, tt):
            io.write(f"{x + 1} {yv + 1} {c + 1} {t + 1} {y[t, c, yv, x]:.9g}\n")
    print(f"wrote {n} samples to {FIXTURE}; output range {y.min():.4f}..{y.max():.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
