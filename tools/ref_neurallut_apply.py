"""
Reference for the trilinear LUT apply, from **upstream's own CUDA kernel**.

    uv run tools/ref_neurallut_apply.py

`export_neurallut.py` covers the graph — the CNN and the blend — and its
reference comes from the same PyTorch module that was exported. The apply is not
in the graph (it is `GPUFiltering.lut3d!`), so it needs its own reference, and a
reimplementation in PyTorch would only prove that two readings of
`trilinear_kernel.cu` agree. This builds and runs the kernel itself.

**Why it needs patching to build at all.** The extension is from 2020 and
includes `<THC/THC.h>`, a header PyTorch removed in 1.11. Nothing in either file
actually calls a THC function — the include is dead, and the one CUDA-runtime
symbol used, `at::cuda::getCurrentCUDAStream`, lives in `ATen/cuda/CUDAContext.h`.
`Tensor::data<T>()` also became `data_ptr<T>()`. So the fix is a header swap and
a rename, applied to a *copy* under `gen/` so the checkout stays pristine and the
next person gets the same build from a clean clone.

The build also needs `gcc-13`: CUDA 12.4's `nvcc` refuses a host compiler newer
than 14, and this box defaults to 15.
"""

import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from common import find_root
ROOT = find_root()
GEN = ROOT / "gen"
CHECKOUT = ROOT / "dev" / "Image-Adaptive-3DLUT"
BUILD = GEN / "trilinear_build"

DIM = 33
# Upstream's constant, fudge factor and all. `lut3d!` documents why it is not
# 1/(DIM-1); the reference has to carry the same one or the interior samples
# land in different bins.
BINSIZE = 1.000001 / (DIM - 1)


def build_trilinear():
    """Upstream's extension, patched enough to compile against a current torch."""
    src = CHECKOUT / "trilinear_cpp" / "src"
    if not src.is_dir():
        raise SystemExit(f"no upstream extension sources at {src}")
    BUILD.mkdir(parents=True, exist_ok=True)

    files = []
    for name in ("trilinear_cuda.cpp", "trilinear_kernel.cu", "trilinear_kernel.h"):
        text = (src / name).read_text()
        # The dead include, and the one header that actually provides
        # `getCurrentCUDAStream`.
        text = text.replace("#include <THC/THC.h>", "#include <ATen/cuda/CUDAContext.h>")
        # `data<float>()` -> `data_ptr<float>()`, but not if it is already the
        # new spelling.
        text = re.sub(r"(?<!_ptr)\.data<float>\(\)", ".data_ptr<float>()", text)
        dst = BUILD / name
        if not dst.is_file() or dst.read_text() != text:
            dst.write_text(text)
        if name.endswith((".cpp", ".cu")):
            files.append(str(dst))

    # CUDA 12.4 rejects a host compiler >= 14; this box defaults to 15.
    for var, exe in (("CC", "gcc-13"), ("CXX", "g++-13")):
        if shutil.which(exe):
            os.environ.setdefault(var, exe)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")   # RTX 3070 is sm_86

    from torch.utils.cpp_extension import load
    return load(name="trilinear_ref", sources=files,
                build_directory=str(BUILD), verbose=False)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this reference is upstream's CUDA kernel; it needs a GPU")

    tl = build_trilinear()
    out_dir = GEN / "graphs" / "neurallut"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The LUT the exporter's reference input predicted, so the apply is checked
    # against the same table the graph produces rather than a synthetic one.
    from safetensors.torch import load_file
    ref = load_file(str(out_dir / "reference.safetensors"))
    lut = ref["lut"].cuda().contiguous()            # (3, D, D, D)

    # A deterministic image with the corners of the cube in it: pure black,
    # pure white and the primaries all land on a table boundary, which is where
    # an off-by-one in the indexing shows up and a random image would not
    # reliably hit.
    torch.manual_seed(0)
    H, W = 256, 256
    img = torch.rand(1, 3, H, W)
    corners = torch.tensor([[0., 0., 0.], [1., 1., 1.], [1., 0., 0.],
                            [0., 1., 0.], [0., 0., 1.], [1., 1., 0.],
                            [0., 1., 1.], [1., 0., 1.], [0.5, 0.5, 0.5]])
    img[0, :, 0, :corners.shape[0]] = corners.T
    img = img.cuda().contiguous()

    out = img.new_zeros(img.shape)
    shift = DIM ** 3
    assert 1 == tl.forward(lut, img, out, DIM, shift, BINSIZE,
                           img.size(2), img.size(3), img.size(0))
    torch.cuda.synchronize()

    save_file({"img": img.cpu(), "graded": out.cpu(), "lut": lut.cpu()},
              str(out_dir / "reference_apply.safetensors"))
    print(f"apply reference: {tuple(img.shape)} -> {tuple(out.shape)}, "
          f"lut {tuple(lut.shape)}, binsize {BINSIZE:.9f}")
    print(f"  graded range {out.min().item():.4f} .. {out.max().item():.4f}")
    print(f"  wrote {out_dir / 'reference_apply.safetensors'}")


if __name__ == "__main__":
    main()
