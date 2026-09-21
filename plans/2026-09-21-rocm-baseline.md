# A baseline: PyTorch/ROCm on the same GPU

Measured 2026-09-21 on the Radeon 8060S (gfx1151, 20 WGPs, 2 MB L2) with
`torch 2.13.0+rocm7.1`, against the same shapes this tree runs. Every number in
`2026-09-21-qwen-denoiser-layer.md` before this was self-relative — this tree
against itself — and two of its conclusions do not survive contact with a
vendor library.

## Getting it to run at all

The PyTorch ROCm wheels bundle their own ROCm runtime, and **on this box that
runtime segfaults at the first kernel launch** — allocation and
`torch.cuda.get_device_properties` succeed, `torch.empty(...).fill_(1)` dies
with SIGSEGV. Both the rocm7.0 and rocm7.1 wheels do it, and `gfx1151` code
objects ARE present in `libtorch_hip.so`, so it is not a missing architecture.

The system ROCm is fine: a plain `hipcc --offload-arch=gfx1151` program
allocates, launches and reads back correctly. So preload the system pair and
the wheel works:

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libhsa-runtime64.so.1:/usr/lib/x86_64-linux-gnu/libamdhip64.so.7

`tmp/baseline/preload.sh` is that, and the venv it wraps is built with

    uv pip install --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
        "torch==2.13.0+rocm7.1"

The `+rocm7.1` local version is load-bearing: without it uv resolves torch from
PyPI and installs the CUDA build.

## What the machine does

| | torch | what this tree gets |
| --- | --- | --- |
| fp16 GEMM `4096³` | **24.87 TFLOP/s** | Mantle `coopmat_gemm` 18.37 |
| fp16 GEMM `12288x4096x4096` | **26.21** | — |
| int8-weight GEMM at the denoiser's shapes | — | `q8gemm` 22.5 TOP/s |
| device copy, 512 MiB | **212 GB/s** | `copyto!` 152 (fp16) / 193 (fp32) |
| SDPA, `Lq=4096 Lk=4118 H=32 E=128` | **8.79 ms, 31.4 TFLOP/s** | 37-42 ms, ~7.5 |
| conv `288→288 @1024²` 3x3 | **69.2 ms, 22.6 TFLOP/s** | 218 ms (was 1482) |
| conv `144→144 @1024²` | **18.6 ms, 21.0** | — |
| conv `576→576 @512²` | **63.6 ms, 24.6** | — |
| conv `1152→1152 @128²` | **12.7 ms, 30.9** | 20.2 ms |

The SDPA number is real: checked against an fp32 reference at **4.7e-4**
relative, and the same 8.8 ms from `FLASH_ATTENTION` and `EFFICIENT_ATTENTION`
explicitly (`MATH` is 223 ms). It is above the dense GEMM rate because flash
attention on RDNA3 accumulates the score product in fp16, which is 2x the
fp16→fp32 WMMA rate this tree's kernel uses — so it is not a like-for-like
kernel, but it IS what the ecosystem delivers on this silicon.

## What this changes

**The products were the honest claim.** `q8gemm` at 22.5 TOP/s against
rocBLAS's 24.87-26.21 fp16 is 86-90%, and it reads int8 weights so it moves
less memory. "At the ceiling" stands.

**The attention conclusion was wrong.** `2026-09-21-qwen-denoiser-layer.md`
says the tiling space is exhausted and what is left is a K/V bandwidth wall.
The bandwidth analysis is right and the conclusion drawn from it is not: torch
does the same shape **4.3x faster**, and the ragged key axis that costs this
tree 56% and an entire tail-split mechanism costs torch nothing (`Lk = 4096`
and `Lk = 4118` both measure 8.87 ms). The attention is 27.8% of a denoising
step, so closing that gap is worth ~22% of the step.

**The convolutions are still 3.2x off** after today's 7x. 218 ms against 69.2
on the shape that dominates the VAE, and the well-aligned `1152→1152` case is
1.6x off. The im2col traffic floor this file computes (~72 ms of the 218 for
`conv_37`) is a floor for THIS design, not for the problem — torch does the
whole convolution in 69 ms.

**Even `copyto!` is 1.1-1.4x off**, which matters for every "at bandwidth"
claim made anywhere in these notes: they were measured against 152-193 GB/s
when the machine does 212.

## What is NOT measured here

End to end. A generation is 234.6 s in this tree and there is no comparable
number for the same checkpoint: the Comfy-Org int8 weights are local
(`~/.cache/JuliaVision/Qwen-Image-2.1-Comfy-Org`, 14 GB) but running them needs
ComfyUI, and the diffusers path needs the bf16 model that is not. The component
table above is what exists.
