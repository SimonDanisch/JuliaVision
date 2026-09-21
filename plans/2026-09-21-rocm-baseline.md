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

## The gap, per shape, re-runnable

`tools/gap_vs_rocm.jl` states the shapes and times them here; `preload.sh
tools/gap_vs_rocm.py` times the same ones in torch and prints the ratio. One
JSON between them so the shapes are written down once.

    kind       name                     ours ms  torch ms     gap  ours TF  torch TF
    attention  qwen-denoiser              41.40      8.82   4.69x     6.68     31.34
    attention  qwen-denoiser-square       37.71      8.83   4.27x     7.29     31.11
    attention  sam2-encoder-global         5.63      1.18   4.79x     6.10     29.21
    conv3x3    vae-288-1024              200.50     72.99   2.75x     7.81     21.45
    conv3x3    vae-144-1024               88.78     10.93   8.13x     4.41     35.82
    conv3x3    vae-288-144-1024          164.71     36.52   4.51x     4.75     21.43
    conv3x3    vae-576-512               137.68     64.06   2.15x    11.37     24.44
    conv3x3    vae-1152-256              114.44     64.63   1.77x    13.68     24.22
    conv3x3    vae-1152-128               37.95     10.61   3.58x    10.31     36.90
    gemm-fp16  square-4096                 6.80      5.65   1.20x    20.21     24.31
    gemm-fp16  denoiser-qkv               24.97     19.44   1.28x    17.03     21.87
    gemm-fp16  denoiser-gateup            48.74     39.84   1.22x    17.45     21.35
    copy       copy-fp16-512MiB            5.06      5.06   1.00x
    copy       copy-fp32-512MiB            5.05      5.06   1.00x

Three things to read off it.

**The copy is at parity**, 1.00x on both element types. An earlier draft of this
file said 1.1-1.4x from a `copyto!` on differently shaped arrays; on the same
512 MiB buffer the two are the same number. Every "at bandwidth" claim in
`2026-09-21-qwen-denoiser-layer.md` is therefore sound as written.

**Attention is 4.3-4.8x off on every shape tried**, including SAM 2's, so it is
the kernel and not the Qwen-Image shape.

**Convolution ranges 1.77x to 8.13x**, and the worst is not the biggest: `144 ->
144 @ 1024²` is 88.78 ms against 10.93, where torch reaches 35.8 TFLOP/s — above
the dense fp16 GEMM rate, so MIOpen is running an algorithm this tree does not
have for 3x3 (Winograd, or an fp16 accumulate). The im2col route's cost is
proportional to `CRS`, and at `Cin = 144` there is the least arithmetic to
amortise writing the matrix over.

## Where the missing time actually is, for THIS product

| | share of a generation | gap | recoverable |
| --- | --- | --- | --- |
| denoiser attention | ~28 s of 234.6 | 4.7x | **~22 s** |
| denoiser products (`q8gemm`, int8 weights) | ~62 s | ~parity | ~0 |
| VAE convolution | ~2-3 s | ~3x | ~1.5-2 s |
| everything else | ~40 s | 1.0x | ~0 |
| model builds (Julia compilation) | ~100 s | n/a | blocked, see the other file |

So for Qwen-Image the prize is **attention, by an order of magnitude over
convolution** — the VAE is only 2% of a generation once its convolutions took
today's 7x. Convolution is the prize for the conv-heavy models instead:
DepthAnything, RIFE, ProPainter, BasicVSR++, SAM 2's encoder.

Note the products row: the denoiser does NOT use `Mantle.coopmat_gemm`, it uses
`DNNKernels.q8gemm` over int8 weights, which measures 22.5 TOP/s against
torch's 21.4-24.3 fp16. The 1.2x in the table is the fp16 library GEMM, which
the denoiser does not run.

## There is already a ROCm backend, and convolution is the hole in it

`AMDGPU.jl` is installed and `Mantle` has a `MantleROCmExt` — 1224 lines —
whose whole purpose is this: `deviceview` hands back a genuine borrowed
`ROCArray` over a Mantle region, so **rocBLAS, MIOpen and rocFFT can read a
planned graph's transients with no copy**, and `LibOp` is how a library call
becomes a pass in the graph.

What it wires up today is rocBLAS (and hipBLASLt for a bias epilogue). Its own
comment says the rest:

> rocBLAS remains the fallback library GEMM, and convolution still uses
> DNNKernels where no suitable vendor plan is integrated.

Its recorded four-model comparison against Lava, steady state, same GPU:

    DepthAnything  ViT + DPT, conv-heavy       85 ms  against  131 ms
    RIFE           optical flow, 1920x1152    162 ms  against  141 ms
    NeuralLUT      26 ops, 33^3 output       1.57 ms  against 2.00 ms
    SAM 2.1        encoder, 1082 passes       322 ms  against  269 ms

DepthAnything is already 1.5x there **without** MIOpen convolution — that is
rocBLAS on its GEMMs alone. Adding a MIOpen `LibOp` for `convolution.default`
is the smallest step to the numbers in the table above, and it costs nothing in
portability: the extension is not `src/`, so it is not vendor-conditional code
in the sense `CLAUDE.md` forbids — Lava keeps its own kernel and the ROCm
backend calls the vendor's.

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
