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
    attention  qwen-denoiser              22.10      8.64   2.56x    12.50     31.97
    attention  qwen-denoiser-square       22.09      8.69   2.54x    12.44     31.62
    attention  sam2-encoder-global         3.59      1.18   3.05x     9.58     29.24
    conv3x3    vae-288-1024              191.27     66.97   2.86x     8.18     23.38
    conv3x3    vae-144-1024               85.34     19.32   4.42x     4.59     20.26
    conv3x3    vae-288-144-1024          158.89     35.54   4.47x     4.93     22.02
    conv3x3    vae-576-512               122.75     63.67   1.93x    12.75     24.59
    conv3x3    vae-1152-256               93.29     64.15   1.45x    16.78     24.41
    conv3x3    vae-1152-128               24.43     11.31   2.16x    16.02     34.61
    gemm-fp16  square-4096                 7.59      5.60   1.36x    18.11     24.53
    gemm-fp16  denoiser-qkv               24.72     16.61   1.49x    17.20     25.61
    gemm-fp16  denoiser-gateup            48.82     35.67   1.37x    17.42     23.84
    copy       copy-fp16-512MiB            4.98      5.06   0.99x
    copy       copy-fp32-512MiB            4.92      5.06   0.97x

The `conv3x3` torch column is from a run where MIOpen's autotuning had settled;
ours and the other rows are the latest. See the note below on why that
distinction has to be made for convolution and for nothing else.

Three things to read off it.

**The copy is at parity**, 0.99x on both element types. An earlier draft of this
file said 1.1-1.4x from a `copyto!` on differently shaped arrays; on the same
512 MiB buffer the two are the same number. Every "at bandwidth" claim in
`2026-09-21-qwen-denoiser-layer.md` is therefore sound as written.

**Attention is 2.5-3.1x off**, on every shape tried including SAM 2's, so it is
the kernel and not the Qwen-Image shape. It was 4.3-4.8x when this file was
written; what closed the first half is below, and what does NOT close the rest
is recorded beside it in `flash.jl`.

**Convolution ranges 1.45x to 4.47x** and is still the widest gap in the tree.
The worst is not the biggest: `144 -> 144 @ 1024²` is 85.34 ms against 19.32.
The im2col route's cost is proportional to `CRS`, and at `Cin = 144` there is
the least arithmetic to amortise writing the matrix over.

Two things in that route were plain waste and both are fixed, worth 10% to 36%
depending on the shape — `1152 -> 1152 @ 128²` went 37.96 ms to 24.43:

  * `im2col_kernel!` took the output width and height as runtime arguments and
    then divided a pixel index by `OW` and by `OH*OW`. Two integer divisions
    per element, on a device with no integer divide. It ran at 62 GB/s against
    `fill!`'s 125 on the same volume, which is the ratio that says "bug, not
    limit". As `Val`s they become multiply-shifts: **-15.5% to -19.5% on every
    VAE shape, output compared and bit-identical**, and no extra specialisation
    because `Val{MP}` beside it already keys the kernel per shape.
  * The chunking took the largest chunk the workspace budget allowed, which put
    the whole remainder in the last chunk **at full cost** — `MP` is the leading
    dimension, so `im2col` writes every row of the buffer and the GEMM
    multiplies every row of it whether the pixels are real or not. `1152 @ 128²`
    ran two chunks of 12864 for 16384 pixels, 57% padding. Spreading the pixels
    evenly over the same number of chunks is **-3.0% to -35.8% of the row-work**,
    which is arithmetic rather than a measurement.

**im2col cannot win the narrow shapes, and the arithmetic says so before any
tuning does.** At `144 -> 144 @ 1024²` the matrix im2col materialises is
`H*W` by `Cin*9` = 1048576 x 1296 fp16 = **2.72 GB**. Writing it once and
reading it once, at the 213 GB/s this device moves under `copyto!`, is
**25.5 ms** — against torch's **19.32 ms for the whole convolution**. No GEMM,
however good, gets under the vendor from there; the route has to stop
materialising. That is what an implicit GEMM is: address the input directly
from `(n, p, q, c, r, s)` and never write the matrix. Measured beside it, ours
is 85.34 ms, so the matrix is ~30% of it and the skinny `N = 144` product —
nine 16-wide column tiles — is most of the rest. The two fixes above took the
waste out of the route; they do not change that the route itself has a floor
above the vendor on this shape.

**The obvious implicit GEMM was built and it loses**, by a lot: 110.4 ms
against 85.3 at `144 @ 1024²`, and 428.3 against 93.3 at `1152 @ 256²`. The
addressing is not the problem — a spike agrees with the host to 4.5e-06 once
the weight is permuted channel-fastest. Arithmetic intensity is: a kernel that
loads both operands from global every k-step has no reuse, and reuse is exactly
what materialising the matrix buys. Raising the per-subgroup tile from 4x1 to
4x4 to fix the ratio measures worse still, on registers. `conv_coopmat.jl` has
the table.

So the win, if it is taken, is to fuse the im2col ADDRESSING into a blocked
GEMM's shared-memory staging rather than to remove the staging — and both that
and the narrow-`N` tiling live in `Mantle.coopmat_gemm!`, which this project
pins.

The same arithmetic is why the gap narrows as `Cin` grows: at `Cin = 1152` the
matrix is 8x larger but the arithmetic is 64x, so the write amortises, and that
row is 1.45x rather than 4.42x.

**An earlier reading of this row was wrong, and the way it was wrong is worth
keeping.** The first run of the harness had torch at 10.93 ms on that shape —
35.8 TFLOP/s, *above* its own dense fp16 GEMM rate — from which this file
concluded that MIOpen must be running an algorithm this tree does not have,
Winograd or an fp16 accumulate. Re-running the same shape gives 18.72 and then
19.32 ms, 20.3 TFLOP/s, comfortably below the GEMM rate. The first number was
MIOpen before its autotuning had settled, and nothing about the algorithm
follows from it. **Torch's convolutions need several runs of the harness before
they stop moving; the attention and GEMM rows do not.**

## What closed a third of the attention gap, and what AOTriton said

The first version of this file concluded that attention was memory-bound and
tiling-exhausted. Reading AMD's own kernels said otherwise. `dev/aotriton`'s
tuning database has an entry for this architecture, and for head dimension 128,
fp16, non-causal it picks **`BLOCK_M = 32, BLOCK_N = 16, num_warps = 4`** —
after an extended search that included every larger tile. 128 threads, the
smallest block in the space, and `qk` accumulated in **fp32** (`fwd_kernel_inner.py`).

Three things followed, in the order they were measured:

1.  **Precision is not the gap.** The reference reaches 32 TFLOP/s with an fp32
    score accumulator, so the `s16` option added here cannot be what is missing
    — and measured, it is worth 4.3%. Left off by default.

2.  **The staging is.** `OpCooperativeMatrixLoadKHR` takes a pointer in any
    storage class, so K and V can be read straight from the tensors instead of
    being stored to shared memory, barriered, and loaded back on every key
    block. That alone is **37.36 ms -> 25.56** at Qwen-Image's shape, with
    bit-identical output. See `globalkv` in `flash.jl`.

3.  **Which tiling is fastest then inverts.** `FLASHCM_TILINGS` was measured
    with staging, and staging is what paid for a wide workgroup. With the
    operands read as tiles, `(16, 32)` at 128 threads beats the staged table's
    pick at every shape — the same 128 threads AOTriton tunes to. The two
    tables are separate and the plan carries which one it used.

A ragged key axis also stopped costing a separate launch and a merge pass over
134 MB: the clamp is now confined to the one block that is short, and that
block's tile slides back onto the last whole tile so the load stays in bounds.

End to end, with the change stashed and unstashed in one session so the machine
is held constant: **251.8 s against 239.2 s**, and the whole of the difference
is the denoising steps — 5.01 to **4.49 s/step**. The three build rows do not
move, so the per-shape tiling costs nothing in compilation.
`QwenImageRunner/README.md` carries both tables and why they disagree with the
older figures.

## What a denoising step is made of, and two tools that lie about it

Profiled with `planfor(...; profile = true)` at 1024², 32 layers, 4118 tokens:
the step is **4.42 s** and 4180 passes. The budget, built from INDEPENDENT
per-kernel measurements rather than the profiler:

| | per step | share | against the vendor |
| --- | --- | --- | --- |
| products (`mm`, 57.5 TFLOP at ~23 TOP/s) | ~2.5 s | ~57% | parity |
| the products' ConvRot and B-pad passes | ~0.3 s | ~7% | — |
| attention (32 x 24.9 ms) | ~0.8 s | ~18% | 2.5x |
| elementwise, norms, rope, copies | ~0.35 s | ~8% | at bandwidth |
| unattributed | ~0.5 s | ~11% | — |

**Attention is the only row with room**, and the rest of that row needs the
rewrite `flash.jl` records as not worth building. The products are at the
vendor's int8 rate, and the elementwise passes are at `copyto!`.

The `ConvRot` and `padB` row looks like per-step work on static weights and is
not: in `q8gemm` the INT8 operand is the weight and `B` is the activation, so
both the rotation and the column pad are over data that changes every step.
Checked, not assumed.

**`Mantle.timings(...).gpu_ms` cannot attribute per-pass time in a plan this
size.** It is a median of a timestamp interval that includes waiting for earlier
work, so the values do not add up: the 4180 medians sum to **8777 ms against a
4419 ms step**. Two checks show how far off an individual one can be — the
`fused.swiglu` pass reads **39.69 ms** and measures **1.56 ms** standalone, at
195 GB/s against `copyto!`'s 200 on the same volume. The one pass whose profile
figure does hold is the longest in its chain: `fused.sdpa` reads 24.93 ms and
measures 22.1 ms alone. Use the profile to RANK, and a standalone launch to
price.

**Pass count is not a lever either.** The text branch's `bmm` is emitted one
dispatch per batch plane — 64 ops become **2048 passes**, each 0.0043 ms, and
`BMM_N_KERNELS` has no entry for its `N` of 22 or 128 even on the interpreted
path. That looks like 2000 barriers to delete. A recorded plan of N dispatches
of a trivial kernel, all writing one buffer so every pass is barriered against
the last, costs **0.0009 ms per pass** (256 to 2048 passes, slope-fitted), so
those 1984 extra dispatches are **~2 ms of a 4419 ms step**. Not worth a
batched kernel.

## Where the missing time actually is, for THIS product

| | share of a generation | gap | recoverable |
| --- | --- | --- | --- |
| denoiser attention | ~15 s of 239.2 (was ~26 of 251.8) | 2.7x | **~10 s** |
| denoiser products (`q8gemm`, int8 weights) | ~62 s | ~parity | ~0 |
| VAE convolution | ~2-3 s | ~3x | ~1.5-2 s |
| everything else | ~40 s | 1.0x | ~0 |
| model builds (Julia compilation) | ~100 s | n/a | blocked, see the other file |

So for Qwen-Image the prize is **attention, by an order of magnitude over
convolution** — the VAE is only 2% of a generation once its convolutions took
today's 7x. Convolution is the prize for the conv-heavy models instead:
DepthAnything, RIFE, ProPainter, BasicVSR++, SAM 2's encoder.

Note the products row: the denoiser does NOT use `Mantle.coopmat_gemm`, it uses
`DNNKernels.q8gemm` over int8 weights. The table's GEMM rows are the fp16
library GEMM, which the denoiser does not run.

**And int8 is not twice fp16 on this device, so there is no factor of two
waiting in the products.** The obvious suspicion — RDNA3 WMMA does int8 at
twice the fp16 rate, `q8gemm` reaches only 1.34x our own fp16 GEMM, therefore
half the int8 rate is being left on the floor — is wrong, and the vendor
answers it directly (`tmp/baseline/int8_ceiling.py`):

    M x N x K              fp16 ms  fp16 TF   int8 ms  int8 TOP   int8/fp16
     4096 x  4096 x  4096     5.32    25.85      5.37     25.60      0.99x
    12288 x  4224 x  4096    16.27    26.13     16.91     25.14      0.96x
    24576 x  4224 x  4096    33.81    25.15     33.31     25.53      1.02x

`torch._int_mm` and `torch.mm` are the same speed. Against that, `q8gemm` on
the same shapes measures 23.0, 23.6 and 27.1 TOP/s — **0.92x to 1.06x of the
vendor's int8 GEMM**, and ahead of it on the square shape. The products are at
the achievable ceiling, and the 1.34x over our own fp16 GEMM says our fp16 GEMM
is the one with room (19.4 TF against torch's 25.9), not that int8 has any.

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
