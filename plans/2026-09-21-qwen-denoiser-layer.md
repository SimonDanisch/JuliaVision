# One Qwen-Image 2.1 denoiser layer, pass by pass

Measured on a Radeon 8060S (RDNA 3.5, RADV), 2026-09-21, at 1024²: 4096 image
tokens over a 22-token prompt, so 4118 keys, 32 heads of 128, hidden 4096.

## How to measure one layer without the model

The transformer is 3473 ops and 20B parameters; a layer is 107 of them and
about 220M. `tools/`-free recipe, all inside DNNKernels:

* cut `ops[144:250]` out of the exported graph, take as inputs every buffer
  they read that they do not produce, as outputs every buffer produced in the
  range that something after it reads, and mark the inputs `:external`;
* give the nine weight keys RANDOM values of the declared shape (host layout is
  the REVERSE of the graph's torch shape, and transposing them silently changes
  what `fuseqkv` stacks);
* `Model(...; quantize = true)`, `planfor`, `replay!`.

That layer replays in 205.7 ms, against 6.89 s / 32 = 215 ms for the real step,
so the harness is the model to within 5% and iterates in a quarter of a second.

**Profile passes with `maxpasses = 1`.** The default plan overlaps passes, and
`Mantle.timings` then reports intervals that include waiting: `_to_copy_50` read
73 ms that way and 1.97 ms serialised, because it sits behind the attention.
One pass per submission costs 2% of wall time here (228 ms against 233) and the
numbers mean what they say.

## Where the time went, before and after

| pass | before | after |
| --- | --- | --- |
| `bmm_7` — the fused attention | 67.3 | 69.3 |
| `fuseqkv_mm_17` — gate+proj, 24576x4096x4118 | 44.2 | 43.9 |
| `mm_19` — mlp out, 4096x12288x4118 | 19.1 | 18.1 |
| `fuseqkv_mm_13` — qkv, 12288x4096x4118 | 18.0 | 18.1 |
| `mm_16` — attn out | 5.6 | 5.9 |
| `mean_3`, `mean_4` — q/k norm reductions | 10.7 | **gone** |
| `cat_7`, `cat_8` — the rotary interleave | 12.6 | **1.2** |
| `cat_10` — text ++ image keys | 5.0 | 4.9 |
| `permute_31/32/33` — attention operands | 8.8 | 8.6 |
| everything else | 31.8 | 25.2 |
| **total, serialised** | **223.1** | **195.2** |

The four products are 86.9 ms and they are not the problem: standalone,
`q8gemm` runs those three shapes at 21.4, 26.1 and 23.4 TOP/s, which is the
device's fp16 ceiling (see `2026-09-20-int8-tensor-cores.md`).

## What was fixed

* **The q/k norm** (`fusegroupedrms`) — Qwen exports the gain spanning ONE head
  and the narrowing cast BEFORE the gain multiply, neither of which the pass
  recognised. 16.5 ms of chain became a 0.95 ms kernel.
* **The rotary interleave** (`interleave2!`) — a `cat` on the innermost axis
  wrote every other element from each of two dispatches. 12.6 ms became 1.16.

## What is left, in order

### The attention, and a key length that does not divide the tile

`bmm_7` is 30% of the layer and runs at 4 TFLOP/s against the products' 21.
Most of that is the kernel's own efficiency at `E = 128` (5.4 TFLOP/s even at a
dividing length), but a third of it is one bit of arithmetic:

| Lk | blocks of BC=16 | ms |
| --- | --- | --- |
| 4096 | 256 | 50.8 |
| 4100 | 256.25 | 72.1 |
| 4112 | 257 | 52.0 |
| **4118** | **257.375** | **73.0** |
| 4128 | 258 | 53.5 |
| 4224 | 264 | 55.0 |

A partial last key block costs **40%**, and it is NOT the masking. Rewriting
every `KCLAMP` test as a select changed nothing; clamping the load ADDRESS so
the staging loads are unconditional changed nothing; removing every `KCLAMP`
effect from the body so that the two cases compile the same program changed
nothing — `Lk = 4118` was still 73.9 ms against 52.1 for `Lk = 4128` over the
same allocation, the same strides, the same 258 blocks and the same source. The
discriminator is `Val{KCLAMP}` itself, which the body no longer reads.

So the fix is not to make the mask cheaper, it is to not need one: split the
key axis at the last multiple of `BC` and merge. The pieces exist —
`attn_flash_cm_merge!` already merges partial attentions under their own row
maxima — but the split inside the kernel is `kbper = cld(nkb, NSPLIT)`, uniform
slices, and this one is 257 blocks plus 1. Worth ~20 ms a layer, 0.64 s a step.

Two things that are NOT worth it, measured: `(64, 32)` with the held store is
68.5 ms against 70.1 for the chooser's `(64, 16)`, and `rego` is 73.5.

### The copies around the attention

`cat_10` and the three `permute`s are 13.5 ms of layout change for operands the
attention then reads. A contiguous fp16 copy of that size is 0.53 ms.

## A trap this measurement fell into

`Lava.FROZEN_VERSION[]` is `"1"` in a normal session, and the frozen SPIR-V
cache is keyed on `typeof(f)`, the argument types and the workgroup size —
**not on the source**. Editing a kernel body under Revise and re-running
therefore measures the OLD kernel, and an hour of flash-attention results here
were exactly that. Signature changes (a new `Val` argument) mint a new key and
are safe; body edits are not. Bump `Lava.FROZEN_VERSION[]` before believing any
kernel measurement.
