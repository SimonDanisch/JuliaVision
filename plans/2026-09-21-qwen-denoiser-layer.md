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
| `bmm_7` — the fused attention | 67.3 | 54.4 (three passes) |
| `fuseqkv_mm_17` — gate+proj, 24576x4096x4118 | 44.2 | 43.5 |
| `mm_19` — mlp out, 4096x12288x4118 | 19.1 | 17.1 |
| `fuseqkv_mm_13` — qkv, 12288x4096x4118 | 18.0 | 18.3 |
| `mm_16` — attn out | 5.6 | 7.2 |
| `mean_3`, `mean_4` — q/k norm reductions | 10.7 | **gone** |
| `cat_7`, `cat_8` — the rotary interleave | 12.6 | **1.2** |
| `cat_10` — text ++ image keys | 5.0 | **0.6** |
| `permute_31/32/33` — attention operands | 8.8 | **under 1.4 each** |

Five changes, each measured on its own, each leaving the layer's output
bit-identical or at fp16 rounding:

| | layer, replayed, ms |
| --- | --- |
| start | 232.8 |
| the q/k norm fused | 222.1 |
| the rotary interleave in one dispatch | 205.7 |
| a ragged key axis split at the last whole tile | 189.5 |
| permuted copies walked in source order | 180.7 |
| a contiguous slab copied as one run | 174.6 |
| the score pass count by head width | 171.1 |
| the interleaved rotary fused | **167.1** |

The four products are 86.9 ms and they are not the problem: standalone,
`q8gemm` runs those three shapes at 21.4, 26.1 and 23.4 TOP/s, which is the
device's fp16 ceiling (see `2026-09-20-int8-tensor-cores.md`).

## What was fixed

* **The q/k norm** (`fusegroupedrms`) — Qwen exports the gain spanning ONE head
  and the narrowing cast BEFORE the gain multiply, neither of which the pass
  recognised. 16.5 ms of chain became a 0.95 ms kernel.
* **The rotary interleave** (`interleave2!`) — a `cat` on the innermost axis
  wrote every other element from each of two dispatches. 12.6 ms became 1.16.
* **The ragged key axis** (`tailsplit`) — the cliff below, answered with two
  launches: 70.9 ms to 63.1 standalone, 67.3 to 54.4 in the layer.
* **Permuted copies** (`stridedcopy32perm!`) — walking the DESTINATION linearly
  leaves the reads scattered, and a scattered read is the expensive direction:
  21 GB/s against 115 for `(E, H, L) -> (E, L, H)`.
* **Contiguous slabs** (`slabcopy!`) — `blockcopy!`'s coordinate arithmetic cost
  9x what moving the bytes did: 12 GB/s against 109.
* **The score pass count** — one pass over the scores or two is a property of
  the head width, and the kernel had one answer for every shape. Two passes win
  at `E >= 96` and lose below it, six shapes either side of the crossing.
* **The interleaved rotary** (`fusepairrope`) — `fuserope` only knew the
  half-rotation spelling, so Qwen's pair rotation stayed eight ops whose cost
  was two stride-2 materialisations and an interleave, not the arithmetic.

Four of the six share a shape. **Every one was a memory pass running at a
tenth of the device's bandwidth for a reason that had nothing to do with
memory** — index arithmetic, traversal order, or a layout the consumer could
not read so a copy was inserted. None was in the arithmetic, and none needed a
faster kernel. That is worth looking for elsewhere before anything clever is
attempted.

## What is left, in order

### The attention's own efficiency, which is now the whole of it

`bmm_7.1` is ~46 ms for 276 GFLOP, about 6 TFLOP/s, against 21-26 for the
products in the same layer. That is the kernel at `E = 128` and not the ragged
axis: a key length the tile divides still runs at 5.4. `O` lives in shared
memory and is read and written on every key block that moves a row's maximum,
which the kernel's own notes put at two thirds of them, and at `BR = 64`,
`EP = 128` that is 64 KiB of LDS traffic per 1.05 MFLOP of matrix work. `rego`
would put it in registers and measures WORSE here (73.5 ms against 70.1), and
the tiling chooser has no admissible wider tile at this head width. A kernel
design question rather than a tuning one, and the next big one.

### Solved: a key length that does not divide the tile

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

Register ballast does not move it either, so it is not an occupancy edge. What
it follows is `Val{KCLAMP}` itself, and that remains unexplained.

So the fix is not to make the mask cheaper, it is to not need one: one launch
over the blocks that fill a tile with the check compiled out, one over the
remainder with it on, and `attn_flash_cm_merge!` to combine them under their
own row maxima. `PARTOUT` is what made it expressible — which split slot a
launch writes, so a launch can hand back an unnormalised partial without
splitting its own key range. The tail launch is one key block of 258 and can be
as slow as it likes.

Two things that are NOT worth it, measured: `(64, 32)` with the held store is
68.5 ms against 70.1 for the chooser's `(64, 16)`, and `rego` is 73.5.

### Where the 161.4 ms that is left actually sits

Serialised, after everything above (223.1 at the start):

| | ms | |
| --- | --- | --- |
| the four products | 86.3 | 53%, and at the device's fp16 ceiling |
| the attention, three passes | 48.4 | 30%, at ~6 TFLOP/s |
| everything else | 26.7 | 17%, of which 118 passes are under 0.9 ms |

There is no third big thing. What is left above a millisecond, and what each
would take:

* ~~`view_77` + `view_79`, 2.8 ms~~ — **taken**: the declared SwiGLU now reads
  the stacked product's halves where they lie, which is what the interpreted
  runner always did. The strided kernel is not slower for being strided (2.43
  ms against 2.57), so the copies bought nothing.
* `add_17`, **2.9 ms** — a three-operand elementwise over dense fp16, 135 MB at
  47 GB/s against a 109 GB/s copy. Not obviously broken, but 2.3x off.
* `permute_35`, **1.4-3.3 ms** — the attention output's layout change. Varies
  more between plan builds than it should.

Each is worth 1-2% of the layer and each is a change in a path every model
shares, which is the trade to weigh before taking one.

### Taken: the biggest product's tile

`fuseqkv_mm_17` is the stacked gate+proj, `24576 x 4096 x 4224`, and it runs at
19.8 TOP/s where the other three products in the same layer reach 23-25. The
chooser already HAS the tiling it wants — `q8gemm_tile`'s `tall` branch,
"a stacked projection, like SwiGLU's gate+up", picks `(2,4,2,2,32,8)` — but the
threshold is `m >= 8k` and Qwen's stacked pair is `m = 6k`, so it falls through
to `m % 256 == 0` and takes `(4,2,4,2,32,8)`.

Interleaved rounds, minimum of each:

    24576 x 4096    (2,4,2,2) 37.92    (4,2,4,2) 42.74*   -11.3%
    12288 x 4096    (4,2,2,4) 18.90    (4,2,4,2) 19.91*    -5.1%

Taken, as `tall = m >= 6k`. What made it safe to take on a microbenchmark
after warning against exactly that: the tile being REPLACED measured
42.86, 43.28 and 42.74 ms across three independent sweeps, and `(2,4,2,2)`
39.59, 39.36 and 37.92 — the ranking between the two challengers moved, the gap
to the incumbent did not. The output is bit-identical, the three neighbouring
products in the same layer do not reach the branch, and Horizon's pinned picks
are all above the old bound already.

**It is still not layer-confirmed**, because the session that found it could no
longer place a 391 MB plan (below). 44 ms of a 167 ms layer, so build the layer
both ways and read it there when a session can: that repeats to 0.3%, and 11%
of that pass is 4.8 ms.

### A session-scale thing that will bite a generation loop

After a day of building and freeing plans, `Mantle`'s pool reported **81.95 GB
reserved against an 80.89 GB capacity**, so `headroom` was zero and no further
plan could be placed — with the largest free span at 31 MB and
`unified_blocks = 1`. Freeing every binding, `collect_for_pool!`,
`reclaim_empty_pool_blocks!` and `trim_gpu_pool!` moved none of it. Whatever
pins those blocks, the effect is that a long-lived process which keeps building
plans eventually cannot build one while its memory is actually free. Worth a
controlled reproduction: build and free N plans in a fresh session and watch
`reserved(dev.pool)`.

## A note on measuring this

The isolated attention launch is ~60 ms and its run-to-run spread reached
**10%** after a day of benchmarking — enough to make a 5% effect look like
either sign, which is how `held` first looked worth taking and how the score
pass count first looked worth skipping. The LAYER replay is 171 ms of sustained
work and repeats to 0.3%. Decide at the layer, not at the kernel: the pass-count
rule was confirmed there (174.3 ms against 170.5 for the same code with the old
default) after the microbenchmark had said both things.

## A trap this measurement fell into

`Lava.FROZEN_VERSION[]` is `"1"` in a normal session, and the frozen SPIR-V
cache is keyed on `typeof(f)`, the argument types and the workgroup size —
**not on the source**. Editing a kernel body under Revise and re-running
therefore measures the OLD kernel, and an hour of flash-attention results here
were exactly that. Signature changes (a new `Val` argument) mint a new key and
are safe; body edits are not. Bump `Lava.FROZEN_VERSION[]` before believing any
kernel measurement.
