# DNNKernels performance — where it stands and what is next

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


Step time on the autocast graphs, RTX 4000 Ada, 240x128: **10.74 ms
(93.1 steps/s)** with capture/replay + barrier elision; 12.1 ms (82 steps/s)
streaming without replay; 16.1 ms (62 steps/s) with a sync every step. Target is
9.1 ms (110); PyTorch autocast is ~111, CUDA.jl on these graphs was 69-73. Correctness is unchanged throughout: `mean |Δ|` 2.78e-4
autocast / 2.91e-4 fp32 against PyTorch's own alpha, 1 differing pixel in 30720
(the known tie-sensitive predicate), DNNKernels suite 61/61.

## The host and the GPU were running strictly one after the other

This was the whole game, and it had nothing to do with kernels.

`Lava.AUTO_SUBMIT_THRESHOLD` was 0, meaning nothing reached the queue until
`flush!`. So a step *recorded* for 16.3 ms with the card idle, then *executed*
for ~12 ms with the host idle: 28.1 ms for 16.3 ms of work and 12 ms of work
that could have overlapped almost entirely.

It presents as a GPU-bound step — host is only 58% of wall — which is why it
survived so long. The test that exposes it is to burn a known number of
microseconds on the host *after* `step!` returns and before synchronising: if
the GPU were still working, small burns would be absorbed and wall would not
move. Wall moved 1:1 (2 ms burn → +1.82 ms, 10 ms → +10.98 ms). Nothing was
executing.

Submitting every 64 dispatches: **28.1 → 19.3 ms**. The curve is flat from 16 to
96 and degrades outside it. That is now Lava's default, with the reasoning at
the constant.

## Then the host became the wall, and it was mostly ceremony

With the two overlapped, host time *is* the step time, and three fixes took it
from 16.3 to 12.2 ms:

* **`LaunchPlan` cache (`ka_backend.jl`).** `ka_launch!` rebuilt
  `Tuple{map(arg_sigtype, tail(all_args))...}` per dispatch and handed it to
  `GPUCompiler.methodinstance` — interning a fresh `Type`, a method lookup, then
  two hash lookups keyed on that type, to rediscover an already-compiled
  pipeline. `typeof(all_args)` is free and types are interned, so an `IdDict` on
  it is a pointer hash. The world counter is stored per entry and checked on
  every hit, so Revise still forces a recompile. **-1.1 ms.**
* **`find_tlas_in_args` made `@generated`.** It walked the argument tuple with
  `Base.tail` on every launch looking for a ray-tracing acceleration structure.
  Whether one *can* be there is a property of the types, so a compute kernel now
  gets `nothing` and no code. The walk did not fold away on KA's long argument
  tuples and was the second-largest host cost, on kernels that have no TLAS and
  never could. **-2.8 ms, with `hoistconstants`.**
* **`hoistconstants` (DNNKernels).** `scalar_tensor`/`full` read no tensor at all,
  yet ran every step. For the 0-d `scalar_tensor` that meant a real
  `vkAllocateMemory` and free every step — `planslab` reserves nothing for a
  shapeless buffer — for a number that never changes. It was the only
  `vkAllocateMemory` left in the steady-state profile.
* **`Op.tag`.** `runop!` dispatched on `Val(Symbol(op.aten))`, interning a Symbol
  per op per step. The name is fixed at load time, so the tag is too. **-0.4 ms.**

## Where the remaining 12.2 ms of host goes

1019 dispatches a step (the folds cut this well below the 2050 assumed earlier),
so ~12 µs of host each against Lava's bare 3.12 µs for a repeated trivial
dispatch. Splitting the profile tree at the KA launch boundary: **~52% Lava's
record path, ~48% DNNKernels' interpreter.** The interpreter half is diffuse — no
single site above ~4 samples, spread across dozens of `runop!` methods, `dest`,
`makeview`, dict lookups — so it will not yield to more peephole work.

## The budget, measured rather than inferred

Turning auto-submit *off* forces the two phases apart, so `wall - host` is GPU
total exactly. That gives the whole budget without a profiler:

| | ms |
|---|---|
| host (recording) | 12.3 |
| GPU total | 11.6 |
| overlapped wall | 16.1 |
| → residual (batch fill/drain) | 3.1 |

The residual *shrinks* as batches get smaller — 6.8 ms at 4 submits a step, 3.8
at 16, 3.1 at 64 — and asymptotes there, while host cost rises with submit count.
Threshold tuning is finished; 64 is the knee.

**So the floor for this graph is ~12.4 ms (81 steps/s)** even with perfect
overlap, and 110 steps/s (9.1 ms) needs *both* halves cut, not either one.

## Record-once-replay — landed, and it removes the host half

`Lava.capture(f, bq)` runs `f` once and keeps its command buffers;
`Lava.replay!(seq)` re-submits them in a single `vkQueueSubmit2` with no
recording at all. The step's launch sequence is identical every iteration
because the slab pins every device address, so there is nothing to rebuild.

**Bit-exact, not approximately equal.** Snapshotting every device array in
`State`/`MemoryBank`, running a recorded step, restoring, then replaying: alpha
matches at `max|Δ| = 0.0` and every state array is unchanged.

| | ms/step | steps/s |
|---|---|---|
| per-step sync (what `benchsteps` measures) | 16.1 | 62 |
| streaming, recorded (sync every 8-16 steps) | 12.1 | 82 |
| **replay** | **11.07** | **90.4** |
| replay, barriers suppressed (wrong results, ceiling) | 7.72 | 129.5 |

Replay is GPU-bound: 11.07 ms against the 11.6 ms GPU total measured
independently. The host is gone as a constraint.

Three things make it work, and each is a trap if changed:
* command buffers recorded under capture use SIMULTANEOUS_USE, not
  ONE_TIME_SUBMIT, which invalidates a CB on submission;
* `submit!` moves ownership of the CBs to the sequence — clearing
  `sealed_cmd_bufs` and swapping in a fresh `cmd_buf` — or the batch pool
  re-records over a sequence `replay!` still points at;
* `reserve_arg_slabs!` pins the arg-slab range the capture filled. A dispatch's
  arguments live in a bump-allocated slab whose *address* is baked into the
  command buffer as a push constant, so a replay reads whatever those bytes hold
  now. The allocator normally rewinds to slab 1 offset 0 on drain, which would
  let the next recording overwrite exactly those bytes.

`flush!` also had to learn about replays: they signal the timeline without
putting a `CommandBatch` in `in_flight`, so the in-flight scan alone returned
before the GPU had run any of it.

**Not yet wired into `step!`.** The step has two shapes — 4 of 5 take the
non-memory branch, every 5th does a memory update (`memevery`) — so production
use wants one captured sequence per branch, re-captured when the shape changes.
`tools/` has no driver for this yet.

## Barrier elision — landed, and worth less than hoped

`Lava.BARRIER_ELISION[]` drops the barrier in front of a dispatch whose buffers
are disjoint from everything touched since the last barrier. Disjoint memory
cannot alias, so this is sound with no read/write annotation on arguments — which
is the whole reason to key on ranges rather than on direction. The ranges come
from `range_leaves!`, the same walk `pin_leaves!` does, so every buffer reachable
through a BDA is counted, including ones nested in wrapper structs.

**93.1 steps/s, 10.74 ms** (from 90.5 / 11.04). Correct: e2e against PyTorch is
bit-identical to the no-elision run (`mean|Δ|` 2.7817e-4, 1 px), and the elided
replay is bit-identical to the plain replay.

It recovers only 0.33 ms of the 3.35 ms of barrier cost, and that is the real
answer, not a bug: **~90% of the barriers in this graph are genuine chain
dependencies**. A DNN step is mostly a chain where each dispatch reads what the
last one wrote. Making copies reset the tracker instead of poisoning it (their
post-copy barrier already orders them against everything after) changed nothing
measurable, which confirms copies were not the limiter either.

Two soundness rules it needs, both learned the hard way:

* **Anything that does not declare its buffers poisons the tracker.** Only the
  KA launch path knows a dispatch's memory; `lava_launch!` used directly by
  Lava's internals and indirect prepares do not. Left alone that is not
  conservative but *wrong* — a later dispatch reading what an undeclared one
  wrote finds no overlap and drops the barrier it needed. Modelled as one range
  covering the whole address space, so it needs no special case.
* **`TOUCHED_RANGES` must be capped.** It only clears when a barrier fires, so a
  long elided run would make the scan quadratic. Past 512 entries it forces a
  barrier and restarts.

A separate correctness fix fell out of this: with `AUTO_SUBMIT_THRESHOLD` on, the
first dispatch of every new batch emitted *no* barrier, because the guard was
`batch.dispatch_count > 0`. A batch boundary is not a synchronisation point —
`submit!` adds no wait on the previous submission — so that first dispatch could
read what the previous batch's last dispatch was still writing. Now barriers when
`in_flight` is non-empty; after a `flush!` the queue is drained and the genuinely
first dispatch still skips.

**Beware the state-restore trap when checking replay against a record.** Both
`capture` and a plain step mutate `State` *and* `MemoryBank`'s scalar fields
(`nvalid`, `next`, `engaged`), so comparing a replay to a record taken from a
different point in that lineage shows a deterministic ~0.007 difference that
looks exactly like a race. The control that settles it in one line: record twice
from the same restored state — they differed by 0.001, so the restore was
incomplete and nothing was attributable to replay or elision.

### Eliminated: redundant pipeline binds

`vk_dispatch_base!` binds the pipeline on every dispatch. Skipping the bind when
it is already bound is correct and was implemented — and is worth **nothing**,
for two independent reasons measured together: only **70 of 1019** dispatches
(6.9%) repeat the previous pipeline, and removing those 70 binds moved the
replayed step 10.740 -> 10.783 ms, i.e. not at all. `vkCmdBindPipeline` is a
command-buffer write, not a state change with real cost. Reverted — it needs a
field on `CommandBatch` and four separate reset sites (fresh CB, split CB, batch
reclaim, capture handover), each a correctness hazard if missed, since stale bind
state means the wrong pipeline executes.

### Eliminated: duplication fusion

Letting a twice-read value stay lazy and recomputing it per consumer (what
Inductor does) reaches only 6 more dispatches under a depth-1 guard, and measures
1019 @ 11.060 ms vs 1013 @ 11.076 ms. Kept behind `DUPLICATE_FUSION[]`, default
off — whole-group codegen needs the mechanism, but pairwise folding cannot pay
for it.

## Where the GPU time actually goes — measured, and it is not what I assumed

Doubling one op family per capture and diffing replayed step time (`OPDOUBLE`,
zero host noise because replay does no recording):

| family | cost | share of step | dispatches |
|---|---|---|---|
| **convolution** | **6.62 ms** | **59.8%** | 325 |
| addmm | 1.34 ms | 12.1% | 96 |
| `_to_copy` | 0.29 ms | 2.6% | 118 |
| add | 0.22 ms | 2.0% | 69 |
| cat | 0.13 ms | 1.2% | 41 |
| bmm | 0.07 ms | 0.6% | 8 |
| mul | ~0 | 0% | 8 |

**All elementwise work put together is 0.71 ms — 6.4% of an 11.08 ms step.**

This invalidates the priority the sections above were written under. Dispatch
*counts* said elementwise dominated (roughly half of 1019); dispatch *time* says
convolution does, by an order of magnitude. Every fusion avenue explored here —
shape-only views, the Inductor plan, duplication — was competing for at most 6.4%,
and their measured zero is unsurprising in hindsight. Whole-group codegen would
be chasing the same slice.

### Per-shape convolution cost, in situ (the trustworthy numbers)

`DNNKernels.OPDOUBLEFILTER` narrows `OPDOUBLE` to a predicate on `(ctx, op)`, so a
single *shape* can be doubled inside a captured, replayed step. That is the only
convolution measurement on this setup that has proved reproducible — the
perturbation is the only thing that changes and the measurement is a whole step.

| shape | count | in-situ | per conv | TFLOP/s |
|---|---|---|---|---|
| 3x3 256->256 @15x8 | 30 | 0.486 ms | 16.2 us | **8.7** |
| 3x3 64->64 @60x32 | 11 | 0.272 ms | 24.7 us | 5.7 |
| 3x3 128->128 @30x16 | 11 | 0.224 ms | 20.4 us | 6.9 |
| 1x1 256->256 @15x8 | 9 | 0.066 ms | 7.3 us | 2.2 |
| 1x1 1024->256 @15x8 | 7 | 0.111 ms | 15.9 us | 4.0 |

**Two things this changes.**

1. The dominant shape runs at **8.7 TFLOP/s**, not the 0.65 the standalone
   harness reported — a 13x inflation. Convolution on the shapes I had been
   staring at is *not* pathologically slow; roughly 6% of tensor-core peak on a
   tall-skinny GEMM with 128x256 of output is unremarkable.
2. **These five shapes are 68 of 134 convolutions but only 1.16 ms of 6.62 ms.**
   The other ~66 account for ~5.46 ms — nearly 5x the cost for a similar count.
   The expensive convolutions are ones I never enumerated, because I picked the
   shape list by *count* and then measured only those.

### Done, and it is a guard: full-resolution convs fall off the coopmat path

Enumerating all 40 distinct shapes showed what sorting by count hid — the
expensive convolutions are at **240x128 and 120x64**, not the 15x8 ones. Measured
in situ:

| shape | count | in-situ | TFLOP/s |
|---|---|---|---|
| **3x3 64->16 @240x128** | 1 | **1.411 ms** | **0.40** |
| 3x3 16->16 @240x128 | 1 | 0.377 ms | 0.38 |
| 1x1 64->16 @240x128 | 1 | 0.147 ms | 0.43 |
| 1x1 3->64 @240x128 | 1 | 0.104 ms | 0.11 |
| 3x3 128->64 @120x64 | 1 | 0.204 ms | 5.56 |
| 3x3 64->64 @120x64 | 1 | 0.111 ms | 5.12 |

**One convolution — `3x3 64->16 @240x128`, once per step — is 1.411 ms: 21% of
the convolution budget and 13% of the whole step.** The 240x128 group totals
~2.04 ms at ~0.4 TFLOP/s, while 120x64 shapes with fatter channels reach 5.1-5.6.

`conv_coopmat_applicable` explains the 13x gap exactly: it requires
`Cout >= 2*GEMM_TILE` (32) and `CRS % GEMM_TILE == 0`. **Cout=16 fails both 16->16
and 64->16; Cin=3 fails `1x1 3->64` (CRS=3).** So every full-resolution
convolution — the ones with the most pixels and therefore the most work — is
rejected and runs on the scalar `conv2d_igemm!` fallback.

**The fix is channel padding, and unlike the M padding I chased earlier it is not
already done.** Pad `Cout` up to 32 and `CRS` up to a multiple of 16 in the
im2col/GEMM path, writing the extra output channels into scratch the epilogue
ignores. For `3x3 64->16 @240x128` that is 2x wasted output channels to move
1.4 ms onto a path running 13x faster — worth roughly 1.2 ms on its own, over
half the 2.0 ms needed for 110 steps/s.

**Is the coopmat path actually better at M=30720?** No 240x128 convolution passes
the guard, so this cannot be A/B'd in situ without building the padding first.
But the measured points bracket it: coopmat reaches **5.56 TFLOP/s at M=7680**
(`3x3 128->64 @120x64`) and 5.12 at the same M, and M=30720 offers 4x the tile
parallelism, so a padded 240x128 convolution should land at or above that — the
GEMM there is *less* occupancy-starved than the ones already succeeding, not
more.

Expected payoff for `3x3 64->16 @240x128`: 1.411 ms at 0.40 TFLOP/s today; at
5 TFLOP/s the same FLOPs take ~0.11 ms, and padding Cout 16->32 doubles the GEMM
work, so ~0.23 ms. **Saving ~1.18 ms from one convolution** — with the rest of
the 240x128 group (~0.6 ms today) likely another ~0.4 ms. That is 1.5 ms of the
2.0 ms needed for 110 steps/s, from a change confined to the conv path.

**Pad the weights at load time, not per step.** Weights are constants, so a
zero-padded `Cout`/`CRS` copy belongs beside `hoistpermutes` in the load-time
passes rather than in the per-step kernel. That keeps the run-time change to
buffer shapes and the epilogue's channel stride, and costs nothing per step.

### How far convolution is from the hardware

134 convolutions, **16.77 GFLOP/step in 6.62 ms => 2.53 TFLOP/s**, against ~26
TFLOP/s fp32 and ~153 TFLOP/s fp16 tensor-core. Both inputs to that ratio are
solid: the GFLOP count is arithmetic over the graph shapes, and the 6.62 ms is
the in-situ `OPDOUBLE` attribution measured through replay. So convolution really
is running at a few percent of the card.

Shape mix: 30x `3x3 256->256 @ 15x8`, 11x `3x3 64->64 @ 60x32`, 11x
`3x3 128->128 @ 30x16`, 9x `1x1 256->256 @ 15x8`, 7x `1x1 1024->256 @ 15x8`.
The dominant one is M=120 (padded to MP=128), N=256, K=2304 — a tall-skinny GEMM
whose 128x256 output over 16x16 tiles gives poor occupancy no matter what.

### Two dead ends here, and a warning about the harness

**Dead end 1 — "M=120 fails every divisibility test".** It does not:
`conv_coopmat.jl:178` computes `MP = padtile(NPQ)` and passes MP=128, which
selects BLK=2 + splitk=6. I had called `coopmat_gemm_shape` with an M the caller
never passes.

**Dead end 2 — the occupancy-vs-intensity trade.** `coopmat_gemm_shape` takes the
first block reaching `4*cores` = 192 tiles, choosing BLK=2/splitk=6 over
BLK=4/splitk=8 (which reaches only 64). Forcing max-block-when-it-tiles, A/B'd in
one session: **0% to -3%** across all five shapes. The trade is not the problem.

`tools/conv_bench.jl` exists now: it repeats each shape until the running minimum
stops improving (rather than best-of-N, which cannot tell "warm" from "lucky")
and cross-checks the count-weighted total against the in-situ figure. **It reads 13.87 ms against the in-situ 6.62 ms** — 2.1x high. Rotating
destination and workspace across 8 slots, on the self-serialisation theory, moved
it only to 13.10 ms, so that theory was wrong; and identical code gives
`1x1 256->256` at 16.0 us one run and 116.1 us the next while the convergence rule
reports both as settled. Something outside the timed loop dominates and the rule
does not detect it. **Do not tune convolution against standalone timings on this
setup.** Attribute with `OPDOUBLE` through replay, which has survived every
check; the harness is useful only for its cross-check until that total agrees.

**Standalone convolution microbenchmarks in this repo are not trustworthy.** The
same five shapes measured 216/290/200/271/18 us in one session and 31/42/34/17/20
us in the next — 5-7x apart — because per-shape coopmat kernels compile on first
use and best-of-N does not always exclude it. An apparent "1x1 256->256 is 15x
slower than 1x1 1024->256" anomaly came entirely from that and **does not exist**
(16.6 vs 20.4 us when warm). Only the in-situ `OPDOUBLE`-through-replay
attribution should be trusted for conv cost; if a standalone number is needed,
warm each shape to a stable minimum first and re-run in a second session before
believing it.

**The gap to 110 steps/s is 2.0 ms, and convolution alone is 6.6 ms.** A 30%
faster convolution closes it outright. The older per-kernel section below already
recorded `convolution_igemm!` running 2.5-2.8x slower than CUDA on identical
source, which is exactly the headroom this points at: tiling, tensor-core
utilisation and the im2col+GEMM split, not launch structure.

Measure before optimising *and* measure what dominates, not what is numerous —
this session lost several rounds to the second half of that.

## What is left

10.74 ms against the 9.09 ms that 110 steps/s needs. The remaining barrier cost
(~3 ms) is genuine dependencies, so it only comes down by having *fewer, larger*
dispatches — which is the fused-kernel codegen in the section below, now with a
different justification than when it was first proposed: not to cut launch
overhead on the host (replay already removed that) but to shorten the dependency
chain on the device.

## Test status

`dev/Lava/test` has no runnable full suite in this project env: `runtests.jl`
needs `SPIRV_Tools_jll`, and 29 of the 67 `test_*.jl` need `LLVM`, `Atomix` or
similar, none of which are project dependencies. Of the 38 that load, **28 pass
and 9 fail — and all 9 fail identically at clean `HEAD` (69af2c4)**, verified by
stashing: `graphics_pipeline`, `handwritten_spirv`, `instance_writer_kernel`,
`phase4_singlethread`, `rapid_alloc_free`, `repeat_inner_3d`, `source_mapping`,
`struct_alignment_systematic`, `tlas_allow_update`. Pre-existing, not from this
work. `test_caching_and_allocations` fails only under parallel load.

Landed since the first measurement, all verified end-to-end against PyTorch's
own alpha (`tools/lavadnn_e2e_cpu.jl`, mean |Δ| 2.75e-4 — the fp32 path is
2.91e-4, so the fp16 route costs nothing measurable):

* **`foldrelu`** — 50 relus folded into the convolution epilogue that was about
  to write the same elements anyway.
* **Split-K reduction fused into the epilogue** (`coopmat_gemm!(...;
  reduce=false)`), for both convolution and `addmm`: one dispatch and one full
  write-plus-read of `M*N*splitk` floats saved per call.
* **`sdpa` scratch onto the `Workspace`**, halving the pool churn.

Remaining scratch pressure is real: the largest convolution's im2col matrix is
35 MB (`MP=30720, CRS=576`) and the workspace peaks near 70 MB, which is what
OOMs when something else holds the card. Tiling im2col+GEMM over `NPQ` would
bound it.

## Close the CUDA gap first — it is 2.5x and fully diagnosed

The same DNNKernels graph on the CUDA backend runs at **69-73 steps/s (13.4-14.4 ms)**
against Lava's **27.6-30.8 (32.4-36.3 ms)**. Same ops, same dispatch count, same
card, measured in one session. So the launch *structure* is not the wall — the
fusion work below is premature until this is closed.

Two causes, both measured in isolation, and they account for essentially all of it:

**1. Kernel codegen — Lava is 2.5-2.8x slower on identical source.**
`convolution_igemm!`, one dispatch per call, launch overhead negligible against a
200 µs kernel:

| shape | Lava | CUDA | |
|---|---|---|---|
| (15,8,256) 3x3x256 | 0.194 ms | 0.069 ms | 2.8x |
| (60,32,64) 3x3x64  | 0.232 ms | 0.093 ms | 2.5x |
| (120,64,64) 3x3x64 | 0.337 ms | 0.132 ms | 2.5x |

Uniform across shapes, so it is not tiling at one size. A pure-ALU probe
isolates it further: a dependent `muladd` chain measures **4.2 TFLOP/s on Lava
against 24.5 on CUDA.jl**, where this card's fp32 peak is ~26.7 — CUDA is at 89%
of peak, Lava at 16%, on a kernel with no memory traffic in the loop at all.

**What it IS, mostly: Lava never unrolls loops.** LLVM unrolls for NVPTX and
declines for SPIR-V, and on a dependent `muladd` chain that is worth **3.5x**:

| loop body | Lava | CUDA.jl |
|---|---|---|
| 1 fma per iteration | 3905 GFLOP/s | 23445 |
| 8 fma per iteration (`@nexprs`) | **13699** | 23587 |

That closes this kernel's gap from 5.6x to 1.7x, and it is the largest single
lever found. Two ways to take it:

* **Per kernel, by hand** — `Base.Cartesian.@nexprs` with a `while k + U <= K`
  head and a remainder loop, which handles trip counts that are not multiples of
  `U`. Done for `strided_gemm_kernel!` (verified at K = 53 and 255). Note it did
  *not* move the model: those 37 matmuls are dispatch-bound, so pick targets by
  where loop time actually is — the coopmat GEMM's `KPER` loop and
  `conv2d_igemm!`'s reduction are the ones that matter.
* **In the compiler**, which subsumes all of it: `unroll_loops!` in
  `compilation.jl` is wired up but inert, because `LoopUnrollPass` only unrolls
  partially when a target's `TargetTransformInfo` opts in and a SPIR-V pipeline
  has none. Attaching `llvm.loop.unroll.count` metadata to each loop latch
  overrides the cost model outright and is the way in.

  One implementation detail that will otherwise cost an hour: LLVM's loop
  metadata must be a **distinct, self-referential** node (`!0 = distinct !{!0,
  !1}`) — `Loop::getLoopID()` rejects anything whose first operand is not the
  node itself, so `LoopUnrollPass` will simply not see a uniqued node.
  `LLVM.MDNode` in LLVM.jl only builds uniqued nodes; the self-reference has to
  be patched in afterwards via `LLVM.API.LLVMReplaceMDNodeOperandWith`. Finding
  the latches without a `LoopInfo` analysis is the easy part — for
  Julia-generated IR a back edge is a branch to a block earlier in function
  order.

**Beware `LLVM.clopts` for measuring this.** `--unroll-count=8
--unroll-allow-partial` reported 4200 -> 2103 GFLOP/s and led me to write
unrolling off entirely; those options are process-global and perturb every other
compilation in the session. Measure at source level with `@nexprs`.

**What the residual 1.7x is NOT** — each checked and eliminated, do not redo:

* *Not FMA contraction.* The SPIR-V emits a correct
  `OpExtInst GLSL.std.450 Fma`.
* *Not register pressure or occupancy.* `Lava.enable_pipeline_executable_properties!()`
  then `list_compiled_kernels()` reports **16 registers** for that kernel (96 for
  `conv2d_igemm`). 16 registers cannot limit occupancy.
* *Not loop overhead.* The loop is left rolled (`OpLoopMerge`, one FMA per
  iteration) where LLVM-NVPTX would unroll it, which looked like the answer —
  but forcing an 8x unroll via `LLVM.clopts("--unroll-count=8",
  "--unroll-allow-partial")` made it **worse**, 4200 -> 2103 GFLOP/s. A
  dependent chain is latency-bound, not issue-bound, so unrolling only costs
  registers. `unroll_loops!` exists in `compilation.jl` and is off for this
  reason.
* *Not workgroup size.* Flat at ~4.3 TFLOP/s from 32 through 1024.
* *Not shared memory limiting blocks-per-SM.* 16% of peak is suspiciously close
  to one 256-thread workgroup resident per SM (8 warps of 48), which a large
  `Workgroup` allocation would cause — but the emitted SPIR-V declares no
  `Workgroup` storage at all, only the three builtins and the push-constant
  block.

**What it partly IS: 64-bit integer arithmetic.** Julia hands out `Int64`
indices and Lava emits them unchanged; NVIDIA has no native 64-bit integer unit,
so adds and compares are emulated and *division* especially so. Narrowing a
tight loop's counter to `Int32` measured **2100 -> 3282 GFLOP/s (1.56x)**, and
rewriting `im2col_kernel!`'s index math in `Int32` — four divisions and two
remainders per element — took the whole step from **35.7 to 32.7 ms**. Doing the
same to `conv_epilogue_kernel!` gained nothing, so the payoff is specifically
where integer *division* sits in an inner loop, not from narrowing per se.

Remaining candidates, in order:

1. **A general fix in Lava** rather than per-kernel edits: narrow index
   arithmetic to 32 bits during emission when the extent provably fits, or give
   LLVM a TTI that reports 64-bit integer ops as expensive so `IndVarSimplify`
   narrows them itself.

   Two *specific* attempts at this were tried and reverted, so do not repeat
   them without a better plan:

   * Narrowing the linear index in Lava's broadcast kernels produced wrong
     results — **and that turned out to be a Lava miscompile, not a property of
     the optimisation.** It is captured as a failing test in
     `dev/Lava/test/test_int32_cartesian_miscompile.jl` (`@test_broken`), with
     the bisection that isolates it:

     | expression | result |
     |---|---|
     | `cis[I % Int32]`, components written out | correct |
     | `cis[I % Int32]` indexing a dense `LavaArray` | correct |
     | `cis[I % Int32]` indexing a `SubArray` / `PermutedDimsArray` | correct |
     | `cis[I % Int32]` indexing a **`Broadcasted`** | **WRONG** |

     It is the *index* being `Int32` that matters, not the axes: a
     `CartesianIndices` with `Int` axes indexed by an `Int32` fails identically,
     and widening the recovered index straight back to `Int` does not help.
     Crucially it only reproduces after `Broadcast.preprocess` — without the
     `Extruded` wrappers one of the two shapes computes correctly. That points
     at `Base.newindex(::CartesianIndex, keeps, Idefaults)`, which combines the
     index with tuples built from `Int` axes via `ifelse`; mixed-width integers
     through that `map` is the thing to look at in the emitter.

     Fixing it unlocks the narrowing for every broadcast, and removes a silent
     wrong-answer hazard for any kernel reaching for a 32-bit index.
     `test/test_broadcast_paths.jl` covers the five paths that a wrong `_copyto!`
     would break; nothing else in either suite caught this.
   * Flattening `DNNKernels.launch!` to a 1-D `ndrange` was wrong *and* slower
     (matte 2.8e-4 -> 0.27, step 31.7 -> 34.6 ms). See its docstring: the
     broadcast win came from removing `Broadcasted`'s per-element div/mod, not
     from the launch geometry, and `ndmap!` has no such division to remove.

   A hand-rolled 32-bit decomposition (as in `im2col_kernel!`, which is
   verified and worth 9% of a step) is the shape that works.

2. **The rest of the ALU gap**, which none of the above explains, needs
   visibility below SPIR-V — what the driver does with 64-bit
   `PhysicalStorageBuffer` addressing versus PTX's 32-bit offsets, and whether a
   Vulkan compute shader gets the same warp residency as a CUDA kernel.
   **`ncu` cannot do this**: it is installed at
   `/opt/nvidia/nsight-compute/2026.2.1` and runs, but reports "No kernels were
   profiled" against a Vulkan workload — it intercepts CUDA only. Nsight
   Graphics (`ngfx`) is the tool for Vulkan compute shader counters; `nsys` will
   give a Vulkan timeline but not SM-level metrics.

**2. Barriers — 8.25 µs per dependent dispatch, against CUDA's 1.78 µs total.**
A dependent 64-element dispatch: Lava 9.67 µs, Lava with barriers elided
1.42 µs, CUDA 1.78 µs. **Lava's raw launch is faster than CUDA's**; the whole
difference is the barrier's pipeline drain, which CUDA gets for free from stream
ordering. In the real model most of that drain hides behind useful work, so the
measured ceiling for eliding every barrier is 6.13 ms of 36 — still the second
biggest item, and the graph's read/write sets plus the plan's byte ranges are
enough to decide elision.

**Already fixed: broadcast launch geometry (6x).** `GPUArrays._copyto!` launches
`ndrange = size(dest)`, so an N-D destination got an N-D index space and
KernelAbstractions partitioned it into N-D workgroups — consecutive lanes no
longer walking consecutive memory. Worse, `getindex(::Broadcasted, ::Integer)`
on an N-D tree is defined as `bc[CartesianIndices(bc)[i]]`, so even a linear
kernel paid a div/mod chain per element. `Lava` now overrides `_copyto!` to
flatten both the range and the expression tree. `D .= A .+ B` over 16.7M Float16:
54 -> 332 GB/s (CUDA.jl: 206). CUDA.jl never hit this because it replaces
broadcast with its own flat grid-stride loop instead of using the KA path.

## Fusion — the plan now exists; consuming it properly does not

`tools/dump_plan.py` is built and works: it merges Inductor's fusion decisions
into `gen/graphs/aten-autocast/*.json` as `fusion_groups`, **127 groups covering
568 distinct ops**. Julia loads them (`Graph.fusion`, preserved through every
rewrite pass) and `fuse.jl` uses them.

Three things that cost time getting there, so they do not have to again:

* **Compile the exported module, not the original.** Matching Inductor's node
  names against our export fails outright — our export runs
  `run_decompositions()` and lifts views into buffers, so it has 1 `sub` where
  Inductor's post-grad graph has 44, and 0 `view` where Inductor has 176.
  Compiling `export(...).run_decompositions().module()` makes Inductor's
  *pre-grad* graph exactly ours, and `preToPost` is then keyed by our own op ids
  — no name matching at all.
* **Key groups on the call site (`kernel:N`), not the kernel name.** Inductor
  emits one kernel and calls it from many places; collapsing those put six
  unrelated `relu`s in one "group" and would have fused ops that never run
  together.
* **`force_disable_caches`.** A second run hits the FX graph cache, skips
  codegen and emits no debug directory — indistinguishable from "provenance
  unavailable".

**But the payoff so far is zero, measured.** Interleaved A/B: 27.89 ms with the
plan against 27.75 without. It unlocks only 11 extra fused ops (71 vs 60),
because `fuse.jl` fuses by returning a lazy `Broadcasted` up a producer→consumer
chain, and the binding constraint is not the plan — it is that 89 candidate ops
are read through a *real view* (reshape/slice/permute), which needs materialised
storage, and 39 more are multi-use.

Exploiting the plan properly means emitting one kernel per group regardless of
views in between — real codegen, not a peephole. The plan file it needs is in
place.

**Corrected: this is not the order-of-magnitude item.** Measuring where the
blocked candidates actually go, rather than only counting them, changes the
conclusion. Of 344 elementwise ops, 72 already fuse and the rest break down as:

| blocked by | n | fusable in principle? |
|---|---|---|
| multi-use (2+ readers) | 67 | yes, by recomputing as Inductor does |
| `permute` view | 48 | needs a lazy permuted expression |
| consumer is `addmm` | 36 | no — that is prologue fusion into a GEMM |
| consumer is `convolution` | 42 | no — same |
| shape-only view (`view`/`unsqueeze`) | 82 | **yes, and now done** |
| `cat`/`bmm`/`layer_norm`/other | 26 | no |

Shape-only views turned out to be the cheap 82: a reshape of an elementwise
expression is that expression over reshaped operands, exactly the identity
`flat1` already uses in Lava's broadcast path. `lazyreshape` implements it and
`fuse.jl` walks through those views. **It bought 2 fused ops, not 82** — because
what sits *behind* those views is overwhelmingly `addmm`, `cat` or
`convolution`, not another elementwise op. The walk is kept (it is correct, and
group codegen needs it) but it is not a lever.

Ceiling check: even fusing every elementwise dispatch away removes ~520 of 1019
dispatches. At the measured cost that is worth a few ms, not the 6 ms needed.
Dispatch elimination is a second-order lever here; **host-side cost per dispatch
is the first-order one**, which is what the section at the top of this file
attacks.

## Older notes

**2180 dispatches per step, 17.5 µs each on average.** PyTorch reaches 111
steps/s (9 ms) running a few hundred fused kernels; we run an order of magnitude
more of them. No amount of per-kernel tuning closes that — at ~38 ms for 2180
dispatches, even a *free* GPU would leave the launch structure as the wall.

That is what `fusion_groups` in the exported graph JSON is for, and it is still
`[]` in every file because `dump_plan.py` was never built. The route is known
and written up in [[lava-dnn-matanyone2]]: run the model under
`TORCH_COMPILE_DEBUG=1` with the inductor backend, read
`inductor_provenance_tracking_node_mappings.json`, invert `postToCppCode` (its
node names match our op ids exactly — `relu`, `add_1`, `convolution`), group by
fused-kernel name, and emit `fusion_groups`. Then DNNKernels emits one kernel per
group instead of one per op. This is the plan — "let PyTorch do the fusing" —
and it is the only item on this list with the right order of magnitude.

Barriers are 6.13 ms (16%) of the step, measured by ablating every one of them
(`Lava.concurrent_dispatch_group`, wrong results but a valid ceiling). An
earlier reading of 3.4 ms was taken through OOM-reclaim stalls and understated
it. Dependency-aware barrier elision would recover some fraction of that; the
graph's read/write sets plus the static plan's byte ranges are enough to decide
it, but most consecutive ops in a DNN genuinely do depend on each other.

## Measured budget

By differential ablation (`DNNKernels.OPDOUBLE`, see `tools/lavadnn_famcost.jl`);
the families sum to 50.3 ms against a 49.3 ms step, so nothing large is missing.

| family | ms | n/step | note |
|---|---|---|---|
| convolution | 19.6 | 119 | im2col + coopmat GEMM + epilogue = 3-4 dispatches each |
| addmm | 2.75 | 48 | was 6.66 before the tensor-core path |
| cat | 5.18 | 21 | rewritten to write into the plan; NOT re-measured |
| relu | 3.98 | 86 | |
| _softmax | 3.89 | 3 | rewritten as one fused kernel; NOT re-measured |
| add | 3.30 | 80 | |
| _to_copy | 2.23 | 118 | now writes into the plan; NOT re-measured |
| upsample_bilinear2d | 2.10 | 4 | 525 us per call |
| div | 1.98 | 10 | 200 us per call |
| mean.dim | 1.49 | 3 | 500 us per call |

Barriers are **not** the problem: removing every one of them (via
`Lava.concurrent_dispatch_group`, wrong results but a valid ceiling) saves
**3.4 ms of 46**. The ~12 us-per-dependent-dispatch figure from a 64-element
microbenchmark does not transfer — real kernels overlap. `Lava.BARRIER_MODE[] =
:execution` (no memory barrier) is no cheaper either, so the cost is the
execution dependency, not cache maintenance.

## Next, in order of expected value

1. **Fuse `conv -> relu`.** 86 relus a step, each a full read-modify-write pass
   over a conv output plus a dispatch. The pass is the same shape as
   `foldbn.jl`: find a `relu.default` whose operand is a convolution used
   nowhere else, set an attr on the conv, drop the relu, alias its buffer.
   Careful: `conv2d_igemm!` writes back **atomically when `SPLITK > 1`**, so the
   activation cannot be applied per-split there — either restrict the fold to
   the coopmat path (which has a separate epilogue kernel and is trivially safe)
   or force `splitk = 1` on folded convolutions.

2. **Cut conv dispatches from 3-4 to 2.** The split-K reduction is its own pass;
   fold it into `conv_epilogue_kernel!`, which already reads the GEMM result.
   For 1x1 stride-1 convolutions with `NPQ % 16 == 0` the im2col is the identity
   (`x` is already the `(NPQ, Cin)` matrix in the right layout) and can be
   skipped entirely — but only when `NPQ` lands on the tile, or the last
   cooperative-matrix load reads past the end of the buffer.

3. **The long tail of one-op-at-a-time elementwise work.** `relu`/`add`/
   `_to_copy`/`div` are ~11 ms across ~300 dispatches. Julia's own broadcast
   fusion gets this for free if elementwise `runop!` methods return a lazy
   `Broadcasted` when the buffer has exactly one consumer and that consumer is
   also elementwise, materialising into the planned slot at the first
   non-broadcast use. Use counts are already computable per graph.

4. **`upsample_bilinear2d`, `div`, `mean.dim`** are 100-500 us per *call* on
   tiny tensors. Each is worth a look on its own; the `_softmax` rewrite (four
   passes and two `AcceleratedKernels` reductions -> one workgroup per slice)
   is the template.

5. **Stop allocating inside ops.** Lava reports **596 MiB live across 103
   buffers** for this model. `sdpa` (`kernels/extern/attention.jl:95`) allocates
   per call and is what OOMs first when the card is busy; it and every other
   `KernelAbstractions.allocate` in an op body should take a `Workspace` the way
   `convolution_coopmat!` and `matmul_coopmat!` now do. This is a speed item
   *and* what makes the model fragile when something else holds VRAM.

## Do not repeat

- Per-op `KA.synchronize` attribution and per-dispatch Vulkan timestamps both
  measure themselves: 239 ms and 276 ms respectively for a 46 ms step.
- Best-of-5-rounds x 100 iterations minimum for any kernel timing. One warm-up
  call left a coopmat GEMM reading 2.34 TFLOP/s when the truth was 21.45.
- Gating cooperative-matrix tiles behind `if i <= BLK` inside one kernel makes
  the matrices conditionally-defined and pins every shape at ~0.15 ms; generate
  one kernel per block size. A runtime loop bound over K costs the static trip
  count the same way.
- The GPU is shared with Simon's VS Code Julia REPL. Check
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` before
  concluding anything about a timing change.
