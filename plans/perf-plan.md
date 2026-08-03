# SAM 2.1 performance — where it stands and what to do next

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


Parked 2026-07-29 for the repository refactor, which is done: the packages moved
to [JuliaVision](https://github.com/SimonDanisch/JuliaVision) (`LavaDNN` is
`DNNKernels` there) and the editor to
[VideoEditor.jl](https://github.com/SimonDanisch/VideoEditor.jl). Paths below
are written as they were; `dev/LavaDNN` now means `dev/JuliaVision/DNNKernels`.

Everything is measured on the RTX 4000 Ada at 2265 MHz, autocast precision,
against `tools/sam2_pytorch_baseline.py` on the same card.

Every figure below is from a *fresh* session — see "How to measure this", which
is not boilerplate: three separate confident numbers in this document's history
were artefacts of how they were taken.

## 2026-08-02: sizing the attention rewrite before anyone starts it

The remaining encode gap is attention — 34.27 ms against PyTorch's ~19.3 on the
same 203.1 GFLOP, i.e. 5.9 TFLOP/s against 10.5. Six hypotheses are dead and the
seventh on the list is the reference's structure: `row_split`, where each
subgroup owns a slice of rows so the row maximum is a `subgroupMax` instead of a
shared-memory reduction behind barriers. The kernel takes **about eight barriers
per key block**, which is 1024 per workgroup on the global shape, so the theory
has a shape.

**It is worth about 2 ms, not 14.** The two switches that already change the
softmax's structure bound it, measured in context:

    shipped (one-pass + lazy rescale)   34.27 ms
    two-pass softmax                    36.30   +2.03
    eager rescale on every block        35.12   +0.85

Those are the softmax's *whole marginal structure* — the passes and the rescale
policy — and they move attention by 5.9% and 2.5%. A rewrite that removes the
shared reduction cannot plausibly be worth much more than the 2.03 ms that
removing a whole pass is worth.

So the gap is the two products, not the softmax around them. At `E = 72` they are
`64x32x80` and `64x80x32` tiles interleaved with a running softmax — a shape that
cannot reach a GEMM's arithmetic intensity, and where cuDNN's 1.8x comes from
warp-level primitives and a much larger effective tile rather than from a better
softmax. That is a different kernel, not a refactor of this one.

**What this changes**: `row_split` is worth doing for ~2 ms, not as the route to
90%. The route to 90% needs either that different kernel or the 4.6 ms found
elsewhere — decode's replay (1.0, measured), and the elementwise tail (~2.9 at
60% of bandwidth, no single cliff).

## 2026-08-02: the encode's remaining buckets, sized against bandwidth

With `addmm` and attention worked over, the rest of the encode was checked the
only way that separates "slow" from "as fast as the memory allows": bytes it
must move against what it takes.

    layer norm   96 ops   2255.5 MB   6.01 ms    38% of bandwidth   AT PARITY
    max-pool      6 ops    264.2 MB   1.81 ms    15%                CLIFF
    add+clone+
      _to_copy   285 ops  4397.2 MB   7.34 ms    60%                no cliff

**Layer norm is done.** 38% looks poor until you read the kernel: it re-reads its
input twice for the two reduction passes, but a normalisation group is 144 to
1152 elements, so those come out of L1. Its own docstring carries the number that
settles it — PyTorch does all 96 in 5.64 ms and we now do them in 6.01, so the
fused kernel closed a 37.6 ms gap to 0.4. This bucket is closed.

**~~Max-pool is a cliff~~ — it is not, and the 15% was the wrong yardstick.**
The reasoning wrote itself: 2x2 at stride 2 with no padding means the windows do
not overlap, so 264.2 MB is exact rather than an estimate, 1.81 ms against 0.26
at 1 TB/s is 15%, and every input is a `permute.default` view — the same wrapper
indexing `densify` existed to work around and that cost the flash kernel 11 ms.

**Built, and it is 16x SLOWER**: 1.05 -> 17.15 ms, encode +12.4. Pinning the
workgroup to `(64,1,1,1)` — on the theory that a 4-D `ndrange` partitions into
4-D workgroups and decoalesces, which is a real effect `mm_epilogue_kernel!`
documents — changed nothing: 16.34 ms.

The yardstick was wrong. `x` is a permute of a **C-fastest** layout, so
`strides(x)[1]` is `C`, i.e. 288: consecutive `ox` are 288 elements apart and the
read is scattered *however it is written*. 1 TB/s assumes a coalescing this
access pattern cannot have, and 264.2 MB in 1.05 ms is 251 GB/s on a fully
strided gather — not a cliff at all. Reverted.

Two things generalise. **A "% of bandwidth" figure is only a diagnosis if the
access can reach that bandwidth** — read the strides before believing the
denominator. And the wrapper-indexing story, right twice today, is not a general
explanation for a slow kernel touching a view: here the wrapper cost nothing,
because the memory pattern and not the index arithmetic was the limit.

**The elementwise tail is not a cliff either, and the "60%" was meaningless.**
Broken out, with each operand's layout checked:

    _to_copy   97 ops  1584.7 MB  1.48 ms  1071 GB/s   96 dense operands
    add        98 ops  1897.9 MB  2.97 ms   639 GB/s  191 dense, 5 permuted
    clone      90 ops  1347.2 MB  2.89 ms   466 GB/s   90 permuted (all of them)

`clone` reads a permuted view every time, so 466 is a gather rate and the blocked
gather already tuned it. `add` at 639 looked like the one gap — until the control:
a plain three-stream add on 64 MB arrays reaches **318 GB/s** on this card, and a
two-stream copy 306, because those are DRAM-bound.

**The encoder's ops are not.** 1071 GB/s is above this card's DRAM peak, which
means `_to_copy` is running out of L2 — its tensors are 4 to 33 MB against a
48 MB cache. So 639 against 1071 is a difference in how resident two working sets
are, not a kernel to fix, and a "% of bandwidth" for a cache-resident op needs to
say *which* memory before it means anything. Same error as the max-pool
denominator, in a different disguise.

With that, every bucket in the encode has been measured: `addmm` at 35 TFLOP/s
against a kernel that demonstrably reaches 44, attention at 5.9 against PyTorch's
10.5, layer norm at parity, convolution routed, gelu folded, the elementwise tail
cache-bound, max-pool strided by construction. **The remaining gap is attention's
kernel design and nothing else.**

## 2026-08-02: a 1x1 convolution is a GEMM on the input as it already lies

Five of the encoder's seven convolutions are 1x1, and as a family the seven ran
at **2.1 TFLOP/s** — against the GEMM's 44 and attention's 6. Taking the largest,
`convolution_4` (144 -> 256 channels over 256x256, 4.83 GFLOP): 0.860 ms of which
the GEMM is 0.411. **The other 52% is `im2col` and the epilogue.**

Neither is doing anything at 1x1. `im2col` gathers each output pixel's receptive
field into a row, and when the field is one pixel the column matrix *is* the
input reshaped. The scatter is the identity for the same reason: `C` is
`(NPQ, Cout)` with `NPQ` contiguous and the reversed output `(W, H, Cout, N)` is
those bytes in that order. So the whole convolution is `matmul!` on a `reshape`.

    convolution.default   6.06 -> 4.90 ms   (in context, interleaved)
    VRAM                  1220 -> 1161 MiB  (the im2col scratch is gone)
    encode                unchanged, +0.19 ms — inside the noise
    masks                 1.00000 / 0.99958 / 0.97727, second one marginally better

**Kept although the encode did not move**, which is the opposite of the call made
on the attention tiling above, and the difference is worth being explicit about.
That one bought no measurable time *and* moved a mask IoU the wrong way. This one
buys no measurable time, costs no accuracy, frees 59 MB, and deletes a pass — the
op itself is 1.17 ms cheaper and something else is overlapping it. A change that
is free on every axis and simpler is worth keeping even when the wall clock
cannot see it; one that trades accuracy for nothing is not.

The bias does not come free: `coopmat_gemm!`'s bias is per-*row* of `C`, which is
per pixel here, and a convolution's is per output channel. So it stays a separate
broadcast — one pass over the output against the two this removes.

**And a K that does not tile costs more than the padding would.** Four of the
seven fall off the staged GEMM entirely, `convolution_4` because `K = Cin = 144`
is not a multiple of `BK = 32`. Padding K to 160 with zeros — which contribute
nothing to a dot product — takes that shape from 0.411 to **0.278 ms**, 32%
faster while doing 11% more arithmetic. Not implemented: at 1x1 the operand is
the input itself and padding it needs the copy this change just removed. It is
the right move for the 7x7 and for any `Cin` that misses the tile.

## 2026-08-02: decode's last 40% is host, and `replay!` takes 1 ms of it off

The packaged decode is 3.9 ms and **40% of it is not any graph op** — 149
dispatches' worth of host recording. `Lava.capture`/`replay!` exists for exactly
that and had already removed the host half of MatAnyone's step. Measured on SAM
2's decoder, captured once and replayed:

    decode  record+run 4.211 ms   replay 3.208 ms   -1.003 ms
            50% of PyTorch -> 65%, and bit-exact

**`CACHE_DECODER_INPUTS` turns out to be the precondition, not an optimisation.**
`capture`'s contract is that every device address is the same next time — "an
input buffer written in place rather than reallocated" — and the cache is what
makes the decoder's converted inputs stable across clicks. Measured on its own it
is worth +0.008 ms and was nearly filed as pointless; it is worth 1 ms as the
thing that makes a replay legal. That is the second switch this week whose
recorded justification was the wrong way round (see `FLASHCM_DENSIFY`).

### The bug that made it impossible: replaying next to ordinary recording

The intended use of `capture`/`replay!` is a drained queue with nothing else in
flight, and that is all any test covered. The real use is not that — an editor
replays a captured decode on every click while a preview render, a thumbnail or
the next encode records around it. That mixture **did not work**:

    AssertionError: batch signal desync: 859 vs 860

`ensure_active_batch!` reserves `bq.next_timeline + 1` as a new batch's signal
value and `submit!` asserts the reservation still holds; `replay!` bumps the same
counter, so a batch that was open at the time is stale and dies on its *next*
submit — naming neither the replay nor the batch. `replay!` now closes an open
batch before touching the counter, which is free in the drained case and is the
whole of what makes interleaving legal.

`test_replay_interleaved.jl` covers it, and the test was checked the only way a
regression test is worth anything: with the fix removed it fails, "batch signal
desync: 5 vs 9". Draining between the two operations hides the bug completely,
which is why every earlier test passed.

**What is left to ship it**: SAM 2 needs a captured sequence cached per
embedding, invalidated when `feats` changes, and `prompt` has to write its two
tensors into persistent buffers instead of allocating a pair per click. The
kernel work is done and measured; that part is API.

## 2026-08-02: the autocast decoder is now the default — decode 11.3 -> 3.9 ms

`gen/graphs/sam2-large/sam2_decoder.json` is the autocast export. The two
encoders are **structurally identical** — same 1353 op ids, same outputs, so the
reference activations and `bench_sam2.jl`'s node lookups are untouched — and only
the decoder differs, 174 ops becoming 308 as autocast's dtype casts appear.

    decode  11.30 -> 3.93 ms    52% of PyTorch, from 18.6%
    encode  unchanged at 102.5
    VRAM    1348 -> 1220 MiB
    masks   1.00000 / 0.99952 / 0.97727, from 0.98750 / 0.99964 / 0.95455

Two of the three mask IoUs improved and the first is now exact. The mechanism is
on the record above: half the fp32 decode was seven 23-token attentions that
`flashcm_applicable` had to refuse because their operands were fp32, so they ran
the three-pass fallback. In fp16 they take the cooperative-matrix path, and
`decode` turns `FLASHCM_CLAMP` on for its own call only, so the encoder pays
nothing for the padded tiles the decoder needs.

**The old decoder is kept as `sam2_decoder.fp32.json.bak`**, and one thing it is
still needed for: `refs.safetensors` holds *node-level* references dumped from
the fp32 decoder, so `verify_sam2.jl`'s node-by-node decoder pass now compares
two different graphs and reports a mismatch that means nothing. The end-to-end
check in `bench_sam2.jl` is unaffected — it compares the decoder's *outputs*,
`slice_2` and `slice_3`, which both exports produce. Re-dumping the references
against the autocast decoder is what would restore the node-by-node pass.

### A dispatch bug this uncovered: capability is not location

`verify_sam2.jl` could not run the encoder at all — "passing non-bitstype
argument", a slab-backed `Vector{UInt8}` array handed to a cooperative-matrix
kernel. `coopmat_sdpa_applicable` gated on `Lava.coopmat_gemm_available()`, which
asks the **device**, and never asked where the operands live; on the CPU backend
of a machine that has a Vulkan device — which is every run of that tool — it said
yes. Fixed with `ondevice`, which is `stridedroot(a) !== nothing`.

The flash path had the identical hole and closed itself earlier the same day:
taking operands by root-plus-strides means `stridedroot` must succeed, and it
returns `nothing` for anything not backed by a `LavaArray`. That was a side
effect, not a design, and it is worth noticing that the fix for a performance
problem removed a correctness one.

## 2026-08-02: gelu into the GEMM's store — 107.2 -> 102.8 ms, and 85.2% of PyTorch

The encoder's 48 `gelu`s were 4.42 ms and 1094.7 MB of output, every byte of it a
tensor the GEMM had just written being read back, transformed and written again.
Folding the activation into the store removes the pass.

**The reason it had never been done is that Lava could not express it.** The
cooperative-matrix intrinsics were load, store, convert, zero and muladd — there
is no way to reach an accumulator's components, so an elementwise epilogue was
not a matter of plumbing, it was inexpressible. SPIR-V has no
extract-from-cooperative-matrix instruction either; what it has is a rule that a
matrix in a `Function`-storage variable may be indexed by `OpAccessChain`, which
is what GLSL's `mat[i]` lowers to. So:

    coopmat_length   OpCooperativeMatrixLengthKHR (takes the TYPE, not a value)
    coopmat_getcomp  OpStore to a Function var, OpAccessChain, OpLoad
    coopmat_setcomp  the same, plus OpStore the component and reload the matrix

`accstore!` then takes an ordinary unary function, passed as a kernel argument —
a singleton, so it inlines and `identity` folds away. Keeping the activation on
the caller's side is what stops the runtime from having to know what a `gelu` is.

    encode  107.2 -> 102.83 ms   85.2% of PyTorch, from 81.7%
    masks   0.98750 / 0.99964 / 0.97727, from 0.95455 on the third
    add_129 relative error 7.72e-02 -> 7.24e-02

Accuracy moved the right way, which is worth a sentence because it need not have.
The activation is applied **after** the fp32→fp16 convert, not before: an
activation on the fp32 accumulator is a *different function* from the same
activation on the rounded value, and the graph rounds between the matmul and the
activation. Matching the graph is the point; being closer to the real number is
not. `actfn` lives next to `gelu`'s own op so the two expressions cannot drift.

### Two emitter bugs, both latent, both found by this

**A `Function` `OpVariable` allocated from inside a loop body is emitted after
its use.** `rayquery.jl` had already hit this and solved it with a pre-scan;
cooperative-matrix components needed the same. `spirv-val` says "ID '96' has not
been defined".

**`entry_function_locals` were appended to the preamble, and must be prepended.**
The preamble is already `[alloca OpVariables..., @localmem unwrapping
OpAccessChains...]`, and SPIR-V requires every `OpVariable` to be among the first
instructions of the first block. Appending put them after the access chains.
Latent because the only previous user was the ray-query variable — and a
ray-query kernel has no `@localmem` to unwrap. A staged GEMM has both.

### What made the graph pass fail silently at first

`foldrelu` folded 0 of 48. A relu reads its convolution directly; a gelu does
not — the graph reshapes the `addmm` result, so the chain is
`gelu <- view.default <- addmm`, and `resolvealias` follows only `alias` and
`detach`, stops at the view, and finds no producer. Resolving the whole chain is
what makes the fold reachable, and it is also why the alias has to target the
*view*: `addmm_2` is `[65536, 576]` and the gelu it feeds is
`[1, 256, 256, 576]`.

Verified bit-exact at the Lava layer before any of this was wired up: on SAM 2's
own `2304 x 4096 x 576` staged shape, `x -> 2x` and `x -> -x` through the
epilogue match applying them afterwards on all 9 437 184 elements. Exact-in-binary
functions on purpose — an affine `3x+1` differs by one ulp on ~6% of elements
because the kernel contracts it to an FMA and the host does not, which is a
correct epilogue and a wrong test.

## 2026-08-02: the decoder, attributed for the first time — it is attention, 49% of it

Every decode figure in this file had come from the serialised table or from
before the render unification. Measured in context (`OPDOUBLE` per aten, warmed,
synchronised), the shipped fp32 decoder at **11.36 ms**, 77.9% accounted:

    _scaled_dot_product_efficient_attention   7   5.587 ms   49.2%
    addmm.default                            47   2.326      20.5%
    convolution.default                       2   0.906       8.0%
    add.Tensor                               30   0.450       4.0%
    everything else                              ~0

**Seven calls are half the decode**, and they are the 23-token attentions. In the
fp32 export their operands are fp32, so `flashcm_applicable` refuses them and
they run the three-pass fallback — which is exactly why the autocast-decoder
package is worth what it is, and now there is a reason on the record rather than
a number. The packaged decoder measures **3.91 ms**:

    _scaled_dot_product_flash_attention       7   1.378 ms   35.3%
    addmm.default                            47   0.357       9.1%
    add.Tensor                               30   0.325       8.3%
    accounted                                     2.343      59.9%

**40% of it is not any graph op.** 1.57 ms unattributed on a 149-op graph. The
obvious suspect was `decode`'s dtype conversion of the encoder features, which
happens outside the graph — and it is not: `CACHE_DECODER_INPUTS` removes that
conversion and is worth **+0.008 ms**, i.e. nothing. 12.6 MB at 250 GB/s is 0.05
ms; the docstring's "12.6 MB of the 22.3 MB each decode allocates" was about
allocation churn, and it was read as if it were about time. What is left is
per-dispatch host cost on a graph with 149 of them, which is what
`Lava.capture`/`replay!` exists for — it took the host half off MatAnyone's step
once already, bit-exact.

### `CACHE_DECODER_INPUTS`: the open question is answered, and the answer is moot

It shipped off with a note that it needed "a decode result read back, and every
attempt at that hit the flush hang". Read back now: the cached first call and the
cached *second* call both differ from the uncached result by **0.000e+00**. That
settles what the note was waiting for — no decoder op writes into an input
buffer — and it agrees with the static argument from this morning (no op names
its own output among its inputs, `escaping` gives externals no slab space).

It stays off anyway, because it buys no time. The switch is now off for a
measured reason instead of an unverified fear.

**A measurement caveat, learned by breaking it.** `decode` returns masks that
live in the slab and are valid only until the next call. Materialising two calls'
results and then comparing them compares one array against itself after being
overwritten — which produced an all-NaN diff and a moment of thinking the cache
was corrupting output. Copy immediately or compare nothing. And 20 decodes in
flight without a sync is 446 MB of allocation the pool has to find: it left this
session's decode at a p50 of 15.1 ms with a 2.1-second outlier, against a true
3.95. Both artefacts were mine.

## 2026-08-02: attention round 6 — decomposed, then the densify removed: 118.3 -> 107.0 ms

Attention is the whole remaining gap, so this is the decomposition it never had.
In context (`OPDOUBLE` on the aten, warmed and clock-gated) it is **43.4 ms**.
Summing the flash kernel measured in isolation over the shapes and counts the
encoder actually runs — taken from a probe inside `flashcm_tiling`, not from
reading the graph — gives **30.1 ms**. The rest:

    flash kernels (isolated sum)   30.1 ms
    densify of K and V              4.65
    unaccounted                     8.6     <- in-context slowdown, not a pass
                                   -----
                                   43.4

**`densify` copies 646.4 MB per encode and not one operand is already dense.**
96 calls, two per attention, every one a `PermutedDimsArray` the graph hands over
and `sdpa` materialises. Timed on the dominant windowed layout it runs at 139
GB/s written / 278 read+write, which is the fast blocked-gather path working
properly — so 646 MB costs 4.65 ms and the fix is not to speed the copy up but to
not make it. `flash_attn_cm1.comp` shows how: it `coopMatLoad`s K straight from
global memory with a stride and only stages through shared when alignment or
quantisation forces it. We always stage.

### The small shapes: 41% of the isolated time for 5% of the arithmetic

Thirteen of the encoder's 48 attention calls are tiny — `Lq = Lk = 16` at
`B = 1024`, `Lq = 4`, and the like. Isolated and clock-gated they are 17.8 ms of
34.3 for 9.8 of 203.1 GFLOP: **0.55 TFLOP/s against the big shapes' 7.1**. The
worst, `Lq = Lk = 16` (five calls), had **no valid tiling at all** — the table's
smallest entry is `(16, 32, 4)` — so it fell to the three-pass path at 0.15
TFLOP/s.

Adding `(16, 16, 4)` fixes that in isolation, and the sweep is clean. Against
`(64, 32, 8)` at fixed `H = 8, B = 16`, clock gated:

    L      32     64    128    256  |   512   4096
    16x16  .045  .055   .143  .424  | 1.565  5.959
    64x32   --   .087   .187  .434  | 1.345  4.476

Small wins to 256, large wins from 512. On the encoder's own shapes it is worth
**3.76 ms isolated**, 3.4x on the worst one.

**It was built, measured and reverted.** In context it is worth 0.25 ms of
attention and 0.38 ms of encode — both inside the noise — and it moves mask 3's
IoU from 0.95455 to 0.93333, deterministically, because a different tile changes
the order the online softmax accumulates in and a 650-pixel mask has a dozen
boundary pixels to lose. No measurable speed for a visible accuracy change is not
a trade worth making, so the table is unchanged.

The probe that mattered was the one on the chooser: it proved the switch changed
which kernel ran (all six small shapes moved to `(16, 16, 4)`, the 4096 one did
not), so the null result is about the *workload*, not about the change failing to
apply. Isolated kernel time does not predict in-context behaviour — that is the
fifth entry in "How to measure this", and this is the sixth time it has held.

### Don't densify — DONE, and it was worth more than the copies

**118.27 -> 107.0 ms encode, 74.7% -> 81.9% of PyTorch, masks bit-identical.**

The kernel now takes `q`, `k` and `v` as a root array plus strides —
`stridedroot` for the base, `strides()` for the rest — so the staging index is a
dot product instead of a 4-D index into a wrapper. `flash_attn_cm1.comp` does the
same thing (`coopMatLoad` straight from global with a stride, staging through
shared only when alignment or quantisation forces it), and `transposeLE` and
Lava's scalar GEMM already used the trick here.

    encode     densify on 115.60 ms   off 107.26   -8.35
    attention  densify on  42.36       off  33.61   -8.75

**The copies were never the whole cost.** Removing them is the 8.35 ms above;
the other ~2.7 ms is `q`, which was never copied and was read through the wrapper
all along — 118.27 with the old kernel against 115.60 with the strided one and
the copies still in place. `FLASHCM_DENSIFY`'s docstring had recorded the copy as
"not an optimisation, a precondition", with the measurement that proved it
(137.64 densified against 169.42 raw). That was true of a kernel that indexed
wrappers. It stopped being true the moment the wrapper left the kernel, and the
switch is now off by default.

Nothing about the arithmetic changed — same elements staged, same order, same
accumulation — so the three mask IoUs are unchanged to the digit, which is the
difference between this and the tiling experiment above.

**One bug, and it was mine.** The first port measured 0.054 relative error, and
the fault was a name collision: the kernel body already binds `qb` from
`grp[1]` (the query-block index) and `kb` from the key loop, and the new base
offsets were called `qb`/`kb`. The staging read the block counter as a base
address. It produced *plausible* output — every head and every row block wrong by
a similar few per cent — which is exactly the signature that invites an
explanation about precision rather than a look at the variable names. Renaming to
`qbase`/`kbase`/`vbase` fixed it; 213/213.

### `O` in registers: the recorded reason was wrong, and the real one closes it

`FLASHCM_REGO` had lost 26% and the file blamed "a `splitidx` per element, per
key block, to recover `(row, e)` from the flat index", concluding that the
reference "is written for a compiler where getting at the same value does not
cost a division". **Nobody had removed the thing that named.**

The row index is loop-*invariant*: `idx = tid + (s-1)*NT`, so `idx % BR` is
`tid % BR` for every `s` whenever `NT` is a multiple of `BR`, which every shipped
tiling is — and `BR` is a power of two, so the call was a mask, not a division.
The compiler was recomputing a constant. Hoisting it out is free and changes
**nothing**: 107.12 ms shared against 117.32 registers, +9.5%, interleaved.

What it actually costs is a pass the shared form does not have. With `O` in
shared, `pvs` **is** the cooperative-matrix accumulator — each key block loads
it, `MulAdd`s into it, stores it back, so accumulation across blocks is free
inside the store, and the rescale on top is lazy, skipped on the third of blocks
where no row's maximum grew. With `O` in registers, `pvs` holds one block's `P·V`
and an **unconditional** `BR x EP` sweep folds it into the registers every block,
behind a barrier. The register form does not save a pass, it adds one.

So the reference is answering a different constraint: in LLM decode `HSV` reaches
256 and shared memory binds, so `Of` has to live in registers and the round-trip
is the price of fitting. Here shared is not binding — 48 900 bytes of ~100 KB per
SM — and the accumulator belongs where the cooperative-matrix store can reach it.
This lead is closed with a tested reason rather than an assumed one.

### What this leaves

Two leads, in order of what the evidence supports:
  * **The elementwise tail.** `gelu` into the GEMM epilogue is 4.5 ms and fully
    scoped; see the 2026-08-01 entry.
  * **The gap that is not a pass.** In-context kernels are slower than the same
    kernels in isolation. Removing the densify took part of it; what is left is
    unattributed, and L2 contention is a guess.

## 2026-08-01 (later): the GEMM tiling table was ordered by an average that hid a cliff — -3.7 ms

`GEMM_TILINGS` is "fastest first" and `gemm_tiling` takes the first block that
divides the shape. The order came from a weighted mean over SAM 2's six `addmm`
shapes, which put 64 x 128 first at 28.3 against 96 x 128's 27.x.

Swept per shape instead of averaged, that is the wrong way round. 96 x 128 is
faster on **four of the six**, by 8.6% to 21.0%. It loses on exactly one —
`576 x 4096 x 2304`, by 18.3% — and that shape is 24.4% of the encoder's GEMM
arithmetic, which is enough to carry the mean and bury the other four.

Sweeping K with M = 576, N = 4096 fixed shows it is not a trend but a wall:

    K       576  1152  1728  2016  2112  2208 | 2304 | 2400  2496  2880  3456
    96/64  +22%  +16%  +14%  +16%  +15%  +15% | -18% | +14%  +16%  +17%  +17%

One point wide. The neighbours at ±96 are both fine. Extending to every K that is
a multiple of 256 — 1024, 1280, 1536, 2048, 2560, 3072 — the 96-row block loses
at **all of them** (-10% to -18%) and at none of the seventeen values that are
not. At the bad K its throughput pins at 30-33 TFLOP/s however large K grows,
while 64 x 128 scales past 40. Eighteen measurements, clock-gated at 2280 MHz,
interleaved per shape, correctness-checked against a Float32 reference each run.

`A` is `M x K`, so `K % 256 == 0` makes its row stride a multiple of 512 bytes —
the classic power-of-two-stride aliasing shape. **That is a suspicion, not a
finding**, and two obvious alternatives are ruled out: both blocks are
register-limited to two workgroups per SM (120 and 128 registers, neither
spilling, read out of `VK_KHR_pipeline_executable_properties`), so it is not
occupancy; and the 96-row block re-reads `B` six times where the 64-row block
reads it nine, so it is not traffic. Nailing it needs Nsight.

So: 96 x 128 leads the table, and `gemm_aliasing` passes over it when
`K % 256 == 0`. Keyed on the block not being a power of two rather than on the
literal 96, because that is what the suspected mechanism turns on; the
generalisation is untested, but it can only decline a tiling in favour of a
measured one, so its failure mode is a missed win.

    encode  -3.70 ms, -2.9%      Interleaved in one session at 2265 MHz by
                                 swapping the table's first two entries in
                                 place. **Quote the delta, not an absolute**:
                                 repeated `bench_sam2.jl` runs of the *same*
                                 build read 117.7, 118.9, 122.4 and 123.2 ms
                                 here, a 5% spread that widens as the card
                                 stays hot, so any single reading can be picked
                                 to tell either story. The per-aten split
                                 (addmm -7.9%, convolution -1.7%) sums to -3.73
                                 and corroborates it independently.
    VRAM    1333 MiB, unchanged over four runs
    accuracy  bit-identical: same six encoder outputs, same three mask IoUs,
              same worst logit

That is **74.4% of PyTorch**, from 72.2%. `test_gemm_staged.jl` gains 63
assertions covering the rule as *selection* — not as speed, since a timing
assertion on a card that idles at 210 MHz would fail for unrelated reasons.

**The lesson is the averaging, not the stride.** A mean over shapes is the right
summary for choosing between two kernels that are uniformly better or worse, and
the wrong one when a kernel has a cliff: it reports the cliff as a mild deficit
spread over every shape. Order the table per shape, then let the picker decline
the exceptions.

### Everything else that shares the table, checked

`gemm_tiling` is reached by `matmul!` *and* by `conv_coopmat!`, so the encoder's
convolutions and MatAnyone both move with it.

    SAM 2   addmm         45.83 -> 42.20 ms   (-7.9%)
            convolution    6.10 ->  6.00 ms   (-1.7%)
                                              sums to -3.73, against the -3.70
                                              measured on the whole encode

MatAnyone reported +2.6-2.8% on a step, twice, and **it is not real**. Running
the same interleaved loop with the pair order reversed gave +0.4%: whatever runs
second in a pair carries a bias of about that size, so an interleaved A/B is only
evidence once both orders agree. What settled it was structural rather than
timing — every MatAnyone matmul returns `nothing` from `gemm_tiling` (its N is 16
or 120, under the 128 block), so its GEMMs *cannot* change tiling at all, and
only 3 of its 43 convolutions do. Timed directly, those three are neutral to
faster: -0.3%, -0.4% and -9.1%, about 0.015 ms of a 15.8 ms step.

### M and N have no cliff, and the kernel is at cuBLAS parity

The K rule is keyed on K because that is where the pathology is. Swept with the
other two dimensions moving and the clock gated per point (`warmclock()` before
each, both endpoints printed — a fixed `heat()` between points is not enough,
the allocation and GC per point let the card idle back to 840 MHz and produced a
first sweep that read 17.5 TF/s for a shape worth 42.8):

    M   576 -> 3072, N=4096, K=576    33.6 -> 43.8 TF/s, smooth
    N  1024 -> 8192, M=2304, K=576    34.4 -> 44.3 TF/s, smooth

No dip at any `M % 256 == 0` (768, 1536, 2304, 3072 all sit on the curve), so
this is not a general power-of-two-stride effect — only `A`'s row stride, which
is what `gemm_aliasing` tests.

The other thing those sweeps say is worth more than the rule: **the staged GEMM
is at cuBLAS parity.** 44.0-44.3 TF/s at large N against `CUBLAS_TFLOPS = 44.6`,
and 42.8 on SAM 2's dominant `2304 x 4096 x 576`. The aggregate 38.1 TF/s the
encoder sees is not the kernel falling short — it is SAM 2's smaller shapes
sitting below the asymptote (33.6 at M = 576). So "cuBLAS gets 50.7 TFLOP/s on
the same work", inferred earlier from PyTorch's recorded per-op excess, does not
survive contact with a direct measurement, and the GEMM is a much smaller target
than that inference implied. **Attention is where the remaining gap is.**

### Sizing attention, with the right yardstick

The encoder's 48 attentions are **203.1 GFLOP** (counting both matmuls, `4·Lq·Lk·E·H·B`),
so 43.83 ms is **4.6 TFLOP/s**. Two shapes are 95% of it: three global
`Lq=Lk=4096, H=8, B=1` calls at 116.0 GFLOP, and thirty-two windowed
`Lq=Lk=256, H=8, B=16` at 77.3 GFLOP.

**Do not read that against the GEMM's 44.** Flash attention interleaves two small
matmuls (`64x32x80` tiles at `E=72`, padded from 72 to 80) with an online softmax
and its barriers; it cannot reach a GEMM's arithmetic intensity, and calling the
difference "9.5x available" would be the serialised-table error wearing new
clothes — comparing a number to a ceiling from a different kernel class.

The yardstick that means something is PyTorch doing *the same shapes*: its
recorded excess of 24.5 ms puts cuDNN at 203.1 GFLOP / 19.3 ms = **10.5 TFLOP/s**.
So the realistic headroom is about **2.3x, worth 24.5 ms** — which is now the
whole remaining gap to 90%, since the GEMM turned out to be at parity.

That is also the fifth attention round, and the first four are on record above:
the cooperative-matrix port (1.9x, shipped), parallel softmax (built, reverted —
not the bottleneck), held-`O` accumulators (built, lost, because `grew` fires on
67-100% of blocks) and the per-row-tile `grew` flag (simulated first, not worth
building). Whatever the 2.3x is, it is not any of those, and it is not the tiling
either — `(64, 32, 8)` measures optimal on both dominant shapes by 27% and 11%.

Attention was swept the same way and needs nothing: `(64, 32, 8)` is already
optimal on both of SAM 2's shapes, ahead of the next tiling by 27% on the global
blocks and 11% on the windowed ones, with every candidate agreeing numerically to
2.3e-4. `FLASHCM_TILINGS` was tuned per shape to begin with, so the averaging
mistake never entered it.

## 2026-08-01: the encoder was recomputing a constant every frame — 133.3 -> 121.3 ms, 1610 -> 1333 MiB

The largest single item left in the encoder was not a kernel. It was 164 ops, 19%
of the graph, computing **the same answer on every encode**: the bilinear resize
of the Hiera trunk's position embedding, which depends on the weights and the
resolution and on nothing else.

It cost **11.7 ms of a 133 ms encode and 604 MB of a 698 MB slab**. The slab share
is the part worth looking at twice: the resize is written as sixteen full-size
`[1, 144, 256, 256]` fp32 gathers, and all sixteen are live at once because a
weighted sum at the end consumes them together. Sixteen times 37.75 MB is 604 MB
that the planner has to hold simultaneously — 86% of the slab, to carry a value
that never changes.

`hoistconstants` already existed and already folds constants; it just folded
*nullary* ops only (`scalar_tensor`, `full` — a literal in an attribute). Taking
the same idea to fixpoint — an op is constant if every input is — turns 164 ops
into four weights.

    encoder ops    852  ->  688
    slab         698.4  -> 273.9 MB
    encode       133.1  -> 121.5 ms   (interleaved A/B, one process, -8.7%)
    VRAM          1610  -> 1333 MiB   (spare against the 1951 ceiling: 341 -> 618)

Accuracy is unchanged where it was already right and exact where it now is: the
three encoder outputs that are themselves constants (`repeat_2`, `_to_copy_595`,
`_to_copy_599` — positional encodings the decoder reads) now match PyTorch at
`0.000e+00`. Masks 0.98750 / 0.99964 / 0.95455, worst logit 3.846e-01.

### Two things this cost, both stated rather than buried

**Load time, +5.8 s once.** Load goes 5.7 -> 48.5 s, which reads alarming and
mostly is not: the pass *runs* the ops it folds, so it pays their JIT at load
instead of leaving it to the first encode. Measured end to end in fresh
processes, time-to-first-encode is 70.6 s -> 76.4 s. The residual is real, and it
buys 11.7 ms on every encode after the first. `FOLD_CONSTSUBGRAPHS[] = false`
turns it off.

**604 MB of intermediates at load.** Transient in the real sense — allocating and
freeing a block that size moves `nvidia-smi` by `+577` and back to `+0`, so Lava
returns it to the driver rather than parking it in the pool. Checked before
relying on it, because if the pool had kept it the whole exercise would have
saved slab and lost the same memory again.

### The measurement that was wrong first, and why

The first reading of this said **0.55 ms**, i.e. "not worth doing". It came from
`OPDOUBLE` over the backward cone of the sixteen `index.Tensor` ops — 68 ops —
and the instrument was fine. The op set was not. A backward cone from the gathers
finds the gathers and the index arithmetic feeding them and stops precisely where
the work begins: everything that *consumes* a gather is downstream of it. Those
68 ops really do cost 0.55 ms, which is 604 MB at bandwidth and exactly right.
The seventy muls and adds that then combine the sixteen gathers move about 8 GB,
and they are the other 11 ms.

The fixpoint set is the right one and two independent instruments agree on it —
doubling says 11.13 ms, deleting says 11.72 ms. A third measurement rules out the
tempting alternative explanation: doubling *only the 688 ops both graphs share*
costs 115.52 ms before the fold and 115.25 ms after, so the surviving ops did not
get faster and none of the win is a locality effect from the smaller slab. It is
the removed ops, all of it.

Lesson for the next one of these, since it is the second time in this document
that a cone has been mistaken for a cost: **a backward closure answers "what does
this need", not "what does this cost"**. For cost, close in both directions, or
close on the property you actually mean — here "is constant", which is a fixpoint
over inputs and naturally sweeps up the consumers too.

### Where the 121.3 ms goes now

In context, by doubling each aten family in a warmed clock-gated run — not the
serialised table, which inflates by dispatch count:

| aten | n | ms | share |
|---|---|---|---|
| `addmm.default` | 195 | 45.77 | 37.7% |
| `_scaled_dot_product_flash_attention` | 48 | 43.83 | 36.1% |
| `convolution.default` | 7 | 6.15 | 5.1% |
| `native_layer_norm.default` | 96 | 5.96 | 4.9% |
| `gelu.default` | 48 | 4.47 | 3.7% |
| `clone.default` | 90 | 2.61 | 2.2% |
| `add.Tensor` | 98 | 2.50 | 2.1% |
| `max_pool2d_with_indices` | 6 | 1.64 | 1.4% |
| `_to_copy.default` | 97 | 1.63 | 1.3% |
| **accounted** | | **114.56** | **94.4%** |

Two kernels are three quarters of it, and both have had a full pass this session.
90% of PyTorch means 78.9 ms, so 42 ms still has to come out of `addmm` and
attention.

Sizing the two big ones against the hardware rather than against each other: the
encoder's 195 `addmm`s are **1606.3 GFLOP**, so 45.77 ms is **35.1 TFLOP/s** —
which lands on the 34.8 that `elementwise_profile.jl` measured a different way, so
the static count and the in-context attribution agree. Four shapes carry 72.6% of
it, the same four Lava's GEMM source already names at 72.7%.

*(Superseded below: I inferred from PyTorch's recorded excess that cuBLAS gets
~50.7 TFLOP/s on this work, making the GEMM a 14 ms target. Measured directly
after the tiling fix, our kernel reaches 44.0-44.3 against cuBLAS's 44.6 — it is
at parity, and the aggregate is held down by SAM 2's smaller shapes rather than
by the kernel. Attention is the target.)*

**The next tractable item is `gelu` into the GEMM epilogue, ~4.5 ms.** All 48
gelus consume an `addmm` output with exactly one reader — checked, 48 of 48 —
carrying 1094.7 MB of output, so each one is a full read and rewrite of a tensor
the GEMM has just written. `foldrelu` is the precedent and generalises directly
(it already keys on a producer-aten set and writes `attrs["act"]`). The work is
not in the pass, it is in the kernel: for every shape the encoder runs, the fused
path writes `out` straight from `Lava.coopmat_gemm!` with the bias in the
accumulator's initial value, so the activation has to go in there too — and the
staged kernels live in a table keyed by tiling config, so a `Val{ACT}` doubles
that table. A runtime flag in the store is the cheaper shape: one uniform branch
per stored element, against 2.2 GB of traffic removed.

`clone.default` looks like a similar target and is not. All 90 read a
`permute.default` view — they are `contiguous()` after a permute, real work, 535.6
MB of output. 1071 MB of traffic in 2.61 ms is ~41% of bandwidth on the permuted
copy path that was already tuned earlier this session.

### What it also settled

Folding an encoder output into a weight is only safe if nothing downstream writes
into it, which `sam2.jl` records as an open question (`CACHE_DECODER_INPUTS`,
off, "the planner is free to alias"). It is answerable statically and the answer
is no: no op in either graph names its own output among its inputs, and
`escaping` gives externals no slab space, so neither of the two structural routes
to aliasing an input exists. That is the check `CACHE_DECODER_INPUTS` was waiting
on, minus the numerical comparison it wanted a readback for.

## 2026-07-31: the ledger after the GEMM work — attention is now the whole story

`tools/sam2_kernel_table.jl`, re-run with everything below landed. Serialised, so
the total (259 ms) exceeds a free-running encode (175 ms) — `OPTIMES` syncs around
every op, which is the only way to attribute device time to a source-level op.
Compare *columns*, not the total.

    aten op                       calls   ours ms  torch ms   ratio   ourTF/s  thTF/s
    _scaled_dot_product_*            48     93.03      9.59    9.7x       2.2       -
    addmm.default                   195     72.19     35.99    2.0x      22.3    44.6
    _to_copy.default                199     18.01     12.89       -         -       -
    native_layer_norm.default        96     16.62      5.64    2.9x         -       -
    add.Tensor                      130     15.62      5.23    3.0x         -       -
    clone.default                    90     10.88         -       -         -       -
    gelu.default                     48      7.69      7.48    1.0x         -       -
    convolution.default               7      7.27      0.86    8.4x       1.9       -

**Attention is 93 ms against torch's 9.6.** That is bigger than the GEMM's entire
excess over torch (72.2 - 36.0 = 36 ms) and it runs at **2.2 TFLOP/s** on 203
GFLOP of arithmetic — the GEMM next to it does 22.3, and cuBLAS 44.6. Everything
else on the list is small by comparison, and `gelu` is already at parity.

This is worth stating plainly because it overturns the obvious next move: with the
GEMM at 79% of cuBLAS the temptation is to chase the last 1.26x there, and that
is worth at most ~9 ms of a serialised 259. Attention is worth ~83. The kernel
table exists precisely to stop that kind of guess, and it earned its keep here.

**And the first 11 ms of it came from deleting a stale constant.** `COOPMAT_MINL`
is the sequence length above which attention takes the cooperative-matrix path
rather than the three scalar kernels. It was 512, set by measurement, and correct
when it was set. But it is a property of *the GEMM underneath it*, and that GEMM
went from 20.6 to 35.3 TFLOP/s today — so the crossover moved, and the constant
did not. Re-measured, interleaved, clock warmed:

    L=64  H16 B16   0.198 -> 0.280 ms   0.71x   still loses
    L=128 H16 B16   0.645 -> 0.606 ms   1.06x   a wash
    L=256 H8  B16   1.227 -> 0.847 ms   1.45x   now WINS, was 0.81x
    L=512 H8  B4    1.159 -> 0.586 ms   1.98x

512 -> 256 moves **32 of SAM 2's 48 attention calls** onto the tensor cores and
takes the encode **175.1 -> 164.0 ms, 50.1% -> 53.4% of PyTorch**. Parity got
marginally *better* — mask 3 IoU 0.95455 -> 0.95556 — because the coopmat path
accumulates the scores in fp32.

Any constant that separates two implementations has this failure mode: it is a
measurement, it is silently invalidated by work on either side of it, and nothing
tells you. The test now asserts the boundary *relative to* `COOPMAT_MINL` rather
than at a literal 256, so it tracks the constant instead of pinning it.

The `addmm` row also reads lower than the 35.3 TF/s measured in isolation, and
that is not a contradiction: 195 calls include the batched and split-K shapes that
never take the staged path, and per-op serialisation charges each one its launch.

## 2026-07-31: two silent miscompiles, and how to tell them apart

Both were "computes the wrong answer, no other symptom", both found the same
week, both vanish under instrumentation — and they have nothing to do with each
other. Kept together because the *method* that separated them is the reusable
part.

**One was ours.** `OpUDiv` in a shared-memory store index drops stores once a
cooperative-matrix `muladd` is in scope. `Lava.splitidx` — magic-number division,
no `OpUDiv` — fixes it; see the section below and
`test_shared_index_division.jl`. It had also been sitting in `flash.jl` for
months as a documented "unexplained" blocker.

**One was NVIDIA's**, and proving that took a second driver rather than an
argument. The same SPIR-V module, two independent Vulkan implementations
(`tools/narrow_index_second_driver.jl`):

                                     rank 2 (control)   rank 3 (the bug)
    NVIDIA RTX 4000 Ada                   exact              WRONG
    lavapipe (llvmpipe, LLVM 22.1.8)      exact              exact

The rank-2 control passes on both, so the drivers are comparable. Reproducer:
`identity` over a plain rank-3 array, `Broadcast.preprocess` applied, indexed
with a narrow `Int32` linear index. Nothing else — no view, no permutation, no
arithmetic operand. `test_int32_cartesian_miscompile.jl`.

**The lesson is the ordering.** "The SPIR-V validates, so it must be the driver"
was asserted twice on this session's first bug and was wrong twice; it was the
`OpUDiv`. What settles the question is not the validator but an *independent
consumer* — and it is cheap: lavapipe is one environment variable. Reach for it
before reading disassembly, not after. Two tools came out of it and both are
kept: `LAVA_SKIP_PASSES` (bisect the structurization pipeline, off by default)
and `tools/narrow_index_second_driver.jl`.

**When the consumer cannot be varied, vary the producer.** That check does *not*
reach the `OpUDiv` bug: a cooperative-matrix `muladd` is one of its necessary
conditions and lavapipe has no cooperative matrices (`coopmat available: false`,
subgroup size 8). This machine has exactly two Vulkan devices — NVIDIA RTX 4000
Ada on 595.84, and llvmpipe on Mesa 26.1.5 — and only one of them can host the
reproducer, so that attribution stays open. The complement is to hold the
consumer fixed and swap the producer: write the kernel in GLSL, compile it with
glslang, and run *that* module on the same driver. Correct there means our SPIR-V
differs in a way that matters; wrong there means the driver. Either axis works;
the point is that one of them has to move.

## 2026-07-31 (later): the staged GEMM lands — 200.6 -> 182.7 ms

Encode p50 **200.6 -> 182.7 ms** (44% -> **48.0%** of PyTorch's 87.6). The GEMM
itself, weighted over the encoder's six `addmm` shapes, **20.6 -> 31.2 TFLOP/s**
— 1.52x, and **70% of cuBLAS's 44.6** where it had been sitting at 46%. Masks
unchanged at IoU 0.98125 / 0.99964 / 0.95455, VRAM unchanged at 1904 MiB.

Those absolutes were taken on a quiet machine. The contention-proof form, staged
against register-blocked **interleaved in one session** on a box with eight other
compute processes: **221.6 -> 198.2 ms, 23.4 ms / 10.6% of the encode**. Quote
that one; the absolute figure moves 20 ms with desktop load and has twice been
mistaken for a regression in this document's history.

Three things made it, and only the first was the intended work.

**1. The warp grid, not the warp tile.** `mul_mm.comp` is tuned for a 4x4 block
of cooperative-matrix tiles per warp. On this card that is the *worst*
configuration — 5.9 TFLOP/s — because sixteen 16x16 fp32 accumulators are 128
registers per lane and the driver spills them into workgroup memory, 31 KB of it,
which `VK_KHR_pipeline_executable_properties` will tell you if you ask:

                      registers   shared/WG   of which the shader's own
    direct 4x4 block    255 (cap)     25344         0    <- all spill
    staged 2x2           96            8960      8960
    staged 4x4          255 (cap)     48896     17664

The register-blocked kernel we had been shipping was spilling its *entire*
accumulator block and had been all along. Keeping the warp tile at 2x2 and
widening the **warp grid** to 8 warps (64 x 128 block, 256 threads) is what wins:
this kernel is latency-bound, not intensity-bound, which inverts the reference's
premise and is worth re-deriving on any other device.

**2. Sixteen bytes of leading-dimension alignment.** `GEMM_PAD` 4 -> 8, so the
shared blocks' row strides are 144 and 80 bytes instead of 136 and 72. Worth 23%
on its own, and it is alignment rather than bank conflicts — the 8-byte-aligned
strides are the conflict-free ones by the usual arithmetic, and they are slower.

**3. Removing an `OpUDiv` from the staging index.** See below. It cost most of a
day, produced three confident wrong diagnoses on the way, and was twice very
nearly written off as the driver's problem.

### `OpUDiv` in a shared-store index drops stores

**Final answer, after three wrong ones.** A staging loop whose shared-memory
store address goes through a real division by a non-power-of-two constant *loses
stores*, when a cooperative-matrix `muladd` is in scope and the loop runs more
than one iteration. `splitidx` — the same `init_fastdiv_values` magic-number port
the broadcast path already used — removes the division and every geometry becomes
exact at every K. 96 x 128 went from unusable to shipped, and it is the tiling the
288-row shape needs (27.4 TFLOP/s against 20.3 for the 32 x 128 it had to fall
back to, and 14.3 for the register-blocked kernel).

The measurement that isolates it — 96 rows into a 104-wide block, 3072 elements:

                        K = 32   K = 64  K = 128  K = 256
      while + OpUDiv      3072      240      256      240
      while + fastdiv     3072     3072     3072     3072
      for   + OpUDiv      3072      240      256      240
      for   + fastdiv     3072     3072     3072     3072

The loop form does not matter — **that was a red herring, and the section below
records how convincing it was**. What identified the division was diffing the
emitted modules for `BM = 112` (lossy) and `BM = 128` (exact, because 128 folds to
a mask and a shift): identical opcode for opcode except that one contains
`OpUDiv` and the other `OpShiftRightLogical` + `OpBitwiseAnd`.

Our SPIR-V holds a *rolled* loop with a single `OpStore`, so whatever unrolls it
and drops stores is downstream of us; whether that is the driver is still open,
and `splitidx` makes it moot. `test_shared_index_division.jl` holds the case, with
the `OpUDiv` rows as `@test_broken` so a driver fix would announce itself.

**It had already bitten somewhere else.** `flash.jl` carried a documented
blocker — "`BQ = 32` is wrong and unexplained ... that bug is the gate on this
whole optimisation" — refused by an `iseven(div(BQ*E, NT))` guard in `flashfits`
and attributed to an odd accumulator-slot count, since `BQ = 32` was exact at
`NT = 128`. It was the same `OpUDiv`: `E = 72` is not a power of two. The tell
was in the old note without being recognised — "exact for *constant* inputs",
which is precisely the condition under which the bug does not bite, because a
constant store needs no global load to feed it. Same session, same inputs, only
the arithmetic:

    BQ/NT    BQ*E/NT      OpUDiv    splitidx
    64/256   18 even     7.8e-07     7.1e-07
    32/256    9 odd      7.1e-02     8.6e-07
    32/128   18 even     8.4e-07     7.0e-07
    64/128   36 even     7.8e-07     6.8e-07

The guard is gone, twelve regression assertions replace it, and the `BQ = 32`
route to two workgroups per SM is open again. Any kernel staging through
`@localmem` with a non-power-of-two extent should use `splitidx` — which for
attention means the cooperative-matrix flash port too, since `E = 72`.

### The three wrong answers, kept because each was convincing

One tiling — 96 x 128 — lost 4 of every 32 k-terms per row, deterministically,
while 96 x 64 and 160 x 128 were exact. Everything that could be checked said the
shader was right: `spirv-val` passes, the barrier is emitted inline in the right
block with `AcquireRelease | WorkgroupMemory | MakeAvailable | MakeVisible`, the
accesses are tagged `NonPrivatePointer`, the strides and offsets decode correctly,
and the driver reports 125 registers, no spill, and a workgroup-memory size equal
to the shader's own two arrays.

The minimal case has no GEMM in it at all: stage an array, barrier, execute **one**
`OpCooperativeMatrixMulAddKHR` whose operands are `OpConstantNull` — no pointer,
no stride, no memory layout, nothing the shader addresses — and read the array
back. 240 of 3072 elements survive. Delete that one instruction and all 3072 do.

That looks exactly like a driver writing workgroup memory it never reserved, and
it is not. The variable is the **loop form**:

    while k0 < K;  stage; @synchronize; muladd; k0 += BK; end   ->   240 / 3072
    for kb in 0:(K ÷ BK - 1);  stage; @synchronize; muladd; end -> 3072 / 3072

Same instructions, same arrays, same single `muladd`, and both with a genuine
multi-iteration loop in the emitted SPIR-V (checked — the counted form is not
merely unrolled away).

**The loop shape is not the mechanism, and the first version of this section said
it was.** LLVM's structurizer does give `while` a top-tested loop whose header
branches to the *continue target* to exit, where the counted `for` gets a
bottom-tested one, and that was a satisfying story. Adding `loop-rotate` to the
structurization pipeline produces the rotated header for `while` too — verified
in the disassembly — and changes nothing; the GEMM measures 1.53x with it against
1.52x without. The pass was removed again and the negative result recorded in
`passes/structurize_cfg.jl` so it is not re-derived.

Nor is the loop form sufficient on its own. Varying only the loop and KA's
`__validindex` guard over one body, counting how many of 3072 staged elements
survive the barrier:

                        K = 32    K = 64
      while + guard        240       241
      while + unsafe      3072       257
      for   + guard       3072      3072
      for   + unsafe      3072       241

Only `for` + guard survives both, and `for` + `unsafe_indices` — which is what
the GEMM ships — fails *there* at K = 64.

**The mechanism is the unrolled staging loop, and that is confirmed rather than
inferred.** Reduced to one staged array, one `muladd` and a 256-thread workgroup,
varying only the rows staged per step — `trips` is the staging loop's count:

    BM             32  48  64  80  96 112 128 160 192 224 256
    staging trips   4   6   8  10  12  14  16  20  24  28  32
    unrolled        y   y   y   y   y   y   y   n   n   n   n
    BM divides WG   y   n   y   n   n   n   y   n   n   n   y
    exact, K >= 64  .   X   .   X   X   X   .   .   .   .   .

Failures are exactly **unrolled AND non-dividing**, no exceptions — including the
three non-dividing sizes at 160 and above, which have too many trips to unroll and
are exact. The confirmation: making the staging trip count a **runtime** argument,
changing nothing else, takes `BM = 96` from losing data at K = 64 and 128 to
exact at all three. When `BM` divides `WG` each lane's row is fixed across the
unrolled copies and only its column advances; when it does not, both move and the
stores stop landing in distinct slots.

What is lost, with the two k-blocks given distinguishable values: of 3072 slots,
496 exact, **0** holding the previous block's value, 2184 holding a value from
the right block but the wrong row, 392 never written. Nothing is sunk past the
barrier — stores are simply lost.

Three more conditions are each necessary: the stored value must come from a
**global load** (a computed constant is exact at every geometry, 96 included),
the k-loop must run **more than one iteration**, and a cooperative-matrix
`muladd` must be present. Total workgroup memory is *not* a variable — 4 KB to
32 KB are all exact on a dividing geometry. And it cannot be instrumented:
recording the store and load indices shows both correct and injective **and makes
the corruption disappear**.

So `gemm.jl` now **rejects** a tiling whose `BM` does not divide its workgroup:
a hard error at kernel generation, not a comment, because it predicted every
observed case and violating it produces a wrong answer with no other symptom.
All four shipped tilings satisfy it, and on top of that are exact at K = 32, 64,
96, 128, 160, 288, 320, 576, 1152 and 2304 — one iteration to seventy-two —
which `test_gemm_staged.jl` sweeps.

Ruled out along the way, each by experiment rather than by argument: the merged
`Block` workgroup struct (one array behaves identically), explicit layout versus
plain `Workgroup` variables (the glslang form — implemented, and it changes
nothing), `NonPrivatePointer` (a genuine spec violation, fixed, not this), the
number of `muladd`s (one is enough), `OpConstantNull` operands (loading them from
global is the same), and loop rotation.

`test_shared_index_division.jl` holds the isolated case, with the `while` rows
as `@test_broken` rather than deleted. 96 x 128 — the geometry it uses — stays
out of `GEMM_TILINGS`. **Root cause still open**, and any kernel that stages
through `@localmem` and uses a cooperative-matrix op should be K-swept rather
than assumed.

Two things fell out of it that matter beyond the GEMM:

  * **`NonPrivatePointer` was missing everywhere.** Lava emits
    `OpMemoryModel ... Vulkan`, under which a barrier's `MakeAvailable` /
    `MakeVisible` apply *only* to accesses tagged non-private. Every `@localmem`
    load and store was untagged and every cooperative-matrix access carried no
    memory operand at all, so a workgroup barrier was granting them nothing. It
    happened to work. Fixed in `emit.jl` and `spirv/coopmat.jl`.
  * **`unsafe_indices=true` on the staged kernel.** KA's `__validindex` guard is
    dead there — the launch is an exact multiple of the workgroup size — and with
    the counted `for` the structurizer failed to flatten it and left the `muladd`
    under an `OpSelectionMerge`. That cost **3x**: 8.8 against 26.1 TFLOP/s.

## 2026-07-31: 256.0 -> 200.6 ms, and the memory target reached

Seven changes, in the order they happened, because several of them only exist
because an earlier one's "win" turned out to be a lie. Every speed figure below
is a **same-session interleaved A/B behind a `Ref`**, not a before-and-after:
this machine's encode moved 40 ms in both directions across sessions today
without any code changing.

Net: encode p50 **256.0 -> 200.6 ms** (34% -> 44% of PyTorch) and −148 MB of
encoder memory. Mask IoU 0.98125 / 0.99964 / 0.95455. The memory *goal* is not
met — see the retraction below; the decoder holds 870 MB nobody has examined.

**1. `permutedims!` deleted, routed at the broadcast.** The dedicated kernel lost
to the generic broadcast on every shape the encoder runs — 2.6-5.9x — because it
launched an N-D ndrange with an N-D workgroup, which is exactly the geometry the
note at the top of `gpuarrays.jl` measures at a quarter of achievable bandwidth.
It is gone; `lava_permutedims_kernel!` no longer appears in the profile and its
51 dispatches moved onto the broadcast path (`mixed` 203 -> 254).

**2. A workgroup above 256 silently runs part of the launch.** Found by chasing a
"3.4x" on the Hiera permute that had written a *quarter* of its destination. The
device accepts `LocalSize 512 1 1` and runs local invocations 1-256 only, while
the grid is computed from the size asked for — so the output comes back with a
periodic hole. Ruled out: index arithmetic (participating lanes are correct),
`spirv-opt`, stale pipelines, invalid SPIR-V (`spirv-val` passes), and
`LocalSize` itself (the module is byte-identical to the working one except that
execution mode). **Not a resource limit**: with
`VK_KHR_pipeline_executable_properties` wired up the driver reports *identical*
Register Count (40), Stack Size, Local and Shared Memory for a body that fails at
512 and one that succeeds at 1024. No statistic separates them, so
`Lava.WORKGROUP_LIMIT[] = 256` now throws instead of truncating
(`test_workgroup_limit.jl`, which also pins the driver behaviour so the limit can
be lifted if a future driver fixes it).

Reading those statistics needed a Vulkan.jl fix of its own —
`get_pipeline_executable_statistics_khr` never returned anything, because
`_initialize_core` patches `sType`/`pNext` through
`ConstructionBase.setproperties`, which refuses structs with overloaded
`propertynames`, which this one has because of its C union.

**3. Elements per thread, from `contig_copy.comp`.** Raising the *workgroup* does
nothing (64 and 256 measure the same). What llama.cpp actually does is `U`
elements per thread strided by the workgroup, so each iteration is still one wide
coalesced run; it buys memory-level parallelism, not fewer instructions. Now in
all three broadcast kernels via `broadcastlaunch`, which backs `U` off from 8
when the grid would be too small to fill the card.

| kernel family | before | after | |
|---|---|---|---|
| `lava_broadcast_flat_mixed!` | 47.11 | **38.51** | 1.22x |
| `lava_broadcast_flat!` | 20.21 | **13.31** | 1.52x |
| elementwise bucket | 99.5 | **81.4** | |
| matmul + attention (untouched) | 162.1 | 145.9 | 0.900 |

Encode 264.7 -> 230.2 ms in the dispatch-timing profile, **which is not the
win**: the untouched buckets moved 10% too, so that session is simply faster.
Worse, the free-running `bench_sam2.jl` read **260.0 ms p50 against a recorded
256.0** — i.e. end to end it looked like nothing had happened, which is the trap
this document already records three times.

So it was settled the only way that counts, with `Lava.BROADCAST_UNROLL[]`
flipped **inside one session**, 15 encodes per variant, round-robin, clock at
2265 MHz throughout:

    BROADCAST_UNROLL = 1    p50 305.92 ms    min 291.59
    BROADCAST_UNROLL = 8    p50 292.03 ms    min 270.40
                            ----------------------------
                            13.89 ms, 4.5%

That is the number. Both cross-session readings were noise — one flattering
(-34 ms), one damning (+4 ms) — and neither was worth anything. Mask IoU is
unchanged at 0.98750 / 0.99941 / 0.95349.

**4. The division chain was the cost all along.** With the unroll in, the
permuted copies were still at 32-49% of a plain copy, so the next question was
what the remaining gap actually is. Isolated — one kernel that only writes, one
that also runs `cart32`, no memory traffic in the way, 2.36 M elements:

    rank 2, 1 division      11.7 -> 33.5 us
    rank 4, 3 divisions     11.7 -> 59.6 us
    rank 6, 5 divisions     10.9 -> 85.0 us

Linear in the number of divisions, ~15 us each, and at rank 6 that is **74 us of
arithmetic against a ~42 us memory floor**. A second check agreed: computing the
index while reading *linearly* costs 400 -> 58 GB/s at rank 6, while the
permutation itself only costs 58 -> 52.

**This document said the opposite** — "the addressing is not the cost, so
anything that helps has to change the access pattern" — from an end-to-end delta
of −0.5 ms on a kernel whose access pattern hid it. The note is kept in
`gpuarrays.jl` above the correction.

`generic_unary_head.glsl` has the answer and `copy.comp` is otherwise our exact
algorithm: **never divide, multiply by a magic number**. `FastDiv32` is
`init_fastdiv_values` / `fastdiv` ported straight over — `L = ceil(log2(d))`,
`mp = 2^32 (2^L − d)/d + 1`, then `n/d == (mulhi(n, mp) + n) >> L`. NVIDIA has
`mul.hi.u32` and no integer divide at all, so it is ~5 cycles where the divide
was ~25.

| case | before | after | of floor |
|---|---|---|---|
| attn `(72,8,256,16)` | 93.4 | **225.2** | 41% -> 86% |
| attn `(72,4,16,1024)` | 117.7 | **248.7** | 49% -> 92% |
| hiera `(576,16,4,16,4,1)` | 72.1 | **212.5** | 32% -> 77% |
| hiera `(288,4,32,4,32,1)` | 81.0 | **250.5** | 34% -> 94% |

Same-session interleaved on the encode, 15 each, `Lava.BROADCAST_FASTDIV[]`:

    divides    p50 284.90 ms    min 270.63
    multiplies p50 263.35 ms    min 249.96
               ----------------------------
               21.55 ms, 7.6%

A wrong quotient here is an out-of-bounds index, not a wrong pixel, so it is
checked exhaustively rather than sampled: every extent the model decomposes by,
every `n` in `0:2^25` — past the largest array the encoder builds — quotient
*and* remainder, plus the kernels diffed against the dividing path they replace
(`test_fastdiv.jl`). Mask IoU unchanged.

**It also kills my blocked gather.** That kernel avoided divisions by taking an
N-D ndrange and paid for it in coalescing; with divisions no longer expensive the
trade is simply bad, and it now measures **slower** than the broadcast (173 vs
225, 198 vs 249 GB/s). Third time in this document that porting the reference
beat something reasoned from first principles — after `copy_transpose.comp` (7.0x
vs my 1.6x) and `toLE_tiled`.

**5. The GEMM has a harness now, and its first result is a negative one.**
`tools/gemm_bench.jl` times the encoder's six real shapes through
`DNNKernels.matmul!` — the path the model actually takes — interleaved, with a
correctness column. Two wrong ways to call it, both found by that column:
`Lava.coopmat_gemm!` bare returns garbage (its `C` must be **fp32 with `splitk`
planes**, passed as `partials = C, reduce = false`; handed an fp16 `M x N` it
gave 18946 NaNs against a reference whose largest element was 7.87), and
`LinearAlgebra.mul!` takes the **scalar** fallback at 1.3 TFLOP/s because Lava's
own coopmat path wants an fp32 destination and autocast writes fp16.

Baseline, one session, clock at 2280:

| M x N x K | share | direct | staged | of cuBLAS |
|---|---|---|---|---|
| 2304 x 4096 x 576 | 24.4% | 16.7 | **18.1** | 41% |
| 576 x 4096 x 2304 | 24.4% | 20.7 | **23.0** | 52% |
| 1728 x 4096 x 576 | 17.8% | 16.7 | **18.8** | 42% |
| 576 x 4096 x 576 | 6.1% | 15.9 | **18.6** | 42% |
| 288 x 16384 x 1152 | 4.1% | 14.1 | 14.1 | 32% |
| 1152 x 16384 x 288 | 4.1% | 13.1 | 13.1 | 29% |

TFLOP/s. The staged kernel — switched off long ago on a recorded 1.03x/0.56x —
is **1.08-1.17x faster on all four dominant shapes**. In the encode it is
**3.7% slower**: 210.61 ms against 218.34, same session, interleaved, output
bit-identical.

**That comparison was confounded and the −3.7% is withdrawn.** It ran with the
fused epilogue on the direct path only. Re-run with the epilogue fused on *both*,
same session, interleaved, output bit-identical:

    GEMM_STAGED = false    p50 205.26 ms
    GEMM_STAGED = true     p50 204.10 ms      1.16 ms, 0.6%

A wash. It stays off — 0.6% does not pay for a second kernel with shape
restrictions — but "staging loses" was never true; the epilogue was doing the
losing. **The encode is the arbiter; the bench is for iterating.**

### The warp level, and the wall it hits

`GEMM_SUBTILES` — tiles per subgroup per axis, the reference's warp level — was a
constant of 2 with "4 measured worse" beside it, and the body's `@nexprs` were
**hardcoded to 2**, so raising it only moved the tile offsets. It is a real
parameter now (`GEMM_STAGED_KERNELS`, generated per `ST`), and 4 is genuinely
worse — 7.0 TFLOP/s against 21.3 on the largest shape. The driver says why:

    direct  (16 acc/subgroup)   wg=64   regs=255  smem=25344   4 wg/SM
    ST = 2  ( 4 acc/subgroup)   wg=128  regs= 96  smem= 8960   5 wg/SM
    ST = 4  (16 acc/subgroup)   wg=128  regs=255  smem=48896   2 wg/SM

**Our SPIR-V declares no `Workgroup` variables at all.** The direct kernel's
25 KB and ST=4's extra 31 KB are NVIDIA's own doing: 16 live fp32 accumulators is
128 registers per lane before anything else, so the driver spills cooperative
matrices to shared. At ST=4 that lands on 48896 bytes — 47.75 KB, against the
48 KB a workgroup gets — and occupancy halves.

So the reference's shape is the one to match, and it already is: `mul_mm.comp`'s
coopmat path carries `cms_per_row * cms_per_col = 2 * 2 = 4` accumulators per
warp, which is exactly ST=2. Both of our configurations sit at 21-23 TFLOP/s
against cuBLAS's 44.6, so the remaining factor of two is **not** the tile
hierarchy.

### Wide staging loads: ported, measured, removed

The reference **vectorises the global -> shared copy** (`LOAD_VEC_A_EFF`) and our
staging loop moved one `Float16` at a time, so this was the last structural
difference. It ports directly — `unsafe_load` through a `Ptr{NTuple{4,Float16}}`
works on this backend and is bit-exact — and **it loses at every width**, against
the register-blocked kernel on the four dominant shapes:

    scalar   1.04 - 1.09x
    2-wide   0.80 - 0.92x
    4-wide   0.69 - 0.81x

The global side is coalesced either way; the shared side is the problem. Ours is
`Float16`, so a lane that loaded `V` elements writes them with `V` scalar stores
whose addresses stride by `V` across the warp — a `V`-way bank conflict on every
store, against none for the stride-1 scalar loop. The monotonic 1 > 2 > 4
ordering is the conflict count, not noise.

The reference does not hit this because it widens the **shared array** at the same
time (`FLOAT_TYPEV2 buf_a[]`, `SHMEM_STRIDE` counted in `vec2`). Porting that
needs a wide store into `Workgroup` memory, which `@localmem` does not expose, and
a cooperative-matrix load that reads a `vec2`-typed shared array — our `loadw` is
typed by the matrix's `Float16`. **Half of it is worse than none.** That is the
real next step, and it is two new intrinsics rather than a loop rewrite.

### Both memory layouts, which the emitter did not have

`mul_mm.comp` stages A and B into **one** shared block and reads A `RowMajor`, B
`ColumnMajor` — A is `(M, K)`, B is `(K, N)` and they share the k axis, so one
staging pattern serves both. Lava's emitter hardcoded `ColumnMajor` on every
`OpCooperativeMatrix{Load,Store}KHR`, which forecloses that: one of the two
operands has to be transposed while staging, or copied twice.

Now a parameter (`AcceleratedMatrix{...}(src, offset, stride, Val(true))`, an
optional `_row` tag on the intrinsic name). Verified by loading one buffer both
ways and multiplying by the identity: column-major returns it, row-major returns
its transpose.

**No measured win yet, and it is not expected to give one on its own.** Our A is
column-major in global where ggml's is row-major, so the reference's layout
choice follows from *its* data, not from a shared-memory optimum — copying it
would buy a transpose we do not currently pay. This is a removed constraint, and
it is a prerequisite for two things that do want it: a faithful `mul_mm.comp`
staging pass, and flash attention, where Q, K and V do not share one layout.

### A correctness bug the benchmark shapes could never have found

`staged_gemm_applicable` asked about extents and batching and not about `splitk`.
**The staged kernel does not split K** — it walks the whole of it and writes one
plane — but `coopmat_gemm_shape` picks `splitk` from the shape, and at 64x64x64 it
picks **4**. The caller then allocates four partial planes and sums them, three of
which were never written: 0.83 relative error.

It was latent because `GEMM_STAGED` was off, and invisible because every shape the
kernel had ever been benchmarked on chooses `splitk == 1`. It only surfaced when
`GEMM_SUBTILES` became a real parameter and the correctness sweep widened to small
shapes. Guard added, `test_gemm_staged.jl` pins it — deliberately built from
shapes a GEMM is *not* fast at, because a suite made of the fast ones cannot find
this.

**6. Where the GEMM's time actually is, and the epilogue is half the answer.**
Splitting the two kernels per shape:

| M x N x K | splitk | gemm ms | epilogue ms | epilogue % | gemm TF/s |
|---|---|---|---|---|---|
| 2304 x 4096 x 576 | 1 | 0.507 | 0.178 | 26% | 21.4 |
| 576 x 4096 x 2304 | 1 | 0.483 | 0.038 | 7% | 22.5 |
| 1728 x 4096 x 576 | 1 | 0.365 | 0.098 | 21% | 22.3 |
| 576 x 4096 x 576 | 1 | 0.128 | 0.036 | 22% | 21.2 |
| 288 x 16384 x 1152 | 1 | 0.696 | 0.066 | 9% | 15.6 |
| 1152 x 16384 x 288 | 1 | 0.466 | 0.363 | 44% | 23.3 |

Two things fall out. The GEMM alone is a **consistent 21-23 TFLOP/s**, half of
cuBLAS — the 16.7 in the table above was the epilogue dragging it down. And
**`splitk == 1` on every shape**, so `mm_epilogue_kernel!` is not reducing
anything: it reads `M x N` fp32, adds a bias and writes `M x N` fp16, for
**23% of matmul time** and 25.3 ms of the encode.

*Occupancy is not the lever.* The driver reports the shipped kernel at **255
registers and 25 KB shared — 4 workgroups per SM, 17% occupancy** — and the
obvious conclusion is wrong: forcing a smaller register block makes it worse on
every shape (blk 4 / 2 / 1 = 16.8 / 12.9 / 7.1 TFLOP/s on the largest). The
kernel wants work in flight per thread, not more threads, and the chooser's
preference for the biggest block that divides is right.

*The epilogue can be deleted, and the mechanism is proven* (`test_coopmat_epilogue.jl`):

  * `convert` between cooperative matrix types — `OpFConvert`, new intrinsic —
    so an fp32 accumulator stores straight into an fp16 destination.
  * a **stride-0 load**, which broadcasts a length-`M` vector across every column
    of a tile, so the accumulator is *initialised with the bias* instead of zero
    and the bias add costs nothing. Not something the spec spells out; measured.

**7. The epilogue is deleted.** `accinit`/`accstore!` in the kernel, gated in
`matmul_coopmat!` on `splitk == 1 && NP == N` behind `DNNKernels.MATMUL_FUSED[]`:

    MATMUL_FUSED = false    p50 257.66 ms
    MATMUL_FUSED = true     p50 228.34 ms      29.32 ms, 11.4%

Same session, interleaved, 15 encodes each, clock at 2265. **`mm_epilogue_kernel!`
no longer appears in the profile at all** — 195 calls and 25.3 ms gone — and the
GEMM grew 65.6 -> 74.1 for the bias load and the convert, so the trade is 25.3
out for 8.5 back. Encode total 235.1 -> 220.8 ms.

Every other caller is untouched: attention and the convolution pass no bias, so
they get `zero` and an identity convert.

*One bug, and it is the interesting part.* The first version loaded the bias as
an **fp32** accumulator. Under autocast the model's biases are **fp16**, so it
reinterpreted the bytes — and the unit test passed, because the unit test used an
fp32 bias. SAM 2's masks went to IoU 0.0. `accinit` now loads in the bias's own
element type and converts, and `test_coopmat_epilogue.jl` covers both.

Accuracy is unchanged: maximum error against a **Float64** reference is identical
to four significant figures on every shape, and the two paths differ on 31
elements of a 2.36 M output by at most **0.5 ulp of fp16** — the bias now joins
the fp32 accumulation chain instead of being added after it. Mask IoU
0.98125 / 0.99964 / 0.95455 against 0.98750 / 0.99941 / 0.95349; the encoder
amplifies fp16-level perturbations through three global attention blocks, which
this document already records.

### What the full probe corrected

`tools/elementwise_profile.jl` printed only the top 20 broadcast shapes, and the
tail is where the plan was wrong:

* **`perm[1] != 1` is 8 dispatches of 254, not 25 of 203.** Each is a single
  call, and `toLE_tiled` already covers the big `(256,256,144,1)` family
  elsewhere. The ported `copy_transpose.comp` is real (5.3x, 101% of the linear
  copy floor) but worth ~1-2 ms here, not ~5.
* **`perm[1] == 1` is 228 dispatches** — 135 rank-4 attention, 89 rank-6 Hiera.
  That is the prize, and the blocked gather (1.46-1.54x on rank 4, measured
  bit-exact) is still the best candidate for it.
* **18 are not permutes at all**, 16 of them `(256,256,144,1)` copies from a
  `SubArray` whose `IndexStyle` is not linear, ~19 MB each. Nothing in this
  document mentioned them.

## Where we are

| | session start | 2026-07-30 | now | PyTorch | goal (90%) |
|---|---|---|---|---|---|
| encode, in-context profile | — | 264.7 ms | 220.8 ms | — | |
| encode p50, `bench_sam2.jl` | 552.6 ms | 256.0 ms † | **164.0 ms** | 87.6 ms | ~97 ms |
| of PyTorch's speed | — | 34% | **53.4%** | 100% | 90% |
| GEMM, weighted over the 6 `addmm` shapes | — | ~20 TF/s | **35.3 TF/s** (79% of cuBLAS) | cuBLAS 44.6 | — |
| VRAM live, encode+decode | 16 136 MB | 2 234 MB | **1 904 MB** ‡ | 1 756 MB | ≤1 951 MB |

‡ `tools/memcheck.jl`, steady state, dead pool capacity returned. Eleven repeats
across two fresh sessions agree. Meets the ceiling by 47 MB — read the three
caveats beside it before treating that as done.
| encode min | 465.8 ms | **255.2 ms** † | 80.8 ms | |
| editor click, new frame | — | **279.7 ms** | — | |
| editor click, same frame | — | **21.0 ms** | — | |
| VRAM steady | 16 136 MB | **2 234 MB** | 1 756 MB reserved | |

Memory is at **1.27x** of PyTorch and speed at **2.9x**. Neither is done:
"within 90%" puts the memory ceiling at **1 951 MB**, so 283 MB still has to go,
and the only change that clears it is flash attention (−402 MB → 1 832, below
PyTorch's own figure). Shrinking `COOPMAT_QCHUNK` gets to 2 033 and costs encode
time, which is the wrong direction on the other target. See "The memory target
has one answer" below — an earlier version of this line said memory was
effectively done, which was wrong: 1.27x is not within 90%.

† Session-dependent. The same binary measured 269.9 ms p50 on 2026-07-29, and a
same-session A/B showed the session's Lava changes are not the cause (see "The
session's Lava fixes cost nothing"). Treat these as a snapshot, not a target to
diff against across sessions.

Mask IoU against the PyTorch reference is 0.9875 / 0.99941 / 0.95349. It was
0.9875 / 0.99964 / 0.95556 until the fused layer norm, which is *more* accurate
than the expression it replaced (1.15e-07 against 4.91e-07 versus a Float64
reference) — the encoder amplifies a 5e-7 perturbation through three global
attention blocks whose own fp16 error is 5.8e-2 relative, so the last digits
move. Worth knowing before reading a mask diff as a regression.

## The kernel ledger — keep this current

One row per kernel: what it costs, what it has to beat, the best open
implementation to port, and where it stands. **Update the state column when a
kernel moves**; a stale ledger is how "elementwise is the biggest bucket" stayed
in this document after it had stopped being true.

Costs are in-context GPU ms from `tools/elementwise_profile.jl` (2026-07-31,
**220.8 ms** total). PyTorch column is from `tools/sam2_pytorch_kernels.py`, per
aten op, and several of ours map onto one of theirs — noted where so.

| kernel | calls | ms | target | state of the art to port | state |
|---|---|---|---|---|---|
| `coopmat_gemm_staged_kernel_1_v2!` | 178 | 74.1 -> ~45 | `addmm` 36.0 total; cuBLAS 44.6 TF/s vs our **35.3** (was 20.6) | `mul_mm.comp` + `mul_mm_funcs.glsl` — block -> warp -> thread, shared padded, `vec2` staging | **done, 1.68x**, **79% of cuBLAS**. 64 x 128 block, 8 warps, 2x2 tiles per warp, `PAD=8`, plus 96 x 128 for the 288-row shape, and `vec2`-typed staging buffers (1.09x on top of scalar; 1.51x on the 288-row shape, which stages most per unit of arithmetic). The reference's 4x4 warp tile is the *worst* config here (5.9 TF/s: 128 accumulator registers per lane, spilled to shared); widening the **warp grid** is what wins. Remaining 1.26x to cuBLAS: no double-buffering, and the k-loop reloads B from shared per `muladd` |
| `lava_broadcast_flat_mixed!` | 254 | 21.7 | no single aten op; 228 of its dispatches are permuted copies | `contig_copy.comp` **and** `generic_unary_head.glsl`'s `fastdiv`, both done; `copy_transpose.comp` for the 8 that transpose | **done** — 47.1 -> 38.5 -> 21.7, now 77-94% of a plain linear copy of the same bytes, which is the ceiling for a permuted one. Blocked gather **abandoned**: it beat the dividing path and loses to the multiplying one |
| `attn_flash_cm!` | 41 | 57.5 -> 28.7 | `_scaled_dot_product_flash_attention` 9.6 for **all** attention | `flash_attn_cm1.comp` + `flash_attn_base.glsl` — `O` in registers (not a coopmat), online softmax, `Br`/`Bc`/`HSK`/`row_split` spec constants | **done, 2.0x, −27.7 ms of encode measured interleaved**, and −281 MiB of VRAM because the score matrix stops existing. `64x32/8w`; **subgroups dominate every other parameter** (the same block at 4 warps is 7.34 ms against 5.09 at 8, and `32x64/2w` is 24.7 — five times the best), which is the GEMM's "warps in flight" result reached from the other side. Our `(E, L, H, B)` layout makes both products load with no transpose, so this is ~200 lines against the reference's 650. Still 7.6 TF/s global / 5.2 windowed against torch's ~21: the remaining items are the softmax using 64 of 256 threads, and `O`'s rescale + accumulator round trip through shared costing 60 KB of shared traffic per key block |
| `ndmap_flat!` (+ `ndmap!`) | 92 | 18.8 | — | `contig_copy.comp` shape + `fastdiv` | **done** — 25.1 -> 18.8. `LAUNCH_FLAT` was off because flattening cost N-1 divisions; with `FastDiv32` it is 11.4 ms/4.3% *faster*, so it is on |
| `attn_scores_b32!` | 37 | 16.4 | ditto attention | ditto — flash deletes this kernel, it only materialises `L x L` | **superseded** for the 41 calls a flash tiling divides; still the path for the 7 whose query count no tiling divides (`Lq` of 4 and 16) |
| `lava_broadcast_flat!` | 428 | 14.5 | part of `_to_copy`+`clone`+`add` ~19 | `contig_copy.comp` | **done**, 20.2 -> 14.5 |
| `attn_softmax_rows!` | 6 | 14.1 | ditto attention | ditto — 2 ms *per call*, global blocks only | not started |
| `coopmat_gemm_kernel_1!` | 12 | 13.8 | with `_4!`/`_2!` above | `mul_mm.comp` | not started |
| `coopmat_gemm_kernel_2!` | 20 | 10.6 | ditto matmul | `mul_mm.comp` | not started |
| `layernorm_kernel!` | 96 | 7.9 | `native_layer_norm` 5.6 | — | **done** — and more accurate than what it replaced |
| `conv2d_igemm!` | 4 | 5.3 | `convolution` 0.94 | `mul_mm.comp` (it is an implicit GEMM) | 5x behind but 2% of the encode; do it with the GEMM |
| `toLE_tiled_Float16!` | 45 | 2.5 | — | `copy_transpose.comp` (already the same 32x32 shape) | **done**, 7.1x when it landed |
| ~~`mm_epilogue_kernel!`~~ | ~~195~~ | **0** | folded inside `addmm` for PyTorch | bias in the accumulator's initial value + `OpFConvert` on the store | **DELETED** — 25.3 ms, gone from the profile |

Buckets, and how the gap now splits:

| bucket | ours | PyTorch | gap | share of gap |
|---|---|---|---|---|
| matmul | 103.9 | 36.0 | **68** | 49% |
| attention | 50.4 | 10.0 | **40** | 29% |
| elementwise + layout | 62.9 | ~32 | **31** | 22% |

**The elementwise bucket is no longer the problem** — it was 99.5 ms and 42% of
the gap this morning and is 62.9 and 22% now, within touching distance of
PyTorch. What is left is two ports: `mul_mm.comp` covers rows 1, 8, 9 and 11
(104 ms), `flash_attn_cm1.comp` covers rows 3, 5 and 7 (50 ms). Two reference
files address **154 of the 221 ms**.

## The measured table

**Measured, 2026-07-29.** `tools/sam2_pytorch_kernels.py` for torch (profiler,
device time) and `tools/sam2_kernel_table.jl` for ours (`DNNKernels.OPTIMES`,
which synchronises per op, so the totals exceed a free-running encode and only
the *relative* rows should be read as speed). Torch's
79.8 ms reconciles with its 87.6 ms baseline.

| aten op | ours ms | torch ms | gap | our TF/s | torch TF/s |
|---|---|---|---|---|---|
| `_scaled_dot_product_flash_attention` | 121.68 | 10.04 | **111.6** | 1.7 | 20.2 |
| `addmm` | 125.52 | 35.99 | **89.5** | 12.8 | 44.6 |
| `_to_copy` + `clone` + `add` | 79.36 | ~19 | **60** | — | — |
| `native_layer_norm` | 49.66 | 5.64 | **44.0** | — | — |
| `gelu` | 14.88 | 7.48 | 7.4 | — | — |
| `convolution` | 7.47 | 0.94 | 6.5 | 1.8 | — |

### What this corrected

Two things written above the line before the table existed, both wrong:

* *"PyTorch achieves 20.8 TFLOP/s on GEMM"* — that divided the whole 87.6 ms
  encode by the total arithmetic. GEMM is 35.99 ms of it, so PyTorch's GEMM runs
  at **44.6 TFLOP/s** and the target is more than twice as far away as stated.
* *"Attention is 11% of the arithmetic and will inherit any GEMM improvement"* —
  it is the **largest single gap in the table**, 111.6 ms, at 1.7 TFLOP/s. It
  was never going to inherit anything; it needed measuring.

Arithmetic share is a bad proxy for time share, which is the whole reason for
the table. `addmm` is 88% of the FLOPs and 30% of our time; attention is 11% of
the FLOPs and 29% of our time.

### Attention: not a threshold, a different algorithm

43% of attention's 203.1 GFLOP never reaches the tensor cores, because SAM 2's
Hiera encoder is mostly *windowed*:

| q `(E,Lq,H,B)` | calls | GFLOP | tensor cores? |
|---|---|---|---|
| `(72, 256, 8, 16)` | 32 | 77.3 | no — `Lq < COOPMAT_MINL` |
| `(72, 4096, 8, 1)` | 3 | 116.0 | yes |
| six smaller shapes | 13 | 9.8 | no |

The gate is not stale — the table in `attention.jl` measured `L=256 B=16` at
**0.81x**, i.e. the cooperative-matrix path genuinely loses there. But the path
it loses to is also slow, and the path it wins with is only 4.1 TFLOP/s on the
global blocks. Both of ours materialise the `L x L` score matrix; PyTorch's
flash attention never does. That is the difference, and no amount of moving
`COOPMAT_MINL` closes it.

A flash kernel was tried and reverted for being 2x slower — a 44.8 KB shared
tile left one workgroup per SM. Worth revisiting *after* the GEMM work below,
because it failed on shared-memory occupancy and that is the same constraint
being addressed there.

### The GEMM was tried first, and it was the wrong target

Built the shared-memory staged kernel, measured it, switched it off (Lava
8a3b72b). Two things the table above got wrong, both found by measuring the
kernels in isolation instead of through `OPTIMES`:

* **We are not at 12.8 TFLOP/s.** That is the kernel *plus* a per-op fence.
  Alone, after a warm-up long enough to lift the clock off its 210 MHz idle, the
  register-blocked kernel does **14-22 TFLOP/s** — 2.2x behind cuBLAS, not 3.5x.
* **Staging does not help on this hardware.** 1.03x at 64×64, 0.56x at 128×128
  (17.2 KB of workgroup memory admits two workgroups per SM). NVIDIA's L1 serves
  the repeated tile loads; staging only buys two barriers per k-step. The full
  per-shape table is in `gemm.jl` beside the constants.

Measure the clock. The first run of that comparison was taken at 210 MHz and
reported one shape as a 9.99x win.

Building it was still worth it: cooperative matrices could not address
`Workgroup` memory at all before, and that is what a flash-attention kernel
needs. It also uncovered an emitter bug that would have blocked one — a coopmat
accumulator carried across a loop containing a barrier got `i32 0` on its
loop-exit phi edge.

### Attention: measured, and it needs flash

Every distinct shape the encoder runs, both paths, one session, best-of-30 after
a 2 s warm-up:

| q `(E,L,H,B)` | calls | three-pass | coopmat | winner |
|---|---|---|---|---|
| (72, 256, 8, 16) | 32 | **1.16** | 1.25 | three-pass |
| (72, 4096, 8, 1) | 3 | — | **12.48** | coopmat only |
| (72, 16, 4, 1024) | 5 | **1.15** | 1.62 | three-pass |
| (72, 64, 16, 16) | 3 | **0.22** | 0.50 | three-pass |
| (72, 64, 2, 1024) | 2 | **1.33** | 3.63 | three-pass |
| (72, 4, 8, 1024) | 1 | 0.54 | **0.52** | coopmat |

    as shipped        84.2 ms
    all-coopmat       94.7 ms
    best-per-shape    84.2 ms      <- no gate tuning available
    PyTorch           10.04 ms

**8.4x behind, and `COOPMAT_MINL` cannot fix it** — the shipped gate already
picks the winner on every shape. The 84 ms splits evenly: 37 ms in the 32
windowed blocks, 37 ms in the three global ones.

An earlier single measurement showed coopmat *winning* the windowed shape
(1.24 vs 1.54); the sweep says otherwise. At ~1.2 ms the two paths are inside
the noise, so treat single points at that scale as unmeasured.

Both of our paths materialise the `L×L` score matrix; flash attention never
does. That is the whole difference and it is algorithmic. The previous flash
attempt was reverted for occupancy — a 44.8 KB shared tile, one workgroup per SM
— and it predates cooperative matrices being able to read shared memory at all,
so a coopmat flash kernel with a small tile is a genuinely new option rather
than a retry.

The `addmm` shapes, all of them tile-aligned (`M,N,K ≡ 0 mod 16`), so none of
them fall off the cooperative-matrix path:

| M × N × K | calls | GFLOP | share |
|---|---|---|---|
| 2304 × 4096 × 576 | 36 | 391.4 | 24.4% |
| 576 × 4096 × 2304 | 36 | 391.4 | 24.4% |
| 1728 × 4096 × 576 | 35 | 285.4 | 17.8% |
| 576 × 4096 × 576 | 36 | 97.8 | 6.1% |
| 288 × 16384 × 1152 | 6 | 65.2 | 4.1% |
| 1152 × 16384 × 288 | 6 | 65.2 | 4.1% |

Four shapes are 72.7% of the arithmetic. Optimise for those.

## Prior art: read it before writing another kernel

`reference/llama.cpp-vulkan/` (fetched 2026-07-29 from ggml-org/llama.cpp,
`ggml/src/ggml-vulkan/vulkan-shaders/`). This is the right reference for us and
not CUTLASS, because it solves our problem on **our API** — `VK_KHR_cooperative
matrix`, SPIR-V, no CUDA — and it is production code that tracks cuBLAS.

| file | what it settles |
|---|---|
| `flash_attn_cm1.comp` + `flash_attn_base.glsl` | coopmat**1** flash attention. `O` in registers, online softmax, `Br`/`Bc`/`HSK`/`row_split` as specialization constants |
| `mul_mm.comp` + `mul_mm_funcs.glsl` | three-level tiled coopmat GEMM: block → warp → thread |
| `copy_transpose.comp` | 32×32 tiled transpose, 32×8 threads × 4 rows, `[TILE][TILE+1]` padding |

**Two conclusions in this document were mine and are wrong.** Both came from
reasoning or from measuring my own implementation, and the reference overturns
both:

1. *"SPIR-V has no per-element coopmat access, so online softmax is impossible;
   use two passes."* No — `O` never needs to be a coopmat. See the retraction in
   step 5 below.
2. *"A shared-memory stage buys the GEMM nothing"* — measured at 1.03x/0.56x and
   switched off. That measured **my** staged GEMM, which went block → coopmat
   directly. The reference stages *and* adds a warp level (`WM/WN = 32×32`,
   `WMITER = 2`) so each subgroup reuses its fragments across sub-tiles, *and*
   pads shared to `BK/2 + 4` against bank conflicts, *and* vectorises the
   global→shared copy (`LOAD_VEC_A_EFF`, `LOAD_VEC_BATCH_A`), *and* makes every
   extent a specialization constant tuned per shape. Missing any one of those can
   erase the benefit, so "staging does not help" is not what my experiment
   established — only "staging without the rest does not help".

That matters because the GEMM is the single largest arithmetic gap: **12.8
TFLOP/s against cuBLAS's 44.6**, ~65 ms. Porting this structure is the plan, not
inventing another tiling.

## Superseded: "our GEMM has no shared-memory stage"

*Kept for the reasoning and the reference notes; the conclusion was wrong and
the measurement that replaced it is under "The GEMM was tried first" above.*


`Lava/src/array/gemm.jl` loads A and B **straight from global memory** into
cooperative matrices:

```julia
AcceleratedMatrix{Float16,GEMM_TILE,GEMM_TILE,MatrixA}(
    pointer(A), 1 + aoff + tm + (i - 1) * GEMM_TILE + k0_u * M, M)
```

With 4×4 register blocking that reads each A tile 4× and each B tile 4× per
k-step, from L2, and every subgroup in the workgroup repeats the B loads. Every
high-performance GEMM instead stages a block of A and B into shared memory once
and loads the cooperative matrices from there.

Checked against `llama.cpp`'s Vulkan backend
(`ggml/src/ggml-vulkan/vulkan-shaders/mul_mm.comp`, MIT, and already cloned at
`dev/llama.cpp`), which is the closest reference — same extension
(`VK_KHR_cooperative_matrix`), same hardware:

* `shared FLOAT_TYPEV2 buf_a[BM * SHMEM_STRIDE]; shared … buf_b[BN * SHMEM_STRIDE];`
* `BM = 64, BN = 64, BK = 32`; warp tile `WM = WN = 32`; register block `TM = 4, TN = 2`
* `SHMEM_STRIDE = BK/2 + 4` — the `+4` is bank-conflict padding, and the coopmat
  path uses 4 where the scalar path uses 1
* `coopMatLoad(cache_a, buf_a, …, gl_CooperativeMatrixLayoutRowMajor)` and
  `…ColumnMajor` for B, both **from shared**
* **no double buffering** — a plain `load → barrier → compute → barrier` loop.
  So pipelining is not what we are missing; staging is.

### The blocker, and it is in the compiler not the kernel

`Lava/src/compiler/spirv/coopmat.jl` hardcodes the storage class:

```julia
encode_instruction!(mod.types_constants, Op.OpTypePointer, ptr_ty,
                    SC.PhysicalStorageBuffer, comp_ty)
```

so `OpCooperativeMatrixLoadKHR` can only address global memory. Shared staging
needs a `Workgroup`-class pointer. The pieces for that already exist: `@localmem`
lowers to an LLVM global in `addrspace(3)`, and `emit.jl` already maps
`addrspace(3)` to `SC.Workgroup` for ordinary loads and stores.

The Julia side is `coopmat_load` in `Lava/src/device/coopmat_intrinsics.jl:44`,
an `@generated` function that emits a call to `_lava_coopmat_load_<dtype>_<M>x<N>_<use>`
taking a `Ptr{S}` — an integer address. A shared-memory variant cannot go
through `Ptr`, since there is no device address to convert: it has to keep the
`addrspace(3)` pointer so the emitter can see the storage class. So the change
is a second intrinsic name (or a suffix on the existing one) plus the branch in
`coopmat_base_pointer!` that emits `SC.Workgroup` instead of converting an
integer.

Also hardcoded: the layout is `ColumnMajor` for both load and store. The
reference uses row-major for A and column-major for B; our kernel compensates in
index arithmetic, which is fine for global but will need revisiting for shared.

## Plan, in order

1. ~~**Build the kernel table first.**~~ Done — it is above, and it changed the
   order of everything below it. Re-run both tools after each change; that is
   what makes "did that help" answerable per kernel instead of per encode.
2. ~~**Teach `coopmat_load`/`coopmat_store` the `Workgroup` storage class.**~~
   Done (Lava 6925d7d). `loadw`/`storew` take a `Core.LLVMPtr{T,3}`; the pointer
   must come from `OpAccessChain` and never `OpBitcast`, which is illegal for
   `Workgroup` in Vulkan's Logical addressing model, validates clean, and
   **segfaults NVIDIA's shader compiler**.
3. ~~**Restructure `coopmat_gemm_kernel!` to stage into shared.**~~ Done and
   switched off; see above.
4. **Take apart the elementwise/layout bucket — it is bigger than attention and
   nobody has looked at it.** The in-context breakdown below puts it at **113 ms
   of 260 (43%)** against attention's own kernels at 33 ms (13%), and the whole
   PyTorch encode is 87.6 ms. Arithmetic first, then work: even a *perfect* flash
   attention removes ~33-58 ms from 255, landing near 200 — nowhere near the 90%
   target. That target is unreachable without this bucket, so it goes first.

   `tools/elementwise_profile.jl` is the starting point: it runs one encode under
   `Lava.with_dispatch_timing` **and** `DNNKernels.LAUNCH_PROBE`, ranks the
   kernels, and lists every recorded launch whose grid cannot fill 48 SMs. The
   note under the table already flags two launches putting 64 workgroups on a
   48-SM card; a kernel that cannot occupy the device is not a kernel problem.
   The direction that has worked before is a **tiled** access pattern
   (`toLE_tiled`, 7.1x), not better addressing, which measured at −0.5 ms.

   **The specific first job, already quantified** (see the long note above
   `BROADCAST_PROBE` in `gpuarrays.jl`): `dest .= PermutedDimsArray(a, perm)` is
   **203 of 203** dispatches on the `flat_mixed` path — 40.1 ms moving ~2.4 GB at
   **60 GB/s, where a plain copy of the same bytes runs at 150-270**. Closing
   that to even 150 GB/s is ~24 ms, which is more than a *perfect* flash
   attention would return. In 178 of them `perm[1] == 1`, so it is a permutation
   of whole contiguous rows.

   And it is not one kernel but two: `lava_permutedims_kernel!` (11.95 ms, 51
   calls) is the *same* naive per-element gather —
   `dest[I...] = src[ntuple(d -> I[IP[d]], ...)...]`, no staging — so
   `permutedims!` and `dest .= PermutedDimsArray(...)` are two spellings of one
   unoptimised operation. Together **52 ms, 46% of the elementwise bucket**, and
   one tiled kernel serves both. That makes this the single largest identified
   opportunity in this document.

   **Measured in isolation, 2026-07-29** (`tools/permute_bench.jl`, SM clock
   2175 MHz — the first run of it read 495 MHz and had to be thrown away):

   | source shape | perm | linear copy | `.=` broadcast | `permutedims!` | of floor |
   |---|---|---|---|---|---|
   | (72, 8, 256, 16) | (1,3,2,4) | 50.6 | 12.3 | 4.3 | **24%** |
   | (72, 4, 16, 1024) | (1,3,2,4) | 58.7 | 13.1 | 4.4 | **22%** |
   | (576, 16, 4, 16, 4, 1) | (1,2,4,3,5,6) | 161.0 | 62.3 | 27.8 | 39% |
   | (288, 4, 32, 4, 32, 1) | (1,2,4,3,5,6) | 261.0 | 71.8 | 30.3 | 28% |
   | (256, 256, 144, 1) | (3,1,2,4) | 392.8 | 44.8 | 50.0 | **13%** |

   GB/s, read+write. Three things fall out of it:

   * The headroom is **larger than the encode suggested** — 13-39% of a plain
     copy of the same bytes, not the ~40% inferred from 60 vs 150. The 72-row
     attention family is worst in relative terms and is 88 of the 203 dispatches.
   * `permutedims!` is **~3x slower than the ordinary broadcast** on the rank-4
     shapes (4.3 vs 12.3 GB/s), where both are correct. The dedicated kernel
     loses to the generic one, so routing `permutedims!` at the broadcast path is
     a win on its own before any new kernel exists.
   * `permutedims!` was also **WRONG on both rank-6 shapes** while the broadcast
     was right — a silent miscompile, since fixed and written up below.
     **Their two `permutedims!` numbers above are therefore void**: 27.8 and
     30.3 GB/s were measured on a kernel writing half and one sixteenth of its
     output, and it was *still* slower than the broadcast. Re-measure them now
     that the guard routes those shapes through the dynamic path — which is
     itself ~2x slower than a correct static launch, so expect worse, not
     better.

   **The blocked candidate is measured and it works** (2026-07-29,
   `tools/permute_bench.jl`, bit-exact against `permutedims` on every shape):

   | source shape | broadcast | `permutedims!` | **blocked** | vs broadcast |
   |---|---|---|---|---|
   | (72, 8, 256, 16) | 11.7 | 4.2 | **18.9** | **1.62x** |
   | (72, 4, 16, 1024) | 12.5 | 4.4 | **22.7** | **1.82x** |

   GB/s, interleaved, SM clock 2175. An earlier non-interleaved run at a uniform
   780 MHz gave 1.76x and 2.11x on the same two shapes — different absolutes,
   same conclusion, which is what makes it believable.

   **The linear-copy "floor" column is NOT trustworthy and is dropped from this
   table.** `(256,256,144,1)` measured 23.8 GB/s on one run and 402.8 on the
   next — a 17x swing on one shape — so `copyto!` between two flat `LavaArray`s
   is not the stable baseline it was meant to be. Do not quote "% of floor" until
   that is understood; the blocked-vs-broadcast ratio is measured within a single
   interleaved run and does not depend on it.

   GB/s, at 780 MHz of 2265 — depressed, but every variant in that run saw the
   same clock, so the ratios hold.

   **A second run at a higher clock produced nonsense, and that is the useful
   part.** It reported the blocked kernel at **254% of the linear-copy floor** for
   `(72,4,16,1024)`, which is impossible: a permuted copy cannot beat a linear
   copy of the same bytes. The clock had warmed to 2175 MHz and read 1290 by the
   end — it drifts *during* a run, so measuring copy, then broadcast, then
   permutedims, then blocked in sequence compares four different machines.
   `tools/permute_bench.jl` now **interleaves**: every variant is timed once per
   round, round-robin, so they share one clock history. This is the same rule
   already recorded for LavaDNN A/B work — only interleaved same-session
   comparison is evidence — and it had to be re-learned here because a *floor*
   looked like a constant and is not.

   That confirms the mechanism — a bigger per-workgroup footprint turns scattered
   144-byte rows into contiguous streams — on 88 of the 203 dispatches, and takes
   them from 22-24% of floor to ~50%. The kernel is a rank-4 specialisation in the
   benchmark, not yet in Lava. Next: generalise to the rank-6 Hiera family (89
   more dispatches, and mind the trailing-singleton trap below), then upstream it
   as the `flat_mixed` path rather than a separate entry point.

   **The ported transpose is 7.0x — four times what my own design got.** From
   `reference/llama.cpp-vulkan/copy_transpose.comp`, specialised to the encoder's
   `(3,1,2,4)` family and measured in the same interleaved run, bit-exact:

   | source shape | perm | broadcast | mine (blocked) | **ported tiled** |
   |---|---|---|---|---|
   | (72, 8, 256, 16) | (1,3,2,4) | 12.1 | 19.8 — 1.6x | — |
   | (72, 4, 16, 1024) | (1,3,2,4) | 13.5 | 23.9 — 1.8x | — |
   | (256, 256, 144, 1) | (3,1,2,4) | 18.3 | — | **128.7 — 7.0x** |

   GB/s. It lands within noise of `toLE_tiled`'s independently measured 7.1x,
   which is corroboration rather than coincidence — both are the same 32x32
   shared-tile shape. The trick that makes it work is the one in the shader's own
   comment: the thread index runs along the **source's** unit-stride axis on the
   read and the **destination's** on the write, and the shared tile absorbs the
   transpose, so neither side is ever strided. Blocking alone cannot do that,
   which is why my kernel got 1.6x on a case where this gets 7.0x.

   Worth ~5 ms: three dispatches, but ~19 MB each, so ~113 MB of traffic at
   18.3 GB/s (6.2 ms) becomes 0.9 ms.

   **The lesson is the ratio, not the milliseconds.** Reading one 67-line shader
   beat a design I reasoned my way to over several rounds. Do that first from now
   on — see "Prior art" above.

   **Two different kernels, and the split matters** — do not write one and claim
   the bucket:

   * `perm[1] != 1` (25 of 203): the fast axis genuinely moves, so this is the
     classic **tiled staging transpose** — read a tile coalesced into
     `@localmem`, `@synchronize`, write it out coalesced along the other axis.
     `DNNKernels`' `toLE_tiled` is the working example (32×33 to break bank
     conflicts, 7.1x).
   * `perm[1] == 1` (178 of 203, the majority): the fast axis does **not** move.
     Each row is already contiguous in *both* source and destination, so there is
     nothing to transpose and a transpose kernel buys nothing. What costs here is
     that the rows are *scattered* across the outer axes, so what is needed is a
     **blocked gather**: give one workgroup a 2D block of the permuted outer
     axes, so the rows it reads are near each other in DRAM and the rows it
     writes are one contiguous span. **No shared memory is needed for this case**
     — nothing inside a row is reordered, so there is nothing to stage; what has
     to change is how much of the array one workgroup covers. This is the case
     worth doing first, because it is 88% of the dispatches.

     Concretely, the existing kernel *already* blocks a little: `staticgroup`
     hands it `(64,2,2,1)`, i.e. a 2×2 block of rows, 64 elements at a time. The
     lever is to make that block much bigger — 8×8 rows say — which a workgroup
     cannot do with one element per thread (8×8×96 is 6 144), so each thread has
     to loop over ~24 elements. Bigger per-workgroup footprint, same indexing,
     no staging.

   **Why blocking helps here, since it is not obvious.** A row is `size(dest,1)`
   elements — 96 fp16 is 192 bytes — and a permutation scatters whole rows, so
   the hardware sees isolated ~192-byte accesses on one side no matter which side
   you choose to stream: reordering the loop just swaps *which* side scatters.
   Staging whole rows cannot fix it either (a 16×16 tile of rows is 48 KB, far
   past the shared budget). What a **2D block of `B2 × B3` rows** does is turn one
   scattered line into `B` *streams* of `B × 192` contiguous bytes on **both**
   sides at once — at `B = 8` that is 1.5 KB per stream instead of 192 B, which
   is the difference between random-line and streaming DRAM behaviour, and
   therefore the difference between the observed 60 GB/s and the 150-270 a plain
   copy gets. The shared tile only has to hold `B2 × B3 × chunk` elements, not
   whole rows, so it fits.

   Two hypotheses already **ruled out by inspection**, so nobody spends a run on
   them: the launch shape is fine — `staticgroup((96,64,8,16))` gives
   `(64,2,2,1)` = 256 threads and a 4 096-workgroup grid, far more than the 48
   SMs need — and the accesses are already coalesced, since 64 contiguous fp16
   along the unit-stride axis is a full 128-byte line on both sides. What is left
   is that those 128-byte lines are *scattered*: a permutation sends consecutive
   destination lines to non-consecutive source lines, and scattered line-sized
   reads run at roughly half of streaming bandwidth, which is exactly the 60 vs
   150-270 GB/s observed. Blocking for locality is therefore the only lever left.

   **Trap for the rank-6 (Hiera) version, which does not apply to rank 4.** Those
   shapes end in a singleton — `(576,16,16,4,4,1)` — so a blocked kernel launched
   on a rank-6 ndrange trips the `trailing_unit_ndrange` guard added below and is
   re-launched dynamically, which costs ~2x and would read as the blocking having
   failed. The current broadcast path avoids it only because it uses a *linear*
   ndrange. So the rank-6 kernel has to drop trailing singleton extents from its
   ndrange (launch at effective rank 5 and treat the last index as 1) rather than
   pass the shape through unchanged. Measure against the broadcast, not against
   `permutedims!`, for the same reason.

   Do **not** retry index arithmetic: that was written,
   verified bit-exact on every encoder shape, confirmed firing on 90 dispatches
   per encode, and was worth −0.5 ms. Both kernels touch memory in the same
   order, and that order is the cost.

5. **Write a cooperative-matrix flash attention.** The largest *algorithmic*
   target — the two paths both materialise the score matrix and flash never does
   — and unreachable by tuning: the shipped `COOPMAT_MINL` gate already picks the
   faster path on every shape the encoder uses. Worth ~33-58 ms of GPU time
   against PyTorch's 10.04, plus 402 MB of scratch (see below), which is why it
   still ranks above everything after it.

   The previous attempt (`kernels/extern/flash.jl`, kept and documented) was
   **scalar** and lost on occupancy: 44.8 KB of shared memory at `E = 72` left
   one workgroup per SM, and its own analysis concludes no valid tiling beats
   the three-pass kernel's full occupancy. That conclusion is sound *for a
   scalar kernel* and does not carry over: with cooperative matrices the
   arithmetic per byte of shared memory is ~20x denser, so one workgroup per SM
   is no longer disqualifying.

   What changed to make it possible: cooperative matrices can now be loaded from
   `Workgroup` memory at all (Lava 6925d7d), and the emitter survives an
   accumulator crossing a barrier inside a loop (Lava 8a3b72b) — which is
   exactly the shape of an online-softmax loop.

   **The staged-GEMM negative result changes the design.** Loading cooperative
   matrices straight from *global* measured exactly as fast as staging them
   through workgroup memory (1.03x at 64×64), so a flash kernel does not need
   q/k/v in shared at all — which is what made the scalar attempt hopeless. It
   needs shared memory for one thing only, and that is forced:

   **SPIR-V gives a cooperative matrix no per-element access.** The online
   softmax has to take a row maximum and an `exp`, so each score tile must be
   stored, reduced in ordinary memory, and re-loaded as a `MatrixA` for the
   second product. `gemm.jl` records the same limitation for split-K
   accumulation.

   **RETRACTED — I reasoned from this to a wrong design, and the reference shows
   why.** I concluded the limitation "rules out the textbook online softmax",
   because rescaling `O` by `exp(m_old - m_new)` is a per-row multiply of a
   cooperative matrix, and recommended a two-pass form that recomputes `QK^T`.
   That is strictly worse and unnecessary. **Read `reference/llama.cpp-vulkan/`
   before writing this kernel** — it is a working coopmat1 flash attention on our
   exact extension (`VK_KHR_cooperative_matrix`), and it does the online form.

   The trick I missed: **`O` never is a cooperative matrix.** In
   `flash_attn_cm1.comp` the accumulator lives in per-thread registers —

       O_TYPEV4 Of[rows_per_thread][d_per_thread];
       float    Lf[rows_per_thread], Mf[rows_per_thread];
       ...
       eMf[r] = exp(Moldf - Mf[r]);
       Of[r][d_local] = O_TYPE(eMf[r]) * Of[r][d_local];   // rescale: trivial

   — and only the two *products* use coopmat: `SfMat = coopMatMulAdd(KMat, QMat,
   SfMat)` for the scores, then `coopMatStore(SfMat, sfsh, ...)` into shared, where
   ordinary scalar code does the max/exp/sum; and a second `coopMatMulAdd` for
   `P·V`, whose result is likewise read back from shared and accumulated into the
   registers. So one pass, no recomputed `QK^T`, and the score matrix still never
   reaches global memory.

   What else that file settles, so none of it has to be guessed:

   * `MatBr = MatBc = 16` — the same tile our `GEMM_TILE` already uses.
   * `HSK_pad = (HSK + 15) & ~15` — head size padded to the tile, exactly our
     `E = 72 → 80`.
   * `Br`, `Bc`, `HSK`, `row_split`, `D_split` are **specialization constants**,
     tuned per shape at pipeline creation rather than fixed — which is the answer
     to "what tile size", and it is *per shape*.
   * Staging K/V through shared is a compile-time toggle (`SHMEM_STAGING`), with
     `coopMatLoad` reading straight from global on the other branch. They kept
     both, which is consistent with our own measurement that global coopmat loads
     cost nothing here.
   * Shared strides carry `+2` (vec4) padding for bank conflicts, and there is an
     explicit "avoid padding for hsk==256 to make it fit in 48KB shmem" — they are
     budgeting against the same 48 KB we are.

   The per-key-block loop is then:

       S_tile  = coopmat(Q_tile) · coopmat(K_tile)     ← operands from global
       store S_tile to shared
       pass 1: reduce the tile's rows into m, l
       pass 2: P_tile = exp(S_tile - m)/l  in shared
               O_acc += coopmat(P_tile) · coopmat(V_tile)   ← P from shared

   Only `S_tile` is resident: at `BQ = BK = 32` that is 4 KB of fp32, ~11
   workgroups per SM — nowhere near the 17.2 KB cliff that cost the staged GEMM
   0.56x, and an order of magnitude under the 44.8 KB that sank the scalar
   attempt.

   What it buys is the score matrix never existing. Today `sdpa_coopmat!`
   materialises **two** full score-sized scratch buffers, not one:

       S = scratch!(ws, backend, Float32, CH, Lk, H, B)   # 268 MB
       P = scratch!(ws, backend, Float16, CH, Lk, H, B)   # 134 MB

   at `CH = 2048, Lk = 4096, H*B = 8`. The first GEMM writes `S`, `attnsoftmax!`
   reads `S` and writes `P`, the second GEMM reads `P`. That traffic is most of
   the 58 ms — and the 402 MB is also the largest single block of scratch we
   hold, against a 2 234 MB total that needs to reach PyTorch's 1 756.

   So flash is the one change that moves **both** targets at once, which is why
   it outranks anything else left on this list. Note the corollary: a flash
   kernel that keeps a chunk of `S` in shared memory but still allocates the full
   `(CH, Lk)` scratch has given up the memory half of the win.

   Two things already tried that do *not* substitute for it: shrinking
   `COOPMAT_QCHUNK` (swept — 1024 halves `S` to 201 MB and is *slower*, 370 vs
   352 ms).

   A cheaper fallback if flash stalls, **not yet ruled out**: `Accumulator` is
   parameterised — `AcceleratedMatrix{TC,M,N,Accumulator}` — so an fp16
   accumulator is expressible in Lava's types; whether it runs is a device
   question, `coopmat_shape(ctx, Float16, 16, 16, 16)` with an fp16 `TC`. If the
   card advertises it, `S` becomes fp16 and `attn_softmax_rows!` can write `P`
   **in place over `S`** (each row is one workgroup, and its writes all follow
   its reads), taking scratch 402 MB → 134 MB and traffic 804 MB → 536 MB for a
   small change. The catch is precision: fp16 accumulation over `K = 80` for
   pre-softmax scores, where PyTorch accumulates in fp32 — check the mask IoU,
   do not assume. It is strictly worse than flash on both axes, so try flash
   first.

6. ~~`native_layer_norm` at **8.81x**.~~ Done (JuliaVision 381ee42): one kernel,
   two passes instead of six, 0.341 -> 0.068 ms on the dominant shape and
   **314.4 -> 269.9 ms end to end**. It is also *more* accurate than the
   expression it replaces (1.15e-07 against 4.91e-07 versus a Float64
   reference), which moved the mask IoU slightly — see the commit; the encoder
   amplifies 5e-7 through attention whose own error is 5.8e-2.

Superseded, kept for the reasoning:

2. ~~**Teach `coopmat_load`/`coopmat_store` the `Workgroup` storage class.**~~
   MWE first: stage a 16×16 tile into `@localmem`, load a cooperative matrix
   from it, compare against the same tile loaded from global. Watch for the
   known `@localmem` trap — a size or type that comes from a local variable
   miscompiles silently and the kernel writes nothing; use `Val` parameters.
3. **Restructure `coopmat_gemm_kernel!` to stage into shared**, with the
   reference's parameters as the starting point rather than a search: BM/BN 64,
   BK 32, and `+4` padding. Measure on the four dominant shapes.
4. **Re-measure the table.** If `addmm` reaches PyTorch's ~21 TFLOP/s the encode
   lands near 110 ms; the remaining gap is then attention and overhead, and the
   table says which. (Both the 21 TFLOP/s figure and the reasoning that attention
   "will inherit any GEMM improvement" are superseded — PyTorch's GEMM is
   44.6 TFLOP/s, and the gap decomposition below says elementwise/layout is the
   larger half. Kept because the *method* — change one thing, re-measure the
   table — is still how to work here.)

## How to measure this, which took three mistakes to learn

Every one of these produced a confident wrong number today. They are cheap to
avoid and expensive to miss.

**Sync between the steps you are attributing.** Timing five dependent ops as one
block gave 2.6 ms where the same five, each synchronised, gave 0.5. Without a
fence you are timing the queue, not the op.

**A long warm-up loop is a different workload.** `lncmp`'s 1.5-second warm-up ran
thousands of iterations, each allocating three reduction temporaries, and
reported layer norm at 2.696 ms. The honest figure is 0.341. The expression
allocates and the fused kernel does not, so the loop flattered the new code by
8x. Warm up enough to raise the clock, not enough to change the allocator's
state.

**Read the clock.** The first staged-GEMM comparison ran at a 210 MHz idle clock
and reported one shape as a 9.99x win. `nvidia-smi --query-gpu=clocks.sm` in the
same measurement, every time.

**Prove the switch switches.** The permuted-copy A/B reported 6.90x with a branch
that never fired — `Broadcast.preprocess` wraps operands in `Extruded`, so the
`isa PermutedDimsArray` test was always false and both halves ran the *same*
kernel. The "gain" was ordering. Put a hit counter on the new path and assert it
moved before believing any number.

**Isolated kernel timings do not predict in-context behaviour here.** Three
separate times a microbenchmark showed a large win that the encode did not move
at all. `Lava.with_dispatch_timing` around a real encode is the arbiter; a
microbenchmark is a hypothesis.

## Where the GPU time actually goes

`Lava.with_dispatch_timing` over one encode, which is timestamps around every
dispatch rather than anything inferred:

    encode wall 265.2 ms | GPU busy 260.5 ms | 1410 dispatches | gaps 4.7 ms (2%)

**98% GPU-bound.** There is no launch overhead, no serialisation, nothing to win
back by batching or by capture/replay. Per-dispatch cost measured separately is
1.5 µs host / 3-5 µs wall, so 1410 dispatches is ~7 ms of the budget at most.
Everything below is real kernel time.

| kernel family | calls | GPU ms | share |
|---|---|---|---|
| `coopmat_gemm_kernel_4!` | 178 | 57.20 | 22.0% |
| `ndmap!` (all `launch!` bodies) | 98 | 42.78 | 16.4% |
| `lava_broadcast_flat_mixed!` | 203 | 40.12 | 15.4% |
| `mm_epilogue_kernel!` | 195 | 21.19 | 8.1% |
| `lava_broadcast_flat!` | 428 | 18.61 | 7.1% |
| `attn_apply_b32!` | 37 | 17.78 | 6.8% |
| `attn_scores_b32!` | 37 | 15.11 | 5.8% |
| `coopmat_gemm_kernel_1!` | 12 | 12.73 | 4.9% |
| `lava_permutedims_kernel!` | 51 | 11.95 | 4.6% |
| `coopmat_gemm_kernel_2!` | 20 | 8.45 | 3.2% |
| `layernorm_kernel!` | 96 | 6.50 | 2.5% |
| `conv2d_igemm!` | 4 | 5.14 | 2.0% |

### Re-measured 2026-07-29 with `tools/elementwise_profile.jl`

One encode, dispatch timing, fresh session. **Total 247.2 ms**:

    matmul         100.8 ms   40.8%
    elementwise     98.9 ms   40.0%
    attention       44.7 ms   18.1%
    other            2.8 ms    1.1%

Which **corrects the table below**: attention's own kernels are 44.7 ms, not 33,
and elementwise is 98.9, not 113. Against PyTorch's per-bucket cost the gap
(247.2 − 87.6 = 160) now splits **matmul ~65, elementwise ~67, attention ~35** —
so matmul and elementwise are effectively *tied* for first, not elementwise
clearly ahead. Both still have to move; attention remains third on time, though
it is first on memory (see below).

Ranked, the same run (top rows, GPU ms / calls):

    coopmat_gemm_kernel_4!      18.35 / 72     matmul
    attn_apply_b32!             16.28 / 32     attention
    lava_broadcast_flat_mixed!  15.54 / 30     permuted copy
    lava_broadcast_flat_mixed!  15.21 / 131    permuted copy
    attn_scores_b32!            14.79 / 34     attention
    coopmat_gemm_kernel_4!      13.12 / 37     matmul
    attn_softmax_rows!          12.11 / 6      attention  (2 ms *per call*)
    lava_permutedims_kernel!     7.14 / 36     permuted copy
    lava_broadcast_flat!         6.52 / 216    elementwise
    lava_broadcast_flat_mixed!   5.82 / 25     permuted copy

Which gives the two actionable totals, and they are **nearly equal**:

    permuted copies   36.6 (mixed, 186 calls) + 7.1 (permutedims!)  = 43.7 ms
    attention         16.3 + 14.8 + 12.1                            = 43.2 ms

Both correct earlier numbers in this document: the permutes are 43.7 ms, not the
52 estimated from the older table, and attention's own kernels are 43.2, not 33.
Flash attention removes all three of its rows — they exist only to materialise
and consume the score matrix — so the two items are worth about the same in time,
and flash is worth 402 MB on top.

**Bucket size is not achievable win, and that matters for the order.** A permuted
copy can never beat a plain copy of the same bytes, so its 43.7 ms has a floor,
not a zero: at the measured 22-39% of floor, recovering to a realistic 60-80% is worth
roughly **15-17 ms**, not 43.7.

   **Caveat on the floor for the small shapes, and do not skip it.** The
   `(72,8,256,16)` copy reads 50.6 GB/s — moving 9.4 MB in 186 µs — while the
   `(256,256,144,1)` copy moves *four times* the bytes in 96 µs at 392.8. A
   smaller transfer taking twice as long in absolute terms is not a bandwidth
   result, and it is far too slow to blame on launch overhead (~5 µs). Something
   about that measurement is wrong — most likely the `reshape(src, prod(sz))`
   the harness copies through. So treat 50.6 as **unverified** and re-measure the
   floor with a genuinely flat allocation before using it as the target the
   permute is judged against; the "24% of floor" figure inherits the same doubt. Attention's 43.2 ms is
different in kind — flash removes the score matrix rather than moving it faster —
but it is not a free 43.2 either: `attn_scores_b32!` and `attn_apply_b32!` *are*
the products, in scalar form, so flash has to **beat** the three-pass path on the
windowed shapes, where the shipped gate says three-pass currently wins
(1.16 vs 1.25 ms). Where it is unambiguous is the global blocks: `L = 4096`, no
three-pass option, and `attn_softmax_rows!` alone is 12.11 ms over 6 calls.

So the honest ranking by *achievable* win is closer than the bucket table
suggests, and flash carries the memory target as well. Do the permute work first
anyway — it is bounded, well understood, and has a harness that iterates in
seconds — but do not expect it to close the gap on its own, and do not read
"elementwise is the biggest bucket" as "elementwise is the biggest win".

The launch probe found only three thin launches — `(256,)` on one workgroup
(2 calls), `(64,16,16)` on 64 (4 calls), `(256,8,16)` on 128 (32 calls) — so
occupancy is a rounding error here, not the story.

### The gap, decomposed — and the two tables agree

The per-op table above and this dispatch breakdown look like they disagree, and
they do not: `OPTIMES` synchronises per op so its absolute totals are inflated,
while this is in-context. Subtract PyTorch's own per-bucket cost from ours and
the 167 ms gap (255.2 vs 87.6) splits cleanly:

| bucket | ours, in-context | PyTorch | gap | share of the gap |
|---|---|---|---|---|
| elementwise + layout | 113 | ~32 (`_to_copy`/`clone`/`add` + layernorm + gelu) | **81** | 48% |
| matmul | 99.6 | 36.0 (`addmm`) | **64** | 38% |
| attention kernels | 33 | 10.0 | **23** | 14% |

81 + 64 + 23 = 168 against an actual 167. Three independent measurements
reconciling to 1 ms is the strongest evidence in this document, and it settles
the order of work: **elementwise/layout first, matmul second, attention third** —
which is the reverse of the order they were being worked on.

It also says the target needs *two* of the three. Our matmul alone (99.6 ms)
already exceeds PyTorch's entire encode (87.6), so no amount of elementwise work
reaches 90% on its own: the GEMM runs at **12.8 TFLOP/s against cuBLAS's 44.6**,
and that 3.5x is the second thing to fix, not an afterthought.

Read the table as three buckets rather than twelve rows:

* **matmul, ~99.6 ms (38%)** — the three `coopmat_gemm_kernel_*` plus
  `mm_epilogue_kernel!`, which is the bias add and the split-K reduction.
* **elementwise and layout, ~113 ms (43%)** — the two broadcast kernels, the
  `launch!` bodies, `permutedims`. This is the bucket nobody has looked at, and
  it is bigger than matmul.
* **attention's own kernels, ~33 ms (13%)** — `attn_scores_b32!` and
  `attn_apply_b32!`, the three-pass path for the windowed blocks.

`LAUNCH_PROBE[]` in `DNNKernels.launch!` records ndrange => workgroup => grid for
every `launch!`; two of the encode's launches put **64 workgroups on a 48-SM
card**. `with_dispatch_timing` says which dispatch is slow, `LAUNCH_PROBE` says
which launch site and what shape.

### Flash attention on the tensor cores — 2026-08-01

**encode 162.7 -> 139.3 ms, VRAM 1904 -> 1623 MiB.** Both measured against the
path it replaces, the encode figure interleaved inside one process.

#### The measurement that chose the design

The shapes come off the exported graph, not from memory. `sdpa` is called 48
times and **two shapes are 95% of the arithmetic**:

| q shape (B,H,L,E) | calls | GFLOP each | total |
|---|---|---|---|
| `(1, 8, 4096, 72)` | 3 | 38.7 | 116.0 (57%) |
| `(16, 8, 256, 72)` | 32 | 2.4 | 77.3 (38%) |
| ten other shapes | 13 | — | 9.8 (5%) |

Then `tools/attn_lab.jl` timed `sdpa_coopmat!`'s stages separately on both:

| shape | padK | padV | padQ | gemm1 | softmax | gemm2 | unpad | total |
|---|---|---|---|---|---|---|---|---|
| 4096x4096 | 0.115 | 0.130 | 0.113 | 2.156 | **5.087** | 3.998 | 0.339 | 11.94 |
| 256x256 | **0.100** | **0.083** | **0.081** | 0.137 | 0.167 | 0.202 | **0.249** | 1.02 |

That table is what settled the question, and it did not say what the ledger
assumed. On the global blocks the **softmax alone is bigger than either GEMM** —
it is a pass over `Lq x Lk` and nothing else. On the windowed blocks the four
padding and unpadding kernels are **half the op**, with the actual arithmetic at
0.34 of 1.02 ms.

So the target was never "make the GEMMs faster". A second candidate existed and
was cheaper — `staged_gemm_tiling` refuses `nbatch > 1`, so all 35 attention
GEMMs still run on the unstaged kernel and never got the 20.6 -> 35.3 TF/s
improvement — but it addresses 0.34 ms of a 1.02 ms op and 6.2 of 11.9. Flash
addresses the softmax, the padding and the score-matrix traffic together, and it
is what PyTorch does.

#### Our layout makes this the short version

`flash_attn_cm1.comp` computes `S` **transposed** (`K·Qᵀ`), purely so an
implementation offering only `16x8` tiles can run it. We have `16x16`, and `q`,
`k`, `v` are `(E, L, H, B)` with `E` contiguous, so the plain orientation loads
every operand out of our own arrays with no transpose:

    S = Q·Kᵀ    A = Q RowMajor stride E     B = Kᵀ ColumnMajor stride E
    O += P·V    A = P RowMajor stride Bc    B = V  RowMajor    stride E

`attn_flash_cm!` is ~200 lines against the reference's 650, and the difference is
almost entirely quantised K/V, GQA, split-K and masks that SAM 2 does not use.

Two things were needed to write it at all. **`Lava.splitidx` everywhere**: `E`
is 72 and `EP` is 80, neither a power of two, so every staging index would
otherwise have emitted the `OpUDiv` that drops shared-memory stores. And a
**row-major `AcceleratedMatrix` load from `@localmem`**, which did not exist —
the layout operand had been wired for global pointers only, so asking for it
from shared produced `jl_f_throw_methoderror` in the middle of otherwise valid
GPU code rather than a missing method at the call site.

`O` is the one thing that cannot be a coopmat: it is rescaled by
`exp(m_old - m_new)` after every key block and there is no elementwise operation
on a cooperative matrix. The reference keeps it in registers and round-trips each
`P·V` tile through shared to add it; we keep it in shared and **load it as the
accumulator's initial value**, so the add is the tensor core's own.

#### Subgroups dominate everything else

| tiling | 4096x4096 | 256x256 |
|---|---|---|
| coopmat (replaced) | 9.50 ms | 0.883 ms |
| **64x32/8w** | **5.09** | **0.468** |
| 32x16/8w | 5.87 | 0.471 |
| 64x16/8w | 5.85 | 0.499 |
| 32x32/8w | 7.36 | 0.567 |
| 64x16/4w | 7.34 | 0.633 |
| 32x64/4w | 16.31 | 1.273 |
| 32x64/2w | 24.71 | 1.717 |

The same `64x32` block goes 7.34 -> 5.09 on four warps against eight, and the
worst thing measured is five times the best. This is the GEMM's result reached
from the other direction — there, widening the warp *grid* won and widening the
warp *tile* lost. What this device wants is warps in flight. `NW` is capped at 8
by `Lava.WORKGROUP_LIMIT`, which refuses more than 256 threads because larger
workgroups silently run only part of themselves here.

#### What it cost and what is left

Masks: 0.98125 / 0.99969 / 0.93333 against 0.98125 / 0.99964 / 0.95556. Two are
the same or better and the **worst-case decoder logit error more than halved,
1.660 -> 0.607** — but mask 3 covers 0.1% of the frame, so a handful of pixels
either side of the threshold move its IoU by 2 points. Called out rather than
averaged away.

Still 7.6 TF/s on the global blocks and 5.2 on the windowed against the ~21
PyTorch sustains over the whole op, so this is a 1.9x on a 4x gap. Two specific
items, in the order they look worth doing:

  * **The softmax uses 64 of 256 threads.** One thread owns a query row, which is
    what avoids a cross-lane reduction, but it leaves six of eight warps idle
    through two serial passes over `BC`. Four threads per row needs a *clustered*
    subgroup reduction (`subgroupMax` reduces all 32 lanes, which spans 8 rows)
    or a shared-memory round trip.
  * ~~**Keep `O` in registers, as the reference does.**~~ **Tried, and it loses.**
    The argument was good on paper: `O` in shared costs a rescale pass that reads
    and writes `BR x EP` plus an accumulator load on top of the store, 80 KB of
    shared traffic a key block against 40 for the register form. Built both
    behind `FLASHCM_REGO` and measured interleaved:

    | tiling | floats/thread | shared | registers |
    |---|---|---|---|
    | 64x32/8w, 4096² | 20 | **5.032 ms** | 6.347 ms |
    | 64x32/8w, 256² | 20 | **0.457** | 0.537 |
    | 32x32/8w, 4096² | 10 | 7.284 | **6.696** |
    | 32x32/8w, 256² | 10 | 0.567 | **0.560** |

    **The register form wins at `32x32` and loses at `64x32`, crossing exactly
    where `BR*EP/NT` goes from 10 floats a thread to 20.**

    I first wrote that up as register pressure. **The driver says it is not.**
    `VK_KHR_pipeline_executable_properties` reports **128 registers for the
    shared form and 122 for the register form, stack size 0 in both** — the
    arrangement blamed for spilling uses *fewer* registers and neither spills.

    What it actually costs is the pass that replaces the accumulator. `O` in
    shared makes the update two cooperative-matrix memory ops, which move a
    16x16 tile in the hardware's own fragment layout. `O` in registers replaces
    them with a scalar sweep over `BR*EP` — a `splitidx` per element per key
    block, to recover `(row, e)` from a flat index — plus a barrier to publish
    `P·V` first. That is `BR*EP/NT` index decompositions a thread a block, which
    is the quantity the crossing point is measured in, and it explains both data
    points rather than just the losing one.

    It also does not free the 20 KB that forces `BC = 32`, because `P·V` needs a
    `BR x EP` landing buffer of exactly the same size.

### The serialised table is not a budget — 2026-08-01

`tools/sam2_kernel_table.jl` puts our column next to PyTorch's, and the two are
**not the same measurement**. Ours serialises — a sync per op — and PyTorch's
comes from its own profiler as device time. On the 2026-08-01 state that reads
203.0 ms against a real encode of 135.6, so every one of our rows is inflated by
about half and every "excess" computed from it is overstated.

It changed which target looked biggest. From the serialised table, `addmm` was
63.62 against PyTorch's 35.99 — a 27.6 ms excess and the obvious second item.
`tools/elementwise_profile.jl` measures the same encode **in context**, with
`Lava.with_dispatch_timing`, and says:

| bucket | ours, in context | PyTorch | excess |
|---|---|---|---|
| matmul | 51.0 | 36.9 | 14.1 |
| elementwise + layout | 43.2 | ~31 | ~12 |
| **attention** | **34.1** | **9.6** | **24.5** |

50.6 against an actual gap of 48.0, so it reconciles. The GEMM is running at
**34.8 TFLOP/s in context** — which is the 35.3 the ledger already claimed from
`gemm_lab`, and not the 25.2 the serialised table computes. There was no 27 ms
sitting in `addmm` to go and find.

Two rules follow. Compare like with like — the serialised table is for *ratios
between our own kernels*, never against PyTorch's column. And it also explains
`foldoutcasts` measuring 1.4 ms against ~7 predicted: in context our `_to_copy`
work is already *faster* than PyTorch's `copy_`, so there was never 7 ms there.

### Skipping the rescale when the maximum has not moved

The flash kernel's largest single consumer of shared traffic is `O`: a rescale
pass that reads and writes it, plus the accumulator load and store — about 82 KB
of the ~110 KB a key block costs. The rescale factor is `exp(m_old - m_new)`,
and it is **exactly one** on every block whose maximum did not beat the running
one, which after the first few blocks is most of them.

One shared word records whether any row grew; the pass is skipped when none did.
Interleaved: 5.094 -> 4.836 ms on the global blocks and 0.471 -> 0.444 on the
windowed, about 5-6%, and the outputs are bit-identical — `@test outs[1] ==
outs[2]`, not a tolerance, because the skipped work is a multiplication by one.

Less than the traffic argument suggests, again, which says the kernel is not
shared-bandwidth-bound either.

#### The softmax is not the bottleneck either — built, measured, reverted

The obvious remaining complaint is that the softmax runs on `BR = 64` of 256
threads: six of eight warps sit at the barrier while two walk `BC` serially. The
fix looked free once the right mapping was found. A **clustered** subgroup
reduction is not needed and neither is shared scratch — store the score tile
row-major so a query row is contiguous, give each *subgroup* a row and each lane
a key, and at `BC = 32` the existing `subgroup_max` **is** the row maximum.

That needed one addition to Lava: a layout operand on the cooperative-matrix
*store* into `@localmem`. The load has taken one since the staged GEMM; the store
never did, though the emitter has always written whichever `MemoryLayout` the
intrinsic name carries. Six lines and a test.

Interleaved, both built:

    tiling      thread-per-row    subgroup-per-row
    4096x4096      4.836 ms          5.097 ms
    256x256        0.444             0.461

**5% the wrong way, so it went back.** Eight rows a subgroup at two subgroup
reductions each is 16 shuffle sequences per key block, and that costs about what
the serial walk did — the idle warps were never the problem. The two halves are
inseparable (a thread-per-row loop over a row-major tile puts all 32 lanes in one
shared bank), so the score tile is column-major again and the loop is back to one
thread a row. Lava keeps the store operand, with a docstring saying plainly that
nothing uses it yet and why.

So three of the four things that looked like the flash kernel's problem are now
measured and none of them was: `O` in registers (26% worse), the rescale pass
(5-6%, kept), and the softmax's idle warps (5% worse).

#### What the driver says about the shipped kernel

`VK_KHR_pipeline_executable_properties`, `64x32/8w`:

    Register Count       128
    Shared Memory Size   48 900
    Stack Size           0        <- nothing spills
    Binary Size          28 032

That fixes the occupancy, and it is **capped at two workgroups per SM by both
resources at once**: 256 threads x 128 registers is 32 768 of the SM's 65 536,
and 48 900 bytes is two of Ada's ~100 KB of shared. 512 of 1 536 resident
threads, 33%.

Two consequences. Cutting shared memory alone buys nothing — registers cap it at
two independently, so a third workgroup needs *both* under 34 KB and under 85
registers a thread, which is a redesign rather than a tuning knob. And the old
`attn_flash!` note that computed occupancy as `48 KB / shared` was using the
wrong budget: 48 KB is Vulkan's per-*workgroup* limit
(`maxComputeSharedMemorySize`), not what the SM has to hand out.

#### The ablation, which finally says where the time is

`tools/attn_lab.jl`'s `ablate()` is a copy of the kernel gated so stages can be
removed one at a time. It computes wrong answers on purpose; only the gaps
between variants mean anything, and `:all` reproduces the shipped kernel to
within 2% so the copy has not drifted.

| stage | 4096x4096 | share |
|---|---|---|
| softmax | **1.630 ms** | 33% |
| `P·V` product | **1.632** | 33% |
| `V` staging | 0.268 | 5% |
| `Q·Kᵀ` + staging + write-out | 1.391 | 28% |
| total | 4.921 | |

Two things fall out.

**The softmax really is a third of the kernel** — so the parallel version was not
attacking a small target, it simply cost as much as it saved. And it is not
`exp`: that lowers to GLSL.std.450 `Exp`, the hardware instruction, and 134 M of
them across the global blocks is ~0.08 ms of SFU throughput. What the loop
actually does is read `ss` **twice** per element, once for the maximum and once
for the exponential.

That one is fixable and is fixed. `FLASHCM_ONEPASS` exponentiates against `mo`,
the maximum from *previous* blocks, which is already known — legitimate because
the online softmax is exact for any reference (it cancels between the numerator
and `l`) — and defers the correction to the sweep that already runs after the
value product. One read a score instead of two:

    4096x4096   5.227 -> 4.383 ms   (16.2%)
    256x256     0.476 -> 0.430      ( 9.5%)

Encode 135.5 -> 133.3, and 65.7% of PyTorch. The bound is that `ps` is fp16, so
`exp(s - mo)` must stay under 65504; a row whose block runs hotter than
`FLASH_EXP_HEADROOM` is redone against its own maximum at two-pass cost, and its
`O` is converted in place by the one thread that owns it rather than by the
sweep. Every row takes that path on the first key block, where `mo` is `-Inf`.

Not bit-identical, unlike the lazy rescale — `ps` now rounds at a different scale
— so the test is a tolerance, and it deliberately includes an `Lk = 4096` case:
128 blocks is where the deferred correction is applied most often and where the
headroom fallback actually fires.

With the one-pass form shipped the kernel is 4.39 ms, and since nothing else in
the table moved, **`P·V`'s 1.63 ms is now 37% of it** — the largest single item,
ahead of a softmax down to roughly a quarter. The ablation harness is pinned to
the two-pass softmax and its `:all` control is compared against
`onepass = false`, so the row that would have silently drifted says so instead.

**`P·V` costs as much as `Q·Kᵀ` for identical arithmetic**, and the tile geometry
says why:

| product | accumulator tiles | muladds each | accumulator shared ops |
|---|---|---|---|
| `Q·Kᵀ` | `RT*CT` = 4x2 = **8** | `ET` = 5 | 8 stores |
| `P·V` | `RT*ET` = 4x5 = **20** | `CT` = 2 | 20 loads + 20 stores |

Same 80 operand-fragment loads either way, but **five times the accumulator
traffic**, because `EP = 80` makes the `O` tile grid five wide while `BC = 32`
gives each accumulator only two muladds to amortise its round trip. That is the
one asymmetry in the kernel and it is worth a third of the runtime.

The lever is `BC`: at 64, `CT = 4` and the round trips halve per unit of
arithmetic. It costs **17 156 bytes** of shared memory the tiling does not have
(66 308 wanted against 49 152), and there are exactly two places to find it —
`ss` in fp16 (−8 192) and `pvs` in fp16 (−10 240). Only both together are
enough.

**`pvs` in fp16 does not survive, so the lever is closed.** That is cheap to
settle without touching the kernel: simulate the accumulator's rounding on the
host — `O` built block by block, rounded to the storage type after each key
block, against the same sum in Float64.

| Lk | BC | blocks | fp32 `O` | fp16 `O` |
|---|---|---|---|---|
| 256 | 32 | 8 | 6.0e-07 | 7.3e-04 |
| 1024 | 32 | 32 | 1.2e-06 | 2.2e-03 |
| 4096 | 32 | 128 | 1.9e-06 | **3.9e-03** |
| 4096 | 64 | 64 | 1.9e-06 | 2.3e-03 |

fp32 is flat at ~1e-6 however many blocks; fp16 grows as roughly the square root
of the block count and reaches **3.9e-03 on the global blocks, against a 5e-03
tolerance**. That is not margin, it is the whole budget spent on one buffer —
and the simulation is *optimistic*, because it keeps the within-block sum in
fp32 where an fp16 cooperative-matrix accumulator would not.

It is tolerable at 8 blocks (7.3e-04), so a shape-dependent choice exists in
principle; it is not worth taking with mask 3 already at 0.93 IoU and 48
attention calls compounding.

So `BC = 64` at `BR = 64` is dead and `ss` in fp16 alone does not reach it.

#### Holding `O` in accumulators: measured before building, and it wins

The remaining idea is to keep each subgroup's `O` tiles in cooperative-matrix
accumulators **across** the key loop, flushing only on the blocks where the
maximum grows — which `FLASHCM_LAZYRESCALE` already shows are rare. The mechanism
is not in doubt: the staged GEMM already carries accumulators across a `for` with
barriers inside it.

I wrote it up as a coin flip on the grounds that three accumulators a subgroup is
~24 more registers a thread on a kernel the driver reports at 128, and two
workgroups an SM needs ≤ 128. **That was wrong, and `tools/attn_lab.jl`'s
`heldacc()` settles it without building the real thing** — a version that never
rescales, so it computes the wrong answer whenever a maximum grows, but which
bounds what the right one can do:

    shape        shipped    held-acc
    4096x4096    4.554 ms   3.116 ms   +31.6%
    256x256      0.439      0.335      +23.9%

    registers    128        122        shared 48 900 -> 48 640, stack 0 both

**Fewer registers, not more.** Holding the tiles removes the per-tile load, the
store, and all their address arithmetic, and the compiler comes out ahead — so
occupancy stays at two workgroups an SM and the 24-32% is not paid for anywhere.

#### …and then the real one loses, which is the actual finding

Built it (`FLASHCM_HELD`, and the softmax's headroom fallback had to become
workgroup-uniform on the way: a fallback row's `ps` is relative to its own new
maximum while a fast row's is relative to `mo`, and with `O` in accumulators
there is no per-row conversion to reconcile them, so if any row overflows every
row redoes the block). Bit-identical results, and **15-19% slower**:

    shape        shared O   held O
    4096x4096    4.494 ms   5.183 ms   -15.3%
    256x256      0.434      0.515      -18.7%

The whole gap is the flush/rescale/reload on blocks where `grew` fires, and the
assumption underneath both this and `FLASHCM_LAZYRESCALE` — "after the first few
blocks the maximum has settled" — is **wrong at this block size**. It is right
per row, and `grew` is an OR over all `BR` of them:

    BR=64  Lk=4096   128 blocks    grew on  67.3%
    BR=64  Lk=256      8 blocks    grew on 100.0%
    BR=16  Lk=4096   128 blocks    grew on  31.7%
    BR=1   Lk=4096   128 blocks    grew on   4.1%

On a growing block held `O` costs a flush, the sweep, a reload and two extra
barriers against shared `O`'s load, sweep and store — strictly more — and it only
wins on the blocks that do not grow. At `BR = 64` two thirds of them do, and on
the windowed shape none of them do.

That also explains a number that had been left as merely disappointing: the lazy
rescale is worth 5-6% rather than the traffic argument's more, because it skips
the sweep on a third of the global blocks and **none** of the windowed ones.

The upper-bound probe was not wrong, it was answering a different question — it
never rescales, so it measured a world where `grew` is never set. A bound is only
as good as the thing it assumes away, and this one assumed away the entire cost.

The obvious repair — a per-row-*tile* flag, so the sweep touches only the 16-row
tiles that actually grew — was the next thing to build, and the same ten lines of
host simulation say **not to**. Fraction of `O` swept per block:

    shape        per-workgroup   per-tile
    Lk=4096         67.3%          32.2%     2.09x less
    Lk=1024         96.4%          67.1%     1.44x
    Lk=256         100.0%          96.2%     1.04x   <- nothing

It halves the sweep on the global blocks and does **nothing** on the windowed
ones, which are the other half of attention's time. Weighted, under 1 ms of
encode — and it does not rescue `FLASHCM_HELD` either, because held `O` loses
worst exactly where the flag does not help.

The reason is the block count, not the flag. At `Lk = 256` there are eight key
blocks: rows are still finding new maxima throughout, because there has not been
time to settle. No amount of finer granularity fixes a distribution that has not
converged, and that is a property of the shape rather than of the code.

Both branches stay tested.

The other suspect, the global traffic flash trades the score matrix for, can be
ruled out by arithmetic: `k` and `v` are re-read once per query block, but per
`(head, batch)` they are 1.18 MB, and with two workgroups an SM the live working
set is ~9.4 MB. That is L2-resident on this card, so the 604 MB of "re-reads" are
cache hits.

### The layer-norm fusion is not the one the plan pointed at — 2026-08-01

Inductor's plan says 123 of its 151 encoder groups are the same shape: residual
`add` + `clone` + `native_layer_norm` + casts. Reading that as "fuse the residual
add into the layer norm" was wrong, and checking before building is what caught
it. In **our** graph, resolving views:

    layernorms                                          96
      whose input is an add.Tensor                       5
      ...where the add's sum has exactly one reader       0

**Five of ninety-six**, and none of those five is foldable, because the residual
sum is read again by the next block — which is exactly why PyTorch's fused kernel
writes it out too. Inductor's groups contain adds because its scheduler fuses a
whole region, not because `add -> layer_norm` is a chain. A fusion pass written
from the group composition alone would have found nothing to do.

What *is* there is the cast on the other side. 96 of the encoder's 199 runtime
`_to_copy`s narrow a value the graph just computed, with nothing else reading the
wide one — 51 layer-norm results and 45 clones, 282 M elements. `foldoutcasts`
declares the producer's result narrow instead, which needs no kernel change at
all: `allocate!` takes the dtype from the buffer and `coerce` forces the declared
dtype after every op, so the declaration *is* the mechanism.

**Measured 1.4 ms** — encode A/B/A in one session, 135.98 on / 137.38 off /
135.95 on — against about 7 ms the traffic argument predicts from 2.25 GB of
removed reads and writes. Masks and the worst-case decoder logit come back
bit-for-bit unchanged, which is the only acceptable result for a pass whose whole
claim is that it is free.

The gap between 7 predicted and 1.4 measured is the finding worth keeping. It is
not the lazy broadcast fuser having got there first — **zero** of the 96 were in
`fusableset`. It is that the serialised per-op table over-attributes these
copies, the same way it inflates every total, and its `_to_copy` row is not a
budget for what removing those copies can pay.

Two traps in the pass itself, both the same trap `foldrelu` records for aliases:
every one of the 96 reads its input **through a view** (a `getitem` of the layer
norm's tuple, or a reshape), so the producer is only found by walking the chain;
and the reader count has to treat views as transparent, or all 96 look twice-read
and nothing folds. A multi-output op's element dtype also lives in an
`attrs["dtypes"]` entry rather than the buffer's own field.

### PyTorch's own fusion plan, merged — 2026-08-01

We do not have to guess which ops may fuse: Inductor decides that for the same
graph, and `tools/dump_plan.py` composes its `preToPost` and `postToCppCode`
provenance maps into `fusion_groups` on the exported JSON. That machinery
already existed. It had **never been run on `sam2-large`** — it was written for
`graphs.py`'s seven-graph decomposition (`aten-autocast/encode_image`, 189 ops)
and the graphs `DNNKernels` actually runs (1353 and 174 ops) carried an empty
plan. `--model sam2` now covers them:

| graph | ops | groups | ops fused |
|---|---|---|---|
| `sam2_encoder` | 1353 | 151 | 657 |
| `sam2_decoder` | 174 | 16 | 98 |

**The tail is not partially fused, it is entirely fused.** Per-op, in the
encoder: `add.Tensor` 130/130, `clone` 90/90, `mul` 46/46, `clamp` 34/34,
`sub` 23/23, `index.Tensor` 16/16, `native_layer_norm` 96/96. That is the
mechanism behind the profile column that reads **0.00** for `clone`, `index`,
`repeat`, `clamp`, `sub` and `mul` — PyTorch never launches them, so it cannot
spend time in them. We launch every one.

The extreme case is one kernel of **165 ops**: the whole bilinear interpolation
of the position embedding (`arange`/`floor`/`clamp`/`index`/`mul`/`add`),
computed inline in a single Triton kernel. We run it as ~165 dispatches, and
that is the `index.Tensor` 4.8 ms and the `clamp`/`sub`/`mul` rows.

`gelu` fuses **0 of 48** — and that is not a mapping failure, it is Inductor
emitting `triton_poi_fused_gelu_view_11` as its own kernel. It is also the one
op in the whole profile where we are already at parity (7.1 ms against 7.5).
The correlation is exact: *we match PyTorch on precisely the op it does not
fuse.* That is the argument for group codegen in one line.

#### A name-matching bug that read as a result

The first merge reported 561 ops fused and `native_layer_norm` **0/96**, which
looked like "Inductor runs layer norm on its own too". It does not. Ops with
more than one result carry the overload in the provenance key —
`native_layer_norm_default_2` where our id is `native_layer_norm_2` — so 150
keys (96 layer norms, 48 attentions, 6 max-pools) matched nothing and were
reported as *unfused* rather than as *unmatched*. `prename` strips the suffix;
657/1353. The old `matched == 0` guard cannot catch this, because partial loss
is the likely failure and total loss is not, so `groups_from` now prints the
unmatched keys by op. The 1460 that remain are all `view`/`permute`/`slice`/
`squeeze` — shape-only ops our export lifts into buffers, which legitimately
have no op id.

#### Merging it changed one buffer, and that is the finding

`fuse.jl` is the only consumer, and it uses the plan for exactly one thing:
relaxing the dtype-boundary guard for pairs Inductor actually fused. With the
real plan in place, `fusableset` goes from **82 to 83** deferred buffers out of
1353 ops. End-to-end unchanged: masks 0.98125 / 0.99964 / 0.95556 and VRAM
1904 MiB, both identical to the recorded baseline.

That is not a disappointment, it is the diagnosis. Our fuser defers a value that
is elementwise, read exactly once, by another elementwise op — a *pairwise chain*.
Inductor's groups are not chains: they contain reductions, indexing and
many-to-one structure that pairwise deferral cannot express. The plan is
therefore not actionable through the current consumer, which is what
`DUPLICATE_FUSION`'s note already predicted: "real group codegen needs exactly
this — recomputing a shared subexpression inside each consumer's kernel."

#### What the groups say to build: one kernel, 123 of 151 groups

Grouping the 151 by composition, the top six shapes are the same kernel:

| groups | composition |
|---|---|
| 23 | 2x`_to_copy`, 1x`add`, 1x`clone`, 1x`native_layer_norm` |
| 20 | 2x`_to_copy`, 2x`add`, 1x`clone`, 1x`native_layer_norm` |
| 20 | 1x`_to_copy`, 1x`add`, 1x`native_layer_norm` |
| 19 | 1x`_to_copy`, 3x`add`, 1x`clone`, 1x`native_layer_norm` |
| 18 | 2x`_to_copy`, 3x`add`, 2x`clone`, 1x`native_layer_norm` |
| 18 | 2x`_to_copy`, 4x`add`, 2x`clone`, 1x`native_layer_norm` |

**Residual add + layer norm + the surrounding casts**, which is the transformer
block's spine and the single most standard fusion there is. Measured cost of the
ops it absorbs: `native_layer_norm` 12.9, `clone` 8.8, `add` 13.9, plus a share
of `_to_copy`'s 15.6 — against PyTorch's 5.6 + 0 + 5.2 for the same work. So the
prize is roughly **25-30 ms of the 164**, from one kernel rather than from a
general fusion engine.

The shape is already close: `layernorm_kernel!` takes `out, mean, rstd, a, γ, β`
and reads `a` twice, once per pass. Fusing means taking the residual as an extra
input, computing `x + r` in pass 1 and writing that sum out — which is needed
anyway, because the next block reads the same residual. That is a kwarg and a
store, not a rewrite. Pull llama.cpp's Vulkan norm shader first: its backend
fuses `ADD + RMS_NORM` and has already answered the layout questions.

Order of work is unchanged by this: attention is 69 ms of excess against this
25-30, and stays first.

## The editor's latency is not the encode

What a user waits for is a *click*, and until 2026-07-29 that was 1068 ms on a
new frame and 764 ms on a frame already embedded — against an encode of 255.
`sam2segmenter` kept its input staging buffer in a `Ref{Any}`, so the resize
loop's three million stores were each a boxed dynamic dispatch: **740 ms per
click, outside the GPU entirely**.

    click, new frame     1068.4 -> 279.7 ms     3.8x
    click, same frame     764.1 ->  21.0 ms    36.3x

### The first mark of a session is not SAM 2 either

Measured through the real `livematte!` path on a fresh process, 320x180 clip:

    framereader      0.0 s
    seedmask         2.5 s     <- SAM 2, cold
    previewmatte    38.6 s     <- MatAnyone, cold
                    -----
    first mark      41.1 s

**94% of it is the other model.** `previewmatte` builds the live preview by
running `MATTEPROPAGATOR[]` — the *propagation* network — on the marked frame, so
the first mark loads MatAnyone's weights and specializes its GEMM tiles for this
clip's matte resolution. Every later mark is warm (~0.6 s).

Two consequences worth keeping straight. First, SAM 2's editor integration is not
the slow part and no amount of encode work moves this number. Second, warming the
propagator when the matte panel *opens* is the right placement precisely because
the first mark needs it anyway — deferring the warm-up until after the first mark
was tried and reverted, since it only moves the same cost inside the click.

The remaining lever, if this is ever worth attacking, is the cold cost itself:
the precompile workload warms MatAnyone at the asset video's matte resolution, so
a clip of any other size re-specializes. That is a MatAnyone question, not a
SAM 2 one, and it is why `interactions.jl`'s first matte assertion needs a
120 s budget where every later one keeps 6.

**Untested hypothesis, cheap to check:** the matte resolution is
`mw = min(480, source_width)`, so it varies with the footage — and the GEMM
specializes per tile shape, so a 320-wide clip misses every frozen kernel the
workload built at 480. If that is where the 38.6 s goes, pinning the matte
resolution to a fixed 480 (padding narrower sources instead of shrinking the
tile) would make the shapes constant and let the frozen cache hit. Measure before
believing it: run the first-mark timing above on a 480-wide clip and on a
320-wide one and compare `previewmatte`. If they are the same, the cost is
weight loading and this idea is worthless.

The second row is the interaction that matters — adding a point to refine a
selection reuses the cached embedding and should be nearly free.

Worth generalising: a GPU profile cannot see this. `with_dispatch_timing` says
the encode is 98% GPU-bound and that is true; this sat in front of it. Before
optimising a kernel, time the whole user-facing call once and check the parts
add up.

### It was the wrapper again — 2026-08-02

The paragraph above used to end "repeated-click latency is now 21 ms, so anything
further there is the encode, not the wrapper". That was wrong, and the same
instrument that found the `Ref{Any}` finds why: time the whole call and check the
parts add up. On 1920x1080, a repeat click through `sam2segmenter` was **16.7 ms
against a decode of 3.3**, so 80% of it was still outside the model.

| | before | after |
|---|---|---|
| `resizeto!` the frame to 1024² | 2.69 ms | **0** — not called |
| `decode`, replayed | 3.30 | 3.30 |
| `Array(logits)` | 0.12 | 0.12 |
| `maskatframe` 256² → 1920x1080 | 9.67 | **1.04** |
| **a repeat click** | **16.7** | **5.05** |

Two things, both of the same kind — work that does not vary the way the code
assumed it did.

**The resize ran on every click and fed nothing.** `img` is consumed by exactly
one expression, the `encode` in the cache-miss branch, but it was computed above
the branch. On the path that matters — the second and later clicks on one marked
frame, which is what refining a selection *is* — it built a 3.1-million-element
buffer nobody read.

**`maskatframe` recomputed its x mapping 2.07 million times.** It wrote the
bilinear resample out literally: two divides, two floors, four clamps and four
scattered loads per output pixel. But `x0`, `x1` and `tx` depend on `i` alone and
are identical down every column, and for a single output row the two source rows
are fixed. Tabulate the x mapping once, blend the two source rows into a
256-long vector once per row, and the inner loop is two loads and two
multiply-adds against something that stays in L1: **8.10 → 1.04 ms**, output
bit-identical. Checked against the original loop kept verbatim as a reference, at
seven frame sizes including 1x1, 1x300 and 3840x2160 — the degenerate ones are
where `x1 == x0` and a tabulated mapping could disagree.

So the ordering lesson holds twice over: **the replayed decode saves 0.6 ms of a
click and this saved 11.6.** Both were invisible to every GPU profile, and the
second one hid behind a sentence in this file asserting there was nothing left.

## Memory, what is left

Unplanned allocation per encode is down from 1 648.6 MB to **56.6 MB, and 100%
of that is `kind=external`** — the encoder's own outputs, which the decoder
reads and which therefore must not share slab space. That is the floor for the
planner; further memory has to come from elsewhere:

* **workspace 624 MB** — dominated by the attention chunk buffers. `COOPMAT_QCHUNK`
  is 2048 (S fp32 + P fp16 = 403 MB); 1024 halves it for roughly 5% of encode
  time. Worth revisiting once the GEMM work changes the balance.
* **slab 698 MB** — against PyTorch's ~655 MB of activations. Effectively done.
* **weights ~530 MB**, **pool 768 MB** after GC.

`DNNKernels.PLAN_MISSES[]` and `Lava.dump_alloc_trace()` are the two diagnostics
that made this tractable; use them together, since the first sees `Recycler`
hits that never reach the pool and the second sees allocations that never go
through `dest`.

### Memory, 2026-07-31: −148 MB real, target still not met, and a retraction

**Retracted:** an earlier version of this section said the 90% memory goal was
reached at 1 940 MB. It was measured over **encodes only**, and the number it was
compared against is not encode-only. The full `bench_sam2.jl` workload — encode
*and* decode, which is what PyTorch's figure covers — is **2 800 MB live**,
849 MB over the 1 951 ceiling. The goal is not met.

What *is* solid is the saving, because both sides were measured the same way:
deleting `mm_epilogue_kernel!` also deleted its `M x NP x splitk` **fp32
scratch**, and over encodes,

    MATMUL_FUSED = false    2 064 MB live
    MATMUL_FUSED = true     1 930 MB live      -148 MB

**The decoder costs ~960 MB, and it is a steady state, not a leak.** Measured
over 450 synchronised decodes in one session, sampled every 50:

    encode only                    1 840 MiB
     50 decodes                    2 864
    100                            2 864
    150 through 450                2 800   (flat, nine samples)

*Second correction of the day on this number.* A coarser earlier series — 1 840,
2 288, 2 288, 2 800, 2 992, 3 184 at 0/25/50/100/200/300 — looked like unbounded
growth and was written up as a leak. It is warm-up to a steady state plus
allocator noise: 450 calls settle flat at 2 800. **Five points spanning a warm-up
are not a trend.** The rule that has held all day for timings — measure the thing
you are claiming, over a range long enough to show it — applies to memory too,
and I broke it twice in one afternoon on this figure.

What survives is the size: the decoder is ~960 MB of steady live allocation on
top of the encoder's 1 840, and **that** is why encode+decode is 2 800 against a
1 951 ceiling.

**And it is in none of the three structures the design accounts for.** At 300
decodes, with live at 3 184 MiB:

    slab              666 MiB   (unchanged from encode-only)
    workspace         403 MiB   (unchanged, `used` 0 between calls)
    recycler banks      2 entries, 2 misses total
    weights          ~640 MiB
    ------------------------------------------------
    accounted       ~1 709 MiB, against 3 184 live

`LIVE_BUFFERS` is 11-12 throughout, so it is a handful of large buffers, not an
accumulation of small ones. Roughly **1.4 GB sits outside the slab, the
workspace and the recycler** — the three things `execute!` reasons about. That is
the next thing to find, and `Lava.dump_alloc_trace()` is the tool for it:
`gpu_memory_usage()` counts buffers, and what is needed here is who is holding
them.

Every memory figure previously in this document — the recorded 2 234 and its
workspace/slab/weights/pool breakdown — is about the encoder, and the encoder is
the half that does *not* have this problem.

### Decode is not dispatch-bound — 2026-08-01

Decode is **14.3 ms against PyTorch's 2.10**, 14.7% of its speed and the worst
ratio in the project — and the editor pays it *per click*, where encode is per
marked frame. It had been recorded as dispatch-bound: "205 ops of one-workgroup
launches". Measured steady state, that is wrong.

Serialised, 23.22 ms over 171 ops; the sync costs about 52 µs an op, so real is
serialised minus `0.052 × n`:

| op | calls | serialised | real | share |
|---|---|---|---|---|
| `_scaled_dot_product_efficient_attention` | 7 | 6.44 ms | **6.08** | 43% |
| `convolution` | 2 | 3.85 | **3.75** | 26% |
| `addmm` | 47 | 5.90 | **3.46** | 24% |
| `add.Tensor` | 30 | 2.34 | 0.78 | 5% |
| ~115 others | — | 4.69 | 0.2 | 2% |

Three families are 93% of it. The one-workgroup elementwise launches the old
reading blamed are about 2% between them — they are real, they are all over the
dispatch log, and they are not where the time is.

**The attention is a different op from the encoder's** —
`_scaled_dot_product_EFFICIENT_attention`, whose signature carries an additive
bias — and it is tempting to conclude that the bias is why the fast paths refuse
it, since both do refuse a bias. **It is not: `bias` is `None` on all seven
calls.** Adding bias support to the fused kernel would buy exactly nothing here.

What refuses them is the extents. Every decoder attention has a **23** in it —
the mask prompt's token count — and 23 does not tile to 16:

| calls | Lq | Lk | E | MFLOP each |
|---|---|---|---|---|
| 3 | **23** | 4096 | 16 | 48.2 |
| 2 | **23** | **23** | 32 | 0.5 |
| 2 | 4096 | **23** | 16 | 48.2 |

242 MFLOP in 6.08 ms is **39.8 GFLOP/s** — about 0.2% of what the same device's
flash kernel reaches on the encoder. At even 2 TFLOP/s those seven calls are
0.12 ms instead of 6.08, so this is worth ~6 ms of a 14.3 ms decode.

The obvious next move is a **bounds-checked** flash kernel — pad the short extent
and mask past the end, which is `flash_attn_cm1.comp`'s `Clamp` and
`KV_bounds_check`, dropped from our port because the *encoder's* shapes all tile.

**That is also not enough, and this is the gate that actually decides it: the
decoder's `q`, `k` and `v` are `float32`.** Cooperative matrices take fp16
operands, so no amount of bounds checking puts these calls on the fused path.
`coopmat_sdpa_applicable` and `flashcm_applicable` both test the element type
first, and they are right to.

#### It is not just the attention: 93% of decode is fp32

Having found the element type gating one op, the obvious question is what else it
gates. All three families:

| family | share | dtype | consequence |
|---|---|---|---|
| attention | 43% | fp32 | refused by both fused paths |
| `convolution` | 26% | fp32, **transposed** | 804 MFLOP at 214 GFLOP/s |
| `addmm` | 24% | fp32 (all 47) | `strided_gemm_kernel!`, not the coopmat GEMM |

**Every one of them is off the tensor cores for the same reason.** The two
convolutions are stride-2 kernel-2 transposed convs upsampling 64x64 -> 128x128
-> 256x256 — cheap arithmetic, and 214 GFLOP/s on a card whose fp16 GEMM path
does 35 TFLOP/s.

That turns "decode's attention is slow" into one decision covering 93% of decode,
which is a much better-posed question than three separate kernel projects.

So decode needs one of two things, and the first is a decision rather than a
kernel:

  * **Export the decoder under autocast — measured, both sides.** Exported to
    `gen/graphs/sam2-large-ad` (weights and refs symlinked, so it costs a few
    hundred KB) and run against the same PyTorch references:

    | decoder | decode | logit max abs | mask IoU |
    |---|---|---|---|
    | fp32 (shipped) | 14.47 ms | 6.624e-01 | 0.98125 / 0.99964 / 0.95455 |
    | autocast | **12.56 ms** | **6.562e-01** | 0.98125 / 0.99952 / **0.97727** |

    **The accuracy fear does not materialise.** The default was chosen because
    fp16 "only costs precision in the mask logits the threshold is taken from";
    measured, the worst-case logit error is slightly *smaller*, mask 1 is
    identical, mask 2 moves 1.2e-4, and mask 3 — the marginal one, 0.1% of the
    frame — goes **up**, 0.95455 -> 0.97727.

    **And the speed-up is 13%, not the 40% the "93% is fp32" framing implies.**
    fp16 makes those ops *eligible* for the tensor-core paths; it does not make
    them *fit*. Attention is still refused by the 23-token extent, and the small
    `addmm` shapes still have to land on the tile. Eligibility and admission are
    different gates, and only the first one changed.

    **Checked on more than the reference frame**, because one click point is not
    evidence about a decoder. 24 click points across the frame, 3 masks each, the
    two decoders compared against *each other*:

    | mask coverage | n | worst IoU | median IoU | median size |
    |---|---|---|---|---|
    | < 0.1% | 3 | 0.8478 | 1.0000 | **5 px** |
    | 0.1-1% | 14 | 0.9320 | 0.9940 | 358 px |
    | 1-10% | 28 | 0.9561 | 0.9967 | 1 290 px |
    | > 10% | 27 | **0.9904** | 0.9995 | 24 997 px |

    Median 0.99820 over all 72, worst logit delta 0.133. The disagreement is
    monotonic in mask size and is threshold noise on a boundary: the 0.8478 is a
    **five-pixel** mask, where one pixel moves IoU by 20 points. On anything a
    user would call a mask — above 10% of the frame — the worst case is 0.9904.

    **With the bounds-checked kernel as well, decode is 8.44 ms.** The autocast
    export alone is 13%, because fp16 makes the attention *eligible* and the
    23-token extent still refuses it; `FLASHCM_CLAMP` is the other half. Together:

    | configuration | decode | of PyTorch |
    |---|---|---|
    | shipped (fp32) | 14.28 ms | 14.7% |
    | autocast | 12.68 | 16.6% |
    | autocast + `FLASHCM_CLAMP` | **8.44** | **24.9%** |

    **41% off the per-click latency**, and the 24-point check holds for the pair
    exactly as it did for the export alone — median IoU 0.99768 against the
    shipped fp32 masks, worst 0.9914 above 10% coverage, and the only sub-0.95
    cases are masks of 5 to 358 pixels.

    Recommended as a pair; neither half is much use alone. Still your call
    because it is the mask logits, and **not switched** — the shipped graphs are
    untouched, `FLASHCM_CLAMP` ships off, and the alternative export sits in
    `gen/graphs/sam2-large-ad`.
  * **A scalar path that likes skinny shapes.** 23 x 4096 x 16 is a shape the
    three-pass kernel handles at 39.8 GFLOP/s; the score matrix is only 753 K
    elements, so this is launch geometry and occupancy, not bandwidth.

#### The two transposed convolutions are a GEMM wearing a disguise

Second item is the two convolutions, and they are **the largest single thing left
in decode**: 3.73 ms of the packaged 8.44, 44%. Interleaved at a verified 2280 MHz:

    conv1  256x64x64   -> 64x128x128    fp32 2.497 ms    fp16 2.677    537 MFLOP
    conv2   64x128x128 -> 32x256x256    fp32 1.229       fp16 1.323    268 MFLOP

**fp16 does not help them — it is 7% worse.** They are the one part of decode the
autocast export does nothing for, which is why they go from 26% of a 14.3 ms
decode to 44% of an 8.44 ms one.

Getting to that number took three tries, and the two failures are the warning:

  * The serialised per-op table stops working at this speed. Before the package
    it inflated 23.22 against a real 14.3 — a workable 1.6x. After, 53.01
    against 8.44, **6.3x**, because the fixed sync per op dwarfs the ops. Its
    21.4 ms for the convolutions is not a number. (Its *pre*-package 3.75 ms
    was fine — two calls carry almost no sync — which is the subtlety: the same
    table is trustworthy for few-call ops and useless for many-call ones.)
  * Standalone timing hit the clock trap twice. Unwarmed it reported fp32 at
    15.4 ms for a **single** convolution, more than the entire decode containing
    it, and made fp16 look 5.7x faster. A second try with a weak warm-up said
    1.46x faster. Both were the clock: fp32 ran first and colder. Only the third
    attempt, gated on `warmclock()` reaching 2280, is above.

Alongside the algebra, which needed no measurement at all:

They are `ConvTranspose2d`, both **stride 2, kernel 2, padding 0**, and they run
through `convtranspose2d`: one thread per output element, no reuse, the same
shape as `convolution_direct!` — which its own docstring calls "the reference the
implicit-GEMM kernel is checked against". 804 MFLOP in 3.75 ms is 214 GFLOP/s.

With stride equal to the kernel and no padding the receptive fields do not
overlap, so each output pixel comes from exactly **one** input pixel and one
`(di, dj)` slice of the weight:

    out[co, 2i+di, 2j+dj] = Σ_ci  x[ci, i, j] * W[ci, co, di, dj]

That is a plain GEMM over `(H*W) x C_in x (C_out*4)` followed by a fixed
depth-to-space interleave — the pixel-shuffle identity:

    conv1  M=4096   K=256  N=256   537 MFLOP   0.015 ms at the GEMM's 35 TF/s
    conv2  M=16384  K=64   N=128   268 MFLOP   0.008 ms

The arithmetic is trivial and the shuffle will dominate, so the pair should land
around 0.1-0.3 ms against **3.73 now** — about 42% of the packaged decode, and
the biggest single item left anywhere in this model. The coopmat GEMM wants fp16,
so it pairs with the autocast export; at fp32 it would go through
`strided_gemm_kernel!` instead, which is still a tiled GEMM against a kernel with
no reuse at all, and worth measuring rather than assuming.

#### Two measurement traps, both hit before the table above was right

`empty.memory_format` — a **zero-element** allocation — first measured at 13.1 ms
and 34% of decode, which read as "one allocation is most of decode". It is a
first-iteration artefact: the first `OPTIMES`-instrumented decode carries ~1500 ms
of one-time cost and it lands on whichever op runs first. Reps 2-4 put that op at
0.007-0.032 ms. **Discard the first instrumented iteration**, every time.

And the explanation that anomaly suggested — that the allocation was firing
`maybe_trim_pool!` — is also wrong, and disproved rather than argued: disabling
the trim entirely makes decode **worse**, 14.5 -> 23.1 ms. The trim is
load-bearing for this workload, not overhead.

### The gap is 5.3 ms, not 24.5 — and the stem convolution was 1.7 of it — 2026-08-02

Two things had to be re-measured before any of this made sense.

**The target is much closer than the framing suggested.** 90% of PyTorch's speed
means an encode of `87.64 / 0.9 = 97.4 ms`, and the encode was **102.65**. So the
gap is **5.3 ms**, not the 24.5 that attention costs. Every plan on file said
"attention is the entire remaining gap"; it is 33% of the encode, but only a
fifth of it needs to go away, and it need not come from there at all.

**Attention is at its shape's ceiling, and that is now proven rather than
assumed.** Its two products, taken out and run as standalone batched GEMMs on
identical shapes:

| product | as a standalone GEMM | inside the flash kernel |
|---|---|---|
| `QK^T`, M4096 N4096 K80 x8 | 1.662 ms, 12.92 TF/s | 1.429 ms, **13.5 TF/s** |
| `P·V`, M4096 N80 K4096 x8 | 2.455 ms, 8.75 TF/s | 1.628 ms, **11.9 TF/s** |
| control, M4096 N4096 K1024 | 1.347 ms, 25.50 TF/s | — |

The fused kernel beats Lava's own best GEMM at both. `E = 72` gives reduction
depths of 80 and 32, and 12-13 TF/s is what this device does at that depth. That
is why five successive kernel hypotheses died: parallel softmax, held `O`,
per-row-tile `grew`, the tiling sweep, and `row_split` (built earlier, 4.836 ->
5.097, and its own source comment says so).

#### Where the 5.3 ms is, measured

`OPDOUBLE` over each op family, in context, summing to 100.1% of the encode:

| family | ops | ms | share |
|---|---|---|---|
| `addmm` | 195 | 44.38 | 43.3% |
| attention | 48 | 34.33 | 33.5% |
| **`convolution`** | **7** | **6.91** | **6.7%** |
| `native_layer_norm` | 96 | 6.09 | 5.9% |
| `clone` | 90 | 3.28 | 3.2% |
| `add.Tensor` | 98 | 3.15 | 3.1% |
| `_to_copy` | 97 | 2.34 | 2.3% |
| `max_pool2d` | 6 | 2.05 | 2.0% |

`addmm` is 1606 GFLOP in 35.8 ms — **44.9 TF/s**, near this GEMM's best, so the
biggest bucket is not the opportunity. (A standalone control at `K = 144` reads
12.7 TF/s and would have said otherwise; the real shapes have `K` of 576-2304.)
Seven convolutions costing 6.91 ms is the outlier: **13.4 GFLOP at 1.94 TF/s**.

#### The stem: 2.800 -> 1.147 ms, 2.44x

SAM 2's patch embed is `7x7x3 -> 144` at 1024², and `CRS = 3·7·7 = 147` does not
land on the 16-tile. `conv_coopmat_applicable` refused it for that reason —
correctly, on the reasoning that `CRS` is the *weight's* extent and padding it
means padding the weight — so it ran on the implicit-GEMM kernel at
**0.99 TFLOP/s**.

The weight can be padded. Both halves are zeros: `im2col` writes zero columns for
the channels that do not exist, and the weight gets a zeroed copy `CRSP` rows
tall. Zero times anything is zero, so the padded product *is* the real one. The
one trap is that the weight's pad must be **written**, not merely reserved —
uninitialised memory there would multiply a possible NaN by zero and give NaN.

    stem, 2.77 GFLOP    implicit GEMM  2.800 ms  0.99 TF/s
                        padded coopmat 1.147 ms  2.42 TF/s     2.44x

Encode **102.65 -> 100.70 ms, 85.4% -> 87.0% of PyTorch**, and the masks got
*better* (mask 3's IoU 0.977 -> 1.000). VRAM +20 MiB, which is exactly the stem's
im2col matrix (65536 x 160 fp16), against 770 MiB of headroom.

Padding is paid for, so it is bounded: `CONV_CRS_PAD = 1.25`. The stem wastes
8.8% and is admitted; MatAnyone's `Cin = 17` layers would waste 88% and are
refused. Its `Cin = 257` and `769` layers waste 5.8% and 2.0% and are newly
admitted — measured in one session, both orders, that model's step is
**1.4% faster**, so the wider rule does not cost the other workload.

#### The 90 clones are an optimisation, not waste — closed

`clone.default` is 3.28 ms and 511 MB written, and every one of them exists to
make a *reshape* valid after a permute. The obvious move is to alias them away
and let the consumer read the view, and `execute.jl` even carried a reason why
that was impossible: the nest "is not recognised as a GPU array by either
backend". **That reason is stale** — Lava's `AnyLavaArray` covers `ReshapedArray`
over `PermutedDimsArray`, the nest gets `LavaArrayStyle`, and reading it on the
device works and is correct.

It is just slow. One broadcast over a 4.5 MB tensor, the same data three ways:

    dense leaf                       0.029 ms   461 GB/s
    PermutedDimsArray leaf           0.050      263
    ReshapedArray{PermutedDims}      0.106      124

`ReshapedArray` carries `SignedMultiplicativeInverse{Int64}` and pays a 64-bit
magic division per element on a device with no 64-bit integer unit — the same
thing that made narrowing `im2col_kernel!`'s counter worth 1.56x. So the fix is
known and cheap to describe: an `Int32` reshape wrapper for broadcast leaves.

**And it would not be worth having.** End to end on the real window-partition
chain, clone-then-read is **0.053 ms** against reading the nest at **0.103**, and
with the `Int64` division fixed the nest would reach about **0.050** — a wash,
because the consumer still pays the permuted read either way and the clone's
dense add runs at 461 GB/s. The clone converts one expensive read into one cheap
read plus one cheap write, and on this hardware that is the better trade.

Measured before building the graph pass, which is the whole point of measuring
first. What is left of the elementwise tail — `add.Tensor` 3.15 ms, `_to_copy`
2.34, layer norm 6.09 (already at parity with PyTorch), max pool 2.05 (strided by
construction, see the earlier misdiagnosis) — has no similar lever behind it.

#### The GEMM is at 35% of the device's ceiling, and the tiling space is exhausted

`addmm` is 43% of the encode, so it is worth knowing whether 38 TF/s is the
kernel or the machine. **The machine does 107.3 TF/s.** Measured, not looked up:
a kernel holding eight independent accumulator chains per lane and doing nothing
but `muladd` on shared operands that never change, 384 workgroups over 48 SMs —
103.1 GFLOP in 0.961 ms. (Spec sheets quote 153 TF/s dense fp16, but that is with
fp16 accumulate; 107 is what fp32 accumulation actually reaches here, and it is
the number to compare a real kernel against.)

Against that, the dominant `addmm` shape runs at 38.03 TF/s — **35%**. So unlike
attention, the GEMM is *not* at its shape's ceiling, and it is the biggest bucket
in the model. The obvious lever is the tiling, and it is spent:

| tiling | tile | warps | acc/warp | ms | TF/s |
|---|---|---|---|---|---|
| `(2,2,2,4,32,8)` | 64x128 | 8 | 4 | **0.286** | **38.03** |
| `(2,2,2,2,32,8)` | 64x64 | 4 | 4 | 0.297 | 36.62 |
| `(1,2,2,4,32,8)` | 32x128 | 8 | 2 | 0.373 | 29.12 |
| `(4,2,2,4,32,8)` | 128x128 | 8 | 8 | 0.432 | 25.14 |
| `(2,2,4,2,32,8)` | 128x64 | 8 | 4 | 0.435 | 25.02 |
| `(2,2,2,4,64,8)` | 64x128, `BK=64` | 8 | 4 | 0.435 | 24.99 |
| `(2,3,2,4,32,8)` | 64x192 | 8 | 6 | 0.441 | 24.66 |
| `(2,2,4,4,32,8)` | 128x128 | 16 | 4 | 0.441 | 24.63 |

The shipped choice wins and **every larger tile is worse**, which rules out the
ILP story: more accumulators per warp costs more than it buys. Note the first
attempt at this table had every row after the second converging on 0.437 — the
clock had drifted down during the sweep. Re-warming to 2280 MHz *before each
candidate* and printing it beside the number is what made the table mean
anything; see the measurement traps at the end of this file.

So the remaining 2.8x is structural, and **it is not something the reference has
that we lack**: `mul_mm.comp` uses one `buf_a`/`buf_b` pair with barriers around
the k-loop, exactly as we do — neither double-buffers. Getting past 35% would
mean a different kernel shape (warp-specialised producer/consumer staging, or a
two-stage shared buffer overlapping the load of tile k+1 with the compute of k),
which is a project, not a tweak, and unproven on this hardware. Recorded here so
that the tiling sweep is not run a third time.

**It is not ILP either, and that is worth stating because it is the obvious next
guess.** A kernel with `NC` independent accumulator chains per lane and nothing
but `muladd`, swept `NC = 1 … 16`, is at **107.3 TF/s for every value including
one**. A single dependent chain saturates the tensor cores, because the latency
is hidden by other *warps*, not by other chains. So "give each warp more
accumulators" — which is what the larger tilings do — cannot help, and measuring
it agrees: the 128x128 tiling has twice the accumulators and 255 registers,
1 workgroup an SM, and 25 TF/s.

The occupancy the tilings actually reach, from `pipeline_exec_stats`:

| tiling | regs | shared | warps/SM | TF/s |
|---|---|---|---|---|
| 64x128 | 118 | 14 848 | **16** | **38.03** |
| 64x64 | 96 | 9 728 | 20 | 36.62 |
| 32x128 | 59 | 12 800 | 32 | 29.12 |
| 128x128 | 255 | 25 344 | 8 | 25.14 |

Not monotone in either direction: 32 warps is *worse* than 16 because the tile is
smaller and the traffic higher. 64x128 is the optimum of that trade, registers
are the binding constraint (14.8 KB of shared would allow six workgroups; 118
registers allow two), and there is no tiling that improves both sides at once.

#### The one thing that did come out of it: Int32 staging addresses, -1.2%

Julia hands out `Int64` indices and Lava emits them as-is; NVIDIA has no 64-bit
integer unit. The same narrowing was worth 1.56x in `im2col_kernel!`, so the
staging addresses were narrowed here too — `GEMM_NARROW`, a second `vec2` kernel
differing from the first only in the width of its address arithmetic.

**It is worth -1.2%, not 1.5x**, weighted over SAM 2's eight `addmm` shapes by
call count, from -4.9% to +0.9%, results bit-identical. The register count went
*up* (118 → 124) and the occupancy did not move, so what it buys is instruction
count in the staging loop and nothing else — the driver was evidently already
narrowing most of this itself.

The number is also a lesson about the measurement. Compared against the *previous
session's* baseline the same change read as **3.5–9%**; interleaved in one
session, both orders, it is 1.2%. A different process on a differently-warmed
card inflated it fourfold. Encode 101.0 → **100.4 ms, 87.3% of PyTorch**.

Both kernels stay, and not only to keep the A/B available: the narrow one is
legal only while `M*K`, `K*N` and `M*N` fit an `Int32` (`gemm_fits32`, strict
because the indices are 1-based), and the wide one is the fallback above that.
The largest this repo runs is 18.9M against a 2.1e9 limit, but the failure would
be a silently wrong answer rather than an error, so it is checked.

### Held `O` loses on occupancy, and both recorded reasons were wrong — 2026-08-02

Attention is the whole remaining encode gap, and the biggest single idea for it
is holding each subgroup's `O` tiles in cooperative-matrix accumulators for the
key loop instead of loading and storing them through shared memory every block.
The prize is not in doubt — a variant that never rescales measures

    4096x4096   4.393 -> 3.025 ms   +31.1%   (8.80 -> 12.78 TFLOP/s)
     256x256    0.425 -> 0.334      +21.4%

and the version that actually rescales loses 13.2% / 19.6%. Two explanations of
that gap were on file. Both were tested this session by removing the named cause,
and **neither moved the number**:

| explanation | what was done | before | after |
|---|---|---|---|
| the flush, reload and two barriers a growing block costs | `Lava.coopmat_setcomp` (built for the GEMM's gelu epilogue) lets the tile be scaled where it lives — no flush, no reload, no barrier | 5.183 ms | 4.950 |
| `getcomp`/`setcomp` each spill the whole tile, so eight components spill eight times | the emitter now stores once for a chain of accesses | 4.955 | 4.950 |

The instrument that settled it is the one that settled the flash kernel's
occupancy a week ago — `Lava.pipeline_exec_stats`:

    shared O   123 registers   48 904 B shared   stack 0   local 16
    held O     192 registers   48 904 B shared   stack 0   local 16

At 256 threads, 192 registers is 49 152 of the SM's 65 536: **one workgroup per
SM against two.** Nothing spills — the component access materialises a second
copy of each held tile — and **register allocation is static**, so it does not
matter that the rescale is skipped on the blocks where no row's maximum grew.

That last point retires a whole line of work. The recorded next move was a
per-row-tile `grew` flag, on the reasoning that `grew` fires on 67.3% of blocks
at `BR = 64` and only 31.7% at `BR = 16`. Making an expensive path rarer does not
recover an occupancy an unrarer path already spent.

Swept over every admissible tiling, held `O` wins only where the tiling is itself
bad — `32x16x8` by 2.8% and `32x32x4` by 11.9%, against baselines 38% and 150%
worse than `64x32x8`'s 4.386 ms. And `REGO`, the reference implementation's shape
(`O` in ordinary per-thread registers, which is what `flash_attn_cm1.comp` does),
costs *fewer* registers than shared `O` — 119 against 123 — and is 44% slower, so
whatever that one pays for is not occupancy either and remains unexplained.

**The one route left to the 31%** was a rescale that costs nothing in
*registers*, and the only such rescale is the one that never runs: fix each row's
reference once so the running maximum never grows. `p` is fp16, so a reference
works exactly while it stays within ~11.09 nats of the row's true maximum — in
either direction, since scaling a row of `P` by a constant cancels in `O/l`.

**Measured, and it does not.** The encoder was run unaliased (no slab, so every
intermediate survives) to get real `q`/`k` for both dominant shapes:

| reference | nats from the true row max | rows outside fp16's window |
|---|---|---|
| `scale·‖q_r‖·max_k‖k_k‖`, L=4096 | median 7.16, p99 14.11, max 17.40 | **20.4%** |
| the same, L=256 | median 6.49, p99 11.94, max 13.00 | 4.7% |
| the first key block's own max, L=4096 | median 4.57, p99 19.98, max 24.23 | **21.1%** |
| the same, L=256 | median 2.57, p99 16.33, max 20.81 | 12.3% |

A fifth of the rows drift out of representable range either way, and a row that
does comes back as zeros. **The online rescale is load-bearing on this data, not
overhead**, so held `O` stays off until a per-row rescale appears that is free in
registers. Two hours and no kernel written, which is what the measurement was for.

One trap in it worth keeping: the same Cauchy-Schwarz bound on the `L = 64`
windows is *tight* — median 0.98 nats, max 5.56 — so measuring the small blocks
alone, which is the cheap thing to do, says yes to an idea the real shapes reject.

Two notes for whoever picks this up. `tools/attn_lab.jl`'s `fused()` was
comparing its "3-pass" and "coopmat" columns against **the same kernel** —
`flashcm_applicable` is gated on `FLASHCM[]`, not on `COOPMAT_MINL[]`, so both
rows took the fused path. The real four-way table, on `E=72`:

    shape          3-pass   staged   flash-cm   flash-scalar
    4096x4096      18.286    8.879      4.423      39.435
     256x256        1.109    0.813      0.444       2.673

And the mapping from an accumulator's components to its rows, which the
cooperative-matrix spec deliberately leaves undefined, is *discoverable*: load a
tile whose element `(r, c)` holds `r` and ask `coopmat_getcomp`. On this card
every lane's eight components lie in exactly two rows, `lane÷4` and `lane÷4 + 8`.
Doing that inside the kernel costs one load at launch and makes a per-row scale
of a held accumulator portable rather than a bet on a driver detail — it is how
the rescale above is written, and it is worth keeping for whatever needs it next.

### The decode, attributed in context at last — 2026-08-02

Every decode number on file came from the serialised per-op table or from before
the render unification, which is to say nobody knew where the time went. Here it
is, `OPDOUBLE` per family with the baseline re-measured **before and after each
one** so drift cancels, 80.6% attributed and 0.165 ms of drift over the run:

| family | ops | ms | share |
|---|---|---|---|
| **attention** | 7 | **1.456** | **35.9%** |
| `add.Tensor` | 30 | 0.457 | 11.2% |
| `addmm` | 47 | 0.374 | 9.2% |
| `convolution` | 2 | 0.140 | 3.4% |
| `native_layer_norm` | 9 | 0.135 | 3.3% |
| `_to_copy` | 33 | 0.133 | 3.3% |
| 17 more families | 52 | 0.578 | 14.2% |
| host + launch gaps | — | 0.787 | 19.4% |

4.06 ms unreplayed, 3.30 replayed, against PyTorch's 2.10.

#### It is one attention shape, and it is a parallelism problem

The decoder's seven attentions do **0.242 GFLOP in 1.456 ms — 0.17 TFLOP/s**,
against the encoder's 6.9. Timed one shape at a time:

    Lq=23   Lk=4096  E=16  H=8   x3   0.462 ms each   0.10 TF/s   <- 95% of it
    Lq=4096 Lk=23    E=16  H=8   x2   0.054           0.89
    Lq=23   Lk=23    E=32  H=8   x2   0.026

`Lq = 23` is the mask prompt's token count. One query block covers all 23 rows,
so the launch is `1 x H*B` = **8 workgroups on a 48-SM card**. The kernel is not
slow; there is almost nothing running. This is the cross-attention from the
prompt tokens to the 4096 image tokens, and it is 36% of the decode.

**Shipped mitigation: the tiling chooser now knows the grid size.**
`FLASHCM_TILINGS` is ordered fastest-first *for a launch that fills the device*,
and `flashcm_tiling` took its first fitting entry without ever knowing how many
workgroups would result — it did not receive `H` or `B`. It does now, and below
`FLASHCM_MINGRID = 48` (one per SM) it prefers the tiling with the larger grid,
keeping the table's order among equals. On the dominant shape that is `16x32x4`
instead of `32x32x8`: 8 workgroups → 16, **0.4492 → 0.3919 ms, −12.8%**.

Decode **3.897 → 3.745 ms (−3.9%)**, interleaved, both orders; a replayed click
3.29 → **3.13**. Masks unchanged. The rule is inert on every encoder shape — the
windowed blocks launch 512 workgroups, the global ones 512 — and that is checked
rather than assumed.

**The real fix is flash-decoding, and it is worth about eight times as much.**
Splitting the *key* axis across workgroups — each taking a slice of the 4096 keys
and producing a partial `(O, m, l)`, then one merge kernel combining them by the
usual log-sum-exp — turns 16 workgroups into 128 or more. The three calls cost
1.386 ms today; the K/V they read is 1 MB total, so the bandwidth floor is
microseconds and the achievable figure is a small fraction of that. **Worth
roughly 1.2 ms of a 3.3 ms decode**, which is the difference between 64% of
PyTorch and parity. It is a real kernel — partials, a merge, and a routing rule —
and it is the single largest remaining item in the project.

#### A trap that ate three attempts at the table above

The first three attributions were garbage: baselines of 18.2 and 23.4 ms against
a real 3.9, and `addmm` attributed 244% of the total. The cause was the
instrumentation. **The un-replayed decode grows the pool from 1181 MiB to a
2461 MiB plateau over its first ~175 calls, and the first ~25 calls after a trim
cost 13.4 ms against a 3.99 ms minimum** — a 3.4x penalty while the pool grows
back. I had been calling `trim_gpu_pool!` between samples to keep memory
comparable, so every sample paid the regrowth.

Once the pool is warm it is flat: 3.88-3.99 ms over 400 consecutive decodes.
So: **warm the allocator, then never touch it inside a measurement loop**, and
treat a baseline that moves between samples as a broken instrument rather than a
noisy one. It is also a real user-facing effect — a click arriving just after a
GC or a memory trim costs 13 ms instead of 3.9.

### The decode as a captured command buffer — 2026-08-02

40% of a packaged decode was never a graph op; it was the host rebuilding a
149-dispatch launch sequence that is identical on every click. `Lava.capture`
records it once, `replay!` re-submits it with one `vkQueueSubmit2`:

| path | p50 | vs PyTorch's 2.10 ms |
|---|---|---|
| `decode` with loose tensors (the parity row) | 3.91 ms | 54% |
| a click — `prompt` + `decode`, replayed | **3.30 ms** | **64%** |

Bit-exact, and steady in a way the recorded path is not: 200 samples at 2280 MHz
give p50 3.17, p99 3.27, max 3.29, **zero** iterations above 2× the median.

The API cost is two fields on `SAM2` and one rule: **a replay reads whatever
those bytes hold now**, so every address it recorded has to be the same address
next click. `prompt` therefore writes each click into one persistent
`(point, label)` pair instead of allocating one, and `CACHE_DECODER_INPUTS` —
switched on for this, worth 0.008 ms on its own — keeps the converted inputs
stable. The slab and the weights were already fixed.

**The capture survives `encode`.** The encoder overwrites its outputs in place,
so a new frame does not move anything the capture recorded and the sequence is
deliberately not rebuilt. That is checked rather than assumed, because the
failure is invisible: a replay over stale features returns a *plausible* mask,
which for a segmentation model is indistinguishable from a slightly different
click. `test_replay_decode.jl` encodes a second image and compares the replayed
masks against the recorded path bit for bit — and does the same for a second
click through the same buffers, which is the load-bearing case.

**It costs 4 MiB**, not the 128 the benchmark first reported. See below.

#### Two bugs, both found by trying to verify the above

**`encode` returns the same tuple every frame, so `===` could never invalidate
anything.** `decode` caches the dtype-converted decoder inputs keyed on
`s.cachekey[] === feats`, with a comment saying a fresh `encode` invalidates them
by identity. It does not: `call` writes the encoder's outputs into the same
statically planned slab buffers, so the tuple `decode` gets back is `===` to the
one it converted from and holds different numbers. Latent for one reason only —
the autocast decoder shipped in #28 declares the dtypes the encoder already
produces, so nothing is converted and the "cache" holds the live features
themselves. Point the same code at the fp32 decoder and frame N decodes frame
N−1's features. `encode` now clears the key, and the conversion writes **into**
the previous destination rather than allocating, so fixing it does not move the
addresses a capture recorded.

**The benchmark measured its rows on an unsettled pool.** `decode` p50 8.76 ms
against a min of 3.85, on three runs of seven, landing on whichever loop happened
to go first — the same shape as a real regression, and it appeared in the run
right after the click row was added. In-session the same call is p50 3.86 / p99
5.07 over 200 samples. The cause is the parity section's temporaries still being
resident when the first timing loop starts, so that loop pays the pool growth.
One `settledlive()` before the cost section; three consecutive runs then agree to
0.1 ms on every row. **Rows meant to be compared with each other have to start
from the same allocator state.**

That also explains the 128 MiB: a held capture pins the transients of the decode
it recorded, and the benchmark is the only thing anywhere that runs a captured
*and* an uncaptured decode in one process — two live sets, 1225 MiB against 1161,
describing a configuration nothing outside that file has. Releasing the capture
afterwards gives back only the empty blocks (1225), so the number is now taken
before the click loop. Over the editor's actual pattern — encode a frame, click
it, move on, four rounds — a held capture costs **4 MiB: 1161 → 1165**.

### The flush hang: found, fixed, and one recurrence after — 2026-08-01

**Cause.** `vk_free!` defers destruction while a batch is open, but that check sat
inside `if last_write !== nothing`, and `last_write` is only written by
`sync_access!` **at submit**. A buffer the *currently recording* batch referenced,
which had never been submitted, read as `nothing` and was destroyed with an open
command buffer still naming it. The pinned case was already covered via
`buf.pins`; this was the unpinned rest. Lava `7ffe8af`, test `b20f850`.

**How it was pinned down.** The trigger is a Julia GC landing inside a recording,
which is why it looked like nothing:

    GC confined to safe points (between decodes)   60/60 clean
    GC live at any time                            hung within 15

Same session, same workload, same probes — only the collector's timing. Note the
first row is not "no GC": collections still ran, at a synchronize.

**Three things that looked like the cause and were not**, each recorded so nobody
spends the afternoon again: the pool trim (disabling it entirely does not help —
it mattered only because it calls `GC.gc()`), `with_dispatch_timing`, and
`LAUNCH_PROBE`. The last two only allocate, and so pull the collector into a
recording. One capture's *stack trace* pointed straight at the trim, and forcing
the trim reproduced on trial 1; both were still wrong.

**Not certainly closed.** After the fix: ~90 clean trials across every
reproduction that used to fail in ten or fewer — 60 probe-decodes with the
collector live, 8 encode+decodes with the trim forced, 6 timing runs in a plain
process, 8 rounds of the exact mixed sequence, 10 timing runs interactively — and
**one** recurrence under `with_dispatch_timing`. The dominant path is closed and
the residual rate is low; that is not the same as fixed, and the code says so.

Since `push!(bq.in_flight, batch)` is the only insertion and it happens after a
successful submit, the four stuck batches are genuinely submitted work the GPU
does not complete — which is what a use-after-free looks like. `FREED_BDA_SCAN_ENABLED`
already scans for a destroyed buffer's address still sitting in a live arg slab;
it is off by default and is the next instrument to reach for.

**Method, which cost two wrong conclusions:** after a hang the batch queue is
wedged and *everything* afterwards hangs, so isolation runs inside an
already-hung session are worthless. Every variant needs a fresh process.

#### What it looked like before any of that was known

Recorded as "twice, not reproducible". It fired **six times** in one day, all in
decode loops: at 86 in-flight batches, at 42, and four times at **4**.

**It was not any of that session's changes**, and the A/B that established it was
sound as far as it went — 60 synchronised decodes under each of five
configurations, 300 clean. What it could not show is that the variable was never
in the kernels at all: five configurations differing only in kernel selection all
sample the same allocation pattern, so the collector's timing never moved.

It briefly looked deterministic: two independent sessions hung at timeline value
**2317** with live at **3 184 MiB**, both after exactly 300 decodes. A later run
of **450** decodes was completely clean, so that was coincidence of workload, not
a threshold. Runs of 200 and 450 have both been clean; six failures sit somewhere
inside roughly 1 500 decode calls today.

**The "dispatch-bound" reading below is wrong and was measured out on
2026-08-01** — see "Decode is not dispatch-bound". Steady-state per-op timing
puts 93% of decode in three families (attention 43%, convolution 26%, `addmm`
24%); the one-workgroup elementwise launches are ~2% between them. They are real
and they are in the log, they are just not where the time is.

The log tail from the 86-batch case is 123 854 dispatches ending in a run of
one-workgroup launches — `lava_broadcast_flat_mixed!` and
`lava_broadcast_flat!` at `groups=(1,1,1)`, `strided_gemm_kernel!` at `(4,1,1)`
and `(1,1,1)`. Those tiny dispatches were taken to be the reason decode runs at
**9.5-14% of PyTorch's speed** against encode's 44%: the decoder is 205 ops on a
64x64 embedding and appears to be dispatch-bound, not compute-bound.

### The pool never gave its blocks back, and that was the memory bug

`live_bytes` counts pool **blocks**, and the pool only ever grew: it reached the
transient high-water mark of whatever ran and held it for the life of the
process. That is why the same workload read 2 800 in one session and 2 096 in
another, why "the decoder needs 384 MiB" looked true, and why the model appeared
273 MiB over the ceiling.

**`maybe_trim_pool!` was defeated by its own collection.** Its comment says it:
"Blocks only become empty once the GC has run their sub-allocations' finalizers,
hence the collection before the scan" — and the collection it ran was
`GC.gc(false)`. Julia's incremental mode sweeps the young generation and leaves
older objects, and a pool chunk that has survived a few frames is exactly an
older object. So the scan found nothing, the trim never fired, and the only paths
that ever returned a block were an OOM retry and an explicit reset.

Fixed by escalating: try the cheap collection, and if it finds no empty block pay
for a full one — on its own longer timer (`POOL_TRIM_FULL_GC_INTERVAL`, 30 s), so
a render loop does not take a full GC every five seconds. Plus an exported
`trim_gpu_pool!()` for "I have finished a batch of work and want the memory
back", which is also what a measurement wants.

Immediately, on SAM 2.1 after 20 decodes: **4 blocks, 256 MiB** handed back,
2 224 -> 1 968 live, idempotent on a second call, and the model still decodes
correctly afterwards.

### The number, with dead capacity returned

`tools/memcheck.jl` now trims before sampling. Eleven repeats across two fresh
sessions, identical in both:

    repeat     encode MiB    enc+dec MiB
    1                1776           1968
    2                1968           1904
    3..n             1904           1904

    steady state             1 904 MiB
    ceiling                  1 951
    PyTorch                  1 756 reserved

**1 904 against a 1 951 ceiling**, so the stated memory goal is met — with three
caveats that belong next to the claim, because this figure has been wrong three
times today:

  * the margin is **47 MiB, 2.4%**, so anything that adds a pool block puts it
    back over;
  * the **first** encode+decode cycle reads 1 968, above the ceiling, and only
    the steady state is under it;
  * this compares our trimmed `live` against PyTorch's **reserved**, which
    includes its own caching allocator. They are not the same quantity, and the
    comparison flatters us by however much Torch is holding in reserve.

### Superseded: 2 224 MiB, spread 0

`tools/memcheck.jl`. Four repeats in one session and three in a fresh one, same
figures to the megabyte:

    repeat     encode MiB    enc+dec MiB
    1                1840           2224
    2                2224           2224
    3                2224           2224

    encoder alone            1 840 MiB
    encode + decode          2 224 MiB   spread 0, within and across sessions
    ceiling                  1 951       -> over by 273 MiB

Repeat 1 reads 1 840 for encode-only because the decoder has not run yet; after
it has, its allocation is retained and every later encode-only sample reads the
full 2 224. So **the decoder adds 384 MiB**, not the 870 or 960 estimated
earlier from partial warm-ups.

**Two things made it look noisy, and neither was the allocator.** `live` is only
meaningful after finalizers have run — a `GC.gc()` pair leaves 2 393 MiB where a
full `GC.gc(true)` reaches the settled value and holds it across six more, so
`memcheck.jl` loops until it stops moving. And the earlier 2 800 / 2 096 pair was
`bench_sam2.jl`, whose workload is 20 timed encodes, 50 timed decodes and two
host readbacks, sampled once — a different quantity measured once, not this one
measured twice.

**So the gap is 273 MiB, not 849, and it is measurable.** A change of 100 MiB
would now show. Three claims about this number were retracted in one afternoon
before it was measured this way; that was the cost of quoting a figure whose
repeatability had never been checked.

**Two numbers, and they are not interchangeable** — this is what made the
retracted claim possible. `Lava.gpu_memory_usage().live_bytes` is what the
allocator has handed out; `nvidia-smi` is what the driver took from the card,
which includes the pool's high-water mark and depends on allocation *order*. Two
runs of the same workload measured 1 940 and 4 849 MB reserved while their live
figures agreed within 10%. `bench_sam2.jl` now prints both, labelled, with the
ceiling checked against `live`.

**What this document got wrong separately.** It claimed the only change that
clears the bar is flash attention (−402 MB), with `COOPMAT_QCHUNK` as a fallback
reaching 2 033. Both figures were fine; the error was treating the *list* as
complete — the fp32 GEMM scratch was not on it, because nothing had asked why an
fp16 matmul needs an fp32 buffer the size of its output. There is no reason to
think the list is complete now either, and the decoder's 870 MB is the obvious
place to look next.

## The session's Lava fixes cost nothing — measured, not assumed

Three changes to the allocator, the submit path and the launch dispatch invite the
question. `bench_sam2.jl` after them read **268.29** then **269.72 ms** p50 —
agreeing to 0.5%, so not noise — against the **256.0** recorded at the top of this
document, which looks like a 5% regression.

It is not. Same-session A/B, `git stash` the four modified Lava files, rebuild,
re-measure, restore:

    without the session's fixes    269.93 ms p50   (min 267.59)
    with them, run 1               268.29          (min 266.96)
    with them, run 2               269.72          (min 267.76)

The baseline is marginally *slower*. So **the 256.0 in the table above is
session-dependent**, not a number this machine reproduces on demand — cross-session
variance here is documented at ~13% and this is 5%. Compare within a session or
not at all; that applies to the encode total exactly as much as it applies to the
kernel microbenchmarks, which is the part that was easy to forget.

Mask IoU is **0.98750 / 0.99941 / 0.95349** on every run, with and without —
digit-for-digit the recorded parity. Changes to an allocator and a submit path
that leave the numerics untouched are the ones you want.

## The editor suite passes end to end

2026-07-29, after the three Lava fixes below and four test corrections:
**40 testsets, 770 assertions, zero failures, `EXIT=0`** — and zero segfaults,
zero `device is lost`, zero `ConcurrencyViolationError`. `UI interactions` is
320/320 inside the full suite; `fuzz`, `matte track`, `restore effect and cache`,
`conform`, `letterbox` and all six `overlays` testsets ran for the first time in
this session's history, because every earlier attempt aborted before reaching
them.

Run it as `julia --project=. dev/VideoEditor/test/runtests.jl` with DISPLAY set,
**not** `Pkg.test` — that forces `--check-bounds=yes`, which changes floating
point association in the fixture `Object lock` builds and fails it, taking
everything after it down.

## The editor segfaulted, and the cause was the +44% auto-submit win

Running the suite past `Object lock` reached `interactions.jl` and **SIGSEGV'd**
inside `vkCmdPipelineBarrier` while SAM 2's weights uploaded. It reduced to a
30-line reproducer (open a `Player`, run the matte job) and was deterministic.

Root cause: `Vulkan.PIPELINE_STAGE_2_ALL_COMMANDS_BIT` is typed as the *sync1*
`PipelineStageFlag` despite the `_2_` in its name, while
`CommandBatch.wait_semaphores` is sync2-typed — so `sync_access!` threw a
`MethodError` on **every cross-queue wait**, and had since that code was written.

It never looked like a `MethodError` because `submit!` calls `sync_access!`
*after* `vkEndCommandBuffer`: the throw left the batch still `bq.active_batch`
with `recording == true` but an ENDED command buffer, the next
`ensure_active_batch!` handed it straight back, and the caller recorded into it.
That is UB, and NVIDIA takes it as a segfault. It stayed latent until
`AUTO_SUBMIT_THRESHOLD = 64` — the +44% overlap win in this plan — put a submit
in the middle of a recording. The crashing copy reported `disp=64` exactly, which
is what finally pointed at it; `AUTO_SUBMIT_THRESHOLD[] = 0` made it vanish.

Four hypotheses were ruled out by experiment first, each cheaply: cross-thread
use (added the missing owner assertion to `cmd_copy_buffer!` — it never fired),
the GL context (crashes with `GLMakie.closeall()` first), GC finalizers (crashes
with GC off), and an out-of-bounds copy from the pool size-class rewrite (traced
in-bounds). Guessing from the stack would not have got there.

**There was a second, independent cause underneath it**, which the new guard
named instead of crashing on: `VideoEditor`'s `@compile_workload` ended with
`registermatte!(prop)`. That writes a module global, and `prop` had already run,
so its `modelref` held a built MatAnyone model — device buffers belonging to the
*precompilation process's* `VkContext`, serialised straight into the package
image. Every session then started with a propagator driving a dead device. The
suite showed it as 5 failures + 1 error in `matte marking`, all downstream of one
`clip.mattetrack` that never got set. Fixed by registering a *fresh, lazy*
propagator from `__init__` and never from the workload; verified by walking the
registered closure — its `modelref` is now `nothing` at load, and
`SAM2Runner.DEFAULT_MODEL[]` / `VideoEditor.SAM2MODEL[]` were already clean.

Fixed in Lava: correctly-typed `STAGE2_ALL_COMMANDS`; `submit!` drops the batch
on any throw from `sync_access!` instead of leaving a live batch on an ended
command buffer; `cmd_copy_buffer!` asserts single-writer like its `flush!`;
`vk_context()` takes a lock — `init_vulkan!` has no idempotency guard, so two
threads racing the lazy singleton would each build a whole VkDevice; that is a
real hole but it is **not** what happened here (the device-creation log shows one
device in the process, the others being precompile-worker output), so treat the
lock as hardening, not as the fix; `sync_access!` refuses a wait
across contexts with a clear error instead of handing the driver a foreign
handle. Regression test: `dev/Lava/test/test_crossqueue_sync.jl`.

## A silent miscompile: static workgroup + a trailing unit extent

Found 2026-07-29 by `tools/permute_bench.jl`, which checks its own reference:
`permutedims!` on a `LavaArray` returns **wrong data** for SAM 2's rank-6 window
shapes while the ordinary broadcast of the same permutation is correct.

It is not a `permutedims!` bug. A kernel whose workgroup lives in its **type**,
launched on an ndrange whose **last extent is 1**, writes only part of its output
and reports nothing:

    N=6  sz=(576,16,16,4,4,1)  wg=(16,2,2,2,2,1)   written 0.500
    N=6  sz=(288,4,4,32,32,1)  wg=(16,2,2,2,2,1)   written 0.062
    N=6  sz=(64,4,4,8,4,1)     wg=(16,2,2,2,2,1)   written 0.500
    N=5  sz=(64,4,4,4,1)       wg=(16,2,2,2,1)     written 0.500
    N=6  sz=(64,4,4,4,4,2)     wg=(16,2,2,2,2,1)   written 1.000  (ok — last extent 2)
    N=7  sz=(64,4,4,4,4,1,1)   wg=(16,2,2,2,2,1,1) written 1.000  (ok — two trailing 1s)

The written fraction is not a clean function of the block counts (0.5 and 0.062
on shapes differing only in extents), so treat the *trigger* as established and
the *law* as not. **Rank 5 is affected too**, which matters because the existing
mitigation was written for rank ≥ 3.

`WORKGROUP_FALLBACK` does not catch it: `interior_unit_workgroup` tests the
workgroup for an **interior** unit extent, and `(16,2,2,2,2,1)` has none — its
unit extent is *trailing*, which the existing note calls harmless. That was true
for the fault it was written for and is not true for this one.

Reachable from `permutedims!` (which passes `staticgroup(size(dest))`) and from
any `launch!` with a multi-dimensional output — `DNNKernels.LAUNCH_FLAT` defaults
to `false`, so every one of them takes the static path. The encoder itself is
clear: `LAUNCH_PROBE` shows its ndranges are all rank ≤ 3, and its rank-6
permutations go through the *broadcast*, which uses a linear ndrange. So this is
latent rather than active — but it is silent, and it corrupts data.

## A data race: the pool free lists, from the finalizer thread

Third silent fault found 2026-07-29, and the one most likely to have been causing
the "transient" device losses above. Julia 1.12 names it outright:

    error in running finalizer: ConcurrencyViolationError("Vector has invalid
    state. Don't modify internal fields incorrectly, or resize without correct
    locks")
      _growend! → push! → return_to_pool!   (memory.jl)     finalizer thread
      ... interleaved with ...
      try_reuse_or_bump → pool_alloc                        allocating thread

`return_to_pool!` ends in `push!(POOL_FREE_LISTS[idx], buf)` — a plain `Vector`
that `pool_alloc` pops from — and `vk_free!` runs from finalizers. `vk_free!`
*does* already hand buffers to the owning thread under `deferred_frees_lock`, but
only when the GPU still has work in flight; an idle buffer falls through and
destroys inline, and a buffer with `last_write === nothing` (allocated, never
written, dropped) does not enter that branch at all.

Fixed by deferring on **thread** as well as on GPU state: if `vk_free!` is not on
`bq.owning_thread`, the buffer goes on the same locked deferred list and
`drain_deferred_frees!` destroys it on the owner. `return_to_pool!` is then
single-threaded and the hot allocation path stays lock-free.

**Unproven but worth holding:** a corrupted free list hands the same chunk out
twice, and two live arrays sharing memory is exactly how a device ends up lost.
If `device is lost` stops appearing in suite logs from here on, that was it.

## Loose ends unrelated to speed

* **The suite loses the Vulkan device transiently; re-run before debugging.**
  Two full runs of identical code on 2026-07-29: one clean at UI interactions
  320/320 in 2m31, the next with three `LavaError during pool_alloc: Vulkan
  device is lost` immediately after device init and 315/4/1 in 5m02. The editor
  behaves correctly under it — logs "matte preview failed" and carries on — so it
  presents as matte assertions timing out with `clip.mattetrack === nothing`,
  which looks exactly like a slow model. `grep -c "device is lost"` on the log
  first. The card is shared with a VS Code REPL, so this is not necessarily ours.

* **Registering the propagator at load makes a CPU API call GPU-bound.** Moving
  `registermatte!` out of the precompile workload and into `__init__` (necessary —
  the workload was baking a dead `VkContext` into the image) has a consequence
  worth deciding on deliberately: `MATTEPROPAGATOR[]` is now non-`nothing` from
  the moment `VideoEditor` loads, so `analyzematte!` — a plain function taking a
  clip and a reader, with no `Player` in sight — silently becomes a GPU call. It
  then inherits the Lava context's thread affinity, and calling it from the main
  thread after a `Player` has claimed the context fails with "BatchQueue is
  single-writer". That is exactly how it surfaced, in a testset asserting the
  *disc fallback's* geometry.

  The test now pins `MATTEPROPAGATOR[] = nothing` for its duration, which is
  right for a test of fallback behaviour. The open product question is whether
  the global default should be the model at all, or whether registration belongs
  to a `Player` — the seam's own docstring says "the editor owns the track, the UI
  and the render path, and the propagator is whatever is installed", which reads
  more like per-editor than per-process.

* **Esc leaves the matte tool armed-but-inert, and nothing says so.** `Keyboard.escape`
  calls `cancelmattecollect!`, which runs `endmattecollect!` (listeners off,
  `:mattecollect` deleted, overlay hidden) and restores the previous track — but
  it never calls `deactivatetool!`, so `player.fxwidgets[:activetool]` still
  holds the matte tool and the toolbar button still reads active. Marking is off,
  preview clicks go back to crop/pick, and the only way to mark again is to
  toggle the tool off and on: `activatetool!` toggles, so pressing it *once* on
  an already-active tool switches it off rather than re-arming. Found by making
  `interactions.jl`'s matte sequence honest — it had continued marking after Esc
  against a detached collect, which can never work, and that assertion had never
  run because earlier failures aborted the testset first.

  **Enter has the same shape**: `finishmattecollect!` also never deactivates, so
  the "rebuild the cards" step after propagating turned the tool off too. Three
  separate places in one testset assumed `activatetool!` is an idempotent
  "activate/refresh"; it is a **toggle** (`activatetool!` deactivates when the
  named tool is already current), and both matte exits leave the tool current.

  **And it costs the user their mark cards.** `deactivatetool!` runs
  `cleartoolcards!`, but `rebuild()` — the only thing that puts cards back — fires
  only on `TOOLSVERSION`, the fold button, or activating a *folded* tool. So
  toggling the matte tool off after marking empties the card list with nothing to
  restore it until some unrelated event bumps `TOOLSVERSION`. The marks are still
  there (`player.mattemarks`); only their rows vanish, which is exactly the
  "panel is a list of the things you made" contract breaking silently. Propagation
  itself is fine — `runmatte!` calls `refreshmattepanel!` on completion, which is
  why the cards appear in the first place.

  Two defensible fixes and it is a product call, not a mechanical one: have the
  matte exits `deactivatetool!` as well (Esc means "out of marking", and a button
  that reads active while nothing responds is the state to avoid), or split the
  toggle from an idempotent `activatetool!` so "refresh this panel" stops meaning
  "turn it off". Not changed here — the tests now toggle off-then-on, which is
  what a user does, and which keeps the assertions honest about the real API.


* **`Object lock` fails inside the VideoEditor suite and passes standalone.**
  `T[1,3] = 136.27` against `133.5 ± 2.0` under `Pkg.test`; run on its own the
  same test gives `134.25` (err +0.75) and is bit-identical across repeats. It
  is not state from earlier testsets — that was the first guess and it is wrong;
  three `analyzemotion!` passes beforehand do not move it.

  Because `Pkg.test` aborts on the first failing top-level `@testset`, it takes
  ~133 UI tests plus the matte, conform and overlay sets down with it. Those can
  still be run — `julia --project=. dev/VideoEditor/test/runtests.jl` uses the
  default `--check-bounds`, where object lock passes.

  **Root-caused: `Pkg.test` forces `--check-bounds=yes`.** That changes
  floating-point association inside `GPUFiltering.smooth!`, which builds the
  test's synthetic video, so the fixture itself differs — a handful of the
  57 600 pixels land on the other side of an `N0f8` rounding boundary and x264
  emits a different bitstream. Reproduced exactly from a standalone run with the
  flag set:

  | | default | `--check-bounds=yes` | suite |
  |---|---|---|---|
  | `sum(bg)` | 28816.68359375 | 28816.69140625 | 28816.69140625 |
  | encoded bytes | 17324 | 17321 | 17321 |
  | tracked `dx` | 134.25 (pass) | 136.27 (fail) | 136.27 (fail) |

  So nothing in this session causes it, and the tracker is not wrong — a 2.7e-7
  relative change in the *input video* moves the NCC argmax 2 px, because a
  σ=1.5 smoothed random blob has a broad correlation peak. On textured footage
  the same code holds 0.15 px.

  Ruled out along the way: the GPU→CPU decode fallback in `graysource` (its
  `@warn` never appears in the log), randomness in the fit (`GPUFiltering` calls
  `rand` nowhere), and prior stabilization work (three `analyzemotion!` passes
  first do not move it).

  Not fixed, because the fix is a judgement call about the test's intent:
  quantising the fixture to 8 bits does *not* work (a 1-ULP difference still
  straddles the `round` boundary), so it needs either a wider tolerance, or a
  sharper fixture — and the comment above the seed says this exact draw is a
  regression guard for a phantom-rotation bug, so changing it is not mine to
  decide.

* Pre-existing suite failures from upstream API drift, untouched: `test_disk_cache`
  (`GPUCompiler.disk_cache_path` is gone), `test_graphics_pipeline` (VulkanCore
  changed `VkRenderingAttachmentInfo`'s signature), one GPUArrays
  broadcasting-with-tuple case that fails inside Base's shape logic.
* `COOPMAT_QCHUNK` sweep and the `POOL_SOFT_CAP` sweep are both recorded in the
  docstrings of the constants they set, not here.

## Rule 0 — the driver is not the suspect

Lava is a few months old and was written fast. The NVIDIA Vulkan driver is years
old and ships to millions of machines. When a kernel here misbehaves the prior is
overwhelmingly that **the bug is ours**. "Driver bug" is a conclusion that needs
a mountain of evidence, never a working hypothesis.

The section immediately below is why this is stated as a rule rather than left to
judgement. A workgroup cap sat in the codebase for months as a hardware defect,
with `spirv-val` passing, the right `LocalSize` in the dump, and identical
driver-reported register counts as supporting evidence. All of it true; the
conclusion wrong; the cause one line of ours.

**So: anything currently labelled a driver bug is a suspect, not a finding.**
Before that word is used, in order — `spirv-val --target-env vulkan1.3`; GPU
assisted validation (`Lava.enable_gpu_av`); a hunt for undefined behaviour in our
own output (out-of-range access, uninitialised reads, missing `NonPrivatePointer`
or memory semantics, signed/unsigned comparisons LLVM canonicalised under
`nuw`/`nsw`); and then the strongest instrument, which needs no second machine:
**write the same kernel in GLSL, compile it with `glslangValidator`, run that
module through the same dispatch.** If glslang's module is correct and ours is
not, the bug is ours and the disassembly diff localises it.

And the corollary that inverts an easy assumption: when our module behaves
differently on another vendor or another NVIDIA chip, that difference is evidence
about **our** code — that we depend on something unspecified — not evidence
against the driver.

## 2026-08-02 — the 256-thread workgroup cap was a pipeline-cache hash collision

`Lava.WORKGROUP_LIMIT` sat at 256 for months, on the recorded diagnosis that
"above 256 this driver silently runs fewer invocations than the shader declares".
A workgroup of 512 wrote half its output, 1024 wrote a quarter, nothing errored.
The device was never involved.

`get_compute_pipeline` keyed its cache on

    hash((spirv_bytes, entry_name, push_constant_size, needs_tlas_descriptor))

and **`Base.hash` on a large `Vector` samples elements rather than reading all of
them**. The 256- and 512-wide modules of one kernel differ at *exactly one byte*
— index 230, the `LocalSize` x operand — and collide. The 512 launch therefore
looked up the 256 pipeline and dispatched a 256-thread shader over a grid
computed for 512: exactly `256/wg` of the output, silently.

That single fact explains every property that made the old diagnosis look
airtight, and each of which had been recorded as a separate mystery:

* `0.5` at 512 and `0.25` at 1024 — it is `256/wg`, not a lane cap.
* "order dependence" — whichever size compiled first owned the cache slot.
* "body dependence" (64 live values failed, 32 and 128 did not) — whether that
  body's two modules happened to collide.
* "adding an unrelated store fixes it" — different bytes, no collision.
* `spirv-val` passing, the dump showing `LocalSize 512`, identical driver-reported
  Register Count for a failing and a working body — all true, all irrelevant,
  because **the dumped module is not the one that ran**.

What settled it, in order: a raw-builtin kernel (`lava_local_invocation_index()`,
unconditional store, atomic tally) showed every lane running at 1024; then the
correct and the incorrect 512-wide modules dumped **byte-identical** (md5
`8283c9fc5102`), which rules out codegen and leaves only the lookup. A trap along
the way — the dump filenames used the same sampling hash, so the two modules
overwrote each other and the first comparison was a file against itself.

Fixed with `Lava.spirv_content_hash` (FNV-1a over every byte, plus the length),
used by the pipeline cache and the dump filenames. `WORKGROUP_LIMIT` is now 1024,
the device's `maxComputeWorkGroupInvocations`; the static-vs-dynamic launch
distinction that briefly looked meaningful was an artifact of the same collision
and is gone. `test/test_workgroup_limit.jl` (51 asserts) pins full coverage at
every size on both launch spellings and asserts directly that one-byte-different
modules get different keys.

**Scope, measured rather than assumed.** Any two SPIR-V modules differing only in
bytes the sampling hash skips shared a pipeline — in principle any two
instantiations of a kernel that differ in a literal. A census over 240 distinct
modules (the flash kernel across every admissible tiling x shared-O/held x two
rescale primitives) found **zero** collisions under the old key, so variants that
differ in a `Val` change far too much code to be at risk and no earlier A/B needs
re-running. It bit exactly one shape: the workgroup-size instantiations of a
single kernel.

## 2026-08-02 — the encode map, re-measured with controls, and where the last 3.5 ms is

The 90% target is an encode of `87.64 / 0.9 = 97.4 ms`. The encode is **100.2 ms**,
so **3.5 ms** has to go. This re-measures where it could come from, and the first
answer was wrong in a way worth writing down.

### The ablation's noise floor is ~0.8 ms, and it will hand you a finding anyway

Doubling one family at a time reported `div` **1.72 ms**, `sub` **1.35** and
`upsample` **1.14** — three *single* ops, 4.2 ms together, more than the whole
remaining gap. It read as the find of the session, and the fusion to remove it was
about to be written.

Two controls killed it:

| control | true value | measured |
|---|---|---|
| a family name matching **no op** | exactly 0 | **−0.26 ms** |
| `sub.Tensor`, which is **provably lazy** — it is one of only two values in `fusableset(g)`, so it returns an unmaterialised `Broadcasted` | ~0 | **−0.47 ms** |
| three interleaved baselines | — | spread **0.78 ms** |

`sum(families) = 107% of the step` was the other tell, and it was read past.

**So: single-family ablation is sound at the top of the table and noise at the
bottom — which is exactly where a tempting new finding appears.** Treat anything
under ~2 ms as unresolved, and run both controls in the same session.

### What resolves: ablate a GROUP, so small costs sum above the floor

`opdouble = "*"` with `opdoublefilter` doubles an arbitrary set. The empty filter
is the control, and it lands where it should:

    elementwise tail   7 families, 393 ops    9.10 ms
    addmm + attention  2 families, 243 ops   75.59 ms
    CONTROL, filter passes nothing            0.08 ms   <- true 0

Within the tail, both of these are now above the floor and trustworthy:

    add.Tensor      98 ops   3.37 ms
    clone.default   90 ops   2.87 ms

### The next item: 51 of the 98 residual adds can be folded into the GEMM

Chasing each add's operands through the view chain (`Buffer.kind === :view`,
follow `.of` — `resolvealias` stops at the first hop and reports these as inputs):

    51   add.Tensor + addmm.default                    <- foldable
    42   add.Tensor + clone.default
     3   clone.default + max_pool2d_with_indices
     1   convolution + input
     1   convolution + upsample_nearest2d

The 51 are the transformer residual stream: `add_N = add_{N-1} + view(addmm_out)`.
A GEMM that **initialises its accumulator from `add_{N-1}` instead of zero** makes
the residual the tensor cores' own accumulate and the separate pass disappears —
the same trick `attn_flash_cm!` already uses for `O` ("starting from `O` itself
means the accumulate is the tensor core's own"). Note `mul!` currently requires
`iszero(β)` to take the coopmat path, so this needs a `C0` operand in Lava's
cooperative-matrix GEMM, not just a graph pass.

Worth roughly `51/98 x 3.37 = 1.75 ms` gross, less the residual tile read the GEMM
then pays — call it **~1.2 ms**, i.e. a third of the gap, for a change that spans
both repos. Size it against that before starting.
