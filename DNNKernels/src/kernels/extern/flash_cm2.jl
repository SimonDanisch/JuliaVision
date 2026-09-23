"""
Flash attention on **workgroup-scope** cooperative matrices — the coopmat2 form.

`flash.jl` holds two kernels; this is the third, and the difference is not the
algorithm but where the operands live. `attn_flash_cm!` owns a `BR x BC` block
per workgroup, stages `q`, `k`, `v` and the score tile through `@localmem`, and
gives each subgroup its own 16x16 tiles. This one has **no shared memory at
all**: one `Br x HSK` matrix per operand, spanning all `NT` invocations, loaded
straight out of global memory by a tensor layout, with the softmax done in the
matrix itself.

That is `flash_attn_cm2.comp`'s structure. Four `VK_NV_cooperative_matrix2`
features carry it, and Lava grew the last three for this kernel:

  * `tensor_addressing` — the loads, including K read through a **transposing
    view** rather than staged as a transposed copy
  * `workgroup_scope` — a `64 x 80` fp32 accumulator is 40 components a lane at
    128 invocations against 160 at subgroup scope, which is the whole point:
    `attn_flash_cm!` is gated by a step at 128 registers (see `FLASHCM_HELD`)
  * `reductions` — the row maximum and row sum, in the fragment, and the
    shape-changing smear that resizes a `Br x Bc` row value to `Br x EP`
  * `per_element_operations` — `exp`, the scale, and the padding mask

**Routed to, on the strength of a measurement**: -27% on SAM 2's global
attention blocks, -45% on the windowed ones, -33% at `L = 1024` and `2048`, all
against `attn_flash_cm!` and all correct against the three-pass path. The table
and the two rules that came out of it are in [`flashcm2_tiling`](@ref), including
the one shape where it LOSES — the decoder's `Lq = 23`, which it declines.

## Where the time goes, MEASURED BY ABLATION — and read the warning first

**Everything in this section was measured twice and the two runs disagree by
2.6x, because the first was taken on a miscompile.** Every per-element callback
and every reduce combiner was emitted `DontInline`, so the driver made a real
function call per element and per combine step. `:all` on SAM 2's global block,
`E72 L4096x4096 H8 B1`:

    variant       BEFORE (broken)         AFTER (callbacks inlined)
    :all          3.213 ms  12.03 TF/s    1.249 ms  30.96 TF/s
    :noreduce     1.988     -38.1%        1.157      -7.4%
    :noexp        2.780     -13.5%        1.126      -9.8%
    :norescale    2.502     -22.1%        1.175      -5.9%
    :nosoftmax    0.981     -69.5%        1.073     -14.1%

The broken column said the softmax was **70% of the kernel** and the reductions
2.8x dearer per operation than the per-element passes. The fixed column says the
softmax is **14%** and the two primitives cost about the same. Every structural
conclusion drawn from the first column — including the tiling rule, which had
been fitted to an inflated per-block cost — was an artifact. See the `FUSED`
note below for the bug, and `flashcm2_tiling` for what re-measuring did to the
rule.

At 30.96 TF/s this kernel is now **above PyTorch's flash attention** on the same
shape (24.9 TF/s), and the softmax being 14% means there is no large structural
win left inside it: the products are most of the remaining time.

Per key block it issues **eight per-element passes** (scale, mask, neg, nrelu,
expnrelu, neg again, exp, mask0) and **three reductions** (row max, row sum, and
the `Br x EP` smear). Half of those passes exist only because a per-element
callback saw scalars: `max(rowmax, Mold)` is three passes where one matrix
operand would do, and `exp(S - M)` is two.

**Cutting the pass count works, and `FUSED` is the default.**
`Lava.coopmat_perelement` was given a matrix operand (`OpCooperativeMatrixPer-
ElementOpNV` with a second matrix; exact to 0 ulp against the three-pass
identity, `mwe_perelement_matrix.jl`), and `Val{FUSED}` below uses it to write
the block as `M' = max(rowmax, M)`, `eM = exp(M - M')`, `P = exp(S - M')` —
**four passes and no matrix adds, against eight and three**.

**This measured +30% for a whole cycle, and that was a bug in Lava, not a
property of the hardware.** The emitter marks a callback `FuncControl.Inline`
only if `collect_inline_callbacks!` recognises the instruction that names it —
and both that function and `coopmat_perelement_marker` compared the intrinsic op
name to `"perelem"` exactly. The matrix-operand form is `"perelemm"`. So its
callback missed the whitelist, was emitted `DontInline`, and the driver made **a
real function call per element** — the exact 8.5x failure mode
`coopmat_perelement`'s own docstring warns about. Both now test
`startswith(name, "perelem")`.

**Pulling that thread found a second, larger instance of the same bug.** The set
was called `perelement_callbacks` and was filled only from per-element
instructions — so **no reduce combiner was ever in it**, and every
`OpCooperativeMatrixReduceNV` named a function the driver was told not to inline,
for every kernel in the tree. `coopmat_reduce`'s own docstring said "do not mark
it `@noinline` — a `DontInline` callback costs a real function call per element",
which is exactly what the emitter was doing to it. Fixing that alone took this
kernel from 2.406 to 1.223 ms. The set is now `inline_callbacks`, because the
name is what hid the omission: a function called `collect_perelement_callbacks!`
does not obviously owe anything to reductions.

The tell was in the driver statistics and I wrote it down before I understood
it: FUSED's binary was *larger* than the three-pass kernel's (207872 against
193664) when four passes replacing eight must produce less code. After the fix
it is 180608, and `both` is 165760 — the smallest of every variant.

**A technique that should win and does not is a bug until proven otherwise.** A
systematic miscompile reproduces across shapes, tilings and runs, so consistency
is no evidence that the numbers mean what they appear to.

**`OSUM` is the fix the ablation points at, and it is the shipped default.**
`EP` is `E` rounded up to the operand granularity, so at SAM 2's `E = 72` the
`Br x EP` accumulator has **eight columns the clamping store never writes**. Put
a column of ones in V's padding and column `E` of `P·V` accumulates
`sum_j P[i,j]` — and because the `O` rescale multiplies that column along with
the rest, what lands there is exactly the online recurrence `L = eM·L + rowsum`.
That deletes **a row reduction per key block** and the `Br x Bc` running total
with it.

There are two ways to get the ones there, and the cheaper one costs nothing at
all. `:pass` writes them with a per-element pass over `V`. `:fill` writes none:
`V` is loaded `Bc x EP` from a slab only `E` wide, so the driver is *already*
substituting a constant in columns `E:EP-1` — `tensor_setclampvalue` just asks
for one instead of zero ([`cm2slabones`](@ref)). `:off` keeps the reduction, for
a head dimension whose padding has no spare column.

From `tools/flash_cm2_variants.jl`, each row against the three-pass form **at its
own tiling**, so these are the value of the formulation and not of the tile.
Interleaved, spread(med/min) 1.01x, callbacks inlined:

    shape                    tiling    3pass (ms)   FUSED     :fill     BOTH
    E72 L4096x4096 H8 B1    64x32/128       1.529   -7.5%     -4.6%   -20.0%
    E72 L256x256  H8 B16    64x32/128       0.153  -11.6%     +6.1%   -19.5%
    E72 L23x4096  H8 B1     64x32/128       0.228   -2.2%     -2.6%   -18.2%

`fused = true, osum = :fill` is the default: BOTH wins at every tiling SAM 2
uses, and the two are close to additive because they remove different things —
passes and a reduction. Note `:fill` **alone** is now worth much less than it
was under the miscompile (-4.6% where it read -20.4%), and is even a small loss
on the windowed shape. That is the same correction as everywhere else here: a
reduction stopped being expensive once its combiner was inlined, so removing one
buys less. It stays on because BOTH beats FUSED alone on all three.

**`:fill` and `:pass` are within a point at the tilings SAM 2 uses**, and `:fill`
is ~14% ahead at two others (`64x64/128` on the long shape, `128x64/128` on the
windowed one; reproduced across runs). Registers are 255 for every variant of
every tiling — the ceiling — so that is not the mechanism. Shared memory is the
one statistic that moves and only at `64x32/128`, where `:pass` needs **33280
bytes against `:fill`'s 11264**: the per-element pass makes the driver stage `V`.
That is the tiling where the two are the SAME speed, and where `:fill` wins by
14% both report 48640. So the driver's statistics do not explain the 14% and
nothing here should be read as if they did.

`:fill` is the default on the strength of: never slower beyond noise, 14% faster
at two tilings, a third of the shared memory at the shipped one, and no
instructions at all.

`OSUM` also comes out MORE accurate — 1.31e-04 against 1.52e-04 — which is not
luck. The row sum now comes from the same fp16 `P` the numerator is built from,
where the reduce summed the fp32 `p` before the conversion, so numerator and
denominator stop disagreeing about what `P` was.

`EP > E` is not guaranteed: at `E = 64` the padding is empty and there is no
spare column, so `OSUM` falls back to `:off` in the kernel itself rather than at
the call site. The `E = 64` shape in `test_flash_cm2.jl` takes that path, and
explicit `:off` and `:pass` cases cover it at `E = 72` as well — `:pass` being
what `:fill` has to match to be believed, since a clamp value that silently did
nothing would look exactly like a kernel that never needed one.

The `O` rescale is the remaining 22%, and the lazy form cm1 uses (skip it on the
third of blocks where no row's maximum grew) is worth about 8% here — smaller,
and it needs one `getcomp` of a `RowAndColumn`-reduced matrix to get a
workgroup-uniform flag.

(Both A/Bs are interleaved. A first cut of the fused one timed each arm in its
own block, read a 2.305 ms baseline for a 0.235 ms configuration, and
"concluded" fusion won by 85% on one shape while losing on the other.)

## Register footprint, for reference

`pipeline_exec_stats` on the shipped configuration (`64x32/128`, `EP = 80`):

    kernel                  registers   shared
    cm2 :off (3 pass)          255       23 KB    <- the driver's CEILING
    cm2 FUSED                  255       32 KB
    cm2 :pass                  255       33 KB
    cm2 :fill                  255       11 KB
    cm1                        115       49 KB

**Every cm2 variant is pinned at 255**, so none of the 19.9% `OSUM` wins is an
occupancy win — it is fewer instructions on a latency-bound kernel. Do not read
the shared-memory column as a ranking either: at `64x64/128` all five variants
report 48640 bytes exactly, so the spread at `64x32` is the driver placing
spills differently and it does not track the timings (`:pass` uses three times
`:fill`'s and runs within a point of it).

So this kernel is **register-saturated and runs at about half cm1's occupancy**
— 255 x 128 invocations is 32640 of an Ada SM's 65536 registers, i.e. two
workgroups of 128 threads, against cm1's two of 256 — and it wins by 33% anyway.
That is headroom, not a wall: PyTorch's flash attention does the same 203 GFLOP
at **24.9 TFLOP/s** against our 12.0, and attention is still the single largest
item in the encode's remaining gap (+12.4 ms of 24).

Counting the live values at `BR=64, BC=32, EP=80, NT=128` accounts for it: the
persistent set is ~92 registers (`O` 40, `L` 16, `M` 16, `Q` 20) and the loop
adds `S`, `rowmax`, `d`, `eM`, `P`, `rowsum` at 16 each plus **`eMdiag` at 40** —
the `Br x EP` smear that exists only to rescale `O`. That one transient is the
largest single item after `O` itself. Rescaling `O` through a matrix-operand
per-element op would remove it, and that op now exists — but it is the one the
table above measures at +31%, so removing the smear that way would have to buy
back more than the op costs. It does not look like it does; the lazy rescale
(worth ~8%, below) keeps the smear and skips it instead.

**The tile sweep is exhausted** — `BR` in {32, 64, 128} x `BC` in {16, 32, 64,
128} x `NT` in {64, 128, 256} are all measured, and `64x32/128` is the best of
them on the global shape. `OSUM` halves the reductions per key block, which is
the kind of change that can move a tiling verdict, so the four candidates were
re-run under it: the ordering held, and `flashcm2_tiling`'s `Lk >= 512` switch
still names the winner at both of SAM 2's shapes (`64x32` at `L = 4096`, 2.582
ms; `64x64` at `L = 256`, 0.208 ms). `BC = 16` was tried specifically at the register ceiling, since six of the
eight live transients are `Br x Bc`, and it LOSES: -20.3% against cm1 where
`64x32` gets -27.1% (and -33.8% vs -36.1% on the windowed shape). Halving the
tile does not pay for doubling the number of key blocks, each of which costs two
reductions and four per-element passes regardless of its width.

**And the register arithmetic says a structural fix would not cross the step
either.** Occupancy at 128 invocations goes to three workgroups only at <= 170
registers (3 x 128 x R <= 65536). We are at 255. Removing `d` costs 16 and
`eMdiag` 40 — both together land at ~199, still short. So the next attempt here
should not be sold as an occupancy win; it would have to pay off as fewer
instructions on a latency-bound kernel, which is a different and less certain
argument. Note the reference materialises its `eMdiag` too, so there is no known
formulation that avoids it.

**In the model**, per-op device time over SAM 2's 48 encoder attentions:
**40.16 ms -> 24.03 ms, -40.2%** (`tools/sam2_attn_share.jl`) with `OSUM`, from
-33.2% without it. Whole encode **90.61 -> 82.65 ms p50**, which is **80.6% of
PyTorch's 66.63** where the cm1 path was 62%; the same run A/Bs the two kernels
at 96.39 vs 83.58 ms, -13.3%.

Mask IoU against PyTorch moved and is worth stating rather than rounding to
"unchanged": 0.988 -> 0.99375, 0.9997 -> 0.99980, **0.977 -> 0.95556**. The two
that moved cover 0.2% and 0.1% of the image, where a handful of boundary pixels
is worth points of IoU, and this kernel is elementwise MORE accurate than the
one it replaced (1.31e-04 against 1.52e-04) — so this looks like small numerical
differences reshuffling boundaries on tiny masks. That is an explanation, not a
proof; nothing here has shown it harmless.

**What is left, from `tools/sam2_kernel_table.jl`.** Deflating the serialised
total (117.1 ms) to the free-running one (82.65) and comparing per op:

    op            ours    torch     gap
    attention    17.07     8.17    +8.90
    addmm        39.82    35.49    +4.33     <- vs their addmm PLUS their gelu
    layer_norm    8.29     5.12    +3.17
    convolution   3.32     0.79    +2.53
    add           6.06     4.98    +1.08
    max_pool      1.70     1.04    +0.66
    clone         5.61    11.05    -5.44

+15.61 against a measured +16.02, so the accounting closes. **Two corrections
are baked into that table and both change what to work on next.**

**The `ours` column is deflated by 1.417.** `OPTIMES` serialises around every op,
which is the only way to attribute device time to a source-level op and also
inflates the whole encode from 82.65 to 117.1 ms. The raw table therefore reads
`addmm` at 28.5 TF/s against PyTorch's 54.1 — our serialised time against their
clean one, which is not a ratio of anything.

**PyTorch's `gelu` is our epilogue.** Our `gelu` row is 0 calls and 0.00 ms
because the 48 of them are fused into the GEMM's store, so their `addmm` bills
5.80 ms of work that ours already contains. Compared against `addmm + gelu`, the
GEMM is 1.12x behind, not 1.90x — and `tools/gemm_shape_table.jl` measures it
directly at **38.6 TF/s weighted**, against 44.6 for cuBLAS on this card and a
~42 TF/s practical fp16 roof. It is not the kernel running at half speed that
the first reading suggested.

So **attention is still the largest single item at +8.90 ms**, 2.1x behind even
after `OSUM`. And the ablation above already says where its time goes: the
reductions. `:norescale` is 22.1% of the kernel and the lazy rescale is the
next thing to try. Attention is about 30% of the encode, so the whole-encode
number has to move by roughly a third of that or one of the two measurements is
wrong — which is the check that they are measuring the same change. Note what
these tools have to do to get a number at all: `encode` runs a baked Mantle plan, so a
replay reaches neither `timeop!` nor a re-planning decision, and instrumenting it
returns an empty table while a caps flip between two `encode`s returns -0.1%.
Both of those read as "this kernel is worth nothing" and neither was measuring
it.

That caution was warranted rather than misplaced: the tensor-addressed GEMM won a
microbenchmark and lost on Whisper's real shapes, which is why the sweep here
covers five sequence lengths and the decoder rather than the two encoder shapes
that would have been enough to look convincing.

## Two departures from the reference, both forced

**`max(rowmax, Mold)` is `rowmax + relu(Mold - rowmax)`.** The reference gets an
elementwise maximum of two matrices from `coopMatPerElementNV(M, rowmax, Max,
Mold)` — a per-element op whose extra operand is another *matrix*, so the
callback sees that matrix's element. `Lava.coopmat_perelement` passes scalars and
pointers, not matrices, so the identity does it instead. **The anchor is not
free**: `Mold + relu(rowmax - Mold)` is the same thing in exact arithmetic and
wrong here, because on the first key block `Mold = -3e38` and `rowmax - Mold`
rounds to `3e38`, so the sum comes back 0 rather than `rowmax`. Anchored on
`rowmax` the first block is exact — `relu(-3e38)` contributes nothing — and
`eM = exp(Mold - M) = exp(-relu(rowmax - Mold))` falls out of the same difference.

**The scale multiplies `S`, not `Q`.** The reference scales `Q` while it is still
fp32, because ggml keeps `q` in fp32. Ours is fp16 in memory, so scaling there
would round the operand; scaling `S` is exact and costs one per-element pass over
`Br x Bc` against a `Br x Bc x EP` product.
"""

"""
    cm2granularity(dev, nt) -> (M, N, K) | nothing

The `(M, N, K)` multiples a **workgroup-scope** fp16 x fp16 -> fp32 matrix must
be at a workgroup of `nt` invocations, from the device's own table
(`M.DeviceCaps.wggran`). `nothing` means this device will not run a
workgroup-scope matrix at that workgroup size at all.

The lookup itself is `Mantle.wggranularity` — the table is the device's, so the
walk into it belongs beside the query that fills it rather than copied here.
This name stays because the kernels below read better for it.

Read, never assumed. The granularity *coarsens* as the workgroup grows — on an
RTX 4000 Ada, 16/16/16 at 32 and 64 invocations, 32/16/16 at 128, 32/32/16 at
256 — and that is what makes `NT = 128` the right default for `E = 72`: `EP` is
then 80 rather than 96, and the padding is 10% of both products instead of 33%.
Measured, every tiling was faster at 128 for exactly that reason.
"""
@inline cm2granularity(dev::M.DeviceCaps, nt::Int) = Mantle.wggranularity(dev, nt)

"""
    cm2pad(dev, E, nt) -> EP | nothing

`E` rounded up to what an operand extent may be at `nt` invocations. `EP` is both
the `K` of `Q·Kᵀ` and the `N` of `P·V`, so it takes the coarser of the two.
"""
@inline function cm2pad(dev::M.DeviceCaps, E::Int, nt::Int)
    g = cm2granularity(dev, nt)
    g === nothing && return nothing
    step = max(g[2], g[3])
    cld(E, step) * step
end

"""Whether a `(BR, BC, NT)` tiling is legal for this head dimension on this
device. Every extent appears in some product and has to be a multiple of that
product's granularity — `BR` as the M of `S` and `O`, `BC` as the N of `S` and
the K of `P`, `EP` as the K of `Q` and the N of `O`."""
@inline function flashcm2fits(dev::M.DeviceCaps, E::Int, BR::Int, BC::Int, NT::Int)
    NT <= dev.workgrouplimit || return false
    g = cm2granularity(dev, NT)
    g === nothing && return false
    Mg, Ng, Kg = g
    EP = cm2pad(dev, E, NT)
    BR % Mg == 0 && BC % Ng == 0 && BC % Kg == 0 && EP % Kg == 0 && EP % Ng == 0
end

# Per-element callbacks and reduce combiners. Top-level, not closures: the
# instruction names the function and there is no operand slot for an environment.
# None may be `@noinline` — `Lava.coopmat_perelement_thunk` already is, and a
# `DontInline` callback costs a real function call per element (8.5x, measured).
cm2neg(::UInt32, ::UInt32, e::Float32) = -e
cm2exp(::UInt32, ::UInt32, e::Float32) = exp(e)
cm2ninf(::UInt32, ::UInt32, e::Float32) = e - 3.0f38
cm2nrelu(::UInt32, ::UInt32, e::Float32) = max(-e, 0.0f0)
cm2expnrelu(::UInt32, ::UInt32, e::Float32) = exp(-max(e, 0.0f0))
cm2scale(::UInt32, ::UInt32, e::Float32, s::Float32) = e * s
cm2recip(::UInt32, ::UInt32, e::Float32) = e == 0.0f0 ? 0.0f0 : 1.0f0 / e
# The clamping load fills out-of-range elements with zero. Left alone they enter
# the row maximum as 0 and the row sum as `exp(-M)`, which is a wrong answer
# rather than a missing one. `-3e38` and not `-Inf`, so `Mold - M` stays finite.
cm2maskneg(r::UInt32, c::UInt32, e::Float32, nr::UInt32, nc::UInt32) =
    (r >= nr || c >= nc) ? -3.0f38 : e
cm2mask0(r::UInt32, c::UInt32, e::Float32, nr::UInt32, nc::UInt32) =
    (r >= nr || c >= nc) ? 0.0f0 : e

# ── The fused forms, which need a MATRIX operand (`Lava.coopmat_perelement`
# with a second matrix). Each replaces two or three passes of the formulation
# above — and measures +31% for it on both shapes, which is why `fused` is off.
# They are kept because the negative result is the interesting part: see the
# table in the docstring.
cm2max2(::UInt32, ::UInt32, a::Float32, b::Float32) = max(a, b)

"""How the row sum is obtained — see `OSUM` in [`attn_flash_cm2!`](@ref).
`:fill` and `:pass` write the same ones into V's padding and differ only in what
they cost; `:off` keeps the row reduction. Named here so the launcher can reject
anything else, since an unrecognised mode would write no ones at all and divide
by zero rather than fail."""
const OSUM_MODES = (:fill, :pass, :off)

"""The ablation variants. Each computes a WRONG answer on purpose; only the gaps
between them mean anything."""
const ABL_MODES = (:all, :noreduce, :noexp, :norescale, :nosoftmax)

# ── The row sum out of the product instead of a reduction. `cm2onescol` writes
# a column of ones into V's padding; `cm2pickcol` keeps one column of the
# accumulator and zeroes the rest, so a *sum* reduce smears it back across the
# row. See `OSUM` in the kernel.
cm2onescol(::UInt32, c::UInt32, e::Float16, ec::UInt32) = c == ec ? Float16(1) : e
cm2pickcol(::UInt32, c::UInt32, e::Float32, ec::UInt32) = c == ec ? e : 0.0f0
cm2expdiff(::UInt32, ::UInt32, a::Float32, b::Float32) = exp(a - b)
cm2scalemask(r::UInt32, c::UInt32, e::Float32, nr::UInt32, nc::UInt32, s::Float32) =
    (r >= nr || c >= nc) ? -3.0f38 : e * s

cm2max(x::Float32, y::Float32) = max(x, y)
cm2sum(x::Float32, y::Float32) = x + y
# The reference's `smearReduce`. Sound only on a matrix already uniform along the
# reduced axis — the combine order is unspecified — which every use here is.
cm2first(x::Float32, y::Float32) = x

"""
    cm2slab(nrow, ncol, srow, scol, r0, nr, nc) -> TensorLayout

A 2-D clamping layout over one `(E, L)` slab, sliced to the `nr x nc` block
starting at row `r0` **of the matrix**.

The tensor's dimensions are the REVERSE of Julia's — its last dimension is the
fastest-varying one — so a matrix row is the array's *column* index. `nrow` and
`srow` therefore describe the `L` axis and `ncol`/`scol` the `E` axis, which is
the mapping `mwe_tensor_gemm_nonsquare.jl` pinned: a logical `M x K` operand
lives in memory as `K x M`.

The strides are what let an operand be a view. Attention's `q`, `k` and `v` are a
permuted view of one packed `(E, L, H, B)` block, so "the next key" is
`stride(k, 2)` elements away and not `E`.
"""
@inline cm2slabbase(nrow::Int32, ncol::Int32, srow::Int32, scol::Int32) =
    Lava.tensor_setstride(
        Lava.tensor_setdim(
            Lava.tensor_layout(Val(2), Val(Lava.TENSOR_CLAMP_CONSTANT)),
            (nrow, ncol)),
        (srow, scol))

@inline cm2slab(nrow::Int32, ncol::Int32, srow::Int32, scol::Int32,
                r0::Int32, nr, nc) =
    Lava.tensor_slice(cm2slabbase(nrow, ncol, srow, scol),
                      (r0, Int32(0)), (Int32(nr), Int32(nc)))

"""
    cm2slabones(...) -> TensorLayout

[`cm2slab`](@ref) whose out-of-range fill is **one** instead of zero.

That is the whole of `OSUM`'s ones column: `V` is loaded `Bc x EP` from a slab
only `E` wide, so the driver already substitutes a constant in columns `E:EP-1`
— it just substitutes zero. Asking for one instead makes those columns of `P·V`
the row sum of `P`, at no instruction cost at all, where the per-element form
(`cm2onescol`) pays a pass over `V` per key block to write the same ones.

The fill is the component type's BITS, which is why it goes through
`Lava.tensor_clampbits` and not `Int32(1)` — the operand is a 32-bit integer for
a matrix of any type, so passing `1` would fill fp16 with 6.0e-8 and this kernel
would divide by a row sum 1.6e7 times too small, with nothing reporting an error.

Both readings are run side by side in `Lava/test/mwe_tensor_clampvalue.jl` and
asserted in `test_tensor_clamp.jl`, IN FP32 — the device arms there fill a
`Float32` matrix, and what they establish is that the operand is a bit pattern
rather than a number. The fp16 half is covered where it is actually used: `:fill`
has to agree with `:pass`, which writes real `Float16(1)` through a per-element
op, to 1e-5 in `test_flash_cm2.jl`.
"""
@inline cm2slabones(nrow::Int32, ncol::Int32, srow::Int32, scol::Int32,
                    r0::Int32, nr, nc) =
    Lava.tensor_slice(
        Lava.tensor_setclampvalue(cm2slabbase(nrow, ncol, srow, scol),
                                  Lava.tensor_clampbits(1.0f0, Float16)),
        (r0, Int32(0)), (Int32(nr), Int32(nc)))

"""
    attn_flash_cm2!(out, q, k, v, scale, …, Val(BR), Val(BC), Val(E), Val(EP))

One workgroup owns `BR` queries and walks the keys in blocks of `BC`, holding
`O`, `L` and `M` in workgroup-scope matrices for the whole loop.

Every operand arrives as a flat root array plus its base offset and four strides,
exactly as `attn_flash_cm!` takes them — attention's operands are wrappers, and
the strides go into the tensor layouts rather than into hand-written indexing.
"""
function attn_flash_cm2!(
        out, q, k, v, scale::Float32,
        qbase::Int32, qsE::Int32, qsL::Int32, qsH::Int32, qsB::Int32,
        kbase::Int32, ksE::Int32, ksL::Int32, ksH::Int32, ksB::Int32,
        vbase::Int32, vsE::Int32, vsL::Int32, vsH::Int32, vsB::Int32,
        obase::Int32, osE::Int32, osL::Int32, osH::Int32, osB::Int32,
        Lq::Int32, Lk::Int32,
        ::Val{BR}, ::Val{BC}, ::Val{E}, ::Val{EP},
        ::Val{ABL} = Val(:all), ::Val{FUSED} = Val(true),
        ::Val{OSUM} = Val(:fill)) where {BR,BC,E,EP,ABL,FUSED,OSUM}
    WM = Lava.WorkgroupMatrix
    grp = Tuple(KI.get_group_id())
    qb, h, b = grp[1], grp[2], grp[3]
    q0 = Int32((qb - 1) * BR)

    # Element offsets of this (head, batch) slab, then byte addresses.
    #
    # `out` gets the same treatment as the operands — base, strides, element size
    # — and that is not symmetry for its own sake. A dense fp32 buffer at
    # `size(out,3)` stride is what `sdpaout` allocates and NOT what a graph
    # passes in: SAM 2's planner hands attention an **fp16** slot, so
    # the kernel wrote fp32 bytes into it and the whole encoder came back NaN.
    # Neither the MWEs nor the A/B could see it — every one of them allocates its
    # own output.
    hb = Int32(h - 1)
    bb = Int32(b - 1)
    OT = eltype(out)
    obytes = Int32(sizeof(OT))
    qa = UInt64(pointer(q)) + UInt64(qbase - Int32(1) + hb * qsH + bb * qsB) * 2
    ka = UInt64(pointer(k)) + UInt64(kbase - Int32(1) + hb * ksH + bb * ksB) * 2
    va = UInt64(pointer(v)) + UInt64(vbase - Int32(1) + hb * vsH + bb * vsB) * 2
    oa = UInt64(pointer(out)) +
         UInt64(obase - Int32(1) + hb * osH + bb * osB) * UInt64(obytes)

    # Q for this block, once. It stays in registers across every key block.
    mq = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,BR,EP,Lava.MatrixA}), qa,
                          cm2slab(Lq, Int32(E), qsL, qsE, q0, BR, EP))

    # K's slab is `(E, Lk)` exactly as Q's is, so as the B operand it is the
    # wrong way round; the view reads it transposed in place. V wants
    # `(Bc, EP)` and is already right, so it takes the same layout unviewed.
    vt = Lava.tensor_view(Val(2), Val((1, 0)))

    # `OSUM` carries the row sum in the accumulator's PADDING instead of a
    # separate `Br x Bc` running total. `EP` is `E` rounded up to the operand
    # granularity, so at `E = 72, EP = 80` there are eight columns the clamping
    # store never writes; putting a column of ones into V makes column `E` of
    # `P·V` accumulate `sum_j P[i,j]` — with the `O` rescale already applied to
    # it, which is exactly the online recurrence `L = eM·L + rowsum`.
    #
    # It buys the removal of a ROW REDUCTION per key block, and the ablation
    # says reductions are what this kernel costs: the two row reduces are 38.1%
    # of it against the six per-element passes' 13.5%.
    #
    # Two ways to get the ones there, and they are not equally priced:
    #
    #   :fill  the clamping load substitutes ONE instead of zero outside V's
    #          slab (`cm2slabones`). Those columns are being substituted anyway,
    #          so the ones cost nothing.
    #   :pass  a per-element pass over `Bc x EP` per key block writes them
    #          (`cm2onescol`). Kept because it is the only route if a device ever
    #          lacks `OpTensorLayoutSetClampValueNV`, and because it is the
    #          control that says what `:fill` is worth.
    #   :off   the row reduction, as before.
    #
    # `EP > E` is not guaranteed — at `E = 64` the padding is empty — so this is
    # a compile-time branch and the reduce stays for devices and head dimensions
    # that have no spare column.
    osum = (EP > E) ? OSUM : :off
    acc  = Lava.coopmat_zero(WM{Float32,BR,EP,Lava.Accumulator})
    lrun = Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator})
    mrun = Lava.coopmat_perelement(cm2ninf,
               Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator}))
    nr = UInt32(max(Int32(0), min(Int32(BR), Lq - q0)))

    nkb = cld(Lk, Int32(BC))
    for j in Int32(0):(nkb - Int32(1))
        j0 = j * Int32(BC)
        lk = cm2slab(Lk, Int32(E), ksL, ksE, j0, BC, EP)
        mk = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,EP,BC,Lava.MatrixB}),
                              ka, lk, vt)
        s = Lava.coopmat_muladd(mq, mk,
                Lava.coopmat_zero(WM{Float32,BR,BC,Lava.Accumulator}))
        nc = UInt32(max(Int32(0), min(Int32(BC), Lk - j0)))
        # Scale and mask in ONE pass. Rows past `Lq` and columns past `Lk` were
        # filled with zero by the clamping load and must not reach the row
        # maximum; `-3e38` rather than `-Inf` so `Mold - M` stays finite.
        s = FUSED ? Lava.coopmat_perelement(cm2scalemask, s, nr, nc, scale) :
                    Lava.coopmat_perelement(cm2maskneg,
                        Lava.coopmat_perelement(cm2scale, s, scale), nr, nc)

        # ── The ablation, and it computes WRONG ANSWERS on purpose.
        #
        # `:all` is the kernel. `:norescale` keeps the softmax but drops the
        # `O` rescale — the `Br x EP` smear plus the multiply, i.e. the largest
        # live transient. `:nosoftmax` drops the whole online softmax and feeds
        # `S` straight to the second product. `:noreduce` and `:noexp` split
        # what is left between the two primitives: the former drops the two row
        # reductions and keeps every per-element pass, the latter the reverse.
        # Read the DIFFERENCES: each variant is a different amount of work, and
        # only the gaps mean anything.
        #
        # This exists because the tile sweep is exhausted and the register
        # ceiling (255) does not by itself say whether the kernel is occupancy-
        # or instruction-bound. If `:norescale` is worth little, the smear is not
        # the target however many registers it holds.
        p = s
        if ABL !== :nosoftmax
            # `:noreduce` skips the row reduction and uses `S` in its place —
            # same shape, same per-element chain downstream. `:all` minus
            # `:noreduce` is what the row reductions cost: BOTH of them at
            # `osum = :off`, and only this one otherwise, since the row
            # sum then comes out of the product.
            rowmax = ABL === :noreduce ? s :
                Lava.coopmat_reduce(cm2max, WM{Float32,BR,BC,Lava.Accumulator}, s,
                                    Val(Lava.CoopMatReduce.Row))
            if ABL === :noexp
                # The mirror: both reductions stay, the six per-element passes
                # and three adds of the online update go. `:all` minus this is
                # what the per-element machinery costs.
                #
                # `mrun` is assigned even though nothing reads it, so that its
                # live range across the key loop survives the ablation — drop it
                # and the variant also gets a register back, which would show up
                # as per-element cost that is really occupancy.
                mrun = rowmax
                em = rowmax
                p = s
            elseif FUSED
                # Three passes become one each, using a matrix as the extra
                # operand — `max(rowmax, Mold)` and both exponentials.
                #
                # `mask0` is GONE: a masked element is `-3e38`, so `exp(s - M)`
                # is already 0 for any finite `M`. The one case it does not cover
                # is a row that is masked in its entirety, where `M` is also
                # `-3e38` and every `p` comes out 1 — and those rows are the ones
                # past `Lq`, which the clamping store discards. Nothing NaNs:
                # `acc` for them is a finite sum of `V`.
                mold = mrun
                mrun = Lava.coopmat_perelement(cm2max2, rowmax, mrun)
                em = Lava.coopmat_perelement(cm2expdiff, mold, mrun)
                p = Lava.coopmat_perelement(cm2expdiff, s, mrun)
            else
                # d = rowmax - Mold ; M = rowmax + relu(-d) ; eM = exp(-relu(d))
                d = Lava.coopmat_add(rowmax, Lava.coopmat_perelement(cm2neg, mrun))
                mrun = Lava.coopmat_add(rowmax, Lava.coopmat_perelement(cm2nrelu, d))
                em = Lava.coopmat_perelement(cm2expnrelu, d)

                p = Lava.coopmat_perelement(cm2exp,
                        Lava.coopmat_add(s, Lava.coopmat_perelement(cm2neg, mrun)))
                p = Lava.coopmat_perelement(cm2mask0, p, nr, nc)
            end
            if osum === :off
                rowsum = ABL === :noreduce ? p :
                    Lava.coopmat_reduce(cm2sum, WM{Float32,BR,BC,Lava.Accumulator}, p,
                                        Val(Lava.CoopMatReduce.Row))
                lrun = Lava.coopmat_add(Lava.coopmat_mul(em, lrun), rowsum)
            end

            if ABL !== :norescale
                # `eM` resized from `Br x Bc` to `Br x EP` by a reduce that
                # reduces nothing — sound because `eM` is already uniform along
                # its row.
                #
                # …unless the two shapes are ALREADY the same. At `BC == EP` the
                # smear is the identity and its type is `eM`'s own, so the whole
                # reduce goes: a third reduction per key block, gone, for a
                # tiling choice rather than a code change. `flashcm2_tiling`
                # offers `BC = EP` for exactly this reason.
                emd = BC == EP ? em :
                    Lava.coopmat_reduce(cm2first, WM{Float32,BR,EP,Lava.Accumulator},
                                        em, Val(Lava.CoopMatReduce.Row))
                acc = Lava.coopmat_mul(acc, emd)
            end
        end

        pa = Lava.coopmat_convert(WM{Float16,BR,BC,Lava.MatrixA}, p)
        # `:fill` asks the DRIVER for the ones column: those columns are outside
        # V's slab and are already being substituted, so a fill of one costs
        # nothing at all where `:pass` costs a per-element pass over `Bc x EP`
        # per key block to write the same values.
        #
        # Rows of `V` past `Lk` are out of range too, so under `:fill` they
        # arrive as ONES rather than zeros — harmless because the matching
        # columns of `P` are masked to zero (`cm2mask0`, or an exact `exp(-3e38)`
        # on the fused path), which is what makes the ones column sum only the
        # keys that exist. `:off` does not care either way.
        vlay = osum === :fill ? cm2slabones(Lk, Int32(E), vsL, vsE, j0, BC, EP) :
                                cm2slab(Lk, Int32(E), vsL, vsE, j0, BC, EP)
        mv = Lava.tensor_load(Lava.coopmat_zero(WM{Float16,BC,EP,Lava.MatrixB}),
                              va, vlay)
        osum === :pass && (mv = Lava.coopmat_perelement(cm2onescol, mv, UInt32(E)))
        acc = Lava.coopmat_muladd(pa, mv, acc)
    end

    # Smearing the carried column back across the row: zero everything but
    # column `E`, then a *sum* reduce puts that one value in every column. A
    # `cm2first` reduce cannot do it — the value is not in column 0.
    ld = if osum === :off
        Lava.coopmat_reduce(cm2first, WM{Float32,BR,EP,Lava.Accumulator}, lrun,
                            Val(Lava.CoopMatReduce.Row))
    else
        Lava.coopmat_reduce(cm2sum, WM{Float32,BR,EP,Lava.Accumulator},
                            Lava.coopmat_perelement(cm2pickcol, acc, UInt32(E)),
                            Val(Lava.CoopMatReduce.Row))
    end
    acc = Lava.coopmat_mul(acc, Lava.coopmat_perelement(cm2recip, ld))
    # The clamping layout bounds-checks the store the same way it bounds-checks
    # the loads, so an edge tile writes only the rows inside `Lq` and the columns
    # inside `E`.
    olay = cm2slab(Lq, Int32(E), osL, osE, q0, BR, EP)
    # `OT` is a type, so this branch is resolved at compile time and only one
    # store is emitted. The destination decides the component type: a tensor
    # store writes the matrix's own bytes, so an fp32 accumulator into an fp16
    # slot is not a conversion, it is corruption.
    if OT === Float16
        Lava.tensor_store(Lava.coopmat_convert(WM{Float16,BR,EP,Lava.Accumulator}, acc),
                          oa, olay)
    else
        Lava.tensor_store(acc, oa, olay)
    end
    return nothing
end

"""
    FlashCM2Plan

Everything [`attn_flash_cm2!`](@ref) needs to launch, decided once — the same
shape as [`FlashCMPlan`](@ref) and for the same reason: a predicate and a chooser
that have to agree and cannot be made to is how a path silently changes
algorithm.

Shorter than `FlashCMPlan` by every switch that kernel accumulated. There is no
`clamp` (the clamping tensor loads and the masked softmax are unconditional and
free — the mask is one per-element pass either way), no `rego`/`held` (`O` is a
matrix, there is nowhere else to put it), no `epad`/`rpad` (nothing is staged
through shared memory, so there are no banks to pad against) and no `nsplit`
(see the grid rule in [`flashcm2_tiling`](@ref)).
"""
struct FlashCM2Plan
    BR::Int
    BC::Int
    NT::Int          # invocations per workgroup — part of the shape contract
    E::Int
    EP::Int          # E rounded up to this workgroup size's granularity
end

"""
    flashcm2_tiling(dev, E, Lq, Lk, nbatch) -> (BR, BC, NT) | nothing

The tiling to run, or `nothing` when this kernel should not take the call.

Measured 2026-08-11 against the shipped `attn_flash_cm!`, interleaved, warm
clock, correctness checked against the three-pass path first (ms, min of 10
medians; spread(med/min) 1.01-1.05):

    shape (E=72)        cm1      cm2 64x32/128   cm2 64x64/128   cm2 128x64/128
    4096x4096  H8 B1    4.433      3.231 -27%      3.889 -12%      3.640 -18%
    2048x2048  H8 B1    1.240      0.835 -33%      1.017 -18%          –
    1024x1024  H8 B1    0.442      0.295 -33%      0.355 -20%      0.300 -32%
     512x512   H8 B4    0.379      0.247 -35%      0.301 -21%          –
     256x256   H8 B16   0.422      0.267 -37%      0.231 -45%      0.258 -39%
      23x4096  H8 B1    0.288      0.529 +84%      0.557 +94%      1.028 +257%

Two rules come out of it, and neither is a curve fit.

**`BC = 32` everywhere, and no switch.** Every key block pays a fixed cost — two
tensor loads, two reductions, four per-element passes — which argues for a larger
`BC` on a short key axis. That argument is an artifact of the `DontInline`
miscompile above, which charged a real function call for every element of every
per-element pass and every step of every reduction: **make the fixed cost real
and the amortisation argument evaporates.** With the callbacks inlined, `both`,
interleaved, spread 1.01x:

    shape                  BC=16    BC=32    BC=64
    E72 L4096x4096 H8 B1   1.457    1.223    2.697
    E72 L256x256  H8 B16   0.135    0.123    0.160
    E72 L23x4096  H8 B1    0.227    0.186    0.353

`BC = 32` wins at `Lk = 256` by 23%, where `64` wins by 13% under the
miscompile, and taking `64` costs 30% on SAM 2's windowed blocks. A tuning
constant that survives the bug it was fitted to is a slow regression, not a
stale comment.

**Decline when the grid does not fill the device.** The last row is SAM 2's
decoder cross-attention, and it is not a tiling problem: `Lq = 23` is one query
block, so the launch is `1 x H x B = 8` workgroups on 48 SMs no matter how the
block is shaped. `attn_flash_cm!` wins there because it splits the KEY axis
across workgroups and merges the partial softmaxes — flash-decoding, which this
kernel does not have. Adding it is the obvious next step and is why the rule is
"the grid is too small" rather than "`Lq` is small": the day the split lands,
this rule is what changes.

Keys a workgroup takes per block: **16**, with `BR = 128` — see
[`FLASHCM2_BR`](@ref), which carries the measurement. The two move together and
neither is right alone: `128 x 32` is 34-49% SLOWER than `64 x 32` on SAM 2 while
`128 x 16` is 14-21% faster, so this is one tiling and not two constants.
"""
const FLASHCM2_BC = 16

"Workgroups per shader core below which this kernel gives the call back."
const FLASHCM2_MINGRID = 1

"""Invocations per workgroup to prefer when the padding rule does not decide.

Measured at BOTH head dimensions in the tree, and it is not a consequence of
padding: 128 beats 256 by **2.27x** at Whisper's `E = 64` where nothing pads at
either size, and beats 64 by 62% at SAM 2's `EP = 80`. See
[`flashcm2_workgroup`](@ref), whose tie-break is not "widest", which would pick
256 for Whisper."""
const FLASHCM2_NT = 128

"""Queries a workgroup owns: **128**, paired with `BC = 16`.

`128 x 16` and not `64 x 32`, which a sweep varying one of `BR`/`BC` while
holding the other at a power-of-two default never reaches. Interleaved, clock
reported per shape, reproduced across two runs and both orderings:

    shape                     64x32    128x16
    E72 L4096x4096 H8 B1      1.238     0.974   -21.4%   (repeat -21.0%)
    E72 L256x256  H8 B16      0.123     0.106   -13.8%   (repeat -15.4%)
    E64 L1500x1500 H20 B1     0.359     0.322   -10.3%   (Whisper)

`128 x 32` is NOT the same finding: it wins on Whisper (-15.9%) and loses badly
on SAM 2 (+34%, +49%). Only the narrow-key pairing wins everywhere, which is why
the two constants are documented as one choice.

**And the models agree, which is the only reason it shipped**: SAM 2 encode
74.3 -> **73.15 ms (91.1% of PyTorch)**, Whisper 86.13 -> **85.16**, with mask 3's
IoU going 0.935 -> **1.000** — faster and more accurate. Cf. the GEMM padding
constant, which won every microbenchmark shape and did nothing in the model."""
const FLASHCM2_BR = 128

"""
    flashcm2_workgroup(dev, E) -> nt | nothing

The workgroup size to launch: of the sizes this device reports, the one whose
granularity pads `E` LEAST, and the largest such when several tie.

This is the measured rule rather than the measured answer. `NT = 128` beat
`NT = 256` at every tiling on an RTX 4000 Ada — by 26% to 63% — and part of the
cause is that 256 invocations force `N` to a multiple of 32, so `E = 72` pads to
96 instead of 80 and a quarter of both products becomes padding.

**The tie-break is not "then the WIDEST", which is wrong by 2.27x.**
Padding explains the choice only where padding differs. At Whisper's `E = 64`
nothing pads at any workgroup size, so the tie-break decided alone, picked 256,
and produced the third-worst of the fourteen legal configurations:

    Whisper encoder attention, E=64 L=1500 H=20 B=1, interleaved
        128x32/128   0.282 ms   40.91 TF/s   <- best
         64x32/128   0.330      34.91
         64x32/256   0.640      18.01        <- what the rule chose
        128x32/64    0.988      11.66

So `NT = 128` is not a consequence of padding, it is the answer at both head
dimensions measured — 128 over 256 by 2.27x here, and 128 over 64 by 62% at
equal `EP = 80` on SAM 2. The tie-break now says so directly
([`FLASHCM2_NT`]) and falls back to the widest only when 128 is not on offer.
SAM 2 is untouched: at `E = 72` the padding rule already picked 128 before the
tie-break was consulted.
"""
function flashcm2_workgroup(dev::M.DeviceCaps, E::Int)
    best = nothing
    # Rank: fewest padded columns first, then FLASHCM2_NT, then widest. The
    # middle term is the one that matters — see the docstring: as "widest" alone
    # it chose 256 for Whisper and cost 2.27x.
    rank(nt, ep) = (ep, nt == FLASHCM2_NT ? 0 : 1, -nt)
    for (nt, _, _, _) in dev.wggran
        nt <= dev.workgrouplimit || continue
        ep = cm2pad(dev, E, nt)
        ep === nothing && continue
        if best === nothing || rank(nt, ep) < rank(best[1], best[2])
            best = (nt, ep)
        end
    end
    best === nothing ? nothing : best[1]
end

function flashcm2_tiling(dev::M.DeviceCaps, E::Int, Lq::Int, Lk::Int, nbatch::Int)
    NT = flashcm2_workgroup(dev, E)
    NT === nothing && return nothing
    BR = FLASHCM2_BR
    BC = FLASHCM2_BC
    flashcm2fits(dev, E, BR, BC, NT) || return nothing
    # No key-axis split, so the query axis has to fill the device on its own.
    dev.cores > 0 && cld(Lq, BR) * nbatch < FLASHCM2_MINGRID * dev.cores && return nothing
    (BR, BC, NT)
end

"""
    flashcm2_plan(dev, q, k, v, bias; BR = 0, BC = 0, NT = 0) -> FlashCM2Plan | Decline

Whether this call may take the coopmat2 flash kernel, and with what.

A question about the problem and the device only, so it takes a
[`M.DeviceCaps`](@ref) and is answerable for a device the caller does not have —
`dev.wggran` carries both "are there workgroup-scope matrices" and "at what
shapes", so a machine with none can still be asked what this would decide.

Passing a tiling explicitly selects it, and it is still validated: a tiling this
device's granularity forbids gets a `Decline`, not a pipeline-creation failure
three layers away.
"""
function flashcm2_plan(dev::M.DeviceCaps, q, k, v, bias;
                       BR::Int = 0, BC::Int = 0, NT::Int = 0)
    bias === nothing || return Decline(:bias)
    isempty(dev.wggran) && return Decline(:nowgscope)
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 ||
        return Decline(:eltype)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    size(v, 1) == E || return Decline(:headdim)

    # All three or none: a caller naming only `BR` would otherwise get `BC = 0`,
    # which `flashcm2fits` refuses as `:tiling` — a refusal that reads like the
    # device's rather than the call's.
    (BR == 0) == (BC == 0) == (NT == 0) ||
        throw(ArgumentError("name all of BR/BC/NT or none of them"))
    tiling = BR == 0 ? flashcm2_tiling(dev, E, Lq, Lk, H * B) : (BR, BC, NT)
    tiling === nothing && return Decline(:notiling)
    BR, BC, NT = tiling
    flashcm2fits(dev, E, BR, BC, NT) || return Decline(:tiling)

    # Every operand reaches the kernel as a root array plus strides, which the
    # tensor layouts consume directly — so a wrapper is fine and a stack
    # `stridedroot` cannot account for is not. Same recoverable refusal as the
    # cm1 path's, and `sdpa` recovers it the same way, by densifying.
    (stridedroot(q) === nothing || stridedroot(k) === nothing ||
     stridedroot(v) === nothing) && return Decline(:wrapped)

    # The head dimension has to be the contiguous one. A tensor layout's innermost
    # stride is 1 in both reference shaders and in everything tested here
    # (`mwe_tensor_stride.jl` strides the OUTER dimension); a non-unit innermost
    # stride is a shape nothing has verified, and the failure mode would be wrong
    # numbers rather than a refusal. The cm1 path stages by hand and does not care,
    # so this declines to it rather than guessing.
    (strides(q)[1] == 1 && strides(k)[1] == 1 && strides(v)[1] == 1) ||
        return Decline(:stridedE)

    FlashCM2Plan(BR, BC, NT, E, cm2pad(dev, E, NT))
end

"""
    sdpaflashcm2!(ctx, out, plan, q, k, v, scale) -> out

Launch [`attn_flash_cm2!`](@ref). Holding a plan means it runs: every refusal
happened in [`flashcm2_plan`](@ref).
"""
function sdpaflashcm2!(ctx, out, plan::FlashCM2Plan, q, k, v, scale;
                       abl::Symbol = :all, fused::Bool = true, osum::Symbol = :fill)
    # An unrecognised `osum` must not reach the kernel: there it would match
    # neither `:fill` nor `:pass`, so no ones column would be written and the
    # denominator would come out zero — a silently wrong answer from a typo.
    osum in OSUM_MODES ||
        throw(ArgumentError("osum must be one of $(OSUM_MODES); got :$osum"))
    abl in ABL_MODES ||
        throw(ArgumentError("abl must be one of $(ABL_MODES); got :$abl"))
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    rq, rk, rv = stridedroot(q), stridedroot(k), stridedroot(v)
    # The destination goes in as root + strides like the operands. A caller may
    # pass one — a graph always does — and it is neither guaranteed dense nor
    # guaranteed fp32.
    ro = stridedroot(out)
    ro === nothing &&
        throw(ArgumentError("attn_flash_cm2! needs a strided destination; got $(typeof(out))"))
    eltype(out) === Float32 || eltype(out) === Float16 ||
        throw(ArgumentError("attn_flash_cm2! writes Float32 or Float16, not $(eltype(out))"))
    strides(out)[1] == 1 ||
        throw(ArgumentError("attn_flash_cm2! needs a contiguous head dimension in `out`"))
    st(a) = map(Int32, strides(a))
    sq, sk, sv, so = st(q), st(k), st(v), st(out)
    flat(r) = reshape(r[1], length(r[1]))

    KI.Kernel(ctx.backend, attn_flash_cm2!)(
        flat(ro), flat(rq), flat(rk), flat(rv), Float32(scale),
        Int32(rq[2] + 1), sq[1], sq[2], sq[3], sq[4],
        Int32(rk[2] + 1), sk[1], sk[2], sk[3], sk[4],
        Int32(rv[2] + 1), sv[1], sv[2], sv[3], sv[4],
        Int32(ro[2] + 1), so[1], so[2], so[3], so[4],
        Int32(Lq), Int32(Lk),
        Val(plan.BR), Val(plan.BC), Val(plan.E), Val(plan.EP), Val(abl), Val(fused),
        Val(osum);
        ndrange = (plan.NT * cld(Lq, plan.BR), H, B), workgroupsize = plan.NT)
    out
end

"""
    sdpaflashcm2!(ctx, out, q, k, v, scale; BR, BC, NT) -> Bool

Plan and run in one call, for direct callers with no plan in hand — which is
every A/B in `tools/flash_cm2_ab.jl`. `false` means this device or these
keywords describe a launch it cannot make; ask [`flashcm2_plan`](@ref) to find
out which rule refused.
"""
function sdpaflashcm2!(ctx, out, q, k, v, scale; BR::Int = 0, BC::Int = 0, NT::Int = 0,
                       abl::Symbol = :all, fused::Bool = true, osum::Symbol = :fill)
    plan = flashcm2_plan(ctx.dev, q, k, v, nothing; BR, BC, NT)
    plan isa Decline && return false
    sdpaflashcm2!(ctx, out, plan, q, k, v, scale; abl, fused, osum)
    true
end

"""Whether this kernel can address a destination: a strided root, a component
type a cooperative matrix has, and the head dimension contiguous."""
cm2writable(o) = stridedroot(o) !== nothing &&
                 (eltype(o) === Float32 || eltype(o) === Float16) &&
                 strides(o)[1] == 1

# One `sdpa!` method per plan type — a new attention path is a new plan type and
# a new method, and nothing in `attention.jl` has to be edited to admit it.
function sdpa!(ctx, plan::FlashCM2Plan, out, q, k, v, bias, scale)
    Lq, H, B = size(q, 2), size(q, 3), size(q, 4)
    o = sdpaout(ctx, out, q, v, Lq, H, B)
    # The destination is not part of the plan — `sdpa` allocates it or the caller
    # supplies it — so it is checked here, and a slot this kernel cannot address
    # hands the call on instead of throwing. That is the same recovery
    # `sdpaplan` makes for a wrapped operand; a graph reserving an unusual slot
    # should get a slower answer, not an error.
    if !cm2writable(o)
        alt = flashcm_plan(ctx.dev, q, k, v, bias; clamp = ctx.clampattn)
        return sdpa!(ctx, alt isa Decline ? Decline(:threepass) : alt,
                     o, q, k, v, bias, scale)
    end
    sdpaflashcm2!(ctx, o, plan, q, k, v, scale)
end
