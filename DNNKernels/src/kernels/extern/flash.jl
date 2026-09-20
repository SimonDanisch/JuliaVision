"""
Fused attention: one kernel, no score matrix.

**Two kernels live here.** `attn_flash!` comes first and is the scalar form: kept
and still tested, never routed to. `attn_flash_cm!` follows it — both products on
the tensor cores, and **2.2x the two-GEMM path** on the encoder's global blocks
(9.50 -> 4.39 ms), 2.0x on the windowed ones (0.883 -> 0.450). Its switches,
tiling table and launcher are below it, each carrying the measurement that set it.

**A third kernel now takes most of the calls**, from `flash_cm2.jl`: workgroup-
scope cooperative matrices and tensor-addressed loads, measured -27% on the global
blocks and -45% on the windowed ones against this one. `attn_flash_cm!` is still
what runs wherever that declines — chiefly the decoder's `Lq = 23`, where its
key-axis split (`splitcount`) beats the newer kernel by 84%, and on any device
without `VK_NV_cooperative_matrix2`, which is every non-NVIDIA one.

Everything from here to `attn_flash!` is about the scalar form. Its conclusion —
"this shape cannot win here" — is correct **for that arrangement** and was read
for a while as a fact about flash attention on this device, which it is not: the
occupancy wall it hits comes from staging q, k and v as fp32 and from needing the
score tile in shared memory for a scalar reduction to reach it. Neither is true
once the products go through `OpCooperativeMatrixMulAddKHR`.

The three-pass form in `attention.jl` is correct and bandwidth-bound. It writes
`Lq x Lk x H x B` scores, reads them back for the softmax, and reads them a third
time to apply — 268 MB written and read twice for one of SAM 2's global blocks,
and on top of that it re-reads `q` once per key block and `v` once per query
block. Measured: 203 GFLOP of attention taking 260 ms, 0.8 TFLOP/s, against the
20 TFLOP/s the same card's GEMM sustains.

This is the standard flash formulation. A workgroup owns `BQ` queries and walks
the keys in blocks of `BK`, keeping a running max `m`, a running denominator `l`
and an accumulator, so the scores for a block live in shared memory and are gone
before the next block loads. Global traffic becomes q once, k and v once per
query block, out once — the score matrix never exists.

    m' = max(m, rowmax)             ; c = exp(m - m')
    l' = l*c + Σ exp(s - m')
    acc' = acc*c + Σ exp(s - m') v
    out = acc / l                   (once, at the end)

`c` is the rescaling that makes this exact rather than an approximation: every
partial sum accumulated under an older, smaller maximum is corrected the moment a
larger one appears.

Shapes are `Val` parameters so the shared arrays are statically sized — Lava
miscompiles an `@localmem` whose size comes from a local, silently, by writing
nothing at all. The accumulator is `@private`, i.e. registers: `BQ*E/NT` floats
per thread, 18 for the configuration SAM 2 uses.

**Measured slower than the three-pass path, and not used by `sdpa`.** Correct to
3.4e-8 against a CPU reference, and 43.9 ms against 22.6 ms on SAM 2's global
attention — 881 GFLOP/s against 1713. The reason is occupancy, not arithmetic:
at `E = 72` the tile claims 44.8 KB of the 48 KB a workgroup may have, so exactly
one workgroup fits per SM (256 of ~1536 resident threads, ~17%) and there is
nothing to hide memory latency with. The three-pass kernel uses no shared memory
and runs fully occupied, which is worth more here than the traffic flash saves.

Kept because the analysis points somewhere specific, and so does what blocks it:

  * **Occupancy is the whole problem.** 48 KB / 44.8 KB = one workgroup. To get
    two the tile must fit in 24 KB. Staging q/k/v as fp16 (they already are fp16
    under autocast) gets it to 26.75 KB — still one. Adding `ss` in fp16 reaches
    22.75 KB and two workgroups, but `ss` holds *raw* scores before the max is
    subtracted, and fp16 there costs ~1% on the attention weights.
  * **`BK` cannot simply shrink.** `BK*E` must divide the workgroup, and with
    `E = 72`, `NT = 256` that forces `BK` to a multiple of 32. `BK = 16` does not
    tile.
  * **`BQ = 32` works.** What blocked it was `OpUDiv` in the staging index:
    `E = 72` is not a power of two, so `idx % E` emits a real division into a
    shared-store address, which drops
    stores on this driver. `Mantle.splitidx` removes it and every configuration is
    exact, so `flashfits` accepts odd slot counts. See the note there.
    So the `BQ = 32` route to two workgroups is available to try.

**The conclusion, having worked the numbers: this shape cannot win here.**
Occupancy is `NT * floor(48 KB / shared)`, and no valid tiling beats the 256
threads per SM the three-pass kernel gets for free.

**The `48 KB` in that formula is wrong**, and the table below inherits it. 48 KB
is `maxComputeSharedMemorySize`, Vulkan's per-*workgroup* limit; Ada's SM has
about 100 KB to hand out, so "44.8 KB of 48 KB, therefore one workgroup" was
never the right arithmetic — two fit. It does not rescue this kernel, whose
problem is doing the products by hand at 0.8 TFLOP/s, but every "wg/SM" figure
below should be read as a lower bound rather than a count. The cooperative-matrix
kernel's occupancy was measured rather than derived: see `FLASHCM_TILINGS`.

    tiling                     shared     wg/SM   threads/SM
    BQ=64 BK=32 NT=256 fp32     44.8 KB     1        256      (current)
    BQ=32 BK=32 NT=128 fp32     31.4 KB     1        128
    BQ=64 BK=32 NT=256 fp16     26.8 KB     1        256
    BQ=32 BK=32 NT=128 fp16     17.9 KB     2        256
    BQ=64 BK=32 NT=256 fp16+fp16 ss  22.8 KB  2      512   <- the only 2x

Only the last one doubles occupancy, and it requires `ss` in fp16 — which cannot
hold *raw* scores at useful precision, so it needs the block max computed in
registers with a cross-thread reduction before anything is stored. That is a
rewrite. And even at 2x it lands at ~22 ms, which is parity with the three-pass
path, not a win.

So the fused form is set aside deliberately, not abandoned for lack of trying:
at `E = 72` the tile flash needs does not fit shared memory at an occupancy that
beats having no shared memory at all. It would pay off at a smaller head
dimension, or on a device with more shared memory per SM.
"""


"""
The shared memory every Vulkan implementation guarantees a workgroup, in bytes.

Only for the callers that have no [`M.DeviceCaps`](@ref) in hand — the scalar flash
kernel is reachable with a bare backend. Anything holding a context asks
`ctx.dev.sharedbudget`, which is what the device actually reports.
"""
const PORTABLE_SHARED_FLOOR = 48 * 1024

"""Bytes of `@localmem` the kernel needs for a tiling."""
@inline flashshared(E, BQ, BK) = 4 * (E * BQ + 2 * E * BK + BQ * BK + 3 * BQ)

"""
Whether the flash kernel's tiling is one this kernel is known to compute
correctly for this head dimension.

The divisibility conditions are what the index arithmetic needs; the shared
budget is what the device needs. `BQ = 64, BK = 32` is the configuration
validated against a CPU reference to 3e-8 — see `test_flash.jl`. Other shapes
that pass the arithmetic have measured *wrong* (5e-3), so they are refused here
rather than used: the three-pass path is always available and always right.
"""
@inline function flashfits(E::Int, BQ::Int, BK::Int, NT::Int,
                           sharedbudget::Int = PORTABLE_SHARED_FLOOR)
    (BQ * E) % NT == 0 && (BK * E) % NT == 0 && (BQ * BK) % NT == 0 || return false
    BQ <= NT || return false
    # Enforced rather than assumed: a tiling that asks for more than the device
    # has does not fail to launch, it launches and **writes nothing**, which reads
    # as an attention that returns zeros. `BQ = BK = 64` at `E = 72` wants 70 KB
    # and does exactly that.
    flashshared(E, BQ, BK) <= sharedbudget || return false
    # No `iseven(div(BQ * E, NT))` refusal: an odd slot count is not the
    # condition. `BQ = 32, NT = 256` computes wrong results from query row 4 on
    # (9.4e-02) while every even count is exact, including `BQ = 32` at
    # `NT = 128` — and the cause is `OpUDiv`. The staging loops decompose a flat index with
    # `idx % E` / `idx ÷ E`, and `E = 72` is not a power of two, so a real
    # division landed in a shared-memory store address — which drops stores on
    # this driver (`Mantle.splitidx`, and `test_shared_index_division.jl` for the
    # isolated case). It is exact for *constant* inputs, which is precisely the
    # condition under which the division bug does not bite, because a constant store needs
    # no global load to feed it. Same session, same inputs, only the arithmetic:
    #
    #   BQ/NT    BQ*E/NT      OpUDiv    splitidx
    #   64/256   18 even     7.8e-07     7.1e-07
    #   32/256    9 odd      7.1e-02     8.6e-07
    #   32/128   18 even     8.4e-07     7.0e-07
    #   64/128   36 even     7.8e-07     6.8e-07
    #
    # The slot count is not the variable: with `splitidx` every configuration is
    # exact, so `BQ = 32` is available, which matters because it is one of the
    # two routes to two workgroups per SM.
    return true
end

@kernel cpu=false function attn_flash!(out, @Const(q), @Const(k), @Const(v), scale,
                                       ::Val{BQ}, ::Val{BK}, ::Val{E}, ::Val{NT},
                                       Lk::Int32) where {BQ, BK, E, NT}
    tid = @index(Local, Linear)
    grp = @index(Group, NTuple)
    qb, h, b = grp[1], grp[2], grp[3]


    # `Float32` written out, NOT a local `T = Float32`: Lava miscompiles an
    # `@localmem`/`@private` whose element type comes from a local binding, and
    # the failure is silent — the kernel runs, writes nothing, and every output
    # is zero. Cost an afternoon once already; the same rule applies to the size,
    # which is why the extents are `Val` parameters.
    #
    # (E, ·) so the reduction over `e` walks contiguous shared memory, which is
    # also the order the operands already have in global memory.
    qs = @localmem Float32 (E, BQ)
    ks = @localmem Float32 (E, BK)
    vs = @localmem Float32 (E, BK)
    ss = @localmem Float32 (BQ, BK)
    ms = @localmem Float32 (BQ,)
    ls = @localmem Float32 (BQ,)
    cs = @localmem Float32 (BQ,)  # this block's rescale factor, per row

    acc = @private Float32 (div(BQ * E, NT),)
    T = Float32                    # only for the arithmetic below

    @inbounds begin
        q0 = (qb - 1) * BQ

        # q for this block, once. Consecutive threads take consecutive `e`, so
        # the global read coalesces along the contiguous axis.
        for r in 0:(div(BQ * E, NT) - 1)
            idx = tid + r * NT - 1
            e, lq = Mantle.splitidx(idx, Val(E)); e += 1; lq += 1
            qs[e, lq] = T(q[e, q0 + lq, h, b])
            acc[r + 1] = zero(T)
        end
        if tid <= BQ
            ms[tid] = T(-Inf)
            ls[tid] = zero(T)
        end
        @synchronize

        nblocks = div(Lk, BK)
        for kb in 0:(nblocks - 1)
            k0 = kb * BK
            for r in 0:(div(BK * E, NT) - 1)
                idx = tid + r * NT - 1
                e, lk = Mantle.splitidx(idx, Val(E)); e += 1; lk += 1
                ks[e, lk] = T(k[e, k0 + lk, h, b])
                vs[e, lk] = T(v[e, k0 + lk, h, b])
            end
            @synchronize

            # scores for the tile
            for r in 0:(div(BQ * BK, NT) - 1)
                idx = tid + r * NT - 1
                qi, ki = Mantle.splitidx(idx, Val(BQ)); qi += 1; ki += 1
                s = zero(T)
                for e in 1:E
                    s = muladd(qs[e, qi], ks[e, ki], s)
                end
                ss[qi, ki] = s * T(scale)
            end
            @synchronize

            # online softmax, one thread per query row
            if tid <= BQ
                mo = ms[tid]
                mb = T(-Inf)
                for ki in 1:BK
                    mb = max(mb, ss[tid, ki])
                end
                mn = max(mo, mb)
                # A row that has seen nothing finite yet must not produce NaN
                # from `exp(-Inf - -Inf)`; the guard keeps it at zero weight.
                c = isfinite(mo) ? exp(mo - mn) : zero(T)
                sum = zero(T)
                for ki in 1:BK
                    p = exp(ss[tid, ki] - mn)
                    ss[tid, ki] = p
                    sum += p
                end
                ms[tid] = mn
                ls[tid] = ls[tid] * c + sum
                cs[tid] = c
            end
            @synchronize

            # rescale what is already accumulated, then add this block's share
            for r in 0:(div(BQ * E, NT) - 1)
                idx = tid + r * NT - 1
                e, lq = Mantle.splitidx(idx, Val(E)); e += 1; lq += 1
                s = zero(T)
                for ki in 1:BK
                    s = muladd(ss[lq, ki], vs[e, ki], s)
                end
                acc[r + 1] = acc[r + 1] * cs[lq] + s
            end
            @synchronize
        end

        # normalise and write out
        for r in 0:(div(BQ * E, NT) - 1)
            idx = tid + r * NT - 1
            e, lq = Mantle.splitidx(idx, Val(E)); e += 1; lq += 1
            l = ls[lq]
            out[e, q0 + lq, h, b] = acc[r + 1] / (l == zero(T) ? one(T) : l)
        end
    end
end

"""
    sdpaflash!(out, q, k, v, scale; backend, BQ=64, BK=32, NT=256) -> Bool

Run the fused kernel, or return `false` when this shape does not fit its tiling
(head dimension, sequence length or query count not divisible). Falls to the
caller to use the three-pass path in that case, which is always correct.

`q`, `k`, `v` are `(E, L, H, B)` — the graph's own layout, no transpose.
"""
function sdpaflash!(out, q, k, v, scale; backend = KernelAbstractions.get_backend(q),
                    BQ::Int = 64, BK::Int = 32, NT::Int = 256,
                    sharedbudget::Int = PORTABLE_SHARED_FLOOR)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    (Lq % BQ == 0 && Lk % BK == 0 && flashfits(E, BQ, BK, NT, sharedbudget)) || return false
    attn_flash!(backend, NT)(out, q, k, v, Float32(scale),
                             Val(BQ), Val(BK), Val(E), Val(NT), Int32(Lk);
                             ndrange = (NT * div(Lq, BQ), H, B))
    return true
end

# ── the cooperative-matrix form ──────────────────────────────────────────────
#
# The scalar kernel above loses because it does the two products by hand, at
# 0.8 TFLOP/s against the 35 the same card's tensor cores sustain. Its occupancy
# analysis is real but it is an analysis of *that* shape: the tile is large
# because q, k and v are staged as fp32 and because the score tile has to live in
# shared memory for a scalar reduction to reach it.
#
# `flash_attn_cm1.comp` is a different arrangement. Both products go through
# `OpCooperativeMatrixMulAddKHR`, the operands stay fp16, and the score tile is
# a coopmat accumulator — registers — until the softmax needs it.
#
# **Our memory order makes this simpler than the reference's.** `q`, `k` and `v`
# are `(E, L, H, B)` with `E` contiguous, so:
#
#     S = Q·Kᵀ    A = Q  RowMajor    stride E     (r, e) at r*E + e
#                 B = Kᵀ ColumnMajor stride E     (e, c) at c*E + e
#     O += P·V    A = P  RowMajor    stride Bc
#                 B = V  RowMajor    stride E     (c, e) at c*E + e
#
# Every operand loads straight out of that layout with no transpose and no
# permutation — where the reference computes `S` transposed (`K·Qᵀ`) purely so
# that an implementation offering only `16x8` tiles can still run it. We have
# `16x16`, so the plain orientation is available and it is the one that matches
# our arrays. That is why this is ~200 lines and not 650.
#
# The one thing that cannot be a coopmat is `O`. It has to be rescaled by
# `exp(m_old - m_new)` after every key block and there is no elementwise
# operation on a cooperative matrix. The reference keeps it in plain registers
# and round-trips each `P·V` tile through shared memory to add it. We keep `O` in
# shared instead and load it *as the accumulator's initial value*, so the add is
# the tensor core's own accumulate and the round trip disappears.

"""
Bytes of `@localmem` the cooperative-matrix kernel needs.

Term by term, so a new buffer cannot be added to the kernel without this
noticing: `qs` + `kvs` (fp16 staging) + `ss` (fp32 scores) + `ps` (fp16 weights)
+ `pvs` (fp32 `O`) + `ms`/`ls`/`cs` + the two scalar flags, `grew` and `redo`.

Those last eight bytes were missed when the flags were added, which left the
budget check eight bytes optimistic — harmless at the shipped tiling with 248 to
spare, and exactly the kind of drift that makes a tiling launch and write nothing
at the margin.
"""
@inline flashcmshared(EP, BR, BC, epad = 0, rpad = 0) =
    2 * (EP + epad) * BR + 2 * (EP + epad) * BC +           # qs, kvs   (e-major)
    4 * (BR + rpad) * BC + 4 * (BR + rpad) * EP +           # ss, pvs   (r-major)
    2 * BC * BR +                                           # ps
    12 * BR + 8                                             # ms/ls/cs, grew/redo

"""
    flashepad(caps, EP) / flashrpad(caps, BR)

How much to pad each shared stride so the tensor cores' column-strided access
does not land every column in one memory bank. **The kernel takes both as type
parameters; these are the defaults, and they are where the measurement put them
rather than where the arithmetic would.**

Shared memory is 32 banks of 4 bytes. For the fp16 `e`-major pair the bank of
column `c` is `((EPS·c + e) / 2) % 32`, so `EP % 32 == 0` collides two columns
and `EP % 64 == 0` collides all of them; for the fp32 `r`-major pair it is
`(BRS·c + r) % 32`, and every shipped tiling has `BR` of 32 or 64.

Measured at `Lq = Lk = 4096`, 8 head-batches, tiling `64x32/8` — `epad`, before
and after:

    E    EP    unpadded   padded
    16   16    1.785      1.782    (predicate off — a 4-way conflict that gains nothing)
    32   32    2.592      2.222    -14.3%
    48   48    3.546      3.644    (predicate off)
    64   64    5.266      3.614    -31.4%
    72   80    4.384      4.404    (predicate off)

Unpadded, `E = 64` did four cooperative-matrix tiles per block against `E = 72`'s
five and was still 20% **slower** — more time for less work, which is not
arithmetic. SAM 2 runs at `E = 72 -> EP = 80` and stumbles past this by luck;
head dimension 64 is the common case in every other transformer.

`flash_attn_cm1.comp` pads every shared stride the same way — `qstride =
HSK_pad/4 + 2`, `psh_stride = Br/4 + 2`, `kvsh_stride = …/4 + 2`, all `+2` vec4s,
i.e. +8 scalars. We padded none.

## `rpad` is implemented, measured, and OFF — the microbenchmark was the wrong shape

The `r`-major pair conflicts by the same arithmetic, and padding it wins on the
long shapes. On top of `epad`, at `Lq = Lk = 4096`:

    E    EP    rpad=0   rpad=2   rpad=4   rpad=8
    16   16    1.818    1.697    1.638    1.696     -9.9%
    32   32    2.473    2.748    2.920    3.130     +18.5%  (loses)
    48   48    3.657    3.480    3.410    3.485     -6.8%
    64   64    3.796    3.511    3.413    3.490    -10.1%
    72   80    4.637    4.130    4.000    4.115    -13.7%

Four of five win, including SAM 2's own `EP = 80`. **SAM 2's encode then went
100.65 -> 107.4 ms, +6.7%.**

The 4096-long benchmark is not what the encoder runs: most of its attention is
*windowed*, short `L` with many head-batches, and there the extra kilobyte of
`@localmem` costs more residency than the bank conflict costs bandwidth. A pad
that helps every shape in a sweep can still lose the model, which is the same
trap as the GEMM tiling whose weighted mean ranked the tilings backwards.

So `rpad` stays a parameter with a default of zero rather than being deleted:
it is the first thing to re-measure on hardware whose shared memory is banked
differently, and the numbers above are the NVIDIA baseline to compare against.
"""
# The pads above are one card's answer. On a device whose cooperative-matrix
# modules run NARROWER than its own wave — `caps.subgroup 64` against
# `caps.coopmatsubgroup 32` on RDNA 3.5 — an LDS access is two 32-lane phases
# rather than one, and the table does not transfer. Re-measured there at
# `Lq = Lk = 4096`, `E = 72 -> EP = 80`, tiling `64x32/16`, `rego` on, `epad` down
# each column and `rpad` across (ms):
#
#     epad\rpad     1       2       4       8
#     0           9.297   9.433   9.359   9.698
#     2          13.919  14.000  13.954  14.082
#     4           8.180   8.231   8.614   8.368
#     8           8.676   8.909   8.727   8.610
#
# `epad` is worth 1.14x and `rpad` almost nothing, and **the bank arithmetic does
# not explain it**: `epad = 2` makes the fp16 row stride 41 bank words, coprime
# with 32 and therefore conflict-free by the rule above, and it is the worst row
# in the table by 70%. `plans/projects/portability/bankconflict_sweep.jl` wrote
# that prediction down before the sweep ran, so this kills it rather than
# confirming it; what the pad really buys here is unexplained and the numbers are
# the only reason for these values.
@inline narrowcoopmat(caps::M.DeviceCaps) = caps.subgroup > caps.coopmatsubgroup
@inline flashepad(caps::M.DeviceCaps, EP) =
    narrowcoopmat(caps) ? 4 : EP % 32 == 0 ? 8 : 0
@inline flashrpad(caps::M.DeviceCaps, BR) = narrowcoopmat(caps) ? 1 : 0

"""
Whether a `(BR, BC)` tiling is one the cooperative-matrix kernel can run **on this
device**. `dev` supplies the tile, the subgroup width and the shared budget; every
one of those was a literal or a module-level `Ref` before, and each is a property
of the device rather than of the kernel.
"""
@inline function flashcmfits(dev::M.DeviceCaps, EP::Int, BR::Int, BC::Int, NT::Int,
                             helddirect::Bool = false)
    BR % dev.tile == 0 && BC % dev.tile == 0 && EP % dev.tile == 0 || return false
    # The softmax gives one thread a whole query row.
    BR <= NT || return false
    # `attn_flash_cm!` holds `O` in exactly three accumulators a subgroup, and
    # `@nexprs` needs that count to be a literal. A tiling wanting a fourth
    # would silently drop its tiles, so it is refused instead.
    cld((BR ÷ dev.tile) * (EP ÷ dev.tile), NT ÷ dev.coopmatsubgroup) <= 3 || return false
    (BR * EP) % NT == 0 && (BR * BC) % NT == 0 || return false
    # With the pads this device will actually launch with: the budget check and
    # the launcher have to agree, or a tiling passes here and then asks for more
    # `@localmem` than the device has.
    epad, rpad = flashepad(dev, EP), flashrpad(dev, BR)
    shared = flashcmshared(EP, BR, BC, epad, rpad)
    # An unsplit held output is written straight from its fragments. `pvs` is
    # then dead and the compiler allocates none of its BR*EP fp32 scratch.
    helddirect && (shared -= 4 * (BR + rpad) * EP)
    shared <= dev.sharedbudget
end

"""
    flashrescale(row, col, element, cs, base) -> Float32

`element * cs[base + row]` — the rescale a held `O` needs, as the callback of
`Lava.coopmat_perelement`.

Top-level, so it has no captured environment to pass, and deliberately **not**
`@noinline`: it is meant to melt into `Lava.coopmat_perelement_thunk`, which is
the function the instruction names. Marked `@noinline` it stays a separate
`OpFunction` with `DontInline` and the driver then calls it once per element —
8.5x, and enough to make the whole feature read as a loss.

`row` and `col` are 0-based; `row` is the element's row *within its own 16x16
tile*, which `base` shifts to the tile's place in the `BR`-row block. `col` is
unused and must still be in the signature — that is `Lava.coopmat_keepparam`'s job.
"""
function flashrescale(row::UInt32, col::UInt32, e::Float32,
                      cs::Core.LLVMPtr{Float32,3}, base::Int32)
    e * unsafe_load(cs, Int(base + row) + 1)
end

"""
    attn_flash_cm!(out, q, k, v, scale, Val(BR), Val(BC), Val(E), Val(EP), Val(NW), Lk)

One workgroup owns `BR` queries and walks the keys in blocks of `BC`.

`EP` is `E` rounded up to the tile — 72 becomes 80 — and the staging loops write
zero past `E`. That padding is free in the arithmetic (`0 * 0` contributes
nothing to `S`, and `O`'s columns past `E` are never read out) and it is what
lets every cooperative-matrix load be full-width.

`Mantle.splitidx` for every staging index, not `%`/`÷`: `EP = 80` and `E = 72` are
not powers of two, and a real `OpUDiv` in a shared-memory store address drops
stores on this driver whenever a `muladd` is in scope. That is the bug this
kernel would otherwise walk straight into — see `test_shared_index_division.jl`.
"""
@inline flashscore(s, scale, ::Nothing, qi, ki, h, b) = s * scale
@inline function flashscore(s, scale, mask, qi, ki, h, b)
    qi > size(mask,2) && return -65504.0f0
    @inbounds z = mask[ki,qi,size(mask,3)==1 ? 1 : h,size(mask,4)==1 ? 1 : b]
    Float32(Float16(Float16(Float16(s)*scale)+z))
end

@inline function flashoutindex(e, h, l, b, ::Val{E}, H, L,
                               ::Val{0}, ::Val{0}, ::Val{0}, ::Val{0}) where {E}
    e + E * ((h - 1) + H * (l + L * (b - 1)))
end

@inline function flashoutindex(e, h, l, b, ::Val{E}, H, L,
                               ::Val{IW}, ::Val{IH}, ::Val{NX}, ::Val{NY}) where
                               {E,IW,IH,NX,NY}
    ix, iy = l % IW, l ÷ IW
    wb = b - 1
    wx, tail = wb % NX, wb ÷ NX
    wy, batch = tail % NY, tail ÷ NY
    col = ix + IW * wx + (IW * NX) * (iy + IH * wy + IH * NY * batch)
    e + E * ((h - 1) + H * col)
end

@kernel cpu=false unsafe_indices=true function attn_flash_cm_spatial4!(
        out, @Const(q), @Const(k), @Const(v), scale, @Const(mask),
        qbase::Int32, qsE::Int32, qsL::Int32, qsH::Int32, qsB::Int32,
        kbase::Int32, ksE::Int32, ksL::Int32, ksH::Int32, ksB::Int32,
        vbase::Int32, vsE::Int32, vsL::Int32, vsH::Int32, vsB::Int32,
        ::Val{BR}, ::Val{BC}, vE::Val{E}, ::Val{EP}, ::Val{NW}, ::Val{REGO}, ::Val{HELD},
        ::Val{CLAMP}, ::Val{KCLAMP}, ::Val{RSC}, ::Val{BALLAST}, ::Val{SHPAD}, ::Val{NRSC},
        ::Val{PREONLY}, ::Val{RSCBAR}, ::Val{PREFETCHV}, ::Val{OUTPERM},
        vWIW::Val{WIW}, vWIH::Val{WIH}, vWNX::Val{WNX}, vWNY::Val{WNY}, ::Val{NH},
        ::Val{NSPLIT}, ::Val{EPAD}, ::Val{RPAD},
        ::Val{SG},
        Lq::Int32, Lk::Int32, alwaysrescale::Int32,
        onepass::Int32, partial, ml) where {BR,BC,E,EP,NW,REGO,HELD,CLAMP,KCLAMP,RSC,
                                            BALLAST,SHPAD,NRSC,PREONLY,RSCBAR,PREFETCHV,OUTPERM,
                                            WIW,WIH,WNX,WNY,NH,NSPLIT,
                                            EPAD,RPAD,SG}
    # `SG` is `dev.coopmatsubgroup` and not a literal 32: the launcher sizes the
    # workgroup as `NW * dev.coopmatsubgroup`, so a literal disagrees with it on
    # any device where Lava cannot pin a 32-lane subgroup: the launch would ask
    # for `NW * 64` threads while the kernel strided by `NW * 32`, and every
    # subgroup past the first would read another's fragment without crashing.
    NT = NW * SG
    # `EPS` and `BRS`, not `EP` and `BR`, are the STRIDES of the shared arrays: the
    # tensor cores read them by column, and an unpadded stride puts every column in
    # one memory bank. Worth -31.4% at head dimension 64 — see [`flashepad`](@ref)
    # for the arithmetic, the measurement, and why the defaults sit where they do.
    #
    # NOTE the sizes below are written out from the type parameters and not from
    # the local `EPS`/`BRS`. Lava miscompiles an `@localmem` whose size comes from
    # a local binding — silently: the kernel runs and writes nothing. The locals
    # are the same values, and are only ever used for index arithmetic. The pad
    # columns are written by nothing and read by nothing: staging covers the real
    # extent and no 16-wide tile starts inside the pad.
    qs  = @localmem Float16 ((EP + EPAD) * BR,)   # (e, r) at r*EPS + e
    kvs = @localmem Float16 ((EP + EPAD) * BC,)   # (e, c) at c*EPS + e — K, then V
    EPS = EP + EPAD
    # The same argument for the r-major pair, which the tensor cores reach at a
    # column stride of `BR` fp32. Bank = `(BR·c + r) % 32`, so `BR % 32 == 0` puts
    # every column in one bank — and every shipped tiling has `BR` 32 or 64.
    BRS = BR + RPAD
    ss  = @localmem Float32 ((BR + RPAD) * BC,)   # (r, c) at c*BRS + r
    ps  = @localmem Float16 (BC * BR,)            # (r, c) at r*BC + c
    # `REGO == false`: this is `O`, and it persists across key blocks.
    # `REGO == true`:  this is one key block's `P·V`, and `O` lives in `acco`.
    pvs = @localmem Float32 ((BR + RPAD) * EP,)   # (r, e) at e*BRS + r
    ms  = @localmem Float32 (BR,)
    ls  = @localmem Float32 (BR,)
    cs  = @localmem Float32 (BR,)
    # Shared-memory ballast — the other half of the `BALLAST` diagnostic, and the
    # one that works. Registers cannot be forced upward: the driver has its own
    # occupancy target and caps itself at 128 (two 256-thread workgroups per SM)
    # no matter how many live values it is handed. Shared memory it cannot
    # negotiate, so padding the footprint is the only way to hold everything else
    # fixed and vary residency alone.
    shpad = @localmem Float32 (SHPAD < 1 ? 1 : SHPAD,)
    # Did any row's running max move this block? One word, and it decides
    # whether `O` has to be rescaled at all — see the loop below.
    grew = @localmem Float32 (1,)
    # Did any row's one-pass attempt overflow fp16? Workgroup-wide, because
    # the retry has to be taken by every row or they end up on different
    # references — see the softmax below.
    redo = @localmem Float32 (1,)

    # `O` in registers: `BR*EP/NT` floats per thread, 20 for the shipped tiling.
    # Written out as `Float32` rather than through a local `T`, because Lava
    # miscompiles a `@private` whose element type comes from a local binding and
    # does it silently — the kernel runs and writes nothing. `NW * SG` spelled
    # out rather than the local `NT` above for the same reason: the size has to
    # come from the type parameters, and both of these are.
    acco = @private Float32 (div(BR * EP, NW * SG),)
    # Optional software pipeline: fetch the current value tile before Q*K and
    # keep each thread's small slice in registers until the score/softmax phase
    # has finished using `kvs`. This overlaps the otherwise-serial global V read
    # with useful matrix work without requiring a second shared-memory tile.
    # This is deliberately enabled only for plans whose slice is small enough
    # for Lava to scalarise it (see `flash_launches`). Larger dynamically
    # indexed private fp16 arrays still lower through packed integer words.
    vstage = @private Float16 (cld(BC * EP, NW * SG),)

    RT = BR ÷ Mantle.GEMM_TILE
    CT = BC ÷ Mantle.GEMM_TILE
    ET = EP ÷ Mantle.GEMM_TILE

    tid = @index(Local, Linear) - 1
    grp = @index(Group, NTuple)
    # The split index rides in the FIRST grid dimension rather than a fourth:
    # `grp[1]` runs over `Tr * NSPLIT`. Folding it keeps the launch 3-D, which is
    # what `ndrange` and every index helper below already assume.
    gq, h, b = grp[1], grp[2], grp[3]
    Tr = cld(Lq, Int32(BR))
    qb = (gq - 1) % Tr + 1
    sp = (gq - 1) ÷ Tr                      # 0-based split index
    w = tid ÷ 32                            # subgroup within the workgroup

    @inbounds begin
        q0 = (qb - 1) * BR

        # Register ballast — a diagnostic, off unless `BALLAST` is set, and the
        # control that decides whether a register count *causes* a slowdown or
        # merely accompanies one. `BALLAST` values are LOADS (so the driver
        # cannot rematerialise them instead of keeping them live) defined before
        # the key loop and consumed after it, on a branch that never runs. They
        # therefore add registers and **no work at all** — which is exactly what
        # is needed to price occupancy on its own.
        Base.Cartesian.@nexprs 24 i -> bal_i = BALLAST >= i ?
            q[qbase + Int32((i - 1) % E) * qsE + Int32(h - 1) * qsH +
              Int32(b - 1) * qsB] : zero(Float16)

        # Q for this block, once, and it stays in shared for every key block.
        for r in 0:(div(BR * EP, NT) - 1)
            idx = tid + r * NT
            e, lq = Mantle.splitidx(idx, Val(EP))
            # `CLAMP` is what lets a sequence that does not divide the tile run at
            # all: the rows past its end are staged as zero and masked out of the
            # softmax, and never written back. SAM 2's *decoder* is the case —
            # every one of its attentions has a 23 in it, the mask prompt's token
            # count, and 23 does not tile to 16.
            inq = !CLAMP || q0 + lq < Lq
            qs[1 + e + lq * EPS] = (e < E && inq) ?
                q[qbase + Int32(e) * qsE + Int32(q0 + lq) * qsL +
                  Int32(h - 1) * qsH + Int32(b - 1) * qsB] : zero(Float16)
        end
        if REGO
            for s in 1:div(BR * EP, NT)
                acco[s] = 0.0f0
            end
        elseif !HELD
            # `(lq, e)`, not a flat index: with `RPAD > 0` the array is no longer
            # `BR·EP` contiguous elements. The division stays on `BR` — a power of
            # two, so a shift — and the padded stride enters as a multiply.
            # `splitidx(idx, Val(BRS))` would put a real `OpUDiv` in a shared-store
            # index, which is the open miscompile in `test_shared_index_division.jl`.
            # The pad columns are written by nothing and read by nothing.
            for r in 0:(div(BR * EP, NT) - 1)
                lq, e = Mantle.splitidx(tid + r * NT, Val(BR))
                pvs[1 + lq + e * BRS] = 0.0f0
            end
        end
        if tid < BR
            ms[1 + tid] = -Inf32
            ls[1 + tid] = 0.0f0
        end
        # `O`'s tiles, held in cooperative-matrix accumulators for the whole key
        # loop when `held` is set. Three because `cld(RT*ET, NW)` is 3 at every
        # tiling `flashcmfits` admits; the third is guarded and idle for the
        # upper subgroups. They cost *fewer* registers than reloading the tile
        # per block did — 122 against 128, measured — because what goes away with
        # the load and the store is also their address arithmetic.
        Base.Cartesian.@nexprs 3 j ->
            acc_j = zero(Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.Accumulator})

        # Which row of its tile does each of this lane's accumulator components
        # belong to? The answer is what lets `O` be rescaled where it lives, and
        # the cooperative-matrix spec does not define it — so it is *measured*,
        # here, once per launch, rather than assumed. A 16x16 tile whose element
        # (r, c) holds `r` is loaded exactly the way `O` is stored (column-major,
        # M axis contiguous), and `coopmat_getcomp` then reports each component's
        # own row. On this card every lane's eight components turn out to lie in
        # just two rows, `lane÷4` and `lane÷4 + 8`, but nothing below depends on
        # that — only on `orow_i` being right.
        #
        # `ss` is the scratch: it is the score matrix from the first key block
        # onward, and both barriers below are already required.
        #
        # `:comp` needs the row for rescaling.  A held, unsplit output also uses
        # both coordinates to store its accumulator fragments straight to the
        # destination at the end.  That direct store is important on devices
        # with 64 KiB of shared memory: round-tripping held O through `pvs`
        # otherwise keeps this kernel at ~49 KiB and admits only one workgroup.
        # With no held-path reference to `pvs`, the compiler removes that 20 KiB
        # allocation entirely.
        if HELD && (RSC === :comp || NSPLIT == 1)
            for idx in tid:NT:(Mantle.GEMM_TILE * Mantle.GEMM_TILE - 1)
                r, c = Mantle.splitidx(idx, Val(Mantle.GEMM_TILE))
                ss[1 + idx] = Float32(r + c * Mantle.GEMM_TILE)
            end
            @synchronize
            rowmat = Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.Accumulator}(
                        ss, 1, Mantle.GEMM_TILE, Val(false))
            Base.Cartesian.@nexprs 8 i -> begin
                ocoord_i = unsafe_trunc(Int32, Mantle.coopmat_getcomp(rowmat, Int32(i - 1)))
                orow_i = ocoord_i % Int32(Mantle.GEMM_TILE)
                ocol_i = ocoord_i ÷ Int32(Mantle.GEMM_TILE)
            end
        end
        @synchronize

        # `cld`, not `div`: with `CLAMP` the last key block is partial, and at
        # `Lk = 23 < BC = 32` — the decoder's self-attention — `div` gives ZERO
        # blocks and the loop never runs. Identical to `div` when `BC` divides
        # `Lk`, which is every non-clamped call.
        # This split's slice of the key axis. With `NSPLIT == 1` these are
        # `0` and `cld(Lk, BC)`, i.e. exactly the loop that was here before.
        nkb = cld(Lk, Int32(BC))
        kbper = cld(nkb, Int32(NSPLIT))
        kbeg = sp * kbper
        kend = min(nkb, kbeg + kbper)
        for kb in kbeg:(kend - 1)
            k0 = kb * BC
            if tid == 0
                grew[1] = Float32(alwaysrescale)
                redo[1] = 0.0f0
            end
            for r in 0:(cld(BC * EP, NT) - 1)
                idx = tid + r * NT
                if idx < BC * EP
                    e, lk = Mantle.splitidx(idx, Val(EP))
                    ink = !KCLAMP || k0 + lk < Lk
                    kvs[1 + e + lk * EPS] = (e < E && ink) ?
                        k[kbase + Int32(e) * ksE + Int32(k0 + lk) * ksL +
                          Int32(h - 1) * ksH + Int32(b - 1) * ksB] : zero(Float16)
                end
            end
            @synchronize

            if PREFETCHV
                for r in 0:(cld(BC * EP, NT) - 1)
                    idx = tid + r * NT
                    if idx < BC * EP
                        e, lk = Mantle.splitidx(idx, Val(EP))
                        ink = !KCLAMP || k0 + lk < Lk
                        vstage[1 + r] = (e < E && ink) ?
                            v[vbase + Int32(e) * vsE + Int32(k0 + lk) * vsL +
                              Int32(h - 1) * vsH + Int32(b - 1) * vsB] : zero(Float16)
                    end
                end
            end

            # S = Q·Kᵀ. RT*CT tiles handed round the subgroups; the trip count is
            # uniform within a subgroup, which is what a coopmat op requires.
            for t in w:NW:(RT * CT - 1)
                rt = t % RT
                ct = t ÷ RT
                acc = zero(Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.Accumulator})
                for et in 0:(ET - 1)
                    a = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixA}(
                            qs, 1 + rt * Mantle.GEMM_TILE * EPS + et * Mantle.GEMM_TILE, EPS, Val(true))
                    bm = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixB}(
                            kvs, 1 + ct * Mantle.GEMM_TILE * EPS + et * Mantle.GEMM_TILE, EPS, Val(false))
                    acc = muladd(a, bm, acc)
                end
                copyto!(ss, 1 + rt * Mantle.GEMM_TILE + ct * Mantle.GEMM_TILE * BRS, BRS, acc)
            end
            @synchronize

            # Online softmax, one thread per query row: the reduction is over
            # keys and a row lives entirely in this thread, so no lane talks to
            # another. `scale` is applied here rather than folded into `qs`,
            # which keeps `q` bit-exact in fp16 on the way in.
            #
            # **This leaves 64 of 256 threads working, and spreading it is
            # slower.** The obvious complaint about this loop is that six of
            # eight warps sit at the barrier while two walk `BC` serially. The
            # fix looks free: store the score tile row-major instead (a whole row
            # contiguous, which needed a row-major cooperative-matrix store — see
            # `copyto!`), give each *subgroup* a query row and each lane a
            # key, and `subgroup_max` is then exactly the row maximum with no
            # cluster and no shared scratch. Built and measured interleaved:
            #
            #     4096x4096   4.836 ms -> 5.097     256x256   0.444 -> 0.461
            #
            # 5% the wrong way. Eight rows a subgroup at two subgroup reductions
            # each is 16 shuffle sequences per block, and that costs about what
            # the serial walk did — so the idle warps were never the problem, and
            # the softmax is not what this kernel is waiting on. The transposed
            # layout and the parallel reduction are inseparable (a thread-per-row
            # loop over a row-major tile puts all 32 lanes in one shared bank),
            # so both went back.
            # **One pass over `ss` when the running maximum is usable as the
            # reference.** The online softmax is exact for *any* reference: `m`
            # cancels between the numerator and `l`, and it exists only to keep
            # `exp` in range. So exponentiate against `mo` — the maximum from
            # previous blocks, already known — and correct afterwards, which reads
            # each score once instead of once for the maximum and again for the
            # exponential.
            #
            # The bound is `ps`, which is fp16: `exp(s - mo) <= 65504` needs
            # `s - mo <= 11.09`. Past that the values written were `Inf` and the
            # block is redone against its own maximum, at two-pass cost.
            #
            # `kb > 0` rather than `isfinite(mo)`: they say the same thing — `mo`
            # is `-Inf` for every row on the first key block and finite for every
            # row after — but this form is uniform across *all* threads, including
            # the ones with no row, which is what lets `pre` below be uniform.
            onep = onepass != 0 && kb > 0
            mb = -Inf32
            sm = 0.0f0
            if onep && tid < BR
                mo = ms[1 + tid]
                for ci in 0:(BC - 1)
                    # A padded key contributes nothing: it is out of the maximum
                    # and its weight is zero, so `P·V` adds zero for it. Staging
                    # already zeroed its `k`, which would otherwise have given it
                    # a score of 0 and a weight of `exp(-mo)` — not nothing.
                    if !KCLAMP || k0 + ci < Lk
                        s = flashscore(ss[1 + tid + ci * BRS], scale, mask, q0+tid+1, k0+ci+1, h, b)
                        mb = max(mb, s)
                        p = exp(s - mo)
                        ps[1 + ci + tid * BC] = Float16(p)
                        sm += p
                    else
                        ps[1 + ci + tid * BC] = zero(Float16)
                    end
                end
                mb - mo > FLASH_EXP_HEADROOM && (redo[1] = 1.0f0)
            end
            @synchronize

            # **The retry is decided for the whole workgroup, not per row.** A row
            # that overflowed needs its `ps` against its own maximum, which leaves
            # it on a different reference from the rows that did not — and with
            # `O` in accumulators there is no per-row conversion available to
            # reconcile them. So if any row overflowed, every row redoes the
            # block. `onep` is uniform already: `mo` is `-Inf` for every row on
            # the first key block and finite for every row after it, which is
            # what `kb > 0` says without reading `ms`.
            # `PREONLY` forces the early placement, which makes the deferred
            # rescale site below statically dead. Only correct when `onepass` is
            # off — then `onep` is always false and `pre` is always true anyway,
            # so this changes no arithmetic, only how much of the kernel exists.
            # It is the probe for whether *two* rescale sites are what costs the
            # second resident workgroup.
            pre = PREONLY || !onep || redo[1] != 0.0f0
            if pre && tid < BR
                mo = ms[1 + tid]
                mb = -Inf32
                for ci in 0:(BC - 1)
                    (!KCLAMP || k0 + ci < Lk) &&
                        (mb = max(mb, flashscore(ss[1 + tid + ci * BRS], scale, mask, q0+tid+1, k0+ci+1, h, b)))
                end
                mn = max(mo, mb)
                # A row that has seen nothing finite must not make NaN out of
                # exp(-Inf - -Inf); it stays at zero weight.
                cr = isfinite(mo) ? exp(mo - mn) : 0.0f0
                sm = 0.0f0
                for ci in 0:(BC - 1)
                    if !KCLAMP || k0 + ci < Lk
                        p = exp(flashscore(ss[1 + tid + ci * BRS], scale, mask, q0+tid+1, k0+ci+1, h, b) - mn)
                        ps[1 + ci + tid * BC] = Float16(p)
                        sm += p
                    else
                        ps[1 + ci + tid * BC] = zero(Float16)
                    end
                end
                ms[1 + tid] = mn
                ls[1 + tid] = ls[1 + tid] * cr + sm
                cs[1 + tid] = cr
                cr == 1.0f0 || (grew[1] = 1.0f0)
            elseif tid < BR
                # `ps` is relative to `mo`, and so is `O`, so nothing is converted
                # before the product: the correction applies to the old and the
                # new contribution alike and is deferred past it. That deferral is
                # the whole point — it is what lets the pass above read each score
                # once.
                mo = ms[1 + tid]
                mn = max(mo, mb)
                cr = exp(mo - mn)
                ms[1 + tid] = mn
                ls[1 + tid] = (ls[1 + tid] + sm) * cr
                cs[1 + tid] = cr
                cr == 1.0f0 || (grew[1] = 1.0f0)
            end
            @synchronize

            for r in 0:(cld(BC * EP, NT) - 1)
                idx = tid + r * NT
                if idx < BC * EP
                    e, lk = Mantle.splitidx(idx, Val(EP))
                    if PREFETCHV
                        kvs[1 + e + lk * EPS] = vstage[1 + r]
                    else
                        ink = !KCLAMP || k0 + lk < Lk
                        kvs[1 + e + lk * EPS] = (e < E && ink) ?
                            v[vbase + Int32(e) * vsE + Int32(k0 + lk) * vsL +
                              Int32(h - 1) * vsH + Int32(b - 1) * vsB] : zero(Float16)
                    end
                end
            end
            @synchronize

            # `pre` says the reference moved *before* this block's contribution,
            # so `O` has to be converted first; otherwise the conversion covers
            # old and new together and waits until after. Either way it is only
            # needed when some row's factor is not one, which `grew` records.
            if !REGO && pre && grew[1] != 0.0f0
                if HELD
                    # NOTE: hoisting the factor matrix out of this loop is
                    # correct — `t_j % RT == w % RT` for all three tiles whenever
                    # `RT` divides `NW`, which every admitted tiling satisfies —
                    # and it is WORSE: 220 registers against 172. The allocator
                    # would rather reload the tile three times than keep one live
                    # across all three rescales. Measured, not assumed.
                    Base.Cartesian.@nexprs 3 j -> begin
                        t_j = w + (j - 1) * NW
                        # `NRSC` rescales only the first `NRSC` of the three held
                        # tiles. Wrong below 3 — it is a diagnostic, and the one
                        # that priced the rescale and found the 128-register step.
                        if j <= NRSC && t_j < RT * ET
                            base_j = Int32((t_j % RT) * Mantle.GEMM_TILE)
                            if RSC === :fmul
                                # The factor as a matrix: a **stride-0** load, so
                                # all 16 columns read the same 16 factors. No
                                # component is ever named, so nothing is
                                # materialised — one load and one `OpFMul`.
                                acc_j = Mantle.coopmat_mul(acc_j,
                                    Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,
                                                           Mantle.GEMM_TILE,Mantle.Accumulator}(
                                        cs, 1 + base_j, 0, Val(false)))
                            elseif RSC === :perelem
                                acc_j = Lava.coopmat_perelement(flashrescale, acc_j,
                                                                cs.ptr, base_j)
                            else
                                Base.Cartesian.@nexprs 8 i ->
                                    acc_j = Mantle.coopmat_setcomp(acc_j, Int32(i - 1),
                                        Mantle.coopmat_getcomp(acc_j, Int32(i - 1)) *
                                        cs[1 + base_j + orow_i])
                            end
                        end
                    end
                    # `RSCBAR`: a barrier the rescale does not need, to stop the
                    # scheduler hoisting work across it. The register count here
                    # is an allocator *decision* — 128 at one rescaled tile, 231
                    # at two, 172 at three — so the question is whether it is
                    # inflating to software-pipeline the rescale against the
                    # muladd that follows. `grew` and `pre` are both
                    # workgroup-uniform, so this is legal where it sits.
                    RSCBAR && @synchronize
                else
                    for r in 0:(div(BR * EP, NT) - 1)
                        lq, e = Mantle.splitidx(tid + r * NT, Val(BR))
                        pvs[1 + lq + e * BRS] *= cs[1 + lq]
                    end
                    @synchronize
                end
            end

            if HELD && !REGO
                # `O` never leaves the accumulators here: the muladd chain adds
                # straight into the tile this subgroup has been holding since the
                # loop began, so the load and the store that dominated `P·V` are
                # simply not issued.
                Base.Cartesian.@nexprs 3 j -> begin
                    t_j = w + (j - 1) * NW
                    if t_j < RT * ET
                        rt_j = t_j % RT
                        et_j = t_j ÷ RT
                        for ct in 0:(CT - 1)
                            a = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixA}(
                                    ps, 1 + rt_j * Mantle.GEMM_TILE * BC + ct * Mantle.GEMM_TILE, BC, Val(true))
                            bm = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixB}(
                                    kvs, 1 + ct * Mantle.GEMM_TILE * EPS + et_j * Mantle.GEMM_TILE, EPS, Val(true))
                            acc_j = muladd(a, bm, acc_j)
                        end
                    end
                end
            else
                for t in w:NW:(RT * ET - 1)
                    rt = t % RT
                    et = t ÷ RT
                    off = 1 + rt * Mantle.GEMM_TILE + et * Mantle.GEMM_TILE * BRS
                    # Starting from `O` itself means the accumulate is the tensor
                    # core's own; starting from zero means the registers below do it.
                    acc = REGO ?
                        zero(Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.Accumulator}) :
                        Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.Accumulator}(
                            pvs, off, BRS, Val(false))
                    for ct in 0:(CT - 1)
                        a = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixA}(
                                ps, 1 + rt * Mantle.GEMM_TILE * BC + ct * Mantle.GEMM_TILE, BC, Val(true))
                        bm = Mantle.AcceleratedMatrix{Float16,Mantle.GEMM_TILE,Mantle.GEMM_TILE,Mantle.MatrixB}(
                                kvs, 1 + ct * Mantle.GEMM_TILE * EPS + et * Mantle.GEMM_TILE, EPS, Val(true))
                        acc = muladd(a, bm, acc)
                    end
                    copyto!(pvs, off, BRS, acc)
                end
            end
            @synchronize

            # The deferred correction, applied to old and new contributions
            # together now that both are in `O`. With `O` in shared this reads and
            # writes it — 40 KB of the ~110 KB a key block costs. With `O` held it
            # touches shared for the factor alone: eight reads of `cs` and eight
            # multiplies inside the accumulator, and **no barrier**, which is what
            # turned this path from a 15-19% loss into a win. `grew` still keeps
            # it off the blocks where every factor is one.
            if !REGO && !PREONLY && !pre && grew[1] != 0.0f0
                if HELD
                    # NOTE: hoisting the factor matrix out of this loop is
                    # correct — `t_j % RT == w % RT` for all three tiles whenever
                    # `RT` divides `NW`, which every admitted tiling satisfies —
                    # and it is WORSE: 220 registers against 172. The allocator
                    # would rather reload the tile three times than keep one live
                    # across all three rescales. Measured, not assumed.
                    Base.Cartesian.@nexprs 3 j -> begin
                        t_j = w + (j - 1) * NW
                        # `NRSC` rescales only the first `NRSC` of the three held
                        # tiles. Wrong below 3 — it is a diagnostic, and the one
                        # that priced the rescale and found the 128-register step.
                        if j <= NRSC && t_j < RT * ET
                            base_j = Int32((t_j % RT) * Mantle.GEMM_TILE)
                            if RSC === :fmul
                                # The factor as a matrix: a **stride-0** load, so
                                # all 16 columns read the same 16 factors. No
                                # component is ever named, so nothing is
                                # materialised — one load and one `OpFMul`.
                                acc_j = Mantle.coopmat_mul(acc_j,
                                    Mantle.AcceleratedMatrix{Float32,Mantle.GEMM_TILE,
                                                           Mantle.GEMM_TILE,Mantle.Accumulator}(
                                        cs, 1 + base_j, 0, Val(false)))
                            elseif RSC === :perelem
                                acc_j = Lava.coopmat_perelement(flashrescale, acc_j,
                                                                cs.ptr, base_j)
                            else
                                Base.Cartesian.@nexprs 8 i ->
                                    acc_j = Mantle.coopmat_setcomp(acc_j, Int32(i - 1),
                                        Mantle.coopmat_getcomp(acc_j, Int32(i - 1)) *
                                        cs[1 + base_j + orow_i])
                            end
                        end
                    end
                    # `RSCBAR`: a barrier the rescale does not need, to stop the
                    # scheduler hoisting work across it. The register count here
                    # is an allocator *decision* — 128 at one rescaled tile, 231
                    # at two, 172 at three — so the question is whether it is
                    # inflating to software-pipeline the rescale against the
                    # muladd that follows. `grew` and `pre` are both
                    # workgroup-uniform, so this is legal where it sits.
                    RSCBAR && @synchronize
                else
                    for r in 0:(div(BR * EP, NT) - 1)
                        lq, e = Mantle.splitidx(tid + r * NT, Val(BR))
                        pvs[1 + lq + e * BRS] *= cs[1 + lq]
                    end
                    @synchronize
                end
            end

            if REGO
                # `O = O*c + PV` in registers. `pvs` has the same `(BR, EP)`
                # shape the accumulator stored into, so this is a flat index
                # with no tile arithmetic.
                #
                # The row is loop-INVARIANT whenever `NT` is a multiple of `BR`,
                # which every shipped tiling is: `idx = tid + (s-1)*NT`, so
                # `idx % BR == tid % BR` for all `s`, so `splitidx` is hoisted
                # out rather than called per element. The per-element call cost
                # 26% on this branch — an attribution
                # that was never tested. Hoisting it is free; see `FLASHCM_REGO`
                # for what the measurement then said.
                lqfixed = tid % BR
                efixed  = tid ÷ BR
                estep   = NT ÷ BR
                for s in 1:div(BR * EP, NT)
                    idx = tid + (s - 1) * NT
                    lq, e = NT % BR == 0 ? (lqfixed, efixed + (s - 1) * estep) :
                                           Mantle.splitidx(idx, Val(BR))
                    acco[s] = acco[s] * cs[1 + lq] + pvs[1 + lq + e * BRS]
                end
                @synchronize
            end
        end

        # The ballast's only consumer, on a branch the driver cannot fold away
        # and that never runs (`alwaysrescale` is 0 or 1). This is what keeps the
        # loads live across the whole key loop.
        if BALLAST > 0 && alwaysrescale == Int32(9999)
            Base.Cartesian.@nexprs 24 i -> (BALLAST >= i && (out[i] = bal_i))
        end
        # Same trick for the shared pad: a store the driver cannot fold away, so
        # the allocation survives, on a branch that never runs.
        if SHPAD > 0 && alwaysrescale == Int32(9999)
            shpad[1 + tid] = Float32(tid)
            out[1] = Float16(shpad[1])
        end

        if HELD && !REGO && NSPLIT != 1
            Base.Cartesian.@nexprs 3 j -> begin
                t_j = w + (j - 1) * NW
                t_j < RT * ET && copyto!(pvs, 1 + (t_j % RT) * Mantle.GEMM_TILE +
                                         (t_j ÷ RT) * Mantle.GEMM_TILE * BRS, BRS, acc_j)
            end
            @synchronize
        end

        # O has remained in cooperative-matrix fragments throughout the key
        # loop. Extract each lane's components and write them directly rather
        # than storing fp32 to shared memory, synchronising, then loading every
        # value again with scalar threads. `ocoord_i` above discovers the
        # implementation-defined component layout; no lane mapping is assumed.
        if HELD && !REGO && NSPLIT == 1
            Base.Cartesian.@nexprs 3 j -> begin
                t_j = w + (j - 1) * NW
                if t_j < RT * ET
                    rt_j = t_j % RT
                    et_j = t_j ÷ RT
                    Base.Cartesian.@nexprs 8 i -> begin
                        lq_i = rt_j * Mantle.GEMM_TILE + orow_i
                        e_i = et_j * Mantle.GEMM_TILE + ocol_i
                        if e_i < E && (!CLAMP || q0 + lq_i < Lq)
                            l_i = ls[1 + lq_i]
                            o_i = Mantle.coopmat_getcomp(acc_j, Int32(i - 1))
                            ov_i = o_i / (l_i == 0.0f0 ? 1.0f0 : l_i)
                            if OUTPERM
                                oi_i = flashoutindex(e_i, h, q0 + lq_i, b,
                                    vE, NH, Lq,
                                    vWIW, vWIH, vWNX, vWNY)
                                unsafe_store!(pointer(out), convert(eltype(out), ov_i), 1 + oi_i)
                            else
                                out[1 + e_i, 1 + q0 + lq_i, h, b] = ov_i
                            end
                        end
                    end
                end
            end
        end

        # Slots `1 : BR*E/NT` are exactly the ones whose `e` is inside the real
        # head dimension: `idx = tid + (s-1)*NT` and `NT` divides `BR*E`, so the
        # padded columns are all in the slots past that and never written out.
        if !HELD || REGO || NSPLIT != 1
            for s in 1:div(BR * E, NT)
                idx = tid + (s - 1) * NT
                lq, e = Mantle.splitidx(idx, Val(BR))
                if !CLAMP || q0 + lq < Lq
                    l = ls[1 + lq]
                    o = REGO ? acco[s] : pvs[1 + lq + e * BRS]
                    if NSPLIT == 1
                        ov = o / (l == 0.0f0 ? 1.0f0 : l)
                        if OUTPERM
                            oi = flashoutindex(e, h, q0 + lq, b,
                                vE, NH, Lq,
                                vWIW, vWIH, vWNX, vWNY)
                            unsafe_store!(pointer(out), convert(eltype(out), ov), 1 + oi)
                        else
                            out[1 + e, 1 + q0 + lq, h, b] = ov
                        end
                    else
                    # UNNORMALISED, plus this split's row max and sum. The merge
                    # cannot divide yet: `l` here is only this slice's sum, and
                    # the rows' maxima differ between splits, so the rescale has
                    # to happen after every split's `m` is known. Same split as
                    # llama.cpp's `flash_attn_split_k_reduce.comp`.
                        partial[1 + e, 1 + q0 + lq, h, b, 1 + sp] = o
                    end
                end
            end
        end
        # One thread per row writes the pair the merge reduces over.
        if NSPLIT > 1
            for lq in tid:NT:(BR - 1)
                if !CLAMP || q0 + lq < Lq
                    ml[1 + q0 + lq, h, b, 1 + sp, 1] = ms[1 + lq]
                    ml[1 + q0 + lq, h, b, 1 + sp, 2] = ls[1 + lq]
                end
            end
        end
    end
end



#=
── `lazyrescale`: skip the rescale on blocks where no row's maximum moved ──────

On by default, as `sdpaflashcm!`'s literal keyword default. The keyword survives, so the
A/B is still one call away — which is how `test_flash.jl` compares the two.

Skip the `O *= exp(m_old - m_new)` pass on key blocks where no row's running
maximum moved.

The online softmax rescales the accumulator whenever a block contains a larger
score than anything seen so far. That is a read and a write of the whole of `O`
— 40 KB of the ~110 KB of shared traffic a key block costs, and `O` is the
single largest consumer of it. The rescale factor is `exp(m_old - m_new)`, which
is **exactly one** whenever the block's max did not beat the running one.

Worth 5-6%, and that is less than the traffic suggests because the flag is an OR
over all `BR` rows. Per row the maximum does settle; for a workgroup of 64 it
does not:

    BR=64  Lk=4096   grew on  67.3% of blocks
    BR=64  Lk=256    grew on 100.0%
    BR=1   Lk=4096   grew on   4.1%

So the pass is skipped on a third of the global blocks and none of the windowed
ones. See [`FLASHCM_HELD`](@ref), where the same statistic is the whole reason a
promising rearrangement of `O` loses.

One shared word per workgroup records whether any row grew. The threads racing
to set it all write the same value, and the branch is uniform because every
thread reads it after the same barrier.

Exact, not approximate — the skipped work is a multiplication by one.
=#

#=
── `onepass`: read each score once in the softmax instead of twice ─────────────

The other settled `sdpaflashcm!` keyword (was `FLASHCM_ONEPASS`), on, literal
default, keyword kept for the A/B.

The two-pass form reads `ss` for the row maximum and again for the exponential —
`2 * BR * BC * 4` bytes a key block, and the softmax is a third of the kernel.
The online softmax is exact for **any** reference maximum (it cancels between the
numerator and `l`; it exists only to keep `exp` in range), so the one-pass form
exponentiates against `mo`, the maximum from previous blocks, which is already
known, and defers the correction to the sweep that runs after the value product.

`ps` is fp16, so `exp(s - mo)` must stay under 65504, i.e. `s - mo <= 11.09`
([`FLASH_EXP_HEADROOM`](@ref)). A row whose block runs hotter than that is redone
against its own maximum at two-pass cost, and its `O` is converted in place by
the one thread that owns it rather than by the sweep. `mo = -Inf`, which is every
row on the first key block, always takes that path.

Exact either way: the tests compare the two settings with `==`.
=#


"""
    flashcm_perelem_available() -> Bool

Whether the device can rescale a held `O` in place — `VK_NV_cooperative_matrix2`
with `cooperativeMatrixPerElementOperations`.

Where the extension exists, rescaling goes through
`OpCooperativeMatrixPerElementOpNV` rather than a chain of
`coopmat_getcomp`/`coopmat_setcomp`, so this asks the device and has no switch
of its own.

This is the missing piece [`FLASHCM_HELD`](@ref) documents: the 31% is real, and
the portable component access that would buy it costs +69 registers (123 -> 192),
halving resident workgroups per SM. `VK_NV_cooperative_matrix2` gives
the driver the job instead — it walks its own layout, hands the callback the
element's `(row, col)`, and materialises nothing.

Two things fall out beyond the register count. The runtime probe that discovered
each lane's component-to-row mapping (`orow_i`, a 16x16 tile of row indices
staged through `ss`) is not needed: the row arrives as an argument. And the
callback reads `cs` directly through a `Workgroup` pointer passed as an extra
operand, so the rescale still touches shared memory for the factor alone and
still needs no barrier.

NVIDIA-only. Gated on `vk_context().coopmat2.per_element_operations`; with it
false the `getcomp`/`setcomp` path is what runs, and `held` stays off there.
"""
flashcm_perelem_available() = Mantle.vk_context().coopmat2.per_element_operations


"""
How far a block's maximum may exceed the running one before the one-pass softmax
falls back. `exp(11.09)` is fp16's largest finite value; 10 leaves room for the
fact that `ps` rounds.
"""
const FLASH_EXP_HEADROOM = 10.0f0


"""
    FLASHCM_TILINGS

`(BR, BC, NW)`, fastest first subject to the shape rules in
[`flashcm_tiling`](@ref). `NW` is cooperative-matrix subgroups, so the actual
workgroup is `dev.coopmatsubgroup * NW`.

Measured on SAM 2's two dominant attention shapes, clock warmed, interleaved,
against the two-GEMM cooperative-matrix path. **Re-swept after the lazy rescale
and the one-pass softmax**: a measured constant is invalidated by a change to
the thing it was measured against, which is how `COOPMAT_MINL` went 512 -> 256
when the GEMM under it got 1.68x faster:

    tiling        4096x4096      256x256
    coopmat        9.50 ms       0.883 ms     (two-GEMM coopmat)
    64x32/8w       4.39          0.450        <- shipped default
    64x16/8w       5.49          0.504
    32x32/8w       6.21          0.557
    32x16/8w       6.38          0.537
    64x16/4w       6.47          0.572
    32x64/8w       9.92          0.808
    32x32/4w      10.77          0.815

The order shuffled below the top — `32x32/8w` and `32x16/8w` swapped on the
global blocks — but the default is unchanged and its lead widened from 15% to
25%, because both optimisations scale with `BR` and `64` is the largest that
fits.

**Subgroups dominate every other parameter.** The same `64x32` block goes 7.34 ->
5.09 on four warps against eight, and `32x64` at two warps is the worst thing
measured — five times the best. That is the same result the GEMM tuning reached
from the other direction, where widening the warp *grid* won and widening the
warp *tile* lost: what this device wants is warps in flight.

`64x64` at any width wants 66 KB and is refused; `64x32` asks 48 904 bytes of the
48 KB budget, with 248 to spare.

What the driver reports for the shipped tiling
(`VK_KHR_pipeline_executable_properties`):

    Register Count 128    Shared Memory Size 48 900    Stack Size 0

Nothing spills, and **two workgroups fit per SM, capped by both resources at
once**: 256 threads x 128 registers is 32 768 of the SM's 65 536, and 48 900
bytes is two of Ada's ~100 KB of shared. 512 of 1 536 resident threads. So
shrinking the tile alone buys no occupancy — registers cap it at two
independently, and a third workgroup needs both under 34 KB and under 85
registers a thread.

On gfx1151, AMDGPU reports native 16x16 WMMA, wave32 cooperative-matrix
subgroups, 64 KiB LDS and 64 resident waves. Directly storing an unsplit held
output from its accumulator fragments removes the `BR*EP` fp32 `pvs` scratch;
that makes the 128-row tile fit and permits two resident workgroups. Re-swept
inside the recorded SAM 2 encoder graph, `BC=16` wins for its 4096-token global
blocks, while `BC=32` preserves the less order-sensitive recurrence and wins
for its 256-token windows. The 64-token blocks also measurably prefer `BC=16`.
Those choices, plus the dense elementwise routes, put the whole encode at a
139.60 ms median on that device (24 samples), rather than merely optimising an
isolated attention launch.
"""
const FLASHCM_TILINGS = [(128, 16, 8), (128, 32, 8), (64, 16, 8), (64, 32, 8), (32, 32, 8), (32, 16, 8),
                         (32, 32, 4), (16, 32, 4), (16, 16, 4)]

"""
    flashcm_tiling(dev, E, Lq, Lk, nbatch = 0; clamp = false) -> (BR, BC, NW) | nothing

The first tiling in [`FLASHCM_TILINGS`](@ref) that divides this shape and fits
`dev`. `nothing` means the caller keeps whatever path it would otherwise have
used.

Takes a [`M.DeviceCaps`](@ref) rather than reading one, which is what makes it
answerable without a GPU: `flashcm_tiling(M.DeviceCaps(true, 16, 64, 32,
65536, 1024, 40, 0), …)` asks what this chooser would do on a wave64 RDNA3 part
from a machine that has none.

Note the fourth argument. Every table below was measured on one card, and the
subgroup width appears twice with different meanings: the device's *default*
(64 there) and the width a cooperative-matrix module actually runs at, which
Lava pins to 32. The tiling needs the second.
"""
function flashcm_tiling(dev::M.DeviceCaps, E::Int, Lq::Int, Lk::Int, nbatch::Int = 0;
                        clamp::Bool = false)
    # `flashcm_plan` checks this before calling, but this is also reached
    # directly — from tests and from the docstring above. Without matrix hardware
    # there is no tile, and `cld(E, 0)` throws instead of reporting "no tiling".
    dev.coopmat || return nothing
    EP = cld(E, dev.tile) * dev.tile
    fits = NTuple{3,Int}[]
    # `NW` scaled to this device's wave, then the table's own. Every number in
    # `FLASHCM_TILINGS` was measured where a cooperative-matrix module runs at the
    # device's native wave width; where it runs narrower — `coopmatsubgroup 32`
    # against `subgroup 64` — the table's eight subgroups are only four waves, so
    # the workgroup has half the waves in flight it was tuned for. Doubling `NW`
    # restores them, and the table's own conclusion was that "subgroups dominate
    # every other parameter".
    #
    # Measured on that device, `Lq = Lk = 4096`, `E = 72`, `epad 4`/`rpad 1`, `rego`
    # on: `64x32/8` 11.69 ms against `64x32/16` **8.18**, -30%. The windowed shape
    # (256x256, 128 head-batches) agrees: 0.72 against 0.638.
    #
    # The scaled width is tried FIRST and per entry, because it is not always
    # admissible: `32x32/16` fails `(BR * E) % NT == 0` at `E = 72`, and there the
    # table's own eight is what runs.
    # There are two independent reasons to try the wider workgroup first.
    #
    #  * A cooperative-matrix subgroup narrower than the device's ordinary
    #    subgroup needs more matrix subgroups to restore the wave count the
    #    table was tuned at (the original Vulkan/RDNA case).
    #  * A processor capable of keeping 64 or more subgroups resident needs
    #    the same width even when its native matrix subgroup is already wave32.
    #    HIP reports that occupancy fact on gfx1151.  With NW=8 this kernel's
    #    ~49 KiB LDS footprint admits only one workgroup and strands most of the
    #    processor; NW=16 changes neither the tile nor its LDS and, with the
    #    arithmetic-preserving accumulator choice below, measures 18.4 -> 15.2
    #    ms on SAM 2's 4096-token block and 1.31 -> 1.09 ms windowed.
    #
    # `warps == 0` means unknown, not zero, so backends which cannot report
    # residency retain the measured table instead of being guessed at here.
    width_widen = max(1, dev.subgroup ÷ dev.coopmatsubgroup)
    occupancy_widen = dev.warps >= 64 ? 2 : 1
    widen = max(width_widen, occupancy_widen)
    for (BR, BC, NW0) in FLASHCM_TILINGS, NW in unique((NW0 * widen, NW0))
        # Measured shape policy for the new narrow-key entries. The long global
        # loop and the 64-token blocks win with BC=16; the 256-token windows use
        # BC=32, which is both faster there and less sensitive to recurrence
        # order. Other shapes retain the established BC=32 choices.
        BR == 128 && BC == 16 && Lk < 4096 && continue
        BR == 64 && BC == 16 && Lk != 64 && continue
        NT = NW * dev.coopmatsubgroup
        NT <= dev.workgrouplimit || continue
        # Without `clamp` the extents have to divide the tile; with it they are
        # padded and masked, which is what puts the decoder's 23-token
        # attentions on this path at all.
        #
        # **But padding is wasted arithmetic, so it has to earn its place.** A
        # tiling may be taken clamped only if it is at least half occupied on
        # both axes. Without that floor the encoder's `Lq = 4` and `Lq = 16`
        # calls get padded to `BR = 64` — 94% and 75% waste — and take the fused
        # path to their cost: encode measured 133.7 -> 137.97 ms with the floor
        # absent. It also makes the chooser prefer `BR = 32` over `BR = 64` for
        # `Lq = 23`, which is 72% occupied rather than 36%.
        if Lq % BR != 0 || Lk % BC != 0
            clamp || continue
            tiny_exact_key = BR == dev.tile && BC == dev.tile && Lk == BC &&
                             4 * Lq >= BR
            (2 * Lq >= BR && 2 * Lk >= BC) || tiny_exact_key || continue
        end
        # The 128-row tile exists only because a held, unsplit output no longer
        # allocates `pvs`. Keep that footprint exception on encoder-style square
        # attention; split-k and decoder cross-attention retain the conservative
        # shared-memory accounting and therefore the established smaller tiles.
        helddirect = BR == 128 && dev.warps >= 64 && NW >= 16 &&
                     Lq == Lk && Lq >= BR
        flashcmfits(dev, EP, BR, BC, NT, helddirect) && (BR * E) % NT == 0 &&
            !any(c -> c[1] == BR && c[2] == BC, fits) && push!(fits, (BR, BC, NW))
    end
    isempty(fits) && return nothing
    # `FLASHCM_TILINGS` is ordered fastest-first *for a grid that fills the
    # device*, and taking its first entry is right whenever one does. SAM 2's
    # decoder is the case where none does: `Lq = 23` against `Lk = 4096` is one
    # query block, and one block times `H*B = 8` is **8 workgroups on 48 SMs**.
    # The kernel is then latency-bound rather than anything else and measures
    # 0.10 TFLOP/s against the encoder's 6.9 — 95% of the whole decoder's
    # attention bill.
    #
    # So when the leading tiling cannot fill the device, prefer one that fills it
    # better, keeping the table's own order among equals. Measured on that shape:
    # `32x32x8` (8 workgroups) 0.4492 ms, `16x32x4` (16) **0.3919**, -12.8%.
    #
    # This is a mitigation and not the fix. The fix is to split the *key* axis
    # across workgroups and merge the partial softmaxes — flash-decoding — which
    # would give 128 workgroups instead of 16 and is worth about 1.2 ms of the
    # decode rather than 0.24. See `perf-plan.md`.
    nbatch <= 0 && return fits[1]
    grid(c) = cld(Lq, c[1]) * nbatch
    # One workgroup per shader core is the floor for "this launch fills the
    # device", queried rather than written as this card's 48: every encoder shape
    # is far above it (the windowed blocks launch 512) and only the decoder's
    # `Lq = 23` cross-attentions fall under, at 8.
    grid(fits[1]) >= max(1, dev.cores) && return fits[1]
    best = fits[1]
    for c in fits
        grid(c) > grid(best) && (best = c)
    end
    best
end



"""
    splitcount(dev, Lq, Lk, BR, BC, nbatch; allow = true) -> Int

How many ways to split the key axis, from llama.cpp's rule rather than ours.

    split = cores * 2 / workgroups_without_split      # aim at 2 wg per core
    chunk = align_up(Lk / split, alignment)           # a whole number of blocks
    split = cld(Lk, chunk)                            # re-derive from the chunk

Two workgroups per core, not one: a single workgroup per core leaves no other
warp to cover a stall, and this kernel is latency-bound at the decoder's shape.

Re-deriving the count from the rounded chunk is what keeps every split a whole
number of `BC`-wide key blocks, so no split gets a ragged remainder.

**`ROUNDUP_POW2` in ggml is not what its name says**, and reading it as written
costs a third of the win. It is

    #define ROUNDUP_POW2(M, N) (((M) + (N) - 1) & ~((N) - 1))

i.e. round `M` up to a **multiple of** `N`, where `N` happens to be a power of
two — not "round `M` to the next power of two". Ported the second way this
returned 4 splits for the decoder's shape where the rule wants 6, and 4 measures
0.159 ms against 0.095 for 8. Counting the chunk in whole key blocks here makes
the alignment implicit, so there is nothing left to round.
"""
function splitcount(dev::M.DeviceCaps, Lq::Int, Lk::Int, BR::Int, BC::Int, nbatch::Int;
                    allow::Bool = true)
    allow || return 1
    dev.cores > 0 || return 1
    base = cld(Lq, BR) * nbatch
    base >= 2 * dev.cores && return 1          # already fills the device
    want = max(1, (2 * dev.cores) ÷ base)
    want == 1 && return 1
    nkb = cld(Lk, BC)                          # key blocks available to split
    nkb <= 1 && return 1
    chunk = max(1, cld(nkb, want))     # already in whole BC-wide key blocks
    n = cld(nkb, chunk)
    # A split that cannot pay for its merge is not worth the extra global traffic.
    n <= 1 && return 1
    return n
end

"""
    flashcm_plan(dev, q, k, v, bias; kw...) -> FlashCMPlan | Decline

Whether [`sdpa`](@ref) may fuse this call, and with what.

`bias` must be absent for the same reason the two-GEMM path refuses it: the mask
would have to be added between the score product and the softmax, and here that
is inside a cooperative-matrix accumulator.

A question about the problem and the device, and nothing else — so it takes a
[`M.DeviceCaps`](@ref) and can be answered for a device the caller does not have.

`BR`/`BC`/`NW` default to the chooser's pick; passing them selects a tiling
explicitly, which is what the A/B in `test_flash.jl` does. They are still
*validated*: an explicit tiling that does not fit gets a `Decline`, not a kernel
that launches and writes nothing.
"""
function flashcm_plan(dev::M.DeviceCaps, q, k, v, bias;
                      clamp::Bool = false, rego::Union{Nothing,Bool} = nothing,
                      held::Union{Nothing,Bool} = nothing,
                      rescale::Symbol = :fmul, onepass::Bool = true,
                      lazyrescale::Bool = true, split::Bool = true,
                      BR::Int = 0, BC::Int = 0, NW::Int = 0)
    bias === nothing || return Decline(:bias)
    dev.coopmat || return Decline(:nocoopmat)
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 ||
        return Decline(:eltype)

    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    EP = cld(E, dev.tile) * dev.tile

    autotiling = BR == 0
    tiling = if autotiling
        flashcm_tiling(dev, E, Lq, Lk, H * B; clamp)
    else
        (BR, BC, NW)
    end
    tiling === nothing && return Decline(:notiling)
    BR, BC, NW = tiling
    # The pinned coopmat width, not the device default — see `M.DeviceCaps`.
    NT = NW * dev.coopmatsubgroup
    holdtiles = held === nothing ? (dev.warps >= 64 && NW >= 16) : held

    NT <= dev.workgrouplimit || return Decline(:workgroup)
    (clamp || (Lq % BR == 0 && Lk % BC == 0)) || return Decline(:extent)
    # `split=false` makes the direct held store statically certain here.  The
    # automatic split count is decided below; until then use the conservative
    # footprint so a split-k decoder cannot be admitted on memory it still uses.
    helddirect = holdtiles && (split === false ||
                              (autotiling && Lq == Lk && Lq >= BR))
    flashcmfits(dev, EP, BR, BC, NT, helddirect) || return Decline(:tiling)
    # `BR * E` must also tile the write-out loop, which `flashcmfits` cannot check
    # because it does not see the unpadded head dimension.
    (helddirect || (BR * E) % NT == 0) || return Decline(:writeout)

    # Operands as a root array plus strides, not as the wrapper. Attention's q, k
    # and v arrive as `PermutedDimsArray -> ReshapedArray -> SubArray -> LavaArray`,
    # and the two things that costs are the same two `transposeLE` already fixed:
    # the wrapper's `SignedMultiplicativeInverse`s put four integer divisions on
    # every staged element, and avoiding that by materialising k and v first cost
    # **646.4 MB of copies per encode, 96 of them, not one operand already dense**.
    # A dense array's strides are `(1, E, E*L, E*L*H)`, so this is the same
    # arithmetic, minus the divisions and minus the copy.
    #
    # An operand stack `stridedroot` cannot account for is the one refusal the
    # caller can recover from, hence its own reason.
    (stridedroot(q) === nothing || stridedroot(k) === nothing ||
     stridedroot(v) === nothing) && return Decline(:wrapped)

    # Decided last, because it depends on the tiling that was just chosen: `BR`
    # fixes how many query blocks there are, and `BC` fixes how finely the key
    # axis can be cut.
    nsplit = splitcount(dev, Lq, Lk, BR, BC, H * B; allow = split)
    # The reduced footprint is real only for the unsplit direct-store path.
    # Recheck after `splitcount`, so an unexpectedly split plan declines instead
    # of launching a kernel whose actual LDS exceeds the admitted budget.
    flashcmfits(dev, EP, BR, BC, NT, holdtiles && nsplit == 1) ||
        return Decline(:tiling)

    # ── Where `O` lives, decided from the tiling and not from the device ──────
    #
    # `O` in registers costs `BR * EP / NT` floats a thread and saves a read and a
    # write of the whole accumulator per key block — 40 KB of a key block's ~110 KB
    # of shared traffic. The Ada sweep found the crossing and wrote it down: the
    # register form "wins at `32x32` and loses at `64x32`, crossing where
    # `BR*EP/NT` goes from 10 floats a thread to 20", which is why this shipped
    # off with `64x32/8`.
    #
    # The RDNA 3.5 sweep lands on the same number from the other side: at
    # `64x32/16` the same `O` is 10 floats a thread and the register form wins,
    # 10.10 ms against 8.41 at `Lq = Lk = 4096`. So the rule is the ratio, and two
    # vendors' hardware agrees on where it turns — a default of `false` would now
    # be wrong for the tiling this device picks.
    # Widening solely to occupy a high-residency wave32 processor must not also
    # change the numerical algorithm. `rego` changes where the online output
    # accumulator lives and therefore its rounding; enabling it for all 42 SAM
    # 2 calls made final residual drift grow 1.40 -> 2.32. It briefly looked
    # profitable on the three global calls alone (15.47 -> 12.93 ms), but the
    # fragment-held path below is both arithmetic-preserving and faster there,
    # 12.93 -> 10.42 ms. An explicitly requested `rego` still means exactly what
    # the caller asked for.
    occupancy_only_widened = autotiling && dev.subgroup == dev.coopmatsubgroup &&
                             dev.warps >= 64 && NW >= 16
    regnt = occupancy_only_widened ? NT ÷ 2 : NT
    holdregs = rego === nothing ? (BR * EP) ÷ regnt <= 10 : rego
    # Holding O in cooperative-matrix fragments removes its load/store on every
    # key block while retaining the same muladd arithmetic. It used to lose on
    # the narrower launch because the fragments consumed the occupancy it had;
    # on a >=64-resident-subgroup processor the widened launch already supplies
    # that occupancy and holding wins: the 39 windowed SAM 2 calls take attention
    # from 80.01 to 69.17 ms, with bit-identical encoder outputs. Unknown
    # residency (`warps == 0`) keeps the conservative table behavior.
    FlashCMPlan(BR, BC, NW, NT, E, EP, clamp, holdregs, holdtiles, rescale, onepass,
                lazyrescale, nsplit)
end

"""
    flashcm_padded_plan(dev, q, k, v, bias; limit = 1.25) -> FlashCMPlan | Decline

The clamped plan, taken only where the padding is cheaper than not having flash
attention at all.

A tiling has to divide both extents, and a joint attention's key length is
whatever the prompt made it: Qwen-Image 2.1 at 1024² over a 22-token prompt
queries 4096 positions against 4118 keys, and 4118 is 2 x 29 x 71. No tile
divides that, so the strict plan declines and the caller materialises a
4096 x 4118 score matrix per layer instead — measured at **40.2 s per denoising
step against 9.6 s** for the same model at a key length that happens to divide.

Clamping pads the last tile and masks it, so the cost is the padding: 0.3% at
that shape. It is refused by default because padding is not always that cheap —
SAM 2's encoder has `Lq = 16` calls that would pad to 32 and lose 2.12 ms of
encode for nothing — so this asks how much padding the shape actually needs and
declines when it is more than `limit`.

The one exception is a query shorter than a single tile against exactly one key
tile: four real rows in a 16-row tile is 4x padding and still beats writing the
scores plus two padded products, which is why `flashcm_tiling` has a rule for
that shape and this has one too.
"""
function flashcm_padded_plan(dev, q, k, v, bias; limit::Real = 1.25)
    plan = flashcm_plan(dev, q, k, v, bias; clamp = true)
    plan isa FlashCMPlan || return plan
    Lq, Lk = size(q, 2), size(k, 2)
    (Lq < dev.tile && Lk == dev.tile && 4Lq >= dev.tile) && return plan
    padded = cld(Lq, plan.BR) * plan.BR * cld(Lk, plan.BC) * plan.BC
    padded <= limit * Lq * Lk ? plan : Decline(:padding)
end

"""
Merge the per-split partial attentions into the final output.

A port of `flash_attn_split_k_reduce.comp` from llama.cpp
(`ggml/src/ggml-vulkan/vulkan-shaders/`), which is the same algorithm in the same
API. Its structure rather than its indexing: two passes over the splits — first
the row's true maximum, then the rescaled sum — instead of a sequential online
update. Fewer dependencies, and the merge itself parallelises.

Each split wrote `O_s` unnormalised together with the row max `m_s` and row sum
`l_s` it was computed under. Restoring the true value needs the *global* row max,
which no split knew:

    m  = maxₛ m_s
    L  = Σₛ exp(m_s − m) · l_s
    O  = Σₛ exp(m_s − m) · O_s   /   L

**One thread per output element, and the split count at runtime** — both as in
the shader this ports, and both load-bearing. It was one thread per `(row, head,
batch)` looping over `E`, with `NSPLIT` a `Val` so that every loop bound was a
compile-time constant and the split loops unrolled. At the counts `splitcount`
actually returns that is not an optimisation, it is a hang: `NSPLIT = 64` with
`E = 72` is a 4608-iteration triple loop that the driver unrolls whole, and the
19274-instruction shader it produces — 58 spilled VGPRs, 456 spilled SGPRs — does
not come back. `Lk = 4096` was a GPU recovery rather than a wrong answer, and
`NSPLIT = 32` still fits, which is why only the long-key shapes died.

A runtime bound cannot be unrolled to a constant, and giving `e` its own thread
removes the inner loop entirely. The `ml` loads stay cheap because they are
uniform across a wave — every `e` of one row reads the same two floats.
"""
@kernel cpu=false function attn_flash_cm_merge!(out, @Const(partial), @Const(ml),
                                                nsplit::Int32, nheads::Int32)
    # The head and batch axes ride folded in the third grid dimension, so the
    # first can carry `e` and the launch stays 3-D.
    e, lq, hb = @index(Global, NTuple)
    @inbounds begin
        h = (hb - 1) % nheads + 1
        b = (hb - 1) ÷ nheads + 1
        # pass 1: the row's true maximum across splits
        m = -Inf32
        for sp in 1:nsplit
            m = max(m, ml[lq, h, b, sp, 1])
        end
        # pass 2: the sum and this element's value, every split rescaled onto it
        L = 0.0f0
        o = 0.0f0
        for sp in 1:nsplit
            w = exp(ml[lq, h, b, sp, 1] - m)
            L += w * ml[lq, h, b, sp, 2]
            o += w * partial[e, lq, h, b, sp]
        end
        out[e, lq, h, b] = o * (L == 0.0f0 ? 1.0f0 : 1.0f0 / L)
    end
end

# sdpaflashcm!(ctx, out, plan::FlashCMPlan, q, k, v, scale) -> out
# 
# Run the cooperative-matrix fused kernel. `q`, `k`, `v` are `(E, L, H, B)`.
# 
# **This cannot decline.** Every condition is settled by
# [`flashcm_plan`](@ref), which is the point of holding a plan: a `return false`
# here would run *after* the caller has allocated `out` and committed to the
# fused path, and would have to agree with a predicate that already said yes.
# 
# `ballast`, `shpad`, `nrsc`, `preonly` and `rscbar` are the diagnostics from the
# held-`O` investigation (closed — see [`FLASHCM_HELD`](@ref)). They stay keywords
# rather than plan fields because they describe an experiment, not a routing
# decision, and nothing in the library sets them.

"""
    flash_launches(caps, out, plan, q, k, v, scale, partial, ml; …) -> Vector

What this attention IS: one launch, or two when the plan splits the key axis for
flash decoding. No side effects and no allocation — `partial` and `ml` are the
split's scratch and the caller owns them, because a graph's scratch is a declared
transient and an immediate call's is `scratch!`.

Split from the launching for the reason `coopmat_gemm_launches` is: a graph
declares these and `sdpaflashcm!` submits them, and which tiling a shape takes
is measured, so two copies of the decision drifting is a performance regression
nothing would report.
"""
function flash_launches(caps, out, plan::FlashCMPlan, q, k, v, scale, partial, ml;
                      mask=nothing,
                      ballast::Int = 0, shpad::Int = 0, nrsc::Int = 3,
                      preonly::Bool = false, rscbar::Bool = false,
                      # Five values per thread at the 512-thread global tile is
                      # enough latency hiding for a measured win. The 128-thread
                      # window tile needs twenty registers and loses occupancy.
                      prefetchv::Bool = plan.NT >= 512,
                      outperm::Bool = false,
                      outwindow::NTuple{4,Int} = (0, 0, 0, 0),
                      epad::Int = flashepad(caps, plan.EP),
                      rpad::Int = flashrpad(caps, plan.BR))
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    BR, BC, NW, NT = plan.BR, plan.BC, plan.NW, plan.NT
    rego, held = plan.rego, plan.held
    rq, rk, rv = stridedroot(q), stridedroot(k), stridedroot(v)
    sq, sk, sv = flashstrides(q), flashstrides(k), flashstrides(v)
    flat(r) = flashflat(r[1])

    ns = plan.nsplit
    outarg = outperm ? flashoutflat(out) : out
    args = (outarg, flat(rq), flat(rk), flat(rv), Float32(scale), mask,
                                Int32(rq[2] + 1), sq[1], sq[2], sq[3], sq[4],
                                Int32(rk[2] + 1), sk[1], sk[2], sk[3], sk[4],
                                Int32(rv[2] + 1), sv[1], sv[2], sv[3], sv[4],
                                Val(BR), Val(BC), Val(plan.E), Val(plan.EP), Val(NW),
                                Val(rego), Val(held && !rego),
                                # Per AXIS, not per plan. A clamped plan is
                                # taken because ONE extent does not divide its
                                # tile, and the other one usually still does:
                                # Qwen-Image 2.1 queries 4096 positions — which
                                # every tile divides — over 4118 keys, which
                                # none do. Compiling the query bounds check in
                                # anyway put it in the load and all three store
                                # loops for nothing, and those are the loops.
                                Val(plan.clamp && Lq % BR != 0),
                                # Padded queries do not require key checks when
                                # the occupied-cache bucket divides BC exactly.
                                Val(plan.clamp && Lk % BC != 0),
                                # Normalised, so a `rescale` setting cannot key a
                                # second identical pipeline when nothing rescales.
                                Val(held && !rego ? plan.rescale : :comp), Val(ballast),
                                Val(shpad), Val(nrsc), Val(preonly && !plan.onepass),
                                Val(rscbar), Val(prefetchv), Val(outperm),
                                map(Val, outwindow)..., Val(H),
                                Val(ns), Val(epad), Val(rpad),
                                Val(caps.coopmatsubgroup),
                                Int32(Lq), Int32(Lk), Int32(plan.lazyrescale ? 0 : 1),
                                Int32(plan.onepass && !rego ? 1 : 0), partial, ml)
    first = (kern = attn_flash_cm_spatial4!, args = args,
             ndrange = (NT * cld(Lq, BR) * ns, H, B), group = NT)
    ns == 1 && return [first]
    # The merge is a separate dispatch because every split has to have
    # finished before any row's true maximum is known — that is the one real
    # dependency flash-decoding introduces, and it is why the split has to
    # pay for a second pass over `Lq * H * B * E` to buy its parallelism.
    return [first, (kern = attn_flash_cm_merge!,
                    args = (out, partial, ml, Int32(ns), Int32(H)),
                    ndrange = (plan.E, Lq, H * B), group = 0)]
end

"""Submit this attention now."""
function sdpaflashcm!(ctx, out, plan::FlashCMPlan, q, k, v, scale; kw...)
    ns = plan.nsplit
    Lq, H, B = size(q, 2), size(q, 3), size(q, 4)
    partial = ns == 1 ? out : scratch!(ctx, Float32, size(v, 1), Lq, H, B, ns)
    ml      = ns == 1 ? out : scratch!(ctx, Float32, Lq, H, B, ns, 2)
    for l in flash_launches(ctx.dev, out, plan, q, k, v, scale, partial, ml; kw...)
        k_ = l.group == 0 ? l.kern(ctx.backend) : l.kern(ctx.backend, l.group)
        k_(l.args...; ndrange = l.ndrange)
    end
    return out
end

"""
    flash_dispatch!(g, caps, out, plan, q, k, v, scale, partial, ml; name, …) -> out

DECLARE this attention into a graph: one pass per launch.
"""
function flash_dispatch!(g, caps, out, plan::FlashCMPlan, q, k, v, scale,
                         partial, ml; name::AbstractString = "sdpa", kw...)
    ls = flash_launches(caps, out, plan, q, k, v, scale, partial, ml; kw...)
    for (i, l) in enumerate(ls)
        M.dispatch!(g, l.kern, l.args, l.ndrange;
                    group = l.group == 0 ? nothing : l.group,
                    name = length(ls) == 1 ? name : "$name.$i")
    end
    return out
end

"""
The strides a flash launch passes, and the 1-D form of an operand it indexes
through.

Two spellings of one fact, and which one applies is decided by what the operand
IS rather than by a flag: an ARRAY answers from its own strides and reshapes,
and a DECLARED resource is dense by construction (`viewstrides` materialises any
view that moves elements) so its strides come from `size` and it is already
addressed linearly. Written here so the immediate launch and the declared one
cannot disagree about the addressing, which would be a silently wrong answer
rather than an error.
"""
flashstrides(a) = map(Int32, strides(a))
flashstrides(x::Union{M.Buffer,M.TransientBuffer,M.ResourceView,M.BufferRange}) =
    declstrides(x)
flashflat(r) = reshape(r, length(r))
flashflat(x::Union{M.Buffer,M.TransientBuffer,M.ResourceView,M.BufferRange}) = x
flashoutflat(r) = reshape(r, length(r))
flashoutflat(x::Union{M.Buffer,M.TransientBuffer,M.ResourceView,M.BufferRange}) =
    M.viewof(x, (length(x),))

"""
    sdpaflashcm!(ctx, out, q, k, v, scale; kw...) -> Bool

Plan and run in one call, for direct callers that have no plan in hand — which is
every A/B in `test_flash.jl`. `false` means the keywords describe a launch this
device cannot make; ask [`flashcm_plan`](@ref) directly to find out *which* rule
refused.
"""
function sdpaflashcm!(ctx, out, q, k, v, scale; ballast::Int = 0, shpad::Int = 0,
                      nrsc::Int = 3, preonly::Bool = false, rscbar::Bool = false,
                      epad::Union{Nothing,Int} = nothing,
                      rpad::Union{Nothing,Int} = nothing,
                      BR::Int = 64, BC::Int = 32, NW::Int = 8, kw...)
    # The shipped tiling by default rather than the chooser's pick, because this
    # form exists for A/Bs that name their own and the ones that do not were
    # written against these three numbers.
    plan = flashcm_plan(ctx.dev, q, k, v, nothing; BR, BC, NW, kw...)
    plan isa Decline && return false
    sdpaflashcm!(ctx, out, plan, q, k, v, scale;
                 ballast, shpad, nrsc, preonly, rscbar,
                 epad = something(epad, flashepad(ctx.dev, plan.EP)),
                 rpad = something(rpad, flashrpad(ctx.dev, plan.BR)))
    return true
end
