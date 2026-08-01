"""
Fused attention: one kernel, no score matrix.

**Two kernels live here.** `attn_flash_cm!` at the bottom is the one `sdpa` runs:
it does both products on the tensor cores and is 1.9x the two-GEMM path on the
encoder's dominant shapes. `attn_flash!` immediately below is the scalar form,
kept and still tested but not routed to.

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
  * ~~**`BQ = 32` is wrong and unexplained.**~~ **Explained and fixed.** It was
    `OpUDiv` in the staging index — `E = 72` is not a power of two, so
    `idx % E` emitted a real division into a shared-store address, which drops
    stores on this driver. `Lava.splitidx` removes it and every configuration is
    exact; `flashfits` no longer refuses odd slot counts. See the note there.
    **This was the gate, and it is open**, so the `BQ = 32` route to two
    workgroups is available to try.

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
    FLASH_SHARED_BUDGET[]

Shared memory a workgroup may claim, in bytes. 48 KB is the floor every Vulkan
implementation guarantees.

Enforced rather than assumed: a tiling that asks for more does not fail to
launch, it launches and **writes nothing**, which reads as an attention that
returns zeros. `BQ = BK = 64` at `E = 72` wants 70 KB and does exactly that.
"""
const FLASH_SHARED_BUDGET = Ref(48 * 1024)

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
@inline function flashfits(E::Int, BQ::Int, BK::Int, NT::Int)
    (BQ * E) % NT == 0 && (BK * E) % NT == 0 && (BQ * BK) % NT == 0 || return false
    BQ <= NT || return false
    flashshared(E, BQ, BK) <= FLASH_SHARED_BUDGET[] || return false
    # There used to be an `iseven(div(BQ * E, NT))` refusal here, because
    # `BQ = 32, NT = 256` computed wrong results from query row 4 on (9.4e-02)
    # while every even slot count was exact — including `BQ = 32` at `NT = 128`,
    # which is why it was recorded as "the odd slot count" rather than as `BQ`.
    #
    # **It was `OpUDiv`.** The staging loops decomposed a flat index with
    # `idx % E` / `idx ÷ E`, and `E = 72` is not a power of two, so a real
    # division landed in a shared-memory store address — which drops stores on
    # this driver (`Lava.splitidx`, and `test_shared_index_division.jl` for the
    # isolated case). The tell was in the old note without being recognised: it
    # was "exact for *constant* inputs", and that is precisely the condition
    # under which the division bug does not bite, because a constant store needs
    # no global load to feed it. Same session, same inputs, only the arithmetic:
    #
    #   BQ/NT    BQ*E/NT      OpUDiv    splitidx
    #   64/256   18 even     7.8e-07     7.1e-07
    #   32/256    9 odd      7.1e-02     8.6e-07
    #   32/128   18 even     8.4e-07     7.0e-07
    #   64/128   36 even     7.8e-07     6.8e-07
    #
    # The slot count was never the variable. With `splitidx` every configuration
    # is exact, so the refusal is gone and `BQ = 32` is available again — which
    # matters because it is one of the two routes to two workgroups per SM.
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
            e, lq = Lava.splitidx(idx, Val(E)); e += 1; lq += 1
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
                e, lk = Lava.splitidx(idx, Val(E)); e += 1; lk += 1
                ks[e, lk] = T(k[e, k0 + lk, h, b])
                vs[e, lk] = T(v[e, k0 + lk, h, b])
            end
            @synchronize

            # scores for the tile
            for r in 0:(div(BQ * BK, NT) - 1)
                idx = tid + r * NT - 1
                qi, ki = Lava.splitidx(idx, Val(BQ)); qi += 1; ki += 1
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
                e, lq = Lava.splitidx(idx, Val(E)); e += 1; lq += 1
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
            e, lq = Lava.splitidx(idx, Val(E)); e += 1; lq += 1
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
                    BQ::Int = 64, BK::Int = 32, NT::Int = 256)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    (Lq % BQ == 0 && Lk % BK == 0 && flashfits(E, BQ, BK, NT)) || return false
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

"""Bytes of `@localmem` the cooperative-matrix kernel needs."""
@inline flashcmshared(EP, BR, BC) =
    2 * EP * BR + 2 * EP * BC + 4 * BR * BC + 2 * BC * BR + 4 * BR * EP + 12 * BR

"""
Whether a `(BR, BC)` tiling is one the cooperative-matrix kernel can run.
"""
@inline function flashcmfits(EP::Int, BR::Int, BC::Int, NT::Int)
    BR % Lava.GEMM_TILE == 0 && BC % Lava.GEMM_TILE == 0 && EP % Lava.GEMM_TILE == 0 || return false
    # The softmax gives one thread a whole query row.
    BR <= NT || return false
    (BR * EP) % NT == 0 && (BC * EP) % NT == 0 && (BR * BC) % NT == 0 || return false
    flashcmshared(EP, BR, BC) <= FLASH_SHARED_BUDGET[]
end

"""
    attn_flash_cm!(out, q, k, v, scale, Val(BR), Val(BC), Val(E), Val(EP), Val(NW), Lk)

One workgroup owns `BR` queries and walks the keys in blocks of `BC`.

`EP` is `E` rounded up to the tile — 72 becomes 80 — and the staging loops write
zero past `E`. That padding is free in the arithmetic (`0 * 0` contributes
nothing to `S`, and `O`'s columns past `E` are never read out) and it is what
lets every cooperative-matrix load be full-width.

`Lava.splitidx` for every staging index, not `%`/`÷`: `EP = 80` and `E = 72` are
not powers of two, and a real `OpUDiv` in a shared-memory store address drops
stores on this driver whenever a `muladd` is in scope. That is the bug this
kernel would otherwise walk straight into — see `test_shared_index_division.jl`.
"""
@kernel cpu=false unsafe_indices=true function attn_flash_cm!(
        out, @Const(q), @Const(k), @Const(v), scale,
        ::Val{BR}, ::Val{BC}, ::Val{E}, ::Val{EP}, ::Val{NW}, ::Val{REGO},
        Lk::Int32, alwaysrescale::Int32) where {BR,BC,E,EP,NW,REGO}
    NT = NW * 32
    qs  = @localmem Float16 (EP * BR,)      # (e, r) at r*EP + e
    kvs = @localmem Float16 (EP * BC,)      # (e, c) at c*EP + e — K, then V
    ss  = @localmem Float32 (BR * BC,)      # (r, c) at c*BR + r
    ps  = @localmem Float16 (BC * BR,)      # (r, c) at r*BC + c
    # `REGO == false`: this is `O`, and it persists across key blocks.
    # `REGO == true`:  this is one key block's `P·V`, and `O` lives in `acco`.
    pvs = @localmem Float32 (BR * EP,)      # (r, e) at e*BR + r
    ms  = @localmem Float32 (BR,)
    ls  = @localmem Float32 (BR,)
    cs  = @localmem Float32 (BR,)
    # Did any row's running max move this block? One word, and it decides
    # whether `O` has to be rescaled at all — see the loop below.
    grew = @localmem Float32 (1,)

    # `O` in registers: `BR*EP/NT` floats per thread, 20 for the shipped tiling.
    # Written out as `Float32` rather than through a local `T`, because Lava
    # miscompiles a `@private` whose element type comes from a local binding and
    # does it silently — the kernel runs and writes nothing. `NW * 32` spelled
    # out rather than the local `NT` below for the same reason: the size has to
    # come from the type parameters.
    acco = @private Float32 (div(BR * EP, NW * 32),)

    RT = BR ÷ Lava.GEMM_TILE
    CT = BC ÷ Lava.GEMM_TILE
    ET = EP ÷ Lava.GEMM_TILE

    tid = @index(Local, Linear) - 1
    grp = @index(Group, NTuple)
    qb, h, b = grp[1], grp[2], grp[3]
    w = tid ÷ 32                            # subgroup within the workgroup

    @inbounds begin
        q0 = (qb - 1) * BR

        # Q for this block, once, and it stays in shared for every key block.
        for r in 0:(div(BR * EP, NT) - 1)
            idx = tid + r * NT
            e, lq = Lava.splitidx(idx, Val(EP))
            qs[1 + idx] = e < E ? q[1 + e, 1 + q0 + lq, h, b] : zero(Float16)
        end
        if REGO
            for s in 1:div(BR * EP, NT)
                acco[s] = 0.0f0
            end
        else
            for r in 0:(div(BR * EP, NT) - 1)
                pvs[1 + tid + r * NT] = 0.0f0
            end
        end
        if tid < BR
            ms[1 + tid] = -Inf32
            ls[1 + tid] = 0.0f0
        end
        @synchronize

        for kb in 0:(div(Lk, BC) - 1)
            k0 = kb * BC
            tid == 0 && (grew[1] = Float32(alwaysrescale))
            for r in 0:(div(BC * EP, NT) - 1)
                idx = tid + r * NT
                e, lk = Lava.splitidx(idx, Val(EP))
                kvs[1 + idx] = e < E ? k[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
            end
            @synchronize

            # S = Q·Kᵀ. RT*CT tiles handed round the subgroups; the trip count is
            # uniform within a subgroup, which is what a coopmat op requires.
            for t in w:NW:(RT * CT - 1)
                rt = t % RT
                ct = t ÷ RT
                acc = zero(Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator})
                for et in 0:(ET - 1)
                    a = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                            qs, 1 + rt * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(true))
                    bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                            kvs, 1 + ct * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(false))
                    acc = muladd(a, bm, acc)
                end
                copyto!(ss, 1 + rt * Lava.GEMM_TILE + ct * Lava.GEMM_TILE * BR, BR, acc)
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
            # `Lava.copyto!`), give each *subgroup* a query row and each lane a
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
            if tid < BR
                mo = ms[1 + tid]
                mb = -Inf32
                for ci in 0:(BC - 1)
                    mb = max(mb, ss[1 + tid + ci * BR] * scale)
                end
                mn = max(mo, mb)
                # A row that has seen nothing finite must not make NaN out of
                # exp(-Inf - -Inf); it stays at zero weight.
                cr = isfinite(mo) ? exp(mo - mn) : 0.0f0
                sm = 0.0f0
                for ci in 0:(BC - 1)
                    p = exp(ss[1 + tid + ci * BR] * scale - mn)
                    ps[1 + ci + tid * BC] = Float16(p)
                    sm += p
                end
                ms[1 + tid] = mn
                ls[1 + tid] = ls[1 + tid] * cr + sm
                cs[1 + tid] = cr
                # `cr == 1` exactly when this block's max did not beat the
                # running one, and then rescaling `O` multiplies it by one. The
                # write races with the other rows and that is fine: every thread
                # that writes writes the same value.
                cr == 1.0f0 || (grew[1] = 1.0f0)
            end
            @synchronize

            # Refill the staging buffer with V, and — when `O` lives in shared —
            # apply this block's rescale to it. Both sit between the same pair of
            # barriers because the score product is the last thing that read
            # `kvs`, and `pvs` is not touched again until below.
            # **Only when some row's max actually moved.** This pass reads and
            # writes the whole of `O` — 40 KB of the ~110 KB of shared traffic a
            # key block costs — and after the first few blocks the running max
            # has usually settled, so most blocks multiply `O` by one from end to
            # end. The flag makes that case free.
            if !REGO && grew[1] != 0.0f0
                for r in 0:(div(BR * EP, NT) - 1)
                    idx = tid + r * NT
                    lq, _ = Lava.splitidx(idx, Val(BR))
                    pvs[1 + idx] *= cs[1 + lq]
                end
            end
            for r in 0:(div(BC * EP, NT) - 1)
                idx = tid + r * NT
                e, lk = Lava.splitidx(idx, Val(EP))
                kvs[1 + idx] = e < E ? v[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
            end
            @synchronize

            for t in w:NW:(RT * ET - 1)
                rt = t % RT
                et = t ÷ RT
                off = 1 + rt * Lava.GEMM_TILE + et * Lava.GEMM_TILE * BR
                # Starting from `O` itself means the accumulate is the tensor
                # core's own; starting from zero means the registers below do it.
                acc = REGO ?
                    zero(Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator}) :
                    Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator}(
                        pvs, off, BR, Val(false))
                for ct in 0:(CT - 1)
                    a = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                            ps, 1 + rt * Lava.GEMM_TILE * BC + ct * Lava.GEMM_TILE, BC, Val(true))
                    bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                            kvs, 1 + ct * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(true))
                    acc = muladd(a, bm, acc)
                end
                copyto!(pvs, off, BR, acc)
            end
            @synchronize

            if REGO
                # `O = O*c + PV` in registers. `pvs` has the same `(BR, EP)`
                # shape the accumulator stored into, so this is a flat index
                # with no tile arithmetic.
                for s in 1:div(BR * EP, NT)
                    idx = tid + (s - 1) * NT
                    lq, _ = Lava.splitidx(idx, Val(BR))
                    acco[s] = acco[s] * cs[1 + lq] + pvs[1 + idx]
                end
                @synchronize
            end
        end

        # Slots `1 : BR*E/NT` are exactly the ones whose `e` is inside the real
        # head dimension: `idx = tid + (s-1)*NT` and `NT` divides `BR*E`, so the
        # padded columns are all in the slots past that and never written out.
        for s in 1:div(BR * E, NT)
            idx = tid + (s - 1) * NT
            lq, e = Lava.splitidx(idx, Val(BR))
            l = ls[1 + lq]
            o = REGO ? acco[s] : pvs[1 + idx]
            out[1 + e, 1 + q0 + lq, h, b] = o / (l == 0.0f0 ? 1.0f0 : l)
        end
    end
end

"""
    FLASHCM[]

Whether `sdpa` may take the cooperative-matrix fused path.

A `Ref` rather than a constant because it is the only way to measure the two
paths **inside one session**, which is the only measurement this project trusts.
"""
const FLASHCM = Ref(true)

"""
    FLASHCM_DENSIFY[]

Whether `sdpa` copies `k` and `v` into dense scratch before the fused kernel.

They arrive as a `PermutedDimsArray` over a `ReshapedArray`, which costs four
integer divisions per element read, and flash reads all of `k` and `v` once per
*query block* — 64 times over for a 4096-query global block. The copy is two
passes; not making it is 64 passes of wrapper arithmetic on top of reads that
happen anyway.

**Not an optimisation, a precondition.** Encode, interleaved, one process:

    coopmat path (replaced)   162.75 ms
    flash, k/v densified      137.64
    flash, k/v as they arrive 169.42   <- slower than the path it replaces

Without the copy the fused kernel *loses* to the two-GEMM path it is meant to
beat, and the entire difference is index arithmetic on operands whose reads
happen either way. The two-GEMM path never had this problem because its padding
kernels read every element exactly once.

`q` is deliberately not copied: each workgroup stages its own `BR` queries and no
other workgroup touches them, so `q` is read once and a copy would be a whole
extra pass to save nothing.

A `Ref` because the trade reverses with the block count: at `Lq = 256` and
`BR = 64` there are four query blocks, not sixty-four.
"""
const FLASHCM_DENSIFY = Ref(true)

"""
    FLASHCM_REGO[]

Where `O` lives across key blocks: `true` puts it in registers, `false` in
shared memory as the value product's accumulator.

**`false`, and that is the opposite of the reference.** `flash_attn_cm1.comp`
keeps `O` in registers and round-trips each `P·V` tile through shared memory to
add it. The shared form does twice the shared traffic on paper — a rescale pass
that reads and writes `BR x EP`, plus an accumulator load on top of the store,
80 KB a key block against 40 — and loses anyway. See the measurement in
`FLASHCM_TILINGS`.

Kept as a switch rather than deleted, because the paper argument for registers is
sound. Interleaved, one session, both forms of the same kernel:

    tiling      floats/thread    shared      registers
    64x32/8w         20          5.032 ms     6.347 ms     <- shipped
    64x32/8w         20          0.457        0.537
    32x32/8w         10          7.284        6.696
    32x32/8w         10          0.567        0.560

**The register form wins at `32x32` and loses at `64x32`, crossing where
`BR*EP/NT` goes from 10 floats a thread to 20.** The obvious reading is register
pressure, and the driver says it is not:
`VK_KHR_pipeline_executable_properties` reports **128 registers for the shared
form and 122 for the register form, stack size 0 in both** — the register form
uses *fewer* registers and neither spills.

What it actually costs is the pass that replaces the accumulator. Keeping `O` in
shared makes the update two cooperative-matrix memory ops, which move a 16x16
tile in the hardware's own fragment layout. Keeping it in registers replaces them
with a scalar sweep over `BR*EP` — a `splitidx` per element, per key block, to
recover `(row, e)` from the flat index — plus a barrier to publish `P·V` first.
That is `BR*EP/NT` index decompositions a thread a block, which is exactly the
quantity the crossing point is measured in.

So the reference is not wrong; it is written for a compiler where `O`'s home is
plain registers indexed by an unrolled constant, not one where getting at the
same value costs a division. If the sweep ever becomes free the switch should be
flipped back.
"""
const FLASHCM_REGO = Ref(false)

"""
    FLASHCM_LAZYRESCALE[]

Skip the `O *= exp(m_old - m_new)` pass on key blocks where no row's running
maximum moved.

The online softmax rescales the accumulator whenever a block contains a larger
score than anything seen so far. That is a read and a write of the whole of `O`
— 40 KB of the ~110 KB of shared traffic a key block costs, and `O` is the
single largest consumer of it. But the rescale factor is `exp(m_old - m_new)`,
which is **exactly one** whenever the block's max did not beat the running one,
and after the first few blocks the max has usually settled: the pass then
multiplies `O` by one from end to end.

One shared word per workgroup records whether any row grew. The threads racing
to set it all write the same value, and the branch is uniform because every
thread reads it after the same barrier.

Exact, not approximate — the skipped work is a multiplication by one.
"""
const FLASHCM_LAZYRESCALE = Ref(true)

"""
    FLASHCM_TILINGS

`(BR, BC, NW)`, fastest first. `NW` is subgroups, so the workgroup is `32*NW`.

Measured on SAM 2's two dominant attention shapes, clock warmed, interleaved,
against the two-GEMM cooperative-matrix path:

    tiling        4096x4096      256x256
    coopmat        9.50 ms       0.883 ms     (the path this replaces)
    64x32/8w       5.09          0.468        <- shipped default
    32x16/8w       5.87          0.471
    64x16/8w       5.85          0.499
    32x32/8w       7.36          0.567
    64x16/4w       7.34          0.633
    32x64/2w      24.71          1.717

**Subgroups dominate every other parameter.** The same `64x32` block goes 7.34 ->
5.09 on four warps against eight, and `32x64` at two warps is the worst thing
measured — five times the best. That is the same result the GEMM tuning reached
from the other direction, where widening the warp *grid* won and widening the
warp *tile* lost: what this device wants is warps in flight.

`64x64` at any width wants 66 KB and is refused; `64x32` asks 48 896 bytes of the
48 KB budget, with 256 to spare.

What the driver reports for the shipped tiling
(`VK_KHR_pipeline_executable_properties`):

    Register Count 128    Shared Memory Size 48 900    Stack Size 0

Nothing spills, and **two workgroups fit per SM, capped by both resources at
once**: 256 threads x 128 registers is 32 768 of the SM's 65 536, and 48 900
bytes is two of Ada's ~100 KB of shared. 512 of 1 536 resident threads. So
shrinking the tile alone buys no occupancy — registers cap it at two
independently, and a third workgroup needs both under 34 KB and under 85
registers a thread.
"""
const FLASHCM_TILINGS = [(64, 32, 8), (32, 32, 8), (64, 16, 8), (32, 16, 8),
                         (32, 32, 4), (16, 32, 4)]

"""
    flashcm_tiling(E, Lq, Lk) -> (BR, BC, NW) | nothing

The first tiling in [`FLASHCM_TILINGS`](@ref) that divides this shape and fits.
`nothing` means the caller keeps whatever path it would otherwise have used.
"""
function flashcm_tiling(E::Int, Lq::Int, Lk::Int)
    EP = cld(E, Lava.GEMM_TILE) * Lava.GEMM_TILE
    for (BR, BC, NW) in FLASHCM_TILINGS
        NT = NW * 32
        NT <= Lava.WORKGROUP_LIMIT[] || continue
        Lq % BR == 0 && Lk % BC == 0 || continue
        flashcmfits(EP, BR, BC, NT) && (BR * E) % NT == 0 && return (BR, BC, NW)
    end
    nothing
end

"""
    flashcm_applicable(q, k, v, bias, Lq, Lk) -> Bool

Whether [`sdpa`](@ref) may fuse this call.

`bias` must be absent for the same reason the two-GEMM path refuses it: the mask
would have to be added between the score product and the softmax, and here that
is inside a cooperative-matrix accumulator.
"""
function flashcm_applicable(q, k, v, bias, Lq::Int, Lk::Int)
    FLASHCM[] || return false
    bias === nothing || return false
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 || return false
    Lava.coopmat_gemm_available() || return false
    flashcm_tiling(size(q, 1), Lq, Lk) !== nothing
end

"""
    sdpaflashcm!(out, q, k, v, scale; backend, BR, BC, NW) -> Bool

Run the cooperative-matrix fused kernel, or return `false` when the shape or the
device does not admit it. `q`, `k`, `v` are `(E, L, H, B)`.
"""
function sdpaflashcm!(out, q, k, v, scale; backend = KernelAbstractions.get_backend(q),
                      BR::Int = 64, BC::Int = 32, NW::Int = 8, rego::Bool = FLASHCM_REGO[],
                      lazyrescale::Bool = FLASHCM_LAZYRESCALE[])
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    EP = cld(E, Lava.GEMM_TILE) * Lava.GEMM_TILE
    NT = NW * 32
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 || return false
    Lava.coopmat_gemm_available() || return false
    (Lq % BR == 0 && Lk % BC == 0 && flashcmfits(EP, BR, BC, NT)) || return false
    # `BR * E` must also tile the write-out loop, which `flashcmfits` cannot check
    # because it does not see the unpadded head dimension.
    (BR * E) % NT == 0 || return false
    attn_flash_cm!(backend, NT)(out, q, k, v, Float32(scale),
                                Val(BR), Val(BC), Val(E), Val(EP), Val(NW), Val(rego),
                                Int32(Lk), Int32(lazyrescale ? 0 : 1);
                                ndrange = (NT * div(Lq, BR), H, B))
    return true
end
