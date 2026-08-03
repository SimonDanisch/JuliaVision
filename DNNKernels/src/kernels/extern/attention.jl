"""
Scaled dot-product attention: hand-written, shared across models.

Covers both backends PyTorch picks from - `_scaled_dot_product_efficient_attention`
(takes an additive `attn_bias`) and `_scaled_dot_product_flash_attention` (no bias,
explicit scale). Under autocast the same graph contains both, one per call site.

PyTorch's `(B, H, L, E)` is `(E, L, H, B)` here, so a head is a contiguous
`(E, L)` slab and the head/batch axes are the slowest.

    scores[lk, lq] = Σₑ q[e, lq] k[e, lk] * scale   (+ bias)
    p              = softmax over lk
    out[e, lq]     = Σ_lk p[lk, lq] v[e, lk]

Three passes rather than one fused kernel: correctness first, and the scores
buffer is a declared transient the arena can size. Accumulation is fp32 for half
operands, as both reference backends do.

The row-max subtraction makes an all-blocked row produce zeros instead of NaN,
which is why upstream's degenerate-row fixup (patches.py) only has to keep the
mask sane, not the softmax.
"""

"""
    toLE(I, a)

`(E, L, H, B)` read as `(L, E, H, B)` — the transpose that makes the score
kernel's reads coalesce.

The graph's layout puts the head dimension first, so `k[e, lk]` for consecutive
`lk` is `E` floats apart: in `attn_scores` consecutive threads vary `lk`, so
every warp asked for 32 words spread over 32 cache lines and used one word of
each. The global attention measured 215 GFLOP/s at 12 GB/s — neither compute nor
bandwidth bound, just paying 72x for every byte. Transposing q and k once (9 MB
each, against a 536 MB score matrix) makes the inner loop read contiguously.
"""
@inline function toLE(I, a)
    l, e, h, b = I
    @inbounds a[e, l, h, b]
end

# `scores` is `(Lq, Lk, H, B)`, not `(Lk, Lq, H, B)`, for the same reason: the
# softmax runs one thread per query row and consecutive threads must land on
# consecutive addresses. With q/k transposed and scores in this order, all three
# kernels below read and write contiguously.
@inline function attn_scores(I, q, k, bias, scale)
    lq, lk, h, b = I
    @inbounds begin
        T = accum(eltype(q))
        acc = zero(T)
        for e in axes(q, 2)
            acc = muladd(T(q[lq, e, h, b]), T(k[e, lk, h, b]), acc)
        end
        acc *= T(scale)
        # `bias` keeps the graph's own `(Lk, Lq, H, B)` order — it is an input,
        # not something this op is free to lay out, and only the mem-efficient
        # variant has one at all.
        bias === nothing ? acc :
            acc + T(bias[bidx(bias, CartesianIndex(lk, lq, h, b))])
    end
end

"""
One thread per query row; the reduction over keys is sequential in-thread.

Reads and writes `scores` in whatever type it is stored as, but takes the maximum
and the sum in `accum` of it. The store type is the operands' — fp16 under
autocast — because this tensor is the whole cost of attention: 536 MB per global
block in SAM 2, written once and read twice. Halving it halves the traffic of the
only bandwidth-bound part of the op. The arithmetic stays fp32 throughout, which
is what the reference does too; only the *storage* is narrow.
"""
@inline function attn_softmax(I, scores)
    lq, h, b = I
    @inbounds begin
        S = eltype(scores)
        T = accum(S)
        m = typemin(T)
        for lk in axes(scores, 2)
            m = max(m, T(scores[lq, lk, h, b]))
        end
        if !isfinite(m)                       # every key blocked
            for lk in axes(scores, 2)
                scores[lq, lk, h, b] = zero(S)
            end
            return one(T)                     # a zero row, divided by one
        end
        s = zero(T)
        for lk in axes(scores, 2)
            e = exp(T(scores[lq, lk, h, b]) - m)
            scores[lq, lk, h, b] = S(e)
            s += e
        end
        # NOT normalised here. Scaling the row in place is a third full pass over
        # a tensor that is 536 MB in SAM 2's global attention; `attn_apply`
        # already touches every element once and divides its accumulated result
        # by this sum instead, which is the same number for one pass less.
        s
    end
end

@inline function attn_apply(I, p, v, sums)
    e, lq, h, b = I
    @inbounds begin
        T = accum(eltype(v))
        acc = zero(T)
        for lk in axes(p, 2)
            acc = muladd(T(p[lq, lk, h, b]), T(v[e, lk, h, b]), acc)
        end
        acc / T(sums[lq, h, b])
    end
end

"""
Register-blocked score and apply kernels: one thread computes `T` outputs.

The one-output-per-thread versions above are correct and bandwidth-starved. In
`attn_scores` a thread reads `E` values of `q` and `E` of `k` to produce a single
score, so the whole 38.7 GFLOP of SAM 2's global attention moves ~40 GB through
L1/L2 — 271 GFLOP/s on a card that will do far more, limited by cache bandwidth
rather than arithmetic.

Giving each thread `TK` consecutive keys amortises the `q` load across all of
them: per warp per `e` the traffic goes from 132 bytes for 32 outputs to
`128 + 4TK` bytes for `32TK`, which at `TK = 8` is 6.6x less. `attn_apply` has
the same shape with the roles swapped, so it blocks over queries instead.

`TK` is a `Val` so the accumulator tuple unrolls into registers, and the block
only runs when it divides the extent — the tail is not worth a bounds check in
the inner loop when falling back to the simple kernel is exact.
"""
const ATTN_BLOCKS = (2, 4, 8, 16, 32)

# One kernel per block size, generated rather than parameterised on `Val{TK}`.
# The obvious form — a tuple accumulator built with `ntuple(..., Val(TK))` —
# does not compile: the closure over the previous tuple defeats inference inside
# a `@kernel` body and every `getindex`/`muladd` in it becomes a dynamic call.
# `@nexprs` needs a literal, so the literal is supplied here and each accumulator
# is a plain local the compiler keeps in a register.
for TK in ATTN_BLOCKS
    @eval @kernel function $(Symbol("attn_scores_b", TK, "!"))(scores, @Const(q), @Const(k),
                                                               bias, scale)
        lq, kb, h, b = @index(Global, NTuple)
        @inbounds begin
            T = accum(eltype(q))
            base = (kb - 1) * $TK
            Base.Cartesian.@nexprs $TK t -> acc_t = zero(T)
            for e in axes(q, 2)
                qv = T(q[lq, e, h, b])
                Base.Cartesian.@nexprs $TK t -> acc_t = muladd(qv, T(k[e, base + t, h, b]), acc_t)
            end
            Base.Cartesian.@nexprs $TK t -> begin
                s_t = acc_t * T(scale)
                bias === nothing ||
                    (s_t += T(bias[bidx(bias, CartesianIndex(base + t, lq, h, b))]))
                scores[lq, base + t, h, b] = s_t
            end
        end
    end

    @eval @kernel function $(Symbol("attn_apply_b", TK, "!"))(out, @Const(p), @Const(v),
                                                              @Const(sums))
        e, qb, h, b = @index(Global, NTuple)
        @inbounds begin
            T = accum(eltype(v))
            base = (qb - 1) * $TK
            Base.Cartesian.@nexprs $TK t -> acc_t = zero(T)
            for lk in axes(p, 2)
                vv = T(v[e, lk, h, b])
                Base.Cartesian.@nexprs $TK t -> acc_t = muladd(T(p[base + t, lk, h, b]), vv, acc_t)
            end
            Base.Cartesian.@nexprs $TK t -> out[e, base + t, h, b] = acc_t / T(sums[base + t, h, b])
        end
    end
end

"""
    blockfor(n, other; minl = 64) -> block size

Register-block size for a sequence of length `n` against `other`, or 1 for none.

`minl` is the shortest sequence for which blocking is worth it.

Blocking divides the *launched* extent by the block size, and the blocked axis is
the ndrange's second dimension while the first is the sequence length. Below this
the grid stops being able to fill a warp along the fast axis and the blocked
kernel is much slower than the plain one — measured on this card, a
`(72, 16, 4, 1024)` attention goes 2.2 -> 26.8 ms at `TK = 8` while
`(72, 256, 8, 16)` goes 8.9 -> 2.2 ms. 64 sits between the two measured regimes.


Largest generated block that divides `n`, or 1 when blocking does not apply.

`other` is the *opposite* extent, and the block only applies to **self**
attention, `Lq == Lk`. That is what the image encoder does at every one of its 48
attentions; the mask decoder instead has a 23-token prompt attending to 4096
image tokens and back. With those lopsided shapes blocked a decode stops
completing — the queue accumulates in-flight batches until `vkWaitSemaphores`
times out, reproducibly, and reproducibly not when they take the plain path.
That is a bug in the blocked kernels, or in what Lava makes of them at those
extents, and it is bounded rather than diagnosed here; the shapes that need the
speed are square, so the encoder keeps the whole win.
"""
@inline function blockfor(n, other, minl::Int = 64)
    # `n != other` is a BUG WORKAROUND, not a design choice, and it is the one
    # thing in this function that is not about speed: blocking the decoder's
    # lopsided (non-square) shapes reproducibly hangs on `vkWaitSemaphores`, with
    # the queue accumulating in-flight batches until it times out. See the
    # docstring above. The shapes that need the speed are square, so the encoder
    # keeps the whole win — but this is an open bug sitting inside a predicate,
    # and it should be re-tested rather than inherited.
    (n != other || n < minl) && return 1
    for t in reverse(ATTN_BLOCKS)
        n % t == 0 && return t
    end
    1
end

"""
Dispatch to the generated kernel for block size `tk`.

`workgroupsize` is given rather than left to KernelAbstractions for the reason
`launchgroup` documents, and it matters most here: `attn_apply`'s ndrange leads
with `E = 72`, which KA's 64-wide group splits into `64 + 8`, so every second
workgroup runs at 12% occupancy and the axis is only 56% utilised overall.
Taking the head dimension whole keeps a group on one contiguous run of `out`.
"""
# `launchgroup`, NOT `Lava.staticgroup`: the latter avoids interior unit extents
# so the workgroup can go in the kernel's type (2x on index-bound kernels), but
# for `attn_apply` it shapes `(64, 2, 2, 1)` and splits the 72-long head
# dimension into 64 + 8 — the exact fragmentation `launchgroup` exists to avoid.
# Measured: `attn_apply` 17.6 -> 21.6 ms. Contiguity wins here, folding does not.
#
# ── The dispatchers, GENERATED from `ATTN_BLOCKS` rather than written out ─────
#
# These were two hand-maintained `tk == 32 && return …` chains sitting beside the
# loop that generates the kernels, which is `kernel-library-review.md` finding 5:
# adding a block size to `ATTN_BLOCKS` generated a kernel and silently did not
# dispatch to it, so the new size was dead code that read as live. Folding over
# the same tuple means the two cannot disagree.
#
# Still a chain and not `Val` dispatch, deliberately. `tk` comes from `blockfor`
# at runtime, so `f(Val(tk), …)` would buy a dynamic dispatch on every launch to
# save four integer comparisons. Finding 5 asked for the dispatcher to be
# *generated*, not for it to become a method table.
#
# The *kernels* genuinely cannot be `Val`-parameterised — see the note above
# `ATTN_BLOCKS`: the closure over the accumulator tuple defeats inference and
# every `muladd` becomes a dynamic call. That constraint is about the kernel
# body, not about this.
for (name, kern, args) in (("scoresblocked!", "attn_scores_b", (:scores, :q, :k, :bias, :scale)),
                           ("applyblocked!",  "attn_apply_b",  (:out, :p, :v, :sums)))
    # Largest first, smallest as the unguarded fallback — the shape that was
    # written by hand, minus the opportunity to forget an entry.
    guarded = [:(tk == $B && return $(Symbol(kern, B, "!"))(backend)($(args...);
                                                                    ndrange, workgroupsize = wg))
               for B in reverse(ATTN_BLOCKS)[1:end-1]]
    fallback = Symbol(kern, first(ATTN_BLOCKS), "!")
    @eval function $(Symbol(name))(backend, tk, $(args...), ndrange)
        wg = launchgroup(ndrange)
        $(guarded...)
        $(fallback)(backend)($(args...); ndrange, workgroupsize = wg)
    end
end

"""
    densify(ctx, a) -> a, dense

`a` itself when it is already dense, and a workspace copy of it when it is not.

Attention's operands arrive as a `PermutedDimsArray` over a `ReshapedArray` over
a `SubArray` — the shape of `q.transpose(1, 2)` after a `view` in the exported
graph. A `ReshapedArray` carries a `SignedMultiplicativeInverse` per axis, so
**every element read costs four integer divisions**, and `attn_scores` reads a
`q` and a `k` element per multiply-accumulate while `attn_apply` reads `v` once
per output element. Copying first pays the index arithmetic once per element
rather than `Lk` times, for a copy that is tiny next to the score matrix it
feeds: 9 MB for `q` against 512 MB of scores in SAM 2's global attention.

`DenseArray` is the right test rather than a backend-specific one, because
`AbstractGPUArray <: DenseArray` — it admits both a device array and the host
`Array` the verification path uses, and rejects exactly the wrapper stack.
"""
@inline function densify(ctx, a)
    a isa DenseArray && return a
    d = scratch!(ctx, eltype(a), size(a)...)
    d .= a
    d
end

# `toLE` is a batched 2-D transpose written as an elementwise gather: consecutive
# threads vary `L` and read with stride `E`, so every warp issues 32 separate
# transactions. Measured 37.7 GB/s on a dense operand against ~300 for a copy, and
# the operand is never dense — attention's `q` arrives as a `PermutedDimsArray`
# over a `ReshapedArray`, whose `SignedMultiplicativeInverse`s add four integer
# divisions per element on top.
#
# Both problems have the same fix. Stage a 32x32 tile in shared memory so the
# read and the write are each coalesced, and take the operand as a base array
# plus STRIDES rather than as the wrapper — `strides()` answers for this stack
# (`(1, 1728, 72, 442368)` for SAM 2's windowed `q`), so the indexing becomes a
# dot product with no divisions and no wrapper type inside the kernel. That is
# the same trick `Lava`'s scalar GEMM already uses on its operands.
#
# Measured on SAM 2's windowed shape, `(72, 256, 8, 16)`: 0.250 -> 0.035 ms,
# 37.7 -> 266 GB/s, bit-identical output.

"""
    stridedroot(a) -> (root, offset) | nothing

Root dense array of a wrapper stack and the linear offset of `a[1, 1, ...]` in
it, or `nothing` when a layer is something this cannot account for.

The `SubArray` is why this returns an offset rather than just the root: the stack
attention's `q` arrives in is `PermutedDimsArray -> ReshapedArray -> SubArray ->
LavaArray`, and refusing the view meant every shape that matters fell back to the
gather while only the two that happened to wrap a bare array took the fast path.
Reshapes and permutes leave the base element alone; a view does not, and its
offset is `LinearIndices(parent)[first.(indices)...] - 1`.
"""
function stridedroot(a)
    off = 0
    p = a
    for _ in 1:8
        p isa Lava.LavaArray && return (p, off)
        if p isa SubArray
            P = parent(p)
            off += LinearIndices(P)[map(first, p.indices)...] - 1
            p = P
        elseif p isa Union{PermutedDimsArray, Base.ReshapedArray}
            p = parent(p)
        else
            return nothing
        end
    end
    nothing
end

# One kernel per element type: `@localmem` is miscompiled — silently, writing
# nothing — when its type comes from a local binding, so the type has to be a
# literal in the generated body.
for T in (Float16, Float32)
    @eval @kernel cpu=false function $(Symbol("toLE_tiled_", nameof(T), "!"))(
            d, @Const(src), base::Int32, sE::Int32, sL::Int32, sH::Int32, sB::Int32,
            E::Int32, L::Int32, nH::Int32)
        # 33, not 32: with 32 banks a 32-wide tile puts a whole column in one
        # bank and the transposed read serialises 32 ways.
        tile = @localmem $(nameof(T)) (33, 32)
        tx, ty = @index(Local, NTuple)
        gx, gy, gz = @index(Group, NTuple)
        e0 = Int32(gx - 1) * Int32(32)
        l0 = Int32(gy - 1) * Int32(32)
        hb = Int32(gz - 1)
        h = hb % nH + Int32(1)
        b = hb ÷ nH + Int32(1)
        @inbounds begin
            for j in Int32(0):Int32(7)
                e = e0 + Int32(tx)
                l = l0 + Int32(ty) + Int32(4) * j
                tile[tx, ty + 4j] = (e <= E && l <= L) ?
                    src[base + (e - Int32(1)) * sE + (l - Int32(1)) * sL +
                        (h - Int32(1)) * sH + (b - Int32(1)) * sB] : zero($(nameof(T)))
            end
            @synchronize
            for j in Int32(0):Int32(7)
                l = l0 + Int32(tx)
                e = e0 + Int32(ty) + Int32(4) * j
                (l <= L && e <= E) && (d[l, e, h, b] = tile[ty + 4j, tx])
            end
        end
    end
end

"""`(E, L, H, B)` operand as a dense `(L, E, H, B)` one in the workspace."""
function transposeLE(ctx, a)
    E, L, H, B = size(a)
    backend = ctx.backend
    d = scratch!(ctx, eltype(a), L, E, H, B)
    r = eltype(a) in (Float16, Float32) ? stridedroot(a) : nothing
    if r === nothing
        launch!(ctx, toLE, d, a)
        return d
    end
    root, off = r
    st = map(Int32, strides(a))
    k = eltype(a) === Float16 ? toLE_tiled_Float16! : toLE_tiled_Float32!
    # Workgroup in the kernel's TYPE, not as a keyword: it is the literal
    # `(32, 4, 1)` every time, so this costs exactly one extra SPIR-V module and
    # the index arithmetic folds to constants — 3.34 -> 2.01 ms in SAM 2's
    # encoder. Safe because `(32, 4, 1)`'s only unit extent is trailing; see
    # `Lava.interior_unit_workgroup`.
    k(backend, (32, 4, 1))(d, reshape(root, length(root)), Int32(off + 1),
                           st[1], st[2], st[3], st[4], Int32(E), Int32(L), Int32(H);
                           ndrange = (32 * cld(E, 32), 4 * cld(L, 32), H * B))
    d
end

# ---------------------------------------------------------- tensor-core path
#
# The three-pass kernels above are scalar: measured at 2.3 TFLOP/s on SAM 2's
# global attention, against the 13 the same device's cooperative-matrix GEMM
# sustains on the same product. Both halves of attention ARE matrix products —
# `S = qT k` and `O = P vT` — one per (head, batch), so `Lava.coopmat_gemm!`'s
# `nbatch` runs all of them in a single dispatch.
#
# Measured against the three-pass path, whole op including the padding copies
# and the softmax, same total token count throughout:
#
#     L=256 B=16   1.28 -> 1.59 ms   0.81x     <- LOSES
#     L=512 B=8    2.35 -> 2.15 ms   1.10x
#     L=1024 B=4   4.78 -> 3.23 ms   1.48x
#     L=2048 B=2   9.37 -> 5.33 ms   1.76x
#     L=4096 B=1  18.71 -> 9.36 ms   2.00x
#
# So it is gated on sequence length, not used everywhere: below `COOPMAT_MINL`
# the padding and the wider score matrix cost more than the tensor cores save.
# SAM 2 lands on both sides of that — its windowed blocks are L=256 and its
# three global blocks are L=4096.
#
# ── re-measured 2026-07-31: the crossover moved to 256 ──
#
# That threshold is a property of the GEMM underneath it, and the GEMM changed:
# staged cooperative-matrix tiling with `vec2` staging buffers took it from 20.6
# to 35.3 TFLOP/s, 1.68x. So the length at which tensor cores repay the padding
# and the doubled score matrix moved down, and it was re-measured rather than
# assumed — interleaved, clock warmed:
#
#     L=64  H16 B16   0.198 -> 0.280 ms   0.71x   <- still loses
#     L=128 H8  B32   0.700 -> 0.749 ms   0.93x   } a wash
#     L=128 H16 B16   0.645 -> 0.606 ms   1.06x   }
#     L=256 H8  B16   1.227 -> 0.847 ms   1.45x   <- now WINS, was 0.81x
#     L=512 H8  B4    1.159 -> 0.586 ms   1.98x
#     L=1024 H8 B2    2.231 -> 1.234 ms   1.81x
#
# 256 rather than 128 because 128 is a coin-flip and 256 is decisive. That moves
# **32 of SAM 2's 48 attention calls** — every windowed block — onto the tensor
# cores; only the L=64 and L=16 tails stay on the three-pass path.
#
# Worth noticing as a pattern: this constant was correct when it was written and
# silently went stale when something it depended on got faster. Any threshold
# separating two implementations has that property.

# The shortest sequence for which this path beats the three-pass kernels is 256
# as of the re-measurement above; it was 512 when the GEMM under it ran at 20.6
# TFLOP/s rather than 35.3. It is `coopmat_sdpa_plan`'s `minl`.

"""`(E,L,H,B)` read as `(L,EP,H,B)`, zero past `E`."""
@inline function toLEpad(I, a, E)
    l, e, h, b = I
    @inbounds e <= E ? a[e, l, h, b] : zero(eltype(a))
end

"""`(E,L,H,B)` read as `(EP,L,H,B)`, zero past `E`."""
@inline function padE(I, a, E)
    e, l, h, b = I
    @inbounds e <= E ? a[e, l, h, b] : zero(eltype(a))
end

"""`(L,EP,H,B)` accumulator back to `(E,L,H,B)`, normalised by the row sums."""
@inline function fromLEpad(I, c, sums)
    e, l, h, b = I
    @inbounds c[l, e, h, b] / sums[l, h, b]
end

"""
Softmax over keys, fp32 in and fp16 out, returning the row sums.

Two buffers rather than in place: the GEMM accumulates in fp32 and the second
GEMM needs an fp16 `A`, so the narrowing happens in the pass that already reads
and writes every element instead of in one of its own. `scale` is applied here
because `attn_scores` folds it into the multiply and a GEMM has nowhere to put
it. Like [`attn_softmax`](@ref) it does NOT normalise — `fromLEpad` divides.
"""
@inline function attn_softmax16(I, p, s, scale)
    lq, h, b = I
    @inbounds begin
        # The fp32 accumulator goes straight into `exp`. Rounding the scaled
        # score to fp16 first — which is where the three-pass path stores it, and
        # where PyTorch's autocast rounds — was tried on the theory that being
        # *more* precise than the reference is what moved the masks. It changed
        # nothing measurable: identical masks to five decimals and an identical
        # encoder output, because `Float16(e)` below already absorbs a
        # perturbation that small. Left out rather than carried.
        m = -Inf32
        for lk in axes(s, 2)
            m = max(m, Float32(s[lq, lk, h, b]) * scale)
        end
        isfinite(m) || (m = 0.0f0)
        acc = 0.0f0
        for lk in axes(s, 2)
            e = exp(Float32(s[lq, lk, h, b]) * scale - m)
            p[lq, lk, h, b] = Float16(e)
            acc += e
        end
        acc
    end
end

# ── the same softmax, with the reduction spread across threads ───────────────
#
# `attn_softmax16` above gives one thread the whole `Lk` reduction for its query
# row. That is coalesced — `lq` is the fast axis, so a warp reads 32 consecutive
# scores — but there are only `CH * H * B` rows, and on SAM 2's global attention
# that is 16 384 threads: **64 workgroups, ~22% of this card's thread slots**.
# Measured with `Lava.with_dispatch_timing`, those six dispatches were
# **18.08 ms, 3.0 ms each — the second-largest kernel family in the whole
# encode** — for a pass that only reads 268 MB and writes 134 MB.
#
# So keep the layout and split the reduction: a workgroup covers `SM_LQ`
# consecutive query rows × `SM_CH` chunks of the key axis, each thread reducing
# its own chunk, then two shared-memory reductions across the chunk index. Reads
# stay coalesced because `lq` still varies fastest within a warp, and the thread
# count goes up by `SM_CH`.
const ATTN_SM_LQ = 32          # query rows per workgroup — one warp's worth, coalesced
const ATTN_SM_CH = 8           # key-axis chunks per row; 32 × 8 = 256 threads


"""
    attn_softmax_rows!(p, sums, s, scale, nlk)

Softmax over the key axis of `s :: (Lq, Lk, HB)`, writing `exp` to `p` in fp16
and the row sums to `sums :: (Lq, HB)`. Does not normalise — `fromLEpad`
divides, exactly as [`attn_softmax16`](@ref) leaves it.

Two passes over `s`, as before: the maximum has to be known before any `exp`.
Both are chunked, so each pass reads `Lk / ATTN_SM_CH` elements per thread.
"""
@kernel cpu=false function attn_softmax_rows!(p, sums, @Const(s), scale::Float32,
                                              nlk::Int32)
    red = @localmem Float32 (ATTN_SM_LQ * ATTN_SM_CH,)
    t = @index(Local, Linear) - 1
    blk = @index(Group, Linear) - 1
    li = t % ATTN_SM_LQ                    # query row within the tile
    ci = t ÷ ATTN_SM_LQ                    # which chunk of the key axis
    ntile = size(s, 1) ÷ ATTN_SM_LQ
    lq = (blk % ntile) * ATTN_SM_LQ + li
    hb = blk ÷ ntile

    # ── pass 1: the row maximum
    m = -Inf32
    lk = ci
    @inbounds while lk < nlk
        m = max(m, Float32(s[lq + 1, lk + 1, hb + 1]) * scale)
        lk += ATTN_SM_CH
    end
    @inbounds red[t + 1] = m
    @synchronize
    # Reduce along the chunk index only: for a fixed `li` the partial results sit
    # `ATTN_SM_LQ` apart. The barrier is outside the branch, as it must be.
    stride = ATTN_SM_CH ÷ 2
    while stride > 0
        @inbounds if ci < stride
            red[t + 1] = max(red[t + 1], red[t + 1 + stride * ATTN_SM_LQ])
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds mx = red[li + 1]
    isfinite(mx) || (mx = 0.0f0)
    @synchronize

    # ── pass 2: exp into `p`, and the row sum
    acc = 0.0f0
    lk = ci
    @inbounds while lk < nlk
        e = exp(Float32(s[lq + 1, lk + 1, hb + 1]) * scale - mx)
        p[lq + 1, lk + 1, hb + 1] = Float16(e)
        acc += e
        lk += ATTN_SM_CH
    end
    @inbounds red[t + 1] = acc
    @synchronize
    stride = ATTN_SM_CH ÷ 2
    while stride > 0
        @inbounds if ci < stride
            red[t + 1] += red[t + 1 + stride * ATTN_SM_LQ]
        end
        @synchronize
        stride ÷= 2
    end
    @inbounds if ci == 0
        sums[lq + 1, hb + 1] = red[li + 1]
    end
end

"""
    attnsoftmax!(ctx, sums, p, s, scale) -> sums

Launch [`attn_softmax_rows!`](@ref) over `s :: (Lq, Lk, H, B)`, flattening
`(H, B)` so the kernel indexes three dimensions instead of four. Falls back to
the one-thread-per-row form when `Lq` is not a multiple of `ATTN_SM_LQ`, which
the tiling has no masking for.
"""
function attnsoftmax!(ctx, sums, p, s, scale)
    backend = ctx.backend
    Lq, Lk, H, B = size(s)
    # The chunked form always wins where it applies (review finding 3, tier two:
    # the switch that selected between them is gone). It has no masking for a
    # partial tile, so a query count the tiling does not divide still takes the
    # one-thread-per-row kernel below — which is also how to A/B the two now that
    # the switch is not there: call `attn_softmax16` directly. They differ only in
    # the order the row maximum and sum are reduced, so any difference between
    # them is floating-point associativity and should be tiny.
    if Lq % ATTN_SM_LQ != 0
        launch!(ctx, attn_softmax16, sums, p, s, Float32(scale))
        return sums
    end
    s3 = reshape(s, Lq, Lk, H * B)
    p3 = reshape(p, Lq, Lk, H * B)
    sm = reshape(sums, Lq, H * B)
    attn_softmax_rows!(backend, ATTN_SM_LQ * ATTN_SM_CH)(
        p3, sm, s3, Float32(scale), Int32(Lk);
        ndrange = (Lq ÷ ATTN_SM_LQ) * H * B * ATTN_SM_LQ * ATTN_SM_CH)
    sums
end


"""`(E,L,H,B)` read as `(CH,EP,H,B)` for query rows `q0+1 .. q0+CH`, zero past `E`."""
@inline function toLEpadchunk(I, a, E, q0, Lq)
    l, e, h, b = I
    ll = l + q0
    @inbounds (ll <= Lq && e <= E) ? a[e, ll, h, b] : zero(eltype(a))
end

"""
    coopmat_sdpa_plan(dev, q, k, v, bias; chunk = 2048, minl = 256) -> CoopMatSDPAPlan | Decline

Whether [`sdpa`](@ref) should take the cooperative-matrix path, and with what.

`bias` must be absent: the three-pass path adds it inside `attn_scores`, and a
GEMM has no epilogue to add it in — supporting it would mean a third pass, which
is exactly the traffic this path exists to remove. The extents must land on the
16-wide tile, which the mask decoder's 23-token prompt does not.
"""
function coopmat_sdpa_plan(dev::Device, q, k, v, bias; chunk::Int = 2048,
                           minl::Int = 256)
    Lq, Lk = size(q, 2), size(k, 2)
    bias === nothing || return Decline(:bias)
    eltype(q) === Float16 && eltype(k) === Float16 && eltype(v) === Float16 ||
        return Decline(:eltype)
    min(Lq, Lk) >= minl || return Decline(:short)
    Lq % dev.tile == 0 && Lk % dev.tile == 0 || return Decline(:extent)
    # `coopmat_gemm_available` asks the *device*, not the operands. On the CPU
    # backend of a machine that has a Vulkan device — which is every run of
    # `verify_sam2.jl` — it says yes, and the encoder's attention then handed
    # slab-backed `Vector{UInt8}` arrays to a cooperative-matrix kernel:
    # "passing non-bitstype argument", and the CPU reference for SAM 2's encoder
    # could not be produced at all. The operands have to be on the device too.
    ondevice(q) && ondevice(k) && ondevice(v) || return Decline(:host)
    dev.coopmat || return Decline(:nocoopmat)

    E = size(q, 1)
    CH = min(Lq, max(dev.tile, chunk))
    CH = cld(CH, dev.tile) * dev.tile      # the GEMM needs M on the tile
    CoopMatSDPAPlan(CH, cld(E, dev.tile) * dev.tile, size(q, 3) * size(q, 4))
end

"""Whether `a` is backed by device memory, wrappers and all."""
@inline ondevice(a) = stridedroot(a) !== nothing

"""
    sdpa_coopmat!(ctx, out, q, k, v, scale) -> out

Attention as two batched cooperative-matrix GEMMs. `q`, `k`, `v` are `(E, L, H, B)`.

`E` is padded up to the tile — 72 becomes 80 — by the copies that were happening
anyway (`transposeLE` for `q`, `densify` for `k`), so the padding is free. `v` is
transposed as well as padded, which `attn_apply` did not need; that is one extra
pass over `E*L*H*B` and it is in the measured numbers above.

The second product is computed TRANSPOSED, `O(Lq x EP) = P(Lq x Lk) * vT(Lk x EP)`,
because `coopmat_gemm!` needs `M` on the tile and `E = 72` is not; `fromLEpad`
puts it back.
"""
function sdpa_coopmat!(ctx, out, plan::CoopMatSDPAPlan, q, k, v, scale)
    backend = ctx.backend
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    EP = plan.EP
    NB = plan.nbatch
    # k and v are needed whole (the softmax reduces over keys); only q is chunked.
    kp = scratch!(ctx, Float16, EP, Lk, H, B)
    vT = scratch!(ctx, Float16, Lk, EP, H, B)
    launch!(ctx, padE, kp, k, E)
    launch!(ctx, toLEpad, vT, v, E)

    CH = plan.chunk
    qc   = scratch!(ctx, Float16, CH, EP, H, B)
    S    = scratch!(ctx, Float32, CH, Lk, H, B)
    P    = scratch!(ctx, Float16, CH, Lk, H, B)
    sums = scratch!(ctx, Float32, CH, H, B)
    O    = scratch!(ctx, Float32, CH, EP, H, B)

    for q0 in 0:CH:(Lq - 1)
        n = min(CH, Lq - q0)
        launch!(ctx, toLEpadchunk, qc, q, E, q0, Lq)
        Lava.coopmat_gemm!(S, qc, kp, CH, Lk, EP; nbatch = NB)
        attnsoftmax!(ctx, sums, P, S, scale)
        Lava.coopmat_gemm!(O, P, vT, CH, EP, Lk; nbatch = NB)
        # `fromLEpad` unchanged: `launch!` indexes the VIEW, so its `l` already
        # starts at 1 for this chunk and no offset belongs in the kernel.
        launch!(ctx, fromLEpad, view(out, :, (q0 + 1):(q0 + n), :, :), O, sums)
    end
    out
end

"""
    sdpa(ctx, q, k, v, bias, scale)

`bias` may be `nothing` (flash) or an additive mask (mem-efficient).
"""
function sdpa(ctx, q, k, v, bias, scale; out=nothing)
    # Decide once, then dispatch. This used to be `flashcm_applicable` (which ran
    # `flashcm_tiling` and threw the answer away), then `flashcm_tiling` again for
    # the tiling, then six more checks inside `sdpaflashcm!` that could still
    # decline — after `out` had been allocated. See `FlashCMPlan`.
    plan, k, v = sdpaplan(ctx, q, k, v, bias)
    return sdpa!(ctx, plan, out, q, k, v, bias, scale)
end

"""
    sdpaplan(ctx, q, k, v, bias) -> (plan, k, v)

Which attention implementation this call gets, decided once. `k` and `v` come
back because one refusal is recovered from by replacing them.

The fused kernel is tried first: it computes the same thing without ever writing
the score matrix, which is what both other paths spend most of their time on.
`sdpa_coopmat!`'s own stage breakdown on the global blocks was softmax 5.1,
second GEMM 4.0, first GEMM 2.2 — the two passes over `Lq x Lk` cost more than
the arithmetic — and on the windowed blocks the padding kernels alone were half
the op.
"""
function sdpaplan(ctx, q, k, v, bias)
    plan = flashcm_plan(ctx.dev, q, k, v, bias; clamp = ctx.clampattn)

    # The one refusal that is recoverable, and now it actually recovers. An
    # operand stack `stridedroot` cannot account for used to need someone to set
    # `FLASHCM_DENSIFY` by hand, which is to say it never happened and the call
    # silently took a slower path instead.
    #
    # `k` and `v` are the ones worth densifying and `q` is not, which is not an
    # oversight. Flash re-reads the whole of `k` and `v` once per query block — 64
    # times over for a 4096-query global block — so a `PermutedDimsArray`'s four
    # integer divisions per element get paid 64 times. **`q` is read exactly
    # once**: each workgroup stages its own `BR` queries and no other workgroup
    # touches them, so densifying it is a whole extra pass over the array to save
    # nothing. Densifying when it is *not* needed costs 646.4 MB of copies per
    # encode and 8.35 ms, which is why this is a fallback and not the default.
    if plan isa Decline && plan.reason === :wrapped
        k, v = densify(ctx, k), densify(ctx, v)
        plan = flashcm_plan(ctx.dev, q, k, v, bias; clamp = ctx.clampattn)
    end
    plan isa FlashCMPlan && return (plan, k, v)

    cm = coopmat_sdpa_plan(ctx.dev, q, k, v, bias)
    cm isa CoopMatSDPAPlan && return (cm, k, v)
    (Decline(:threepass), k, v)
end

"""Allocate the output if the caller did not."""
@inline sdpaout(ctx, out, q, v, Lq, H, B) =
    out === nothing ?
        KernelAbstractions.allocate(ctx.backend, accum(eltype(q)), size(v, 1), Lq, H, B) :
        out

# ── One method per plan type (review finding 1). A new attention path is now a
# new plan type and a new `sdpa!` method: nothing here has to be edited to admit
# it, and each method is reachable in a test by constructing its plan. That is
# what `kernels-to-port.md` item 1 (flash-decoding for the decoder's `Lq = 23`
# cross-attention) is meant to land as.

function sdpa!(ctx, plan::FlashCMPlan, out, q, k, v, bias, scale)
    Lq, H, B = size(q, 2), size(q, 3), size(q, 4)
    sdpaflashcm!(ctx, sdpaout(ctx, out, q, v, Lq, H, B), plan, q, k, v, scale)
end

function sdpa!(ctx, plan::CoopMatSDPAPlan, out, q, k, v, bias, scale)
    Lq, H, B = size(q, 2), size(q, 3), size(q, 4)
    # The operands go in as they arrive, wrappers and all. Densifying first was
    # tried and is strictly worse here: `densify` is a whole extra pass over each
    # of q, k and v, while the padding kernels read every element EXACTLY ONCE,
    # so the wrapper's four integer divisions are paid once per element instead
    # of once per pass. That is the opposite of `attn_scores`, which reads `q` E
    # times and is why `densify` exists.
    sdpa_coopmat!(ctx, sdpaout(ctx, out, q, v, Lq, H, B), plan, q, k, v, scale)
end

"""The three-pass path: always available, always right, and the slowest."""
function sdpa!(ctx, ::Decline, out, q, k, v, bias, scale)
    backend = ctx.backend
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    T = accum(eltype(q))

    # Before anything else, and before the big scratch allocations, so the
    # workspace hands these out at low offsets and the score matrix follows.
    # q and k are transposed on the way; v is already in the order `attn_apply`
    # wants and only needs to be dense.
    q = transposeLE(ctx, q)
    k = densify(ctx, k)
    v = densify(ctx, v)

    # The scores matrix and the (unused) softmax sums are pure working storage,
    # so they come from the op's `Workspace` — allocating them per call is what
    # kept the pool churning ~6 MB a step and driving the OOM-reclaim path, which
    # is a full device flush plus a GC each time it fires.
    # Stored as the operands' own type, accumulated in `T`. See `attn_softmax`.
    ST = eltype(q)
    scores = scratch!(ctx, ST, Lq, Lk, H, B)
    tk = blockfor(Lk, Lq)
    if tk > 1
        scoresblocked!(backend, tk, scores, q, k, bias, T(scale), (Lq, Lk ÷ tk, H, B))
    else
        launch!(ctx, attn_scores, scores, q, k, bias, scale)
    end

    # normalises `scores` in place; the returned sums are unused
    sums = scratch!(ctx, T, Lq, H, B)
    launch!(ctx, attn_softmax, sums, scores)

    # `out` outlives the op, so it cannot come from the workspace — but it can
    # come from the caller, and when the caller is a graph that is the slot the
    # planner reserved. Allocating it here instead was 48 fresh device buffers
    # per SAM 2 encode, enough to keep Lava's pool in its OOM-reclaim path.
    out === nothing && (out = KernelAbstractions.allocate(backend, T, size(v, 1), Lq, H, B))
    tq = blockfor(Lq, Lk)
    if tq > 1
        applyblocked!(backend, tq, out, scores, v, sums, (size(v, 1), Lq ÷ tq, H, B))
    else
        launch!(ctx, attn_apply, out, scores, v, sums)
    end
    out
end
