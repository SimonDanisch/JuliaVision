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

@inline function attn_scores(I, q, k, bias, scale)
    lk, lq, h, b = I
    @inbounds begin
        T = accum(eltype(q))
        acc = zero(T)
        for e in axes(q, 1)
            acc = muladd(T(q[e, lq, h, b]), T(k[e, lk, h, b]), acc)
        end
        acc *= T(scale)
        bias === nothing ? acc :
            acc + T(bias[bidx(bias, CartesianIndex(lk, lq, h, b))])
    end
end

"""One thread per query row; the reduction over keys is sequential in-thread."""
@inline function attn_softmax(I, scores)
    lq, h, b = I
    @inbounds begin
        T = eltype(scores)
        m = typemin(T)
        for lk in axes(scores, 1)
            m = max(m, scores[lk, lq, h, b])
        end
        if !isfinite(m)                       # every key blocked
            for lk in axes(scores, 1)
                scores[lk, lq, h, b] = zero(T)
            end
            return zero(T)
        end
        s = zero(T)
        for lk in axes(scores, 1)
            e = exp(scores[lk, lq, h, b] - m)
            scores[lk, lq, h, b] = e
            s += e
        end
        for lk in axes(scores, 1)
            scores[lk, lq, h, b] /= s
        end
        s
    end
end

@inline function attn_apply(I, p, v)
    e, lq, h, b = I
    @inbounds begin
        T = accum(eltype(v))
        acc = zero(T)
        for lk in axes(p, 1)
            acc = muladd(T(p[lk, lq, h, b]), T(v[e, lk, h, b]), acc)
        end
        acc
    end
end

"""
    sdpa(q, k, v, bias, scale; backend)

`bias` may be `nothing` (flash) or an additive mask (mem-efficient).
"""
function sdpa(q, k, v, bias, scale; backend=KernelAbstractions.get_backend(q), ws=nothing)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    T = accum(eltype(q))

    # The scores matrix and the (unused) softmax sums are pure working storage,
    # so they come from the op's `Workspace` — allocating them per call is what
    # kept the pool churning ~6 MB a step and driving the OOM-reclaim path, which
    # is a full device flush plus a GC each time it fires.
    scores = scratch!(ws, backend, T, Lk, Lq, H, B)
    launch!(attn_scores, scores, q, k, bias, scale; backend)

    # normalises `scores` in place; the returned sums are unused
    sums = scratch!(ws, backend, T, Lq, H, B)
    launch!(attn_softmax, sums, scores; backend)

    # `out` is the op's result and outlives the op, so it is a real allocation.
    out = KernelAbstractions.allocate(backend, T, size(v, 1), Lq, H, B)
    launch!(attn_apply, out, scores, v; backend)
    out
end
