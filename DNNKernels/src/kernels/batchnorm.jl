"""
Batch norm in training mode, as two declared dispatches.

`_native_batch_norm_legit.no_stats` computes its own statistics: a mean and a
variance per channel over every other axis, then normalises. So it is two passes
with a dependency between them, and declaring it is exactly that: `use` says the
first writes the statistics and the second reads them, and `Barriers` derives
the wait. `runop!` wrote it as five fused broadcasts over `sum(x; dims = rd)`,
which allocated the mean, the centred copy and the variance outside any plan.

Two-pass and not Welford, because torch's own form is two-pass: the mean, then
the variance of `x .- μ`. Matching it keeps the rounding the same where it can
be the same.
"""

"""
    bnstats!(mean, invstd, a, cstride, C, nouter, eps)

The per-channel mean and inverse standard deviation, one thread per CHANNEL.

`cstride` is the product of the axes below the channel and `nouter` the product
of those above it, so a channel's elements are `nouter` runs of `cstride`
consecutive values, `cstride * C` apart. That is the whole of the layout: the
channel axis is torch's dim 1, which in the reversed shape is `ndims - 1`, and
no axis needs naming beyond the two products.

**The accumulator is `accum(eltype(a))`.** Over a `Float16` input this reduction
sums the whole sequence: on a vocoder that is 18,961 elements per channel around
magnitude 27, so the sum of squares reaches ~1.4e7 and saturates to `Inf` in
half precision. `invstd` is then `1/sqrt(Inf) = 0` and every output is silently
zero, for long inputs only.

C threads is a poor shape for a GPU and is the shape the arithmetic has: one
sequential sum per channel is what keeps the rounding comparable to a broadcast
over `sum(x; dims = rd)`. It is the thing to tile if a profile ever names it.
"""
function bnstats!(mean, invstd, a, cstride::Int, C::Int, nouter::Int, eps)
    ch = KI.get_global_id().x
    ch <= C || return
    A = accum(eltype(a))
    n = cstride * nouter
    base = (ch - 1) * cstride
    step = cstride * C
    s = zero(A)
    @inbounds for o in 0:(nouter - 1), t in 0:(cstride - 1)
        s += A(a[o * step + base + t + 1])
    end
    mu = s / n
    v = zero(A)
    @inbounds for o in 0:(nouter - 1), t in 0:(cstride - 1)
        d = A(a[o * step + base + t + 1]) - mu
        v += d * d
    end
    @inbounds mean[ch] = mu
    # Biased, as torch's is: `v / n`, not `v / (n - 1)`.
    @inbounds invstd[ch] = inv(sqrt(v / n + eps))
    return
end

"""
    bnapply!(out, x, mean, invstd, gamma, beta, n, cstride, C)

`out = (x - mean) * invstd * gamma + beta`, per channel.

The channel of a linear index is `((i - 1) ÷ cstride) % C + 1`, which is the
same two products `bnstats!` takes and the reason neither kernel needs the
shape.
"""
function bnapply!(out, x, mean, invstd, gamma, beta, n::Int, cstride::Int, C::Int)
    i = KI.get_global_id().x
    i <= n || return
    ch = ((i - 1) ÷ cstride) % C + 1
    @inbounds out[i] = (x[i] - mean[ch]) * (invstd[ch] * gamma[ch]) + beta[ch]
    return
end
