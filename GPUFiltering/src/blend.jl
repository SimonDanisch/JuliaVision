@kernel function blend_kernel!(out, @Const(a), @Const(b), p::Float32)
    I = @index(Global, Cartesian)
    pa = a[I]; pb = b[I]
    ca = tofloat(pa)
    cb = tofloat(pb)
    q = 1.0f0 - p
    out[I] = topixel(eltype(out), q * ca.r + p * cb.r, q * ca.g + p * cb.g, q * ca.b + p * cb.b,
                     q * alphaof(pa) + p * alphaof(pb))
end

"""
    blend!(out, a, b, p) -> out

Linear cross-mix `out = (1-p)·a + p·b` over three same-size RGB buffers (`out`
may alias `a` or `b`). The pixel core of a cross-dissolve transition, and of a
fade when one side is a constant color.
"""
blend!(out::AbstractMatrix{<:AnyRGB}, a::AbstractMatrix{<:AnyRGB}, b::AbstractMatrix{<:AnyRGB}, p::Real) =
    blend!(device(out), out, a, b, p)
function blend!(dev, out::AbstractMatrix{<:AnyRGB}, a::AbstractMatrix{<:AnyRGB},
                b::AbstractMatrix{<:AnyRGB}, p::Real)
    launch!(dev, blend_kernel!, (out, a, b, Float32(p)), size(out))
    return out
end
