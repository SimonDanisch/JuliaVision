@kernel function blend_kernel!(out, @Const(a), @Const(b), p::Float32)
    I = @index(Global, Cartesian)
    ca = tofloat(a[I])
    cb = tofloat(b[I])
    q = 1.0f0 - p
    out[I] = topixel(eltype(out), q * ca.r + p * cb.r, q * ca.g + p * cb.g, q * ca.b + p * cb.b)
end

"""
    blend!(out, a, b, p) -> out

Linear cross-mix `out = (1-p)·a + p·b` over three same-size RGB buffers (`out`
may alias `a` or `b`). The pixel core of a cross-dissolve transition, and of a
fade when one side is a constant color.
"""
function blend!(out::AbstractMatrix{<:AbstractRGB}, a::AbstractMatrix{<:AbstractRGB},
                b::AbstractMatrix{<:AbstractRGB}, p::Real)
    backend = KA.get_backend(out)
    blend_kernel!(backend)(out, a, b, Float32(p); ndrange = size(out))
    KA.synchronize(backend)
    return out
end
