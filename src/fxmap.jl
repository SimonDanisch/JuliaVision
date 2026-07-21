# Map a CALLBACK over an image with one backend-generic kernel, so effects can be
# written as pure functions instead of kernels. Callbacks work in `Vec3f` (RGB in
# [0,1]); `tofloat`/`topixel` handle the storage round-trip (and the GPU-safe write).
#
# Two categories cover almost everything:
#   • pointwise — one output pixel from one input pixel + its coordinate
#   • stencil   — one output pixel from an n×m neighborhood (border-clamped)
# Anything heavier (separable/multi-pass filters, reductions) is a plain function
# that calls these or writes its own kernel — the escape hatch.

@kernel function pointwise_kernel!(out, @Const(inp), f, invsz::Vec2f)
    I = @index(Global, Cartesian)
    c = tofloat(inp[I])
    uv = Vec2f((Float32(I[1]) - 0.5f0) * invsz[1], (Float32(I[2]) - 0.5f0) * invsz[2])
    r = f(Vec3f(c.r, c.g, c.b), uv)
    out[I] = topixel(eltype(out), r[1], r[2], r[3])
end

"""
    pointwise!(out, inp, f) -> out

Apply a per-pixel callback `f(c::Vec3f, uv::Vec2f) -> Vec3f` to every pixel, where
`uv ∈ [0,1]²` is the pixel's normalized coordinate. `out` may alias `inp` (in place).
Backend-generic: runs on a CPU `Array`, a `LavaArray`, a `CuArray`, …
"""
function pointwise!(out::AbstractMatrix{<:AbstractRGB}, inp::AbstractMatrix{<:AbstractRGB}, f)
    backend = KA.get_backend(out)
    pointwise_kernel!(backend)(out, inp, f, Vec2f(1.0f0 / size(out, 1), 1.0f0 / size(out, 2));
                               ndrange = size(out))
    return out
end

@kernel function stencil_kernel!(out, @Const(inp), f, radius::Int32, w::Int32, h::Int32, invsz::Vec2f)
    I = @index(Global, Cartesian)
    ix = Int32(I[1]); iy = Int32(I[2])
    sample = (di, dj) -> begin
        c = tofloat(inp[clamp(ix + di, Int32(1), w), clamp(iy + dj, Int32(1), h)])
        Vec3f(c.r, c.g, c.b)
    end
    uv = Vec2f((Float32(ix) - 0.5f0) * invsz[1], (Float32(iy) - 0.5f0) * invsz[2])
    r = f(sample, radius, uv)
    out[I] = topixel(eltype(out), r[1], r[2], r[3])
end

"""
    stencil!(out, inp, f, radius) -> out

Apply a neighborhood callback `f(sample, radius, uv) -> Vec3f`, where
`sample(di, dj) -> Vec3f` reads the neighbor at offset `(di, dj)` (clamped at the
border) and `uv ∈ [0,1]²`. `out` must NOT alias `inp`.
"""
function stencil!(out::AbstractMatrix{<:AbstractRGB}, inp::AbstractMatrix{<:AbstractRGB}, f, radius::Integer)
    backend = KA.get_backend(out)
    stencil_kernel!(backend)(out, inp, f, Int32(radius), Int32(size(inp, 1)), Int32(size(inp, 2)),
                             Vec2f(1.0f0 / size(out, 1), 1.0f0 / size(out, 2)); ndrange = size(out))
    return out
end
