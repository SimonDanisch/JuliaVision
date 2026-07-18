@kernel function warp_kernel!(out, @Const(img), M::Mat3f)
    I = @index(Global, Cartesian)
    p = M * Vec3f(Float32(I[1]), Float32(I[2]), 1.0f0)
    x = p[1] / p[3]
    y = p[2] / p[3]
    w = size(img, 1)
    h = size(img, 2)
    x0 = clamp(floor(Int32, x), Int32(1), Int32(w))
    y0 = clamp(floor(Int32, y), Int32(1), Int32(h))
    x1 = min(x0 + Int32(1), Int32(w))
    y1 = min(y0 + Int32(1), Int32(h))
    fx = clamp(x - Float32(x0), 0.0f0, 1.0f0)
    fy = clamp(y - Float32(y0), 0.0f0, 1.0f0)
    c00 = tofloat(img[x0, y0])
    c10 = tofloat(img[x1, y0])
    c01 = tofloat(img[x0, y1])
    c11 = tofloat(img[x1, y1])
    r = (c00.r * (1 - fx) + c10.r * fx) * (1 - fy) + (c01.r * (1 - fx) + c11.r * fx) * fy
    g = (c00.g * (1 - fx) + c10.g * fx) * (1 - fy) + (c01.g * (1 - fx) + c11.g * fx) * fy
    b = (c00.b * (1 - fx) + c10.b * fx) * (1 - fy) + (c01.b * (1 - fx) + c11.b * fx) * fy
    out[I] = topixel(eltype(out), r, g, b)
end

"""
    warp!(out, img, M::Mat3f) -> out

Bilinear-sampled projective warp: each output pixel `(i, j)` samples
`img` at `M * (i, j, 1)` (border pixels replicate). One kernel covers
crop, scale, rotation and stabilization warps. `out` and `img` may have
different sizes.

    warp!(out, img, crop::NTuple{4})

Convenience: sample the normalized crop rect `(x, y, w, h)` of `img`
(measured from the top-left) into all of `out` — crop + resize in one pass.
"""
function warp!(out::AbstractMatrix{T}, img::AbstractMatrix{S}, M::Mat3f) where {T <: AbstractRGB, S <: AbstractRGB}
    backend = KA.get_backend(img)
    warp_kernel!(backend)(out, img, M; ndrange = size(out))
    return out
end

"Sampling matrix mapping output pixels onto the normalized crop rect of the input."
function cropmatrix(crop::NTuple{4, <:Real}, insize::Tuple{Int, Int}, outsize::Tuple{Int, Int})
    x, y, w, h = Float32.(crop)
    W, H = insize
    ow, oh = outsize
    # map out (1:ow, 1:oh) → img (x*W .+ (0:w*W), y*H .+ (0:h*H)), 0.5-centered
    sx = w * W / ow
    sy = h * H / oh
    return Mat3f(sx, 0, 0,
                 0, sy, 0,
                 x * W + 0.5f0 - 0.5f0 * sx, y * H + 0.5f0 - 0.5f0 * sy, 1)
end

"Pure translation sampling matrix: shifts content by `(dx, dy)` pixels."
translationmatrix(dx::Real, dy::Real) = Mat3f(1, 0, 0, 0, 1, 0, -Float32(dx), -Float32(dy), 1)

function warp!(out::AbstractMatrix{<:AbstractRGB}, img::AbstractMatrix{<:AbstractRGB},
               crop::NTuple{4, <:Real})
    return warp!(out, img, cropmatrix(crop, size(img), size(out)))
end
