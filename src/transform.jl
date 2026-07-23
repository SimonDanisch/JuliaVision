# Catmull-Rom weights for the 4 taps at offsets -1..2 around fractional `t`.
# The negative lobes preserve sharpness that bilinear sampling low-passes away —
# a stabilization warp resamples EVERY frame at sub-pixel offsets, so the
# interpolator's quality is directly the output's sharpness.
@inline function catmullrom(t::Float32)
    t2 = t * t
    t3 = t2 * t
    return (-0.5f0 * t3 + t2 - 0.5f0 * t,
            1.5f0 * t3 - 2.5f0 * t2 + 1.0f0,
            -1.5f0 * t3 + 2.0f0 * t2 + 0.5f0 * t,
            0.5f0 * t3 - 0.5f0 * t2)
end

@kernel function warp_kernel!(out, @Const(img), M::Mat3f)
    I = @index(Global, Cartesian)
    p = M * Vec3f(Float32(I[1]), Float32(I[2]), 1.0f0)
    x = p[1] / p[3]
    y = p[2] / p[3]
    w = Int32(size(img, 1))
    h = Int32(size(img, 2))
    xb = floor(Int32, x)
    yb = floor(Int32, y)
    wx = catmullrom(clamp(x - Float32(xb), 0.0f0, 1.0f0))
    wy = catmullrom(clamp(y - Float32(yb), 0.0f0, 1.0f0))
    r = 0.0f0; g = 0.0f0; b = 0.0f0
    @inbounds for j in Int32(0):Int32(3)
        yj = clamp(yb - Int32(1) + j, Int32(1), h)   # replicate borders
        wyj = wy[j + 1]
        for i in Int32(0):Int32(3)
            xi = clamp(xb - Int32(1) + i, Int32(1), w)
            c = tofloat(img[xi, yj])
            wgt = wx[i + 1] * wyj
            r += c.r * wgt; g += c.g * wgt; b += c.b * wgt
        end
    end
    # Catmull-Rom overshoots at hard edges — clamp back into gamut
    out[I] = topixel(eltype(out), clamp(r, 0.0f0, 1.0f0), clamp(g, 0.0f0, 1.0f0),
                     clamp(b, 0.0f0, 1.0f0))
end

"""
    warp!(out, img, M::Mat3f) -> out

Bicubic (Catmull-Rom) projective warp: each output pixel `(i, j)` samples
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

@kernel function areadownscale_kernel!(out, @Const(img))
    I = @index(Global, Cartesian)
    tw, th = size(out)
    w, h = size(img)
    x0 = (I[1] - 1) * w ÷ tw + 1; x1 = max(I[1] * w ÷ tw, x0)
    y0 = (I[2] - 1) * h ÷ th + 1; y1 = max(I[2] * h ÷ th, y0)
    r = 0.0f0; g = 0.0f0; b = 0.0f0; cnt = 0.0f0
    @inbounds for yy in y0:y1, xx in x0:x1
        c = tofloat(img[xx, yy])
        r += c.r; g += c.g; b += c.b; cnt += 1.0f0
    end
    inv = 1.0f0 / cnt
    @inbounds out[I] = topixel(eltype(out), r * inv, g * inv, b * inv)
end

"""
    areadownscale!(out, img) -> out

Area-average downscale: every output pixel is the mean of its full source cell,
so heavy decimation (timeline thumbnails) cannot alias the way point sampling
does. `out` and `img` may live on any KA backend.
"""
function areadownscale!(out::AbstractMatrix{<:AbstractRGB}, img::AbstractMatrix{<:AbstractRGB})
    areadownscale_kernel!(KA.get_backend(out))(out, img; ndrange = size(out))
    KA.synchronize(KA.get_backend(out))
    return out
end
