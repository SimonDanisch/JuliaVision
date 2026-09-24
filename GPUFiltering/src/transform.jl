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

"""
What a warp does with the pixel it resampled.

A PARAMETER of the one kernel, not a second kernel: placing a layer on a canvas
and compositing it there are the same resample. Warping into a canvas-sized
scratch image and running a second full-canvas pass to composite it is one
resample and two copies of where the layer is.
"""
abstract type WarpWrite end

"Store the resampled pixel, replacing whatever was there. The default."
struct Replace <: WarpWrite end
@inline (::Replace)(dst, r::Float32, g::Float32, b::Float32, a::Float32) =
    topixel(typeof(dst), r, g, b, a)

"""
Composite the resampled pixel OVER what was there, at layer opacity `alpha`.

The source is PREMULTIPLIED and carries its own coverage, so this is
`α·src + dst·(1 − α·a)` — not a lerp. Outside the source `skipoutside` leaves
`dst` alone, which is the same rule stated at the other end: a pixel the layer
does not reach is a pixel it does not write.
"""
struct Over <: WarpWrite
    alpha::Float32
end
@inline function (o::Over)(dst, r::Float32, g::Float32, b::Float32, a::Float32)
    α = o.alpha
    k = 1.0f0 - α * a
    return topixel(typeof(dst),
                   α * r + Float32(red(dst)) * k,
                   α * g + Float32(green(dst)) * k,
                   α * b + Float32(blue(dst)) * k,
                   α * a + alphaof(dst) * k)
end

@kernel function warp_kernel!(out, @Const(img), Mp, skipoutside::Bool,
                              boundsp, writep)
    I = @index(Global, Cartesian)
    M = Mat3f(paramvalue(Mp))
    # The source rect and the write mode change per frame in a composition — the
    # layer's crop and its opacity — so both arrive as parameters. ONE value for
    # the rect rather than four: four refs is four stores that can disagree about
    # which frame they describe.
    bnds = paramvalue(boundsp)
    x0, y0, x1, y1 = Int32(bnds[1]), Int32(bnds[2]), Int32(bnds[3]), Int32(bnds[4])
    write = paramvalue(writep)
    p = M * Vec3f(Float32(I[1]), Float32(I[2]), 1.0f0)
    x = p[1] / p[3]
    y = p[2] / p[3]
    # `x0…y1` is the source rect this warp may read — the whole image by default,
    # a CROP when the caller has one. A crop that only chose where to sample from
    # is not a crop: reframe outward and the material it was supposed to remove
    # comes back into view.
    w = x1
    h = y1
    # `skipoutside`: an output pixel the source doesn't reach is LEFT ALONE
    # rather than filled with a replicated border pixel. That is what makes
    # letterboxing one rule instead of two — the caller decides what shows
    # through by what it put there (black for the base layer, the canvas so far
    # for a layer stacked over one, which is how upper-layer bars stay clear).
    # Written as a branch, not an early return: KA kernels forbid `return`.
    covered = (x >= Float32(x0) - 0.5f0) & (x <= Float32(x1) + 0.5f0) &
              (y >= Float32(y0) - 0.5f0) & (y <= Float32(y1) + 0.5f0)
    if covered | !skipoutside
        xb = floor(Int32, x)
        yb = floor(Int32, y)
        wx = catmullrom(clamp(x - Float32(xb), 0.0f0, 1.0f0))
        wy = catmullrom(clamp(y - Float32(yb), 0.0f0, 1.0f0))
        r = 0.0f0; g = 0.0f0; b = 0.0f0; a = 0.0f0
        @inbounds for j in Int32(0):Int32(3)
            yj = clamp(yb - Int32(1) + j, y0, y1)   # replicate the RECT's borders
            wyj = wy[j + 1]
            for i in Int32(0):Int32(3)
                xi = clamp(xb - Int32(1) + i, x0, x1)
                px = img[xi, yj]
                c = tofloat(px)
                wgt = wx[i + 1] * wyj
                r += c.r * wgt; g += c.g * wgt; b += c.b * wgt
                a += alphaof(px) * wgt
            end
        end
        # Catmull-Rom overshoots at hard edges — clamp back into gamut before
        # anything downstream reads it, since `Over` multiplies by the alpha and
        # an alpha above one would darken what is behind the layer.
        @inbounds out[I] = write(out[I], clamp(r, 0.0f0, 1.0f0), clamp(g, 0.0f0, 1.0f0),
                                 clamp(b, 0.0f0, 1.0f0), clamp(a, 0.0f0, 1.0f0))
    end
end

"""
    warp!(out, img, M::Mat3f) -> out

Bicubic (Catmull-Rom) projective warp: each output pixel `(i, j)` samples
`img` at `M * (i, j, 1)` (border pixels replicate). One kernel covers
crop, scale, rotation and stabilization warps. `out` and `img` may have
different sizes.

`skipoutside = true` leaves output pixels the source doesn't cover UNTOUCHED
instead of replicating an edge pixel into them — see [`fitmatrix`](@ref), where
those pixels are the letterbox bars.

`bounds = (x0, y0, x1, y1)` restricts what may be READ to that source rect: with
it, a crop actually removes picture instead of merely choosing where to sample,
so nothing outside it can come back into view when the result is scaled down.

`write` says what to do with the resampled pixel — [`Replace`](@ref) by default,
[`Over`](@ref) to composite it straight into `out` instead of into a scratch
image that something else then composites.

    warp!(out, img, crop::NTuple{4})

Convenience: sample the normalized crop rect `(x, y, w, h)` of `img`
(measured from the top-left) into all of `out` — crop + resize in one pass.
"""
function warp!(out::AbstractMatrix{T}, img::AbstractMatrix{S}, M::Mat3f;
               skipoutside::Bool = false,
               bounds::Union{Nothing, NTuple{4, <:Integer}} = nothing,
               write::WarpWrite = Replace()) where {T <: AnyRGB, S <: AnyRGB}
    backend = KA.get_backend(img)
    # The kernel runs on the SOURCE's backend, so a destination living somewhere
    # else is handed to it as a foreign array. On a GPU backend that surfaces as
    # `KernelError: passing non-bitstype argument / Argument 3 ... Matrix{RGB{N0f8}}`
    # from inside GPUCompiler, naming neither `warp!` nor which of its arguments
    # is wrong — `renderpreview` in the editor hit exactly that and it took a
    # stack trace to attribute. Refuse it here, where the two arrays are visible.
    typeof(KA.get_backend(out)) === typeof(backend) ||
        throw(ArgumentError("warp!: `out` is on $(KA.get_backend(out)) and `img` on \
                             $backend. The kernel runs on the source's backend; \
                             allocate `out` there and copy afterwards."))
    w, h = size(img, 1), size(img, 2)
    b = bounds === nothing ? (1, 1, w, h) : bounds
    warp_kernel!(backend)(out, img, M, skipoutside,
                          Vec4{Int32}(b[1], b[2], b[3], b[4]), write; ndrange = size(out))
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

"""
    fitmatrix(crop, insize, outsize; scale = 1, position = (0, 0)) -> Mat3f

Sampling matrix that places the normalized `crop` rect of an `insize` image into
an `outsize` canvas **without distorting it**: the rect is scaled by
`min(fitx, fity)` so it fits entirely, and centred. Output pixels it doesn't
reach fall outside the source and become letterbox bars — pair this with
`warp!(…; skipoutside = true)`, which leaves them alone.

`scale` zooms about the centre (1 = fit, >1 fills past the edges, cropping) and
`position` shifts it, both as fractions of the canvas — the manual reframe on
top of the automatic fit.

Identical to [`cropmatrix`](@ref) whenever the rect already has the canvas'
aspect and the reframe is neutral, which is every single-format edit: fitting
must not resample material that already fits.
"""
function fitmatrix(crop::NTuple{4, <:Real}, insize::Tuple{Int, Int}, outsize::Tuple{Int, Int};
                   scale::Real = 1, position::Tuple{<:Real, <:Real} = (0, 0),
                   rotation::Real = 0)
    x, y, w, h = Float32.(crop)
    W, H = insize
    ow, oh = outsize
    cw, ch = w * W, h * H                      # the visible rect, in source pixels
    s = min(ow / cw, oh / ch) * Float32(scale) # canvas pixels per source pixel
    inv = 1.0f0 / s
    ox = 0.5f0 * (ow - cw * s) + Float32(position[1]) * ow
    oy = 0.5f0 * (oh - ch * s) + Float32(position[2]) * oh
    # ROTATION about the placed rect's own centre, so turning a clip does not also
    # walk it across the canvas. This is a SAMPLING matrix (canvas pixel → source
    # pixel), so the angle enters inverted: rotating the picture clockwise means
    # sampling along a counter-clockwise-rotated axis.
    if rotation == 0
        return Mat3f(inv, 0, 0,
                     0, inv, 0,
                     x * W + 0.5f0 - inv * (ox + 0.5f0), y * H + 0.5f0 - inv * (oy + 0.5f0), 1)
    end
    θ = -Float32(rotation) * Float32(pi) / 180.0f0
    c, sn = cos(θ), sin(θ)
    a11 = inv * c;  a12 = -inv * sn
    a21 = inv * sn; a22 = inv * c
    qx = ox + 0.5f0 * cw * s + 0.5f0        # centre of the placed rect, canvas px
    qy = oy + 0.5f0 * ch * s + 0.5f0
    px = x * W + 0.5f0 * cw + 0.5f0         # …and the source point it samples
    py = y * H + 0.5f0 * ch + 0.5f0
    return Mat3f(a11, a21, 0,
                 a12, a22, 0,
                 px - (a11 * qx + a12 * qy), py - (a21 * qx + a22 * qy), 1)
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
