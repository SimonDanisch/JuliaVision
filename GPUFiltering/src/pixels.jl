# Pixel formats, luma, 90 degree turns, and the two samplers every resampling
# kernel shares.

"Rec. 709 luma of a pixel, in [0, 1] for one that is."
@inline luma(c::AbstractRGB) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
@inline luma(c::TransparentRGB) = 0.2126f0 * Float32(red(c)) + 0.7152f0 * Float32(green(c)) + 0.0722f0 * Float32(blue(c))
@inline luma(c::AbstractGray) = Float32(gray(c))
@inline luma(c::Real) = Float32(c)

@inline tofloat(c::AbstractGray) = (g = Float32(gray(c)); RGB{Float32}(g, g, g))
@inline topixel(::Type{T}, v::Float32) where {T <: AbstractGray} = T(clamp(v, 0.0f0, 1.0f0))
@inline topixel(::Type{Float32}, v::Float32) = v
@inline topixel(::Type{Gray{N0f8}}, v::Float32) = Gray{N0f8}(fixed8(v))

"""
    floatpixel(c)

A pixel as the value resampling computes with: `RGB{Float32}` for any colour
or grey pixel, `Float32` for a number.
"""
@inline floatpixel(c::Real) = Float32(c)
@inline floatpixel(c::Union{AbstractRGB, TransparentRGB, AbstractGray}) = tofloat(c)

# `a * (1 - t) + b * t`, per channel: ColorTypes alone has no arithmetic.
@inline mixpixel(a::Float32, b::Float32, t::Float32) = a * (1.0f0 - t) + b * t
@inline mixpixel(a::RGB{Float32}, b::RGB{Float32}, t::Float32) =
    RGB{Float32}(mixpixel(a.r, b.r, t), mixpixel(a.g, b.g, t), mixpixel(a.b, b.b, t))
@inline addpixel(a::Float32, b::Float32) = a + b
@inline addpixel(a::RGB{Float32}, b::RGB{Float32}) = RGB{Float32}(a.r + b.r, a.g + b.g, a.b + b.b)
@inline scalepixel(a::Float32, k::Float32) = a * k
@inline scalepixel(a::RGB{Float32}, k::Float32) = RGB{Float32}(a.r * k, a.g * k, a.b * k)

"""
    bilinear(img, x, y)

`img` at `(x, y)` in pixel coordinates (pixel `i` is centred on `i`), bilinear,
with the border replicated outside: a `Float32` for an image of numbers, an
`RGB{Float32}` for colour or grey pixels.
"""
@inline bilinear(img::AbstractMatrix, x::Real, y::Real) = bilinear(img, Float32(x), Float32(y))
@inline function bilinear(img::AbstractMatrix, x::Float32, y::Float32)
    w, h = size(img)
    x, y = clamp(x, 1.0f0, Float32(w)), clamp(y, 1.0f0, Float32(h))
    xa, ya = max(min(unsafe_trunc(Int, floor(x)), w - 1), 1), max(min(unsafe_trunc(Int, floor(y)), h - 1), 1)
    fx, fy = x - xa, y - ya
    xb, yb = min(xa + 1, w), min(ya + 1, h)
    @inbounds a, b = floatpixel(img[xa, ya]), floatpixel(img[xb, ya])
    @inbounds c, d = floatpixel(img[xa, yb]), floatpixel(img[xb, yb])
    return mixpixel(mixpixel(a, b, fx), mixpixel(c, d, fx), fy)
end

"""
    areamean(img, x0, x1, y0, y1)

The mean of the pixels a footprint from `x0` to `x1` and `y0` to `y1` covers,
in pixel EDGE coordinates (pixel `i` spans `i - 1` to `i`), as [`bilinear`](@ref)
returns it. Averaging each output pixel's footprint keeps thin strokes from
aliasing away when shrinking. A footprint edge within a thousandth of a pixel
of a pixel edge counts as on it, so rounding in the footprint never adds a row.
"""
@inline function areamean(img::AbstractMatrix, x0::Float32, x1::Float32, y0::Float32, y1::Float32)
    w, h = size(img)
    xs = clamp(unsafe_trunc(Int, floor(x0 + 1.0f-3)) + 1, 1, w):clamp(unsafe_trunc(Int, ceil(x1 - 1.0f-3)), 1, w)
    ys = clamp(unsafe_trunc(Int, floor(y0 + 1.0f-3)) + 1, 1, h):clamp(unsafe_trunc(Int, ceil(y1 - 1.0f-3)), 1, h)
    @inbounds acc = scalepixel(floatpixel(img[first(xs), first(ys)]), 0.0f0)
    for y in ys, x in xs
        @inbounds acc = addpixel(acc, floatpixel(img[x, y]))
    end
    return scalepixel(acc, 1.0f0 / max(length(xs) * length(ys), 1))
end

@kernel function convert_kernel!(out, @Const(img))
    I = @index(Global, Cartesian)
    @inbounds c = tofloat(img[I])
    @inbounds out[I] = topixel(eltype(out), c.r, c.g, c.b)
end

"""
    convertpixels!([device,] out, img) -> out

`img` in `out`'s pixel format: any RGB or grey pixel to any RGB format, clamped
to [0, 1].
"""
convertpixels!(out::AbstractMatrix, img::AbstractMatrix) = convertpixels!(device(out), out, img)
function convertpixels!(dev, out::AbstractMatrix{<:AnyRGB}, img::AbstractMatrix)
    size(out) == size(img) || throw(DimensionMismatch("out $(size(out)) vs img $(size(img))"))
    launch!(dev, convert_kernel!, (out, img), size(out))
    return out
end

@kernel function luma_kernel!(out, @Const(img))
    I = @index(Global, Cartesian)
    @inbounds out[I] = topixel(eltype(out), luma(img[I]))
end

"""
    grayscale!([device,] out::AbstractMatrix{<:Gray}, img) -> out

Rec. 709 luma of `img` in a grey format.
"""
grayscale!(out::AbstractMatrix{<:AbstractGray}, img::AbstractMatrix) = grayscale!(device(out), out, img)
function grayscale!(dev, out::AbstractMatrix{<:AbstractGray}, img::AbstractMatrix)
    size(out) == size(img) || throw(DimensionMismatch("out $(size(out)) vs img $(size(img))"))
    launch!(dev, luma_kernel!, (out, img), size(out))
    return out
end

@kernel function rotate90_kernel!(out, @Const(img), degreesp)
    I = @index(Global, Cartesian)
    d = Int(paramvalue(degreesp))
    w, h = size(img)
    x, y = I[1], I[2]
    sx = d == 90 ? y : d == 180 ? w - x + 1 : d == 270 ? w - y + 1 : x
    sy = d == 90 ? h - x + 1 : d == 180 ? h - y + 1 : d == 270 ? x : y
    @inbounds out[I] = img[sx, sy]
end

"""
    rotate90!([device,] out, img, degrees) -> out

`img` turned clockwise by `degrees`, one of 0, 90, 180 and 270; a turn by 90 or
270 swaps the sides. `degrees` is a value, or a one-element device array (a
`GPURef`) for a turn that changes between runs of a graph; that one is not
checked here.
"""
rotate90!(out::AbstractMatrix, img::AbstractMatrix, degrees) = rotate90!(device(out), out, img, degrees)
function rotate90!(dev, out::AbstractMatrix, img::AbstractMatrix, degrees)
    checkturn(out, img, degrees)
    launch!(dev, rotate90_kernel!, (out, img, degrees), size(out))
    return out
end

function checkturn(out, img, degrees::Integer)
    degrees in (0, 90, 180, 270) || throw(ArgumentError("can only turn by 0, 90, 180 or 270 degrees, not $degrees"))
    turned = degrees in (90, 270) ? reverse(size(img)) : size(img)
    size(out) == turned || throw(DimensionMismatch("out $(size(out)) for img $(size(img)) turned by $degrees"))
    return nothing
end
checkturn(out, img, degrees::AbstractArray) = nothing
