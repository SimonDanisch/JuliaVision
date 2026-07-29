function gaussianweights(σ::Real)
    radius = max(ceil(Int, 3σ), 1)
    weights = Float32[exp(-Float32(k)^2 / (2.0f0 * Float32(σ)^2)) for k in -radius:radius]
    weights ./= sum(weights)
    return weights, radius
end

# separable passes: dim=1 is the contiguous axis of our (width, height) frames
@kernel function convpass_kernel!(out, @Const(img), @Const(weights), radius::Int32, ::Val{DIM}) where {DIM}
    I = @index(Global, Cartesian)
    i, j = Tuple(I)
    len = Int32(size(img, DIM))
    r = 0.0f0
    g = 0.0f0
    b = 0.0f0
    for k in -radius:radius
        ii = DIM == 1 ? clamp(i + k, Int32(1), len) : i
        jj = DIM == 2 ? clamp(j + k, Int32(1), len) : j
        c = tofloat(img[ii, jj])
        w = weights[k + radius + 1]
        r += w * c.r
        g += w * c.g
        b += w * c.b
    end
    out[I] = topixel(eltype(out), r, g, b)
end

"""
    gaussianblur!(out, img, σ; tmp=similar(img)) -> out

Separable gaussian blur (radius `3σ`, replicate border). `out`, `tmp` and
`img` must have equal size; `img` is not modified.
"""
function gaussianblur!(out::AbstractMatrix{T}, img::AbstractMatrix{T}, σ::Real;
                       tmp::AbstractMatrix{T} = similar(img)) where {T <: AbstractRGB}
    σ <= 0 && return copyto!(out, img)
    backend = KA.get_backend(img)
    weights, radius = gaussianweights(σ)
    wdev = todevice(img, weights)
    kernel = convpass_kernel!(backend)
    kernel(tmp, img, wdev, Int32(radius), Val(1); ndrange = size(img))
    kernel(out, tmp, wdev, Int32(radius), Val(2); ndrange = size(img))
    return out
end

@kernel function unsharp_kernel!(out, @Const(img), @Const(blurred), amount::Float32)
    I = @index(Global, Cartesian)
    c = tofloat(img[I])
    bl = tofloat(blurred[I])
    out[I] = topixel(eltype(out),
                     c.r + amount * (c.r - bl.r),
                     c.g + amount * (c.g - bl.g),
                     c.b + amount * (c.b - bl.b))
end

"""
    unsharpmask!(out, img, σ, amount; tmp=similar(img)) -> out

Sharpen via unsharp masking: `out = img + amount * (img - blur(img, σ))`.
"""
function unsharpmask!(out::AbstractMatrix{T}, img::AbstractMatrix{T}, σ::Real, amount::Real;
                      tmp::AbstractMatrix{T} = similar(img)) where {T <: AbstractRGB}
    amount == 0 && return copyto!(out, img)
    gaussianblur!(out, img, σ; tmp)  # out temporarily holds the blur
    backend = KA.get_backend(img)
    unsharp_kernel!(backend)(tmp, img, out, Float32(amount); ndrange = size(img))
    return copyto!(out, tmp)
end
