"""
    gaussianweights(σ; maxradius = 0) -> (weights, radius)

The separable Gaussian taps for `σ`, normalised, and the radius they span.

`maxradius > 0` caps the radius AND pads the vector to `2·maxradius + 1`, so the
result always fits one buffer of that size — which is what lets a caller own the
weights instead of uploading a fresh array per frame (see `gaussianblur!`'s
`weights`). The taps sit at the FRONT: the kernel reads `weights[k + radius + 1]`
for `k in -radius:radius` and never looks at the padding.

A σ past the cap blurs by the cap rather than by what was asked. It is reachable
only from a keyframe or a parameter edge — a slider cannot go there — and the
alternative is resizing a buffer from a VALUE, which is exactly what must not
decide which graph is compiled.
"""
function gaussianweights(σ::Real; maxradius::Integer = 0)
    radius = max(ceil(Int, 3σ), 1)
    maxradius > 0 && (radius = min(radius, Int(maxradius)))
    weights = Float32[exp(-Float32(k)^2 / (2.0f0 * Float32(σ)^2)) for k in -radius:radius]
    weights ./= sum(weights)
    if maxradius > 0
        append!(weights, zeros(Float32, 2 * Int(maxradius) + 1 - length(weights)))
    end
    return weights, radius
end

# separable passes: dim=1 is the contiguous axis of our (width, height) frames
@kernel function convpass_kernel!(out, @Const(img), @Const(weights), radiusp, ::Val{DIM}) where {DIM}
    I = @index(Global, Cartesian)
    radius = Int32(paramvalue(radiusp))
    i, j = Tuple(I)
    len = Int32(size(img, DIM))
    r = 0.0f0
    g = 0.0f0
    b = 0.0f0
    a = 0.0f0
    for k in -radius:radius
        ii = DIM == 1 ? clamp(i + k, Int32(1), len) : i
        jj = DIM == 2 ? clamp(j + k, Int32(1), len) : j
        px = img[ii, jj]
        c = tofloat(px)
        w = weights[k + radius + 1]
        r += w * c.r
        g += w * c.g
        b += w * c.b
        a += w * alphaof(px)
    end
    # Coverage is blurred WITH the colour. Both are weighted sums of the same
    # taps, so a soft edge stays a soft edge and the premultiplied colour stays
    # consistent with the alpha it is multiplied by.
    out[I] = topixel(eltype(out), r, g, b, a)
end

"""
    gaussianblur!(out, img, σ; tmp = similar(img), weights = nothing) -> out

Separable gaussian blur (radius `3σ`, replicate border). `out`, `tmp` and
`img` must have equal size; `img` is not modified.

`weights` is a DEVICE vector the caller owns and has already filled with
`gaussianweights(σ; maxradius)` for this σ — its length says what the cap is. Pass
it from inside a render graph: without it this uploads a fresh array per call, and
a host→device upload in the middle of a recorded batch forces a `vkQueueSubmit`.
Measured on a four-layer 1080p composite: one extra submit per blur per frame,
five where the graph promises one.
"""
function gaussianblur!(out::AbstractMatrix{T}, img::AbstractMatrix{T}, σ::Real;
                       tmp::AbstractMatrix{T} = similar(img),
                       weights = nothing) where {T <: AnyRGB}
    σ <= 0 && return copyto!(out, img)
    backend = KA.get_backend(img)
    if weights === nothing
        w, radius = gaussianweights(σ)
        wdev = todevice(img, w)
    else
        wdev = weights
        radius = last(gaussianweights(σ; maxradius = (length(weights) - 1) ÷ 2))
    end
    kernel = convpass_kernel!(backend)
    kernel(tmp, img, wdev, Int32(radius), Val(1); ndrange = size(img))
    kernel(out, tmp, wdev, Int32(radius), Val(2); ndrange = size(img))
    return out
end

@kernel function unsharp_kernel!(out, @Const(img), @Const(blurred), amountp)
    I = @index(Global, Cartesian)
    amount = Float32(paramvalue(amountp))
    px = img[I]
    c = tofloat(px)
    bl = tofloat(blurred[I])
    # The coverage is the ORIGINAL's: sharpening decides how the picture looks,
    # not how much of it is there, and adding a high-pass to alpha would ring the
    # matte edge it is supposed to leave alone.
    out[I] = topixel(eltype(out),
                     c.r + amount * (c.r - bl.r),
                     c.g + amount * (c.g - bl.g),
                     c.b + amount * (c.b - bl.b), alphaof(px))
end

"""
    unsharpmask!(out, img, σ, amount; tmp=similar(img)) -> out

Sharpen via unsharp masking: `out = img + amount * (img - blur(img, σ))`.

`weights` is forwarded to [`gaussianblur!`](@ref) — same reason, same buffer.
"""
function unsharpmask!(out::AbstractMatrix{T}, img::AbstractMatrix{T}, σ::Real, amount::Real;
                      tmp::AbstractMatrix{T} = similar(img),
                      weights = nothing) where {T <: AnyRGB}
    amount == 0 && return copyto!(out, img)
    gaussianblur!(out, img, σ; tmp, weights)  # out temporarily holds the blur
    backend = KA.get_backend(img)
    unsharp_kernel!(backend)(tmp, img, out, Float32(amount); ndrange = size(img))
    return copyto!(out, tmp)
end
