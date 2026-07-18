"""
    ColorAdjustments(; brightness=0, contrast=1, saturation=1, temperature=0)

Fused per-pixel color correction parameters (all Float32, isbits):
`brightness` additive in [-1, 1], `contrast` multiplicative around 0.5,
`saturation` mix against luma (0 = grayscale, 1 = unchanged),
`temperature` warm/cool shift in [-1, 1] (positive = warmer).
"""
struct ColorAdjustments
    brightness::Float32
    contrast::Float32
    saturation::Float32
    temperature::Float32
end

function ColorAdjustments(; brightness::Real = 0, contrast::Real = 1,
                          saturation::Real = 1, temperature::Real = 0)
    return ColorAdjustments(Float32(brightness), Float32(contrast),
                            Float32(saturation), Float32(temperature))
end

isneutral(adj::ColorAdjustments) =
    adj.brightness == 0 && adj.contrast == 1 && adj.saturation == 1 && adj.temperature == 0

@kernel function coloradjust_kernel!(img, adj::ColorAdjustments)
    I = @index(Global, Cartesian)
    c = tofloat(img[I])
    # temperature: opposing red/blue gains
    r = c.r * (1.0f0 + 0.25f0 * adj.temperature)
    g = c.g
    b = c.b * (1.0f0 - 0.25f0 * adj.temperature)
    # saturation: mix with Rec.709 luma
    luma = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    r = luma + (r - luma) * adj.saturation
    g = luma + (g - luma) * adj.saturation
    b = luma + (b - luma) * adj.saturation
    # contrast around mid-gray, then brightness
    r = (r - 0.5f0) * adj.contrast + 0.5f0 + adj.brightness
    g = (g - 0.5f0) * adj.contrast + 0.5f0 + adj.brightness
    b = (b - 0.5f0) * adj.contrast + 0.5f0 + adj.brightness
    img[I] = topixel(eltype(img), r, g, b)
end

"""
    coloradjust!(img; brightness=0, contrast=1, saturation=1, temperature=0)
    coloradjust!(img, adj::ColorAdjustments)

In-place fused brightness/contrast/saturation/temperature adjustment.
"""
coloradjust!(img::AbstractMatrix{<:AbstractRGB}; kwargs...) =
    coloradjust!(img, ColorAdjustments(; kwargs...))

function coloradjust!(img::AbstractMatrix{<:AbstractRGB}, adj::ColorAdjustments)
    isneutral(adj) && return img
    backend = KA.get_backend(img)
    coloradjust_kernel!(backend)(img, adj; ndrange = size(img))
    return img
end

@kernel function channellinear_kernel!(img, gain::Vec3f, offset::Vec3f)
    I = @index(Global, Cartesian)
    c = tofloat(img[I])
    img[I] = topixel(eltype(img),
                     c.r * gain[1] + offset[1],
                     c.g * gain[2] + offset[2],
                     c.b * gain[3] + offset[3])
end

"""
    channellinear!(img, gain::Vec3f, offset::Vec3f)

In-place per-channel linear transform `c' = c * gain + offset` — the
building block for frame-to-frame color/exposure stabilization.
"""
function channellinear!(img::AbstractMatrix{<:AbstractRGB}, gain::Vec3f, offset::Vec3f)
    gain == Vec3f(1) && offset == Vec3f(0) && return img
    backend = KA.get_backend(img)
    channellinear_kernel!(backend)(img, gain, offset; ndrange = size(img))
    return img
end

"""
    channelstats(img) -> (mean::Vec3f, std::Vec3f)

Per-channel mean and standard deviation (host-side reduction).
"""
function channelstats(img::AbstractMatrix{<:AbstractRGB})
    n = length(img)
    s = zero(Vec3f)
    s2 = zero(Vec3f)
    for c in img
        v = Vec3f(Float32(red(c)), Float32(green(c)), Float32(blue(c)))
        s += v
        s2 += v .* v
    end
    μ = s ./ n
    σ = sqrt.(max.(s2 ./ n .- μ .* μ, 0.0f0))
    return μ, σ
end
