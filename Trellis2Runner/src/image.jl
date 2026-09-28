"""
    cropobject(rgba) -> Array{UInt8,3}

`Trellis2ImageTo3DPipeline.preprocess_image` for an image that already has an
alpha channel: the square around every pixel with alpha above 0.8, centred on
the object, RGB premultiplied by alpha and truncated back to UInt8 as numpy's
`astype(np.uint8)` does. `rgba` is `(4, W, H)` UInt8, PIL's `(H, W, 4)` rows as
columns; the result is `(3, S, S)`.

The crop box is PIL's: integer corners from a float centre, and pixels outside
the image read as zero.
"""
function cropobject(rgba::AbstractArray{UInt8,3})
    W, H = size(rgba, 2), size(rgba, 3)
    xs = [x - 1 for x in 1:W, y in 1:H if rgba[4, x, y] > 0.8 * 255]
    ys = [y - 1 for x in 1:W, y in 1:H if rgba[4, x, y] > 0.8 * 255]
    x0, x1, y0, y1 = minimum(xs), maximum(xs), minimum(ys), maximum(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = max(x1 - x0, y1 - y0)
    bx, by = cx - s ÷ 2, cy - s ÷ 2
    # PIL rounds a float box to the pixel grid.
    left, top = round(Int, bx), round(Int, by)
    right, bottom = round(Int, cx + s ÷ 2), round(Int, cy + s ÷ 2)
    out = zeros(UInt8, 3, right - left, bottom - top)
    for y in top:(bottom - 1), x in left:(right - 1)
        (0 <= x < W && 0 <= y < H) || continue
        a = Float32(rgba[4, x + 1, y + 1]) / 255
        for c in 1:3
            v = Float32(rgba[c, x + 1, y + 1]) / 255 * a
            out[c, x - left + 1, y - top + 1] = unsafe_trunc(UInt8, v * 255)
        end
    end
    return out
end

"""
    lanczos(img, size) -> Array{Float32,3}

PIL's `Image.resize(size, LANCZOS)` of a `(3, W, H)` UInt8 image to `(3, size,
size)`: separable Lanczos-3, its support widened by the downscale factor so it
antialiases, weights normalised per output pixel, horizontal pass first, each
pass rounded to UInt8 as PIL's 8-bit path does.
"""
function lanczos(img::AbstractArray{UInt8,3}, size_::Int)
    tmp = resample(img, size_, 2)
    return Float32.(resample(tmp, size_, 3))
end

lanczos3(x) = x == 0 ? 1.0 : abs(x) < 3 ? 3 * sin(π * x) * sin(π * x / 3) / (π * x)^2 : 0.0

"""One PIL resampling pass along dimension `dim` (2 = width, 3 = height)."""
function resample(img::AbstractArray{UInt8,3}, outsize::Int, dim::Int)
    insize = size(img, dim)
    scale = insize / outsize
    fscale = max(scale, 1.0)
    support = 3 * fscale
    dims = collect(size(img))
    dims[dim] = outsize
    out = zeros(UInt8, dims...)
    for o in 0:(outsize - 1)
        center = (o + 0.5) * scale
        lo = max(0, floor(Int, center - support + 0.5))
        hi = min(insize, floor(Int, center + support + 0.5))
        w = [lanczos3((i - center + 0.5) / fscale) for i in lo:(hi - 1)]
        w ./= sum(w)
        for idx in CartesianIndices(ntuple(d -> d == dim ? (1:1) : axes(out, d), 3))
            acc = 0.0
            for (k, i) in enumerate(lo:(hi - 1))
                I = Base.setindex(Tuple(idx), i + 1, dim)
                acc += w[k] * img[I...]
            end
            I = Base.setindex(Tuple(idx), o + 1, dim)
            out[I...] = clamp(round(Int, acc), 0, 255)
        end
    end
    return out
end

"""
    dinoinput(rgba; size = 512) -> Array{Float32,4}

The conditioner's input: the cropped object at `size`^2, in [0, 1], normalised
with ImageNet's mean and std, as `(W, H, 3, 1)` (torch's `(1, 3, H, W)`).
"""
function dinoinput(rgba::AbstractArray{UInt8,3}; size::Int = 512)
    img = lanczos(cropobject(rgba), size) ./ 255f0
    mean = (0.485f0, 0.456f0, 0.406f0)
    std = (0.229f0, 0.224f0, 0.225f0)
    out = Array{Float32,4}(undef, size, size, 3, 1)
    for y in 1:size, x in 1:size, c in 1:3
        out[x, y, c, 1] = (img[c, x, y] - mean[c]) / std[c]
    end
    return out
end
