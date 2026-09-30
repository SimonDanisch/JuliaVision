# Helpers the example scripts share: image IO in the runners' (W, H) layout and
# the few compositing steps the README pictures need. Not a package API.
module ExampleCommon

using FileIO, ColorTypes, FixedPointNumbers, ImageCore
using GPUFiltering: areadownscale!, warp!

export REPO, media, loadimage, saveimage, resizewidth, hstrip, vstrip, overchecker,
       tintmask, cutout, marker!

const REPO = normpath(joinpath(@__DIR__, "..", ".."))

"Path under `media/`, creating its directory."
function media(parts::AbstractString...)
    path = joinpath(REPO, "media", parts...)
    mkpath(dirname(path))
    return path
end

# FileIO hands images out as (H, W); every runner indexes [x, y].
loadimage(path::AbstractString) = permutedims(load(path))
saveimage(path::AbstractString, img::AbstractMatrix) = save(path, permutedims(img))

"Resize a (W, H) RGB image to width `w`, keeping the aspect: area mean down, bicubic up."
function resizewidth(img::AbstractMatrix{<:AbstractRGB}, w::Integer)
    W, H = size(img)
    out = similar(img, RGB{N0f8}, w, round(Int, H * w / W))
    w <= W ? areadownscale!(out, img) : warp!(out, img, (0, 0, 1, 1))
    return out
end

"Images side by side along x, top-aligned, `gap` pixels of `fill` between them."
function hstrip(imgs::AbstractMatrix...; gap::Integer = 8, fill = RGB{N0f8}(1, 1, 1))
    H = maximum(size(i, 2) for i in imgs)
    W = sum(size(i, 1) for i in imgs) + gap * (length(imgs) - 1)
    out = Base.fill(fill, W, H)
    x = 0
    for i in imgs
        out[x+1:x+size(i, 1), 1:size(i, 2)] .= RGB{N0f8}.(i)
        x += size(i, 1) + gap
    end
    return out
end

"Images stacked along y, left-aligned."
vstrip(imgs::AbstractMatrix...; kw...) = permutedims(hstrip(permutedims.(imgs)...; kw...))

checker(x, y; size = 16) = isodd(fld(x - 1, size) + fld(y - 1, size)) ? 0.8f0 : 1.0f0

"Straight-alpha RGBA composited over a grey checkerboard."
function overchecker(img::AbstractMatrix{<:TransparentColor})
    return [begin
                c = img[x, y]; a = Float32(alpha(c)); k = checker(x, y)
                RGB{N0f8}(clamp(a * red(c) + (1 - a) * k, 0, 1),
                          clamp(a * green(c) + (1 - a) * k, 0, 1),
                          clamp(a * blue(c) + (1 - a) * k, 0, 1))
            end for x in axes(img, 1), y in axes(img, 2)]
end

"The image with `mask` (UInt8 0..255, same size) tinted in `color`."
function tintmask(img::AbstractMatrix{<:AbstractRGB}, mask::AbstractMatrix{UInt8};
                  color = RGB{Float32}(0.1, 0.55, 1.0), strength::Real = 0.55)
    return [begin
                a = Float32(strength) * mask[x, y] / 255f0
                c = RGB{Float32}(img[x, y])
                RGB{N0f8}(clamp.((1 - a) .* (c.r, c.g, c.b) .+ a .* (color.r, color.g, color.b), 0, 1)...)
            end for x in axes(img, 1), y in axes(img, 2)]
end

"RGB plus a UInt8 matte as straight RGBA."
cutout(img::AbstractMatrix{<:AbstractRGB}, mask::AbstractMatrix{UInt8}) =
    [RGBA{N0f8}(red(img[i]), green(img[i]), blue(img[i]), reinterpret(N0f8, mask[i]))
     for i in CartesianIndices(img)]

"Draw a filled disc with a white rim at pixel (x, y), for marking a click."
function marker!(img::AbstractMatrix, x::Real, y::Real; radius::Real = 9,
                 color = RGB{N0f8}(1, 0.2, 0.2))
    for j in axes(img, 2), i in axes(img, 1)
        d = hypot(i - x, j - y)
        d <= radius && (img[i, j] = d <= radius - 3 ? color : RGB{N0f8}(1, 1, 1))
    end
    return img
end

end
