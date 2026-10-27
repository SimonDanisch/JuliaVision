# Sampling an image along a coarse grid of source coordinates, as torch's
# `grid_sample` with `align_corners=True` and zero padding: the backward map of
# a dewarping network, which predicts the grid at a fraction of the image's
# resolution.

@inline lerp(a, b, t) = a * (1.0f0 - t) + b * t

@inline function pixelorzero(img, x::Int, y::Int)
    1 <= x <= size(img, 1) && 1 <= y <= size(img, 2) || return RGB{Float32}(0, 0, 0)
    return @inbounds tofloat(img[x, y])
end

function gridsample_kernel!(out, img, grid)
    I, inside = workitem(size(out))
    inside || return nothing
    x, y = I[1], I[2]
    w, h = size(out)
    sw, sh = size(img)
    gw, gh = size(grid, 1), size(grid, 2)
    # The grid upsampled to the output bilinearly, with its corners on the
    # output's corners.
    gx = min(1.0f0 + (x - 1) * Float32((gw - 1) / max(w - 1, 1)), Float32(gw))
    gy = min(1.0f0 + (y - 1) * Float32((gh - 1) / max(h - 1, 1)), Float32(gh))
    a = unsafe_trunc(Int, floor(gx)); b = unsafe_trunc(Int, floor(gy))
    c = min(a + 1, gw); d = min(b + 1, gh)
    fx = gx - a; fy = gy - b
    @inbounds u = lerp(lerp(grid[a, b, 1, 1], grid[c, b, 1, 1], fx), lerp(grid[a, d, 1, 1], grid[c, d, 1, 1], fx), fy)
    @inbounds v = lerp(lerp(grid[a, b, 2, 1], grid[c, b, 2, 1], fx), lerp(grid[a, d, 2, 1], grid[c, d, 2, 1], fx), fy)
    # [-1, 1] spans the source's first to last pixel centre.
    px = (u + 1.0f0) * 0.5f0 * (sw - 1) + 1.0f0
    py = (v + 1.0f0) * 0.5f0 * (sh - 1) + 1.0f0
    r = 0.0f0; g = 0.0f0; bl = 0.0f0
    if px > 0 && px < sw + 1 && py > 0 && py < sh + 1
        ix = unsafe_trunc(Int, floor(px)); iy = unsafe_trunc(Int, floor(py))
        tx = px - ix; ty = py - iy
        p00 = pixelorzero(img, ix, iy); p10 = pixelorzero(img, ix + 1, iy)
        p01 = pixelorzero(img, ix, iy + 1); p11 = pixelorzero(img, ix + 1, iy + 1)
        r = lerp(lerp(p00.r, p10.r, tx), lerp(p01.r, p11.r, tx), ty)
        g = lerp(lerp(p00.g, p10.g, tx), lerp(p01.g, p11.g, tx), ty)
        bl = lerp(lerp(p00.b, p10.b, tx), lerp(p01.b, p11.b, tx), ty)
    end
    @inbounds out[I] = topixel(eltype(out), r, g, bl)
    return nothing
end

"""
    gridsample!([device,] out, img, grid) -> out

`img` resampled along `grid`, a `(gw, gh, 2)` or `(gw, gh, 2, 1)` array of
normalised source coordinates `(x, y)` in [-1, 1], -1 and 1 being the centres
of the first and last pixel. The grid is upsampled bilinearly to `out`'s size,
so `out` may be any size; points outside the source are zero. torch's
`grid_sample(…, align_corners=True)` with zero padding, without making the
full-size grid.
"""
gridsample!(out::AbstractMatrix, img::AbstractMatrix, grid::AbstractArray) = gridsample!(device(out), out, img, grid)
function gridsample!(dev, out::AbstractMatrix, img::AbstractMatrix, grid::AbstractArray)
    size(grid, 3) == 2 || throw(DimensionMismatch("grid must be (gw, gh, 2[, 1]), not $(size(grid))"))
    launch!(dev, gridsample_kernel!, (out, img, grid), size(out))
    return out
end
