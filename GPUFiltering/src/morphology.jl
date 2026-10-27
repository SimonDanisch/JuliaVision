# Erosion and dilation with a square window, and the opening built from them.
# A square's minimum is the minimum over rows of the minimum over columns, so
# each is two one-dimensional passes. Pixels outside the image take no part,
# so a region may touch the border.

function minmax_kernel!(out, img, r::Int32, dilate::Bool, ::Val{DIM}) where {DIM}
    I, inside = workitem(size(img))
    inside || return nothing
    i, j = I[1], I[2]
    len = size(img, DIM)
    @inbounds v = img[i, j]
    for k in -r:r
        p = (DIM == 1 ? i : j) + k
        if 1 <= p <= len
            @inbounds m = DIM == 1 ? img[p, j] : img[i, p]
            v = dilate ? max(v, m) : min(v, m)
        end
    end
    @inbounds out[I] = v
    return nothing
end

function minmax!(dev, out, img, r, tmp, dilate::Bool)
    size(out) == size(img) == size(tmp) || throw(DimensionMismatch("out $(size(out)), img $(size(img)), tmp $(size(tmp))"))
    launch!(dev, minmax_kernel!, (tmp, img, Int32(r), dilate, Val(1)), size(img))
    launch!(dev, minmax_kernel!, (out, tmp, Int32(r), dilate, Val(2)), size(img))
    return out
end

"""
    erode!([device,] out, img, r[, tmp]) -> out
    dilate!([device,] out, img, r[, tmp]) -> out

The minimum (erode) or maximum (dilate) of `img` over the `(2r + 1)²` square
around each pixel. `tmp`, the size of `img`, holds the pass between the two
directions; without it one is allocated.
"""
erode!(out::AbstractMatrix, img::AbstractMatrix, r::Integer) = erode!(device(out), out, img, r, similar(img))
erode!(dev, out::AbstractMatrix, img::AbstractMatrix, r::Integer, tmp::AbstractMatrix) = minmax!(dev, out, img, r, tmp, false)
dilate!(out::AbstractMatrix, img::AbstractMatrix, r::Integer) = dilate!(device(out), out, img, r, similar(img))
dilate!(dev, out::AbstractMatrix, img::AbstractMatrix, r::Integer, tmp::AbstractMatrix) = minmax!(dev, out, img, r, tmp, true)

"""
    opening!([device,] out, img, r[, tmp]) -> out

Erosion, then dilation, with the `(2r + 1)²` square: what is narrower than the
square goes, the rest keeps its shape.
"""
opening!(out::AbstractMatrix, img::AbstractMatrix, r::Integer) = opening!(device(out), out, img, r, similar(img))
function opening!(dev, out::AbstractMatrix, img::AbstractMatrix, r::Integer, tmp::AbstractMatrix)
    minmax!(dev, out, img, r, tmp, false)
    minmax!(dev, out, out, r, tmp, true)
    return out
end
