# Sauvola's adaptive threshold: a pixel is ink when it is darker than its
# neighbourhood's mean by more than the neighbourhood's spread allows, so faint
# pencil next to a dark heading both survive. The window sums are two passes,
# along rows and then down columns, of the luma and its square.

@kernel function rowsums_kernel!(s1, s2, @Const(lum), r::Int32)
    I = @index(Global, Cartesian)
    x, y = I[1], I[2]
    w = size(lum, 1)
    a = 0.0f0; b = 0.0f0
    for xx in max(1, x - r):min(w, x + r)
        @inbounds v = lum[xx, y]
        a += v; b += v * v
    end
    @inbounds begin s1[I] = a; s2[I] = b; end
end

@kernel function sauvola_kernel!(out, @Const(lum), @Const(s1), @Const(s2), r::Int32, k::Float32, range::Float32)
    I = @index(Global, Cartesian)
    x, y = I[1], I[2]
    w, h = size(lum)
    a = 0.0f0; b = 0.0f0
    for yy in max(1, y - r):min(h, y + r)
        @inbounds begin a += s1[x, yy]; b += s2[x, yy]; end
    end
    n = Float32((min(w, x + r) - max(1, x - r) + 1) * (min(h, y + r) - max(1, y - r) + 1))
    mean = a / n; sd = sqrt(max(b / n - mean * mean, 0.0f0))
    @inbounds out[I] = Gray{Bool}(lum[I] > mean * (1.0f0 + k * (sd / range - 1.0f0)))
end

"""
    sauvola!([device,] out::AbstractMatrix{Gray{Bool}}, img, window = 31, k = 0.2, range = 0.5[, tmp]) -> out

`img` thresholded per pixel by Sauvola's rule over the `window × window`
square around it (clipped at the border): white where its luma exceeds
`mean * (1 + k * (sd / range - 1))`. `tmp = (luma, rowsums, rowsquares)` are
three `Float32` matrices the size of `img`; without them they are allocated.
"""
function sauvola!(out::AbstractMatrix{Gray{Bool}}, img::AbstractMatrix, window::Integer = 31, k::Real = 0.2f0, range::Real = 0.5f0)
    dev = device(out)
    tmp = ntuple(_ -> KA.allocate(dev, Float32, size(img)), 3)
    return sauvola!(dev, out, img, window, k, range, tmp)
end
function sauvola!(dev, out::AbstractMatrix{Gray{Bool}}, img::AbstractMatrix, window::Integer, k::Real, range::Real, tmp::Tuple)
    size(out) == size(img) || throw(DimensionMismatch("out $(size(out)) vs img $(size(img))"))
    lum, s1, s2 = tmp
    r = Int32(window ÷ 2)
    launch!(dev, luma_kernel!, (lum, img), size(img))
    launch!(dev, rowsums_kernel!, (s1, s2, lum, r), size(img))
    launch!(dev, sauvola_kernel!, (out, lum, s1, s2, r, Float32(k), Float32(range)), size(img))
    return out
end
