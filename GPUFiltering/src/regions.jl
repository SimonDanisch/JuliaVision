# Measuring and editing rectangular regions: per-row statistics inside boxes,
# filling boxes, and the bounding boxes of connected components. Boxes are
# `(x0, y0, x1, y1)`, 1-based and inclusive, any indexable isbits value such as
# a `Vec{4, Int32}`; a box list is a device vector of them.

"""
    RowStats

What [`boxrows!`](@ref) finds in one row of a box: how many pixels the
predicate selects, in how many horizontal runs, the first and last column
holding one (0 for none), and the sum of the image's colour over them.
"""
struct RowStats
    count::Int32
    runs::Int32
    first::Int32
    last::Int32
    sum::RGB{Float32}
end

RowStats() = RowStats(0, 0, 0, 0, RGB{Float32}(0, 0, 0))

function boxrows_kernel!(stats, img, mask, boxes, pred, countp)
    I, inside = workitem(size(stats))
    inside || return nothing
    r, k = I[1], I[2]
    n = Int32(0); starts = Int32(0); firstx = Int32(0); lastx = Int32(0)
    acc = RGB{Float32}(0, 0, 0)
    if k <= paramvalue(countp)
        @inbounds b = boxes[k]
        y = Int(b[2]) + r - 1
        if y <= b[4] && 1 <= y <= size(mask, 2)
            previous = false
            for x in max(Int(b[1]), 1):min(Int(b[3]), size(mask, 1))
                @inbounds m = pred(mask[x, y])
                if m
                    n += Int32(1)
                    starts += Int32(!previous)
                    firstx == 0 && (firstx = Int32(x))
                    lastx = Int32(x)
                    @inbounds acc = addpixel(acc, tofloat(img[x, y]))
                end
                previous = m
            end
        end
    end
    @inbounds stats[r, k] = RowStats(n, starts, firstx, lastx, acc)
    return nothing
end

"""
    boxrows!([device,] stats, img, mask, boxes, pred = !iszero, count = length(boxes)) -> stats

Per row of each of the first `count` boxes, the pixels `pred(mask[x, y])`
selects, as a [`RowStats`](@ref): `stats[r, k]` for row `r` of box `k`,
counted from the box's top. Rows past a box's bottom and boxes past `count` are
empty. `stats` is a `(rows, capacity)` matrix; `count` is a value or a
one-element device array, for a number of boxes that changes between runs of a
graph.

A text line's profile from it: ink per row gives the baseline and the x-height,
pixels per run the stroke width, the first and last columns the ink's extent,
and the sums its colour.
"""
boxrows!(stats::AbstractMatrix{RowStats}, img::AbstractMatrix, mask::AbstractMatrix, boxes::AbstractVector,
         pred = !iszero, count = length(boxes)) = boxrows!(device(stats), stats, img, mask, boxes, pred, count)
function boxrows!(dev, stats::AbstractMatrix{RowStats}, img::AbstractMatrix, mask::AbstractMatrix,
                  boxes::AbstractVector, pred, count)
    size(img) == size(mask) || throw(DimensionMismatch("img $(size(img)) vs mask $(size(mask))"))
    size(stats, 2) <= length(boxes) || throw(DimensionMismatch("$(size(stats, 2)) columns for $(length(boxes)) boxes"))
    launch!(dev, boxrows_kernel!, (stats, img, mask, boxes, pred, count), size(stats))
    return stats
end

function fillboxes_kernel!(img, boxes, value, countp)
    I, inside = workitem(size(img))
    inside || return nothing
    x, y = I[1], I[2]
    covered = false
    for k in 1:paramvalue(countp)
        @inbounds b = boxes[k]
        covered |= b[1] <= x <= b[3] && b[2] <= y <= b[4]
    end
    if covered
        @inbounds img[I] = value
    end
    return nothing
end

"""
    fillboxes!([device,] img, boxes, value, count = length(boxes)) -> img

`value` in every pixel of `img` inside one of the first `count` boxes. One
thread per pixel tries every box, so it suits a few hundred boxes over an image
of a few million pixels; `count` is a value or a one-element device array.
"""
fillboxes!(img::AbstractMatrix, boxes::AbstractVector, value, count = length(boxes)) =
    fillboxes!(device(img), img, boxes, value, count)
function fillboxes!(dev, img::AbstractMatrix, boxes::AbstractVector, value, count)
    launch!(dev, fillboxes_kernel!, (img, boxes, convert(eltype(img), value), count), size(img))
    return img
end

function componentstats_init_kernel!(stats, counter)
    k, inside = workitem(size(stats, 2))
    inside || return nothing
    @inbounds begin
        stats[1, k] = typemax(Int32); stats[2, k] = typemax(Int32)
        stats[3, k] = Int32(0); stats[4, k] = Int32(0); stats[5, k] = Int32(0)
        k == 1 && (counter[1] = Int32(0))
    end
    return nothing
end

function componentstats_kernel!(stats, labels)
    I, inside = workitem(size(labels))
    inside || return nothing
    @inbounds root = labels[I]
    if root != Int32(0)
        x = Int32(I[1]); y = Int32(I[2])
        # Read by the next launch only: no ordering among these is needed.
        KA.@atomic :monotonic min(stats[1, root], x)
        KA.@atomic :monotonic min(stats[2, root], y)
        KA.@atomic :monotonic max(stats[3, root], x)
        KA.@atomic :monotonic max(stats[4, root], y)
        KA.@atomic :monotonic stats[5, root] += Int32(1)
    end
    return nothing
end

function componentlist_kernel!(boxes, counter, stats, minarea::Int32)
    k, inside = workitem(size(stats, 2))
    inside || return nothing
    @inbounds area = stats[5, k]
    if area >= minarea
        slot = KA.@atomic :monotonic counter[1] += Int32(1)   # a unique slot whatever the ordering
        if slot <= size(boxes, 2)
            @inbounds for j in 1:5
                boxes[j, slot] = stats[j, k]
            end
        end
    end
    return nothing
end

"""
    componentboxes!([device,] boxes, count, labels, minarea = 1[, stats]) -> boxes

The bounding boxes of the components [`components!`](@ref) labelled, those of
at least `minarea` pixels: column `j` of `boxes` is `(x0, y0, x1, y1, area)` of
one of them, in no particular order, and `count[1]` how many there are; more
than `size(boxes, 2)` means some did not fit. OpenCV's
`connectedComponentsWithStats`, compacted on the device so only the boxes come
back. `stats` is an `Int32` `(5, length(labels))` matrix; without it one is
allocated.
"""
function componentboxes!(boxes::AbstractMatrix{Int32}, count::AbstractVector{Int32}, labels::AbstractMatrix{Int32},
                         minarea::Integer = 1)
    dev = device(boxes)
    return componentboxes!(dev, boxes, count, labels, minarea, KA.allocate(dev, Int32, (5, length(labels))))
end
function componentboxes!(dev, boxes::AbstractMatrix{Int32}, count::AbstractVector{Int32}, labels::AbstractMatrix{Int32},
                         minarea::Integer, stats::AbstractMatrix{Int32})
    size(boxes, 1) == 5 || throw(DimensionMismatch("boxes must be (5, capacity), not $(size(boxes))"))
    size(stats) == (5, length(labels)) || throw(DimensionMismatch("stats $(size(stats)) for $(length(labels)) labels"))
    launch!(dev, componentstats_init_kernel!, (stats, count), length(labels))
    launch!(dev, componentstats_kernel!, (stats, labels), size(labels))
    launch!(dev, componentlist_kernel!, (boxes, count, stats, Int32(minarea)), length(labels))
    return boxes
end
