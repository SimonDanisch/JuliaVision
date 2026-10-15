# Connected components: which pixels of a mask belong together.

@kernel function components_init_kernel!(labels, @Const(mask), pred)
    k = @index(Global, Linear)
    @inbounds labels[k] = pred(mask[k]) ? Int32(k) : Int32(0)
end

# The root of `k`'s tree, read atomically while other threads hook trees.
@inline function componentroot(labels, k::Int32)
    p = KA.@atomic :monotonic labels[k]
    while p != k
        k = p; p = KA.@atomic :monotonic labels[k]
    end
    return k
end

# A monotone union: the larger root hooks under the smaller, retried when
# another thread changed either root first.
@inline function joincomponents!(labels, a::Int32, b::Int32)
    while true
        a = componentroot(labels, a); b = componentroot(labels, b)
        a == b && break
        hi = max(a, b); lo = min(a, b)
        result = KA.@atomicreplace :monotonic labels[hi] hi => lo
        result.success && break
    end
    return nothing
end

@kernel function components_merge_kernel!(labels, @Const(mask), pred, w::Int32)
    k = @index(Global, Linear)
    x = mod(k - 1, w) + 1; y = (k - 1) ÷ w + 1
    if @inbounds pred(mask[k])
        if x > 1 && (@inbounds pred(mask[k - 1]))
            joincomponents!(labels, Int32(k), Int32(k - 1))
        end
        if y > 1
            for dx in -1:1
                j = k - w + dx
                if 1 <= x + dx <= w && (@inbounds pred(mask[j]))
                    joincomponents!(labels, Int32(k), Int32(j))
                end
            end
        end
    end
end

# Every pixel straight to its root. A thread that replaces its own parent by
# the root keeps every path through it leading to the same root, so the
# threads that still walk through it end where they would have.
@kernel function components_flatten_kernel!(labels)
    k = @index(Global, Linear)
    @inbounds l = labels[k]
    if l != 0
        r = l
        while (@inbounds labels[r]) != r
            @inbounds r = labels[r]
        end
        @inbounds labels[k] = r
    end
end

"""
    components!([device,] labels, mask, pred = !iszero) -> labels

The 8-connected components of the pixels of `mask` for which `pred` holds:
`labels[k]` is the linear index of the component's first pixel, the same for
every pixel of one component, and 0 outside the mask. `labels` is an `Int32`
array with as many elements as `mask`. A union-find with atomics, so the
work is one pass over the pixels however many components there are.
"""
components!(labels::AbstractArray{Int32}, mask::AbstractMatrix, pred = !iszero) =
    components!(device(labels), labels, mask, pred)
function components!(dev, labels::AbstractArray{Int32}, mask::AbstractMatrix, pred)
    length(labels) == length(mask) || throw(DimensionMismatch("labels $(size(labels)) vs mask $(size(mask))"))
    n = length(mask)
    launch!(dev, components_init_kernel!, (labels, mask, pred), n)
    launch!(dev, components_merge_kernel!, (labels, mask, pred, Int32(size(mask, 1))), n)
    launch!(dev, components_flatten_kernel!, (labels,), n)
    return labels
end
