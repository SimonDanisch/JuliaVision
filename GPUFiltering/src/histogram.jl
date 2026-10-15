# Histograms, Otsu's threshold and quantiles, all on the device.

@kernel function zero_kernel!(a)
    i = @index(Global, Linear)
    @inbounds a[i] = zero(eltype(a))
end

# Spread over the columns of `counts`, so the atomics of neighbouring pixels
# rarely meet on one counter.
@kernel function histogram_kernel!(counts, @Const(img), by, lo::Float32, scale::Float32)
    i = @index(Global, Linear)
    nbins, groups = size(counts)
    @inbounds v = Float32(by(img[i]))
    bin = clamp(unsafe_trunc(Int, (v - lo) * scale) + 1, 1, nbins)
    KA.@atomic counts[bin, mod(i - 1, groups) + 1] += Int32(1)
end

"""
    histogram!([device,] counts, img, by = luma, range = (0, 1)) -> counts

How many pixels of `img` fall in each of `size(counts, 1)` equal bins over
`range`, of the value `by(pixel)`; values outside count in the first or last
bin. `counts` is an `Int32` matrix whose columns are partial counts, spread so
that the atomics of neighbouring pixels rarely meet: the histogram is the sum
over its columns ([`otsu!`](@ref) reads it that way). 64 columns is a good
spread.
"""
histogram!(counts::AbstractMatrix{Int32}, img::AbstractMatrix, by = luma, range = (0.0f0, 1.0f0)) =
    histogram!(device(counts), counts, img, by, range)
function histogram!(dev, counts::AbstractMatrix{Int32}, img::AbstractMatrix, by, range)
    lo, hi = Float32.(range)
    hi > lo || throw(ArgumentError("empty range $range"))
    launch!(dev, zero_kernel!, (counts,), length(counts))
    launch!(dev, histogram_kernel!, (counts, img, by, lo, size(counts, 1) / (hi - lo)), length(img))
    return counts
end

# Otsu's method in one thread over the bins: the threshold between bins that
# maximises the variance between the two classes it makes.
@kernel function otsu_kernel!(out, @Const(counts), lo::Float32, width::Float32)
    i = @index(Global, Linear)
    nbins, groups = size(counts)
    total = 0.0f0; sum_all = 0.0f0
    for bin in 1:nbins
        c = 0.0f0
        for g in 1:groups
            @inbounds c += counts[bin, g]
        end
        total += c; sum_all += c * (bin - 0.5f0)
    end
    wb = 0.0f0; sb = 0.0f0; best = -1.0f0; thr = Float32(nbins); mb = 0.0f0; mf = 0.0f0
    for bin in 1:(nbins - 1)
        c = 0.0f0
        for g in 1:groups
            @inbounds c += counts[bin, g]
        end
        wb += c; sb += c * (bin - 0.5f0)
        wf = total - wb
        if wb > 0.0f0 && wf > 0.0f0
            between = wb * wf * (sb / wb - (sum_all - sb) / wf)^2
            if between > best
                best = between; thr = Float32(bin); mb = sb / wb; mf = (sum_all - sb) / wf
            end
        end
    end
    @inbounds out[1] = lo + thr * width
    @inbounds out[2] = (mf - mb) * width
end

"""
    otsu!([device,] out, counts, range = (0, 1)) -> out

Otsu's threshold for the histogram `counts` over `range` (as made by
[`histogram!`](@ref)): `out[1]` is the threshold, the value at the bin edge
that best separates two classes, and `out[2]` the distance between the two
classes' means, which says whether there were two classes to separate at all.
`out` is a two-element `Float32` device array.
"""
otsu!(out::AbstractVector{Float32}, counts::AbstractMatrix{Int32}, range = (0.0f0, 1.0f0)) =
    otsu!(device(out), out, counts, range)
function otsu!(dev, out::AbstractVector{Float32}, counts::AbstractMatrix{Int32}, range)
    length(out) >= 2 || throw(DimensionMismatch("otsu! writes two values, out has $(length(out))"))
    lo, hi = Float32.(range)
    launch!(dev, otsu_kernel!, (out, counts, lo, (hi - lo) / size(counts, 1)), 1)
    return out
end

# A quantile by radix selection on the bits of non-negative Float32 values,
# whose order is the order of the bits: four passes of one byte each, from the
# top, each counting the candidates that agree with the bytes chosen so far.

@kernel function select_init_kernel!(state, k::UInt32)
    i = @index(Global, Linear)
    @inbounds begin state[1] = UInt32(0); state[2] = k; end
end

@kernel function select_count_kernel!(hist, @Const(img), by, @Const(state), stride::Int, shift::Int)
    i = @index(Global, Linear)
    @inbounds bits = reinterpret(UInt32, Float32(by(img[1 + (i - 1) * stride])))
    @inbounds prefix = state[1]
    mask = shift == 24 ? UInt32(0) : typemax(UInt32) << (shift + 8)
    if bits & mask == prefix
        bin = Int((bits >> shift) & UInt32(255)) + 1
        KA.@atomic hist[bin, (i - 1) ÷ 256 + 1] += Int32(1)
    end
end

@kernel function select_sum_kernel!(counts, @Const(hist))
    i = @index(Global, Linear)
    n = Int32(0)
    for g in 1:size(hist, 2)
        @inbounds n += hist[i, g]
    end
    @inbounds counts[i] = n
end

@kernel function select_pick_kernel!(state, out, @Const(counts), shift::Int)
    i = @index(Global, Linear)
    @inbounds k = Int32(state[2])
    bin = 1
    while bin < 256 && k > (@inbounds counts[bin])
        @inbounds k -= counts[bin]; bin += 1
    end
    @inbounds begin
        state[1] = state[1] | (UInt32(bin - 1) << shift); state[2] = UInt32(k)
        shift == 0 && (out[1] = reinterpret(Float32, state[1]))
    end
end

"""
    quantilesizes(n; stride = 1) -> (hist, counts, state)

The sizes of the temporaries [`selectquantile!`](@ref) takes for an image of `n`
pixels: an `Int32` histogram, `Int32` counts and a `UInt32` state.
"""
quantilesizes(n::Integer; stride::Integer = 1) = ((256, cld(length(1:stride:n), 256)), (256,), (2,))

"""
    selectquantile!([device,] out, img, p, by = luma, stride = 1[, tmp]) -> out

`out[1]` becomes the `p`-quantile of `by(pixel)` over every `stride`-th pixel of
`img`, exactly: the k-th smallest value, `k = round(p * samples)`. The values
must not be negative. `tmp = (hist, counts, state)` are the temporaries of
[`quantilesizes`](@ref); without them they are allocated.
"""
function selectquantile!(out::AbstractVector{Float32}, img::AbstractMatrix, p::Real, by = luma, stride::Integer = 1)
    hs, cs, ss = quantilesizes(length(img); stride)
    dev = device(out)
    tmp = (KA.allocate(dev, Int32, hs), KA.allocate(dev, Int32, cs), KA.allocate(dev, UInt32, ss))
    return selectquantile!(dev, out, img, p, by, stride, tmp)
end
function selectquantile!(dev, out::AbstractVector{Float32}, img::AbstractMatrix, p::Real, by, stride::Integer, tmp::Tuple)
    0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1], not $p"))
    hist, counts, state = tmp
    n = length(1:stride:length(img))
    size(hist) == first(quantilesizes(length(img); stride)) || throw(DimensionMismatch("hist $(size(hist)), see quantilesizes"))
    k = UInt32(clamp(round(Int, p * n), 1, n))
    launch!(dev, select_init_kernel!, (state, k), 1)
    for shift in (24, 16, 8, 0)
        launch!(dev, zero_kernel!, (hist,), length(hist))
        launch!(dev, select_count_kernel!, (hist, img, by, state, stride, shift), n)
        launch!(dev, select_sum_kernel!, (counts, hist), 256)
        launch!(dev, select_pick_kernel!, (state, out, counts, shift), 1)
    end
    return out
end
