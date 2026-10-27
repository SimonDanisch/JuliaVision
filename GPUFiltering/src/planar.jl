# An image into the planar float tensor a DNNKernels graph takes as input, the
# view of torch's `(1, 3, h, w)` that Julia's column-major `(w, h, 3, 1)` is:
# resampled, de-interleaved and normalised in one pass, so the frame crosses the
# bus once and the graph gets its input already laid out.

"""
    PlanarMap(scale, offset, extent)

Where a planar tensor's pixels sample their source. Destination pixel `(u, v)`
covers the source from `offset + scale .* (u - 1, v - 1)` to
`offset + scale .* (u, v)`, in source pixel EDGE coordinates (pixel `i` spans
`i - 1` to `i`); bilinear sampling reads the centre of that cover. Only the
destination pixels up to `extent` sample, the rest take the fill: a letterbox,
or a line padded to a fixed width. An isbits value: pass it as is, or in a
device array when it changes between runs of a graph, one per batch slot.
"""
struct PlanarMap
    scale::Vec2f
    offset::Vec2f
    extent::Vec{2, Int32}
end

"How a planar tensor's pixel reads its cover in the source."
abstract type Sampling end
"The source at the cover's centre, bilinearly, the border replicated."
struct Bilinear <: Sampling end
"The mean of the source pixels the cover touches: for shrinking without aliasing thin strokes away."
struct Area <: Sampling end
"""The source at the cover's centre by cubic convolution, without antialiasing,
clamped to [0, 1]: what OpenCV's `INTER_CUBIC` and torch's bicubic resize of an
8-bit image, which saturate the kernel's overshoot at sharp edges, feed a
network trained on them."""
struct Bicubic <: Sampling end

@inline function sampleat(::Bilinear, img, m::PlanarMap, u::Int, v::Int)
    return bilinear(img, m.offset[1] + m.scale[1] * (u - 0.5f0) + 0.5f0,
                         m.offset[2] + m.scale[2] * (v - 0.5f0) + 0.5f0)
end
@inline function sampleat(::Bicubic, img, m::PlanarMap, u::Int, v::Int)
    c = bicubic(img, m.offset[1] + m.scale[1] * (u - 0.5f0) + 0.5f0, m.offset[2] + m.scale[2] * (v - 0.5f0) + 0.5f0)
    return saturate(c)
end
@inline saturate(c::Float32) = clamp(c, 0.0f0, 1.0f0)
@inline saturate(c::RGB{Float32}) = RGB{Float32}(saturate(c.r), saturate(c.g), saturate(c.b))
@inline function sampleat(::Area, img, m::PlanarMap, u::Int, v::Int)
    return areamean(img, m.offset[1] + m.scale[1] * (u - 1), m.offset[1] + m.scale[1] * u,
                         m.offset[2] + m.scale[2] * (v - 1), m.offset[2] + m.scale[2] * v)
end

# The map of batch slot `b`: one value for every slot, or one per slot.
@inline planarmap(m::PlanarMap, b) = m
@inline planarmap(maps::AbstractVector{PlanarMap}, b) = @inbounds maps[b]

function sampleplanar_kernel!(dst, img, maps, sampling, mean::NTuple{3, Float32},
                              invstd::NTuple{3, Float32}, fill::Float32)
    I, inside = workitem((size(dst, 1), size(dst, 2), size(dst, 4)))
    inside || return nothing
    u, v, b = I[1], I[2], I[3]
    m = planarmap(maps, b)
    c = RGB{Float32}(fill, fill, fill)
    if u <= m.extent[1] && v <= m.extent[2]
        c = sampleat(sampling, img, m, u, v)
    end
    @inbounds begin
        dst[u, v, 1, b] = (c.r - mean[1]) * invstd[1]
        dst[u, v, 2, b] = (c.g - mean[2]) * invstd[2]
        dst[u, v, 3, b] = (c.b - mean[3]) * invstd[3]
    end
    return nothing
end

"""
    sampleplanar!([device,] dst, img, maps, sampling = Bilinear(), mean = (0, 0, 0), std = (1, 1, 1), fill = 0) -> dst

`img`, any colour or grey image, resampled into `dst`, a `(w, h, 3, b)`
`Float32` tensor of `b` batch slots, each channel normalised as
`(v - mean) / std`. `maps` is one [`PlanarMap`](@ref) for every slot, or a
device vector of them, one per slot: line `k` of a batch cut out of a page. `fill`
is the source value of the pixels outside a map's extent, normalised like the
rest; a slot whose extent is zero is all fill. Values pass through unclamped: a
network's input legitimately leaves [0, 1] once normalised.
"""
sampleplanar!(dst::AbstractArray{Float32, 4}, img::AbstractMatrix, map, sampling::Sampling = Bilinear(),
              mean = (0.0f0, 0.0f0, 0.0f0), std = (1.0f0, 1.0f0, 1.0f0), fill::Real = 0.0f0) =
    sampleplanar!(device(dst), dst, img, map, sampling, mean, std, fill)
function sampleplanar!(dev, dst::AbstractArray{Float32, 4}, img::AbstractMatrix, map, sampling::Sampling,
                       mean, std, fill::Real)
    size(dst, 3) == 3 || throw(DimensionMismatch("dst must be (w, h, 3, b), not $(size(dst))"))
    checkmaps(map, size(dst, 4))
    any(iszero, std) && throw(ArgumentError("std has a zero component: $std"))
    μ = NTuple{3, Float32}(Float32.(mean))
    invstd = NTuple{3, Float32}(inv.(Float32.(std)))
    launch!(dev, sampleplanar_kernel!, (dst, img, map, sampling, μ, invstd, Float32(fill)), (size(dst, 1), size(dst, 2), size(dst, 4)))
    return dst
end

"""
    resizeplanar!([device,] dst, img; mean = (0, 0, 0), std = (1, 1, 1)) -> dst

`img` resized bilinearly to all of `dst` as torch's
`nn.Upsample(mode='bilinear')` does with its default `align_corners=False`, so a
network whose own resize was stripped at export time sees the pixels it would
have made itself. See [`sampleplanar!`](@ref).

Without a device, a host `img` is uploaded to `dst`'s backend first: decoders
hand out host frames. A device `img` on another backend is an error.
"""
checkmaps(::PlanarMap, slots) = nothing
checkmaps(maps::AbstractVector{PlanarMap}, slots) = length(maps) >= slots ||
    throw(DimensionMismatch("$(length(maps)) maps for $slots batch slots"))

function resizeplanar!(dst::AbstractArray{Float32, 4}, img::AbstractMatrix;
                       mean = (0.0f0, 0.0f0, 0.0f0), std = (1.0f0, 1.0f0, 1.0f0))
    backend = device(dst)
    if KA.get_backend(img) != backend
        KA.get_backend(img) isa KA.CPU ||
            throw(ArgumentError("dst is on $backend, img is on $(KA.get_backend(img))"))
        onbackend = KA.allocate(backend, eltype(img), size(img))
        copyto!(onbackend, img)
        img = onbackend
    end
    return resizeplanar!(backend, dst, img; mean, std)
end
function resizeplanar!(dev, dst::AbstractArray{Float32, 4}, img::AbstractMatrix;
                       mean = (0.0f0, 0.0f0, 0.0f0), std = (1.0f0, 1.0f0, 1.0f0))
    w, h = size(img)
    (w > 0 && h > 0) || throw(ArgumentError("empty source image"))
    W, H = size(dst, 1), size(dst, 2)
    map = PlanarMap(Vec2f(w / W, h / H), Vec2f(0, 0), Vec{2, Int32}(W, H))
    return sampleplanar!(dev, dst, img, map, Bilinear(), mean, std, 0.0f0)
end
