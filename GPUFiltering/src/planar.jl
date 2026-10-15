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
one-element device array (a `GPURef`) when it changes between runs of a graph.
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

@inline function sampleat(::Bilinear, img, m::PlanarMap, u::Int, v::Int)
    return bilinear(img, m.offset[1] + m.scale[1] * (u - 0.5f0) + 0.5f0,
                         m.offset[2] + m.scale[2] * (v - 0.5f0) + 0.5f0)
end
@inline function sampleat(::Area, img, m::PlanarMap, u::Int, v::Int)
    return areamean(img, m.offset[1] + m.scale[1] * (u - 1), m.offset[1] + m.scale[1] * u,
                         m.offset[2] + m.scale[2] * (v - 1), m.offset[2] + m.scale[2] * v)
end

@kernel function sampleplanar_kernel!(dst, @Const(img), mapp, sampling, mean::NTuple{3, Float32},
                                      invstd::NTuple{3, Float32}, fill::Float32)
    I = @index(Global, Cartesian)
    u, v = I[1], I[2]
    m = paramvalue(mapp)
    c = RGB{Float32}(fill, fill, fill)
    if u <= m.extent[1] && v <= m.extent[2]
        c = sampleat(sampling, img, m, u, v)
    end
    @inbounds begin
        dst[u, v, 1, 1] = (c.r - mean[1]) * invstd[1]
        dst[u, v, 2, 1] = (c.g - mean[2]) * invstd[2]
        dst[u, v, 3, 1] = (c.b - mean[3]) * invstd[3]
    end
end

"""
    sampleplanar!([device,] dst, img, map::PlanarMap, sampling = Bilinear(), mean = (0, 0, 0), std = (1, 1, 1), fill = 0) -> dst

`img`, any colour or grey image, resampled through `map` into `dst`, a
`(w, h, 3, 1)` `Float32` tensor, each channel normalised as `(v - mean) / std`.
`fill` is the source value of the pixels outside the map's extent, normalised
like the rest. Values pass through unclamped: a network's input legitimately
leaves [0, 1] once normalised.
"""
sampleplanar!(dst::AbstractArray{Float32, 4}, img::AbstractMatrix, map, sampling::Sampling = Bilinear(),
              mean = (0.0f0, 0.0f0, 0.0f0), std = (1.0f0, 1.0f0, 1.0f0), fill::Real = 0.0f0) =
    sampleplanar!(device(dst), dst, img, map, sampling, mean, std, fill)
function sampleplanar!(dev, dst::AbstractArray{Float32, 4}, img::AbstractMatrix, map, sampling::Sampling,
                       mean, std, fill::Real)
    size(dst, 3) == 3 && size(dst, 4) == 1 ||
        throw(DimensionMismatch("dst must be (w, h, 3, 1), not $(size(dst))"))
    any(iszero, std) && throw(ArgumentError("std has a zero component: $std"))
    μ = NTuple{3, Float32}(Float32.(mean))
    invstd = NTuple{3, Float32}(inv.(Float32.(std)))
    launch!(dev, sampleplanar_kernel!, (dst, img, map, sampling, μ, invstd, Float32(fill)), (size(dst, 1), size(dst, 2)))
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
