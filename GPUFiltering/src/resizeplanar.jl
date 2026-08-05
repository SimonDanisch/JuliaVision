# RGB image -> the planar float tensor a DNNKernels graph takes as input.
#
# Every model port on this machine starts with the same step: the editor holds a
# frame as an `AbstractMatrix{<:AbstractRGB}` at timeline resolution, and the
# exported graph wants a `(w, h, 3, 1)` `Float32` — the Julia column-major view of
# torch's `(1, 3, h, w)` — at the resolution the network was traced for. Doing it
# in one kernel rather than a resize followed by a de-interleave keeps the full
# frame off the bus twice and gives the graph its input already laid out.
#
# ## Matching torch's `align_corners=false`
#
# The classifier in `NeuralLUTRunner` is upstream's `Classifier` with its leading
# `nn.Upsample(size=(256,256), mode='bilinear')` stripped, because exporting a
# resize that throws its input resolution away would bake a frame size into the
# graph. The resize therefore happens here instead, and it has to be the *same*
# resize or the three predicted blend weights drift.
#
# `nn.Upsample(mode='bilinear')` defaults to `align_corners=False`, which maps an
# output index to `src = (dst + 0.5) * in/out - 0.5` and then clamps `src` up to 0
# — the clamp is applied to the coordinate, before the fraction is taken, so a
# destination pixel whose source falls left of the first sample gets fraction 0
# rather than a negative weight. The upper neighbour is clamped to the last index.
# Both clamps are reproduced below; getting either wrong is a half-pixel shift
# that shows up as a small, plausible-looking difference in the predicted look
# rather than as an obvious failure.

@kernel function resizeplanar_kernel!(dst, @Const(img), sx::Float32, sy::Float32,
                                      w::Int32, h::Int32,
                                      mean::NTuple{3,Float32}, invstd::NTuple{3,Float32})
    I = @index(Global, Cartesian)

    # (dst + 0.5) * scale - 0.5 in 0-based coordinates, clamped at 0 as torch
    # does. `I[1] - 1` converts Julia's 1-based index to torch's 0-based one.
    fx = max((Float32(I[1]) - 0.5f0) * sx - 0.5f0, 0.0f0)
    fy = max((Float32(I[2]) - 0.5f0) * sy - 0.5f0, 0.0f0)

    x0 = unsafe_trunc(Int32, fx)          # fx >= 0, so trunc == floor
    y0 = unsafe_trunc(Int32, fy)
    dx = fx - Float32(x0)
    dy = fy - Float32(y0)

    # 0-based -> 1-based; the upper neighbour saturates at the last sample.
    x1 = min(x0 + Int32(2), w); x0 += Int32(1)
    y1 = min(y0 + Int32(2), h); y0 += Int32(1)

    @inbounds begin
        c00 = tofloat(img[x0, y0]); c10 = tofloat(img[x1, y0])
        c01 = tofloat(img[x0, y1]); c11 = tofloat(img[x1, y1])

        w00 = (1.0f0 - dx) * (1.0f0 - dy)
        w10 = dx * (1.0f0 - dy)
        w01 = (1.0f0 - dx) * dy
        w11 = dx * dy

        r = w00 * c00.r + w10 * c10.r + w01 * c01.r + w11 * c11.r
        g = w00 * c00.g + w10 * c10.g + w01 * c01.g + w11 * c11.g
        b = w00 * c00.b + w10 * c10.b + w01 * c01.b + w11 * c11.b

        # Normalisation folded into the same pass. It is one multiply-add on a
        # value already in a register, against a second full read and write of
        # the whole tensor if it were its own kernel — and `invstd` is
        # reciprocated once on the host rather than dividing per pixel.
        dst[I[1], I[2], 1, 1] = (r - mean[1]) * invstd[1]
        dst[I[1], I[2], 2, 1] = (g - mean[2]) * invstd[2]
        dst[I[1], I[2], 3, 1] = (b - mean[3]) * invstd[3]
    end
end

"""
    resizeplanar!(dst, img; mean = (0, 0, 0), std = (1, 1, 1)) -> dst

Bilinearly resize `img` into `dst`, de-interleaving the channels on the way and
optionally normalising them.

`img` is any `AbstractMatrix{<:AbstractRGB}`; `dst` is a `(w, h, 3, 1)` `Float32`
array on the same backend, which is what a `DNNKernels` graph exported from a
torch `(1, 3, h, w)` input expects. The output resolution is `size(dst)[1:2]`, so
the caller states it by allocating the destination.

The sampling matches `torch.nn.Upsample(mode='bilinear')` with its default
`align_corners=False`, so a network whose own resize was stripped at export time
sees the same pixels it would have made itself. See the comment above for the two
clamps that has to get right.

`mean` and `std` are per-channel and applied as `(v - mean) / std` after the
resample — the ImageNet statistics for a DINOv2-style model, or the default
identity for one that wants raw 0..1. They are here rather than in a second
kernel because the value is already in a register: a separate normalisation pass
is a full read and write of the tensor for one multiply-add.

Values pass through unclamped. The graph's first op is a convolution, not a pixel
store, and a model whose input legitimately leaves 0..1 (which is exactly what
normalisation does) must not have it clipped on the way in.

Asynchronous, like every kernel here: synchronize before reading on the host.
"""
function resizeplanar!(dst::AbstractArray{Float32,4}, img::AbstractMatrix{<:AbstractRGB};
                       mean = (0.0f0, 0.0f0, 0.0f0), std = (1.0f0, 1.0f0, 1.0f0))
    size(dst, 3) == 3 ||
        throw(ArgumentError("dst must have 3 channels, got $(size(dst, 3))"))
    size(dst, 4) == 1 ||
        throw(ArgumentError("dst batch must be 1, got $(size(dst, 4))"))
    w, h = size(img)
    (w > 0 && h > 0) || throw(ArgumentError("empty source image"))
    backend = KA.get_backend(dst)
    # A HOST source is uploaded rather than refused. The kernel reads `img` on
    # the device, so both must live there — but every caller in the repo starts
    # from a host frame, because that is what a decoder hands the editor, and
    # `predictlut(model, img)` / `depthmap!(model, img)` both *declare*
    # `AbstractMatrix{<:AbstractRGB}` with no backend in the type.
    #
    # So the documented entry point of two shipped packages could not be called
    # at all: it threw `dst is on LavaBackend(...), img is on CPU()`. Both the
    # check and those callers arrived in the same commit (`28782f1`), so the host
    # path never worked — the numbers in `small-models/REPORT.md` came from
    # somewhere that already had a device image. Neither package's test suite
    # runs a forward pass, which is why nothing said so.
    #
    # `RGB{N0f8}` is an isbits 3-byte element and round-trips through a
    # `LavaArray` unchanged, so this is one upload of the source frame, not a
    # conversion. A *device* image on a different backend is still an error:
    # that is a genuine mixup and uploading would hide it.
    if KA.get_backend(img) != backend
        KA.get_backend(img) isa KA.CPU ||
            throw(ArgumentError("dst is on $backend, img is on $(KA.get_backend(img))"))
        gpuimg = KA.allocate(backend, eltype(img), w, h)
        copyto!(gpuimg, img)
        return resizeplanar!(dst, gpuimg; mean, std)
    end

    any(iszero, std) && throw(ArgumentError("std has a zero component: $std"))
    μ = NTuple{3,Float32}(Float32.(mean))
    invstd = NTuple{3,Float32}(inv.(Float32.(std)))

    sx = Float32(w / size(dst, 1))
    sy = Float32(h / size(dst, 2))
    resizeplanar_kernel!(backend)(dst, img, sx, sy, Int32(w), Int32(h), μ, invstd;
                                  ndrange = (size(dst, 1), size(dst, 2)))
    return dst
end
