"""
Image to the conditioner's input, on the host.

`hunyuan3d_cond` takes a normalised 518x518 tensor and deliberately contains none
of the work that produces one — same call `export_depthanything.py` makes, and for
the same reason: baking a resize into the graph fixes one source resolution. So
this file is the graph's missing prologue, and it is the only place the
statistics live.

Two stages, from two different places upstream:

  * `ImageProcessorV2` (`preprocessors.py`) — recentre the object inside its own
    alpha bounding box with a 15% border, composite it onto white, resize to 512
    and map to (-1, 1). This is the stage that matters semantically: the model was
    trained on objects framed this way, and an image that merely *looks* right
    but is framed differently produces a plausible mesh of the wrong proportions.
  * `ImageEncoder.transform` (`conditioner.py`) — back to (0, 1), bilinear resize
    to 518, centre crop, ImageNet normalise. All of it in fp16, because the
    encoder is fp16 and the tensor is cast before the transform, not after.

**The resampling filters are equivalent, not identical.** Upstream resizes uint8
buffers through OpenCV, whose `INTER_AREA` and `INTER_CUBIC` paths for 8-bit
images run in fixed-point integer arithmetic with their own rounding; reproducing
that bit for bit means reimplementing OpenCV's fixed-point resize, and it would
buy a fraction of a grey level. What is implemented here is the same filters in
floating point — exact area averaging for the downscale, Catmull-Rom with
`a = -0.75` for the cubic, half-pixel centres and replicated borders throughout,
which is what OpenCV's own coefficient generation uses. `test/runtests.jl`
measures what that costs against the dumped reference rather than assuming it is
nothing. The *geometry* — the bounding box, the border, the placement — is
integer arithmetic and is exact.
"""

"""
    DINO_MEAN, DINO_STD, DINO_RES

The ImageNet statistics `DinoImageEncoder` normalises with, and the square it
works at. Restated here because the export leaves the transform out of the graph,
so this is the only place the Julia side can read them from, and getting them
wrong produces a plausible mesh rather than an error. `tools/export_hunyuan3d.py`
writes the same three to `preprocess.json` beside the graph.

518 is not a free parameter: DINOv2 splits the input into 14x14 patches and the
export bakes 37x37 of them plus the class token.
"""
const DINO_MEAN = (0.485f0, 0.456f0, 0.406f0)
const DINO_STD = (0.229f0, 0.224f0, 0.225f0)
const DINO_RES = 518

"""
    PROC_RES, BORDER_RATIO

`ImageProcessorV2(size = 512, border_ratio = 0.15)` from the checkpoint's
`config.yaml` — not the class defaults, which are 512 and `None`.

The object is scaled to occupy `1 - BORDER_RATIO` of the square. Both are read
off the config rather than the constructor because the two disagree.
"""
const PROC_RES = 512
const BORDER_RATIO = 0.15

# ------------------------------------------------------------------- resampling

"""
    areasample(src, w, h) -> dst

Exact area averaging, which is what OpenCV's `INTER_AREA` computes when it is
downscaling: an output pixel is the mean of the input it covers, weighted by how
much of each input pixel falls inside it.

`src` is `(W, H, C)`. Written for the crop-and-shrink inside [`recenter`](@ref),
which is always a downscale — upstream picks `INTER_AREA` there for exactly that
reason.
"""
function areasample(src::AbstractArray{T,3}, w::Integer, h::Integer) where {T}
    W, H, C = size(src)
    dst = zeros(Float32, w, h, C)
    sx, sy = W / w, H / h
    for c in 1:C, j in 1:h
        # Half-open input span of this output row, in input coordinates.
        y0, y1 = (j - 1) * sy, j * sy
        for i in 1:w
            x0, x1 = (i - 1) * sx, i * sx
            acc = 0.0f0
            for jj in floor(Int, y0):min(ceil(Int, y1) - 1, H - 1)
                wy = min(y1, jj + 1) - max(y0, jj)
                wy > 0 || continue
                for ii in floor(Int, x0):min(ceil(Int, x1) - 1, W - 1)
                    wx = min(x1, ii + 1) - max(x0, ii)
                    wx > 0 || continue
                    acc += Float32(wx * wy) * Float32(src[ii + 1, jj + 1, c])
                end
            end
            dst[i, j, c] = acc / Float32(sx * sy)
        end
    end
    return dst
end

"""
    cubicweights(t) -> NTuple{4,Float32}

OpenCV's bicubic kernel at `a = -0.75`, sampled at the four taps around a source
coordinate whose fractional part is `t`.

    |x| <= 1 : ((a + 2)|x| - (a + 3))x^2 + 1
    1 < |x| < 2 : a(|x|^3 - 5|x|^2 + 8|x| - 4)

`a = -0.75` and not the -0.5 Catmull-Rom of most graphics code: OpenCV's
`interpolateCubic` uses -0.75, and the two differ visibly on edges.
"""
function cubicweights(t::Float32)
    a = -0.75f0
    x1, x2 = 1 + t, t                     # distances to the four taps
    x3, x4 = 1 - t, 2 - t
    w1 = a * x1^3 - 5a * x1^2 + 8a * x1 - 4a
    w2 = ((a + 2) * x2 - (a + 3)) * x2^2 + 1
    w3 = ((a + 2) * x3 - (a + 3)) * x3^2 + 1
    w4 = a * x4^3 - 5a * x4^2 + 8a * x4 - 4a
    return (w1, w2, w3, w4)
end

"""
    cubicsample(src, w, h) -> dst

Separable bicubic resize with half-pixel centres and replicated borders — the
mapping `s = (d + 0.5) * scale - 0.5` OpenCV's `resize` builds its coefficient
tables from, and the index clamping it applies at the edges.
"""
function cubicsample(src::AbstractArray{<:Real,3}, w::Integer, h::Integer)
    W, H, C = size(src)
    sx, sy = W / w, H / h
    rows = Array{Float32}(undef, w, H, C)
    for c in 1:C, j in 1:H, i in 1:w
        s = (i - 0.5f0) * Float32(sx) - 0.5f0
        i0 = floor(Int, s)
        ws = cubicweights(s - i0)
        acc = 0.0f0
        for (k, ω) in enumerate(ws)
            acc += ω * Float32(src[clamp(i0 + k - 1, 1, W), j, c])
        end
        rows[i, j, c] = acc
    end
    dst = Array{Float32}(undef, w, h, C)
    for c in 1:C, j in 1:h
        s = (j - 0.5f0) * Float32(sy) - 0.5f0
        j0 = floor(Int, s)
        ws = cubicweights(s - j0)
        for i in 1:w
            acc = 0.0f0
            for (k, ω) in enumerate(ws)
                acc += ω * rows[i, clamp(j0 + k - 1, 1, H), c]
            end
            dst[i, j, c] = acc
        end
    end
    return dst
end

"""
    bilinearsample(src, w, h) -> dst

`torch.nn.functional.interpolate(mode = "bilinear", align_corners = false)`, in
fp32 with the result stored at `src`'s precision — which is what CUDA does for a
half tensor, accumulating in `opmath_t`.

No antialiasing, and that is not an omission: `transforms.Resize(518, ...,
antialias = true)` on a 512 square is an *upscale*, where the filter support is
`max(1, 1/scale)` = 1 and the antialiased path reduces to this one.
"""
function bilinearsample(src::AbstractArray{T,3}, w::Integer, h::Integer) where {T<:AbstractFloat}
    W, H, C = size(src)
    sx, sy = W / w, H / h
    dst = Array{T}(undef, w, h, C)
    for c in 1:C, j in 1:h
        s = clamp((j - 0.5f0) * Float32(sy) - 0.5f0, 0.0f0, Float32(H - 1))
        j0 = floor(Int, s); j1 = min(j0 + 1, H - 1); β = s - j0
        for i in 1:w
            t = clamp((i - 0.5f0) * Float32(sx) - 0.5f0, 0.0f0, Float32(W - 1))
            i0 = floor(Int, t); i1 = min(i0 + 1, W - 1); α = t - i0
            v = (1 - α) * (1 - β) * Float32(src[i0 + 1, j0 + 1, c]) +
                α * (1 - β) * Float32(src[i1 + 1, j0 + 1, c]) +
                (1 - α) * β * Float32(src[i0 + 1, j1 + 1, c]) +
                α * β * Float32(src[i1 + 1, j1 + 1, c])
            dst[i, j, c] = T(v)
        end
    end
    return dst
end

# ------------------------------------------------------------------- the framing

"""
    recenter(rgba; border_ratio = BORDER_RATIO) -> (rgb, alpha)

`ImageProcessorV2.recenter`: crop the object to its alpha bounding box, scale it
to `1 - border_ratio` of a square whose side is the longer input edge, centre it,
and composite onto white.

`rgba` is `(W, H, 4)` `UInt8`. Returns the composited `(S, S, 3)` and the placed
alpha `(S, S, 1)`, both `UInt8`, with `S = max(W, H)`.

Two details that are easy to get wrong and produce a framing rather than an
error. The extents are `max - min`, **not** `max - min + 1`, so the crop is one
pixel short of the bounding box on each axis — upstream's `h = x_max - x_min`
against a Python slice that excludes `x_max`. And the scale is set by the longer
of the two object extents, so the *object* keeps its aspect ratio while being
centred in a square that generally is not the input's.
"""
function recenter(rgba::AbstractArray{UInt8,3}; border_ratio::Real = BORDER_RATIO)
    W, H, C = size(rgba)
    C == 4 || throw(ArgumentError("recenter needs RGBA, got $C channels"))
    S = max(W, H)

    cols = findall(x -> any(@view(rgba[x, :, 4]) .!= 0), 1:W)
    rows = findall(y -> any(@view(rgba[:, y, 4]) .!= 0), 1:H)
    (isempty(cols) || isempty(rows)) && throw(ArgumentError("input image is empty"))
    # `x` is upstream's row index and `y` its column index; named for what they
    # are here instead, since this layout has width first.
    c0, c1 = first(cols), last(cols)
    r0, r1 = first(rows), last(rows)
    w, h = c1 - c0, r1 - r0                 # max - min, as upstream
    (w == 0 || h == 0) && throw(ArgumentError("input image is empty"))

    desired = floor(Int, S * (1 - border_ratio))
    scale = desired / max(h, w)
    w2, h2 = floor(Int, w * scale), floor(Int, h * scale)
    c2, r2 = (S - w2) ÷ 2, (S - h2) ÷ 2

    crop = @view rgba[c0:(c0 + w - 1), r0:(r0 + h - 1), :]
    small = areasample(crop, w2, h2)

    out = zeros(UInt8, S, S, 4)
    for c in 1:4, j in 1:h2, i in 1:w2
        out[c2 + i, r2 + j, c] = round(UInt8, clamp(small[i, j, c], 0, 255))
    end

    rgb = Array{UInt8}(undef, S, S, 3)
    alpha = Array{UInt8}(undef, S, S, 1)
    for j in 1:S, i in 1:S
        a = Float32(out[i, j, 4]) / 255
        for c in 1:3
            # over white, in float, then clipped — `result[..., :3] * mask +
            # bg * (1 - mask)` with `bg` all 255.
            v = Float32(out[i, j, c]) * a + 255 * (1 - a)
            rgb[i, j, c] = round(UInt8, clamp(v, 0, 255))
        end
        alpha[i, j, 1] = round(UInt8, clamp(a * 255, 0, 255))
    end
    return rgb, alpha
end

"""
    prepareimage(rgba; size = PROC_RES, border_ratio = BORDER_RATIO) -> (image, mask)

`ImageProcessorV2.__call__`: [`recenter`](@ref), cubic resize to `size`, and
`x / 255 * 2 - 1`.

Returns `(size, size, 3, 1)` and `(size, size, 1, 1)` `Float32` in (-1, 1) — the
runtime's reversed layout of torch's `(1, 3, size, size)`. The mask rides along
because `prepare_image` returns it and `encode_cond` forwards it to the
conditioner; `DinoImageEncoder` ignores it, and it is returned here so that stays
visible rather than being quietly dropped.
"""
function prepareimage(rgba::AbstractArray{UInt8,3}; size::Integer = PROC_RES,
                      border_ratio::Real = BORDER_RATIO)
    rgb, alpha = recenter(rgba; border_ratio)
    img = cubicsample(rgb, size, size)
    # The mask goes through nearest, not cubic — `cv2.INTER_NEAREST` upstream.
    S = Base.size(alpha, 1)
    msk = Array{Float32}(undef, size, size, 1, 1)
    for j in 1:size, i in 1:size
        si = clamp(floor(Int, (i - 1) * S / size) + 1, 1, S)
        sj = clamp(floor(Int, (j - 1) * S / size) + 1, 1, S)
        msk[i, j, 1, 1] = Float32(alpha[si, sj, 1]) / 255 * 2 - 1
    end
    out = Array{Float32}(undef, size, size, 3, 1)
    for c in 1:3, j in 1:size, i in 1:size
        out[i, j, c, 1] = clamp(round(Float32, clamp(img[i, j, c], 0, 255)), 0, 255) / 255 * 2 - 1
    end
    return out, msk
end

"""
    normalizeimage(image; res = DINO_RES, T = Float16) -> tensor

`ImageEncoder.forward`'s prologue: rescale from (-1, 1) to (0, 1), bilinear
resize to `res`, centre crop, ImageNet normalise. `image` is
`(512, 512, 3, 1)` from [`prepareimage`](@ref); the result is
`(res, res, 3, 1)` and is what [`encodeimage`](@ref) takes.

Everything after the rescale is in `T`, which is fp16, because upstream casts the
tensor to the encoder's dtype *before* the transform runs. That includes the
normalisation constants: `F.normalize` builds `mean` and `std` at the input's
dtype, so the divisor is `Float16(0.229)` and not `0.229`.

The centre crop is a no-op at these sizes — `Resize(518)` on a square already
gives 518 — and is not implemented, because a non-square input would have been
made square by [`recenter`](@ref) two steps earlier.
"""
function normalizeimage(image::AbstractArray{<:Real,4}; res::Integer = DINO_RES,
                        T::Type = Float16)
    w, h, c, n = size(image)
    (c == 3 && n == 1) || throw(ArgumentError("expected (W, H, 3, 1), got $(size(image))"))
    w == h || throw(ArgumentError("expected a square image, got $(w)x$(h)"))
    scaled = Array{T}(undef, w, h, c)
    for k in 1:c, j in 1:h, i in 1:w
        scaled[i, j, k] = T((Float32(image[i, j, k, 1]) + 1) / 2)
    end
    r = bilinearsample(scaled, res, res)
    out = Array{T}(undef, res, res, c, 1)
    for k in 1:c
        μ, σ = T(DINO_MEAN[k]), T(DINO_STD[k])
        for j in 1:res, i in 1:res
            out[i, j, k, 1] = (r[i, j, k] - μ) / σ
        end
    end
    return out
end
