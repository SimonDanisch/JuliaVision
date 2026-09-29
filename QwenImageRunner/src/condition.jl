"""
Reference images: Qwen-Image 2.1's image-conditioned generation.

`QwenImage21Pipeline` takes up to ten reference images (the model card's number;
this admits twelve) next to the prompt, which is how the model edits, restyles
and cuts subjects out of photographs. Each image reaches the model twice:

* through the **vision tower** into the prompt: the template puts a
  `<|vision_start|><|image_pad|><|vision_end|>` slot per image into the user
  turn, the processor widens each `<|image_pad|>` to one token per merged 32x32
  patch, and Qwen3-VL's language model reads the vision tower's features there,
  with M-RoPE grid positions and three DeepStack features added after its first
  three layers;
* through the **VAE encoder** into the denoiser: the image's latents take the
  places of its `<|image_pad|>` tokens in the joint sequence, four latent tokens
  per slot.

The denoiser runs the way the reference does by default (`use_kv_cache=True`):
the prompt and the reference images are modulated from `t = 0` and never see the
target, so their keys and values are computed once, a segment at a time (a text
run causally, an image block bidirectionally), and every denoising step runs the
target's tokens against that cache. See `tools/export_qwenimage21_condition.py`
for why the segments reproduce the reference's block-causal forward exactly.

Everything here that is host-side bookkeeping is a port of the reference's own:
Pillow's resampling for the resize, `Qwen2VLImageProcessorFast`'s patch order,
`get_vision_interpolation_indices_and_weights`, `get_rope_index`, and
`QwenImage21Rope.forward`.
"""

"""The model card's limit is ten; the graphs are sized for this many."""
const MAX_REFERENCE_IMAGES = 12

"""The template with one vision slot per reference image, as
`QwenImage21Pipeline._get_qwen_prompt_embeds` builds it: `<image1>` before the
first slot, and a space and `<imageN>` before every later one."""
function qwen_condition_template(prompt::AbstractString, nimages::Integer)
    slots = join(("$(i == 1 ? "" : " ")<image$i><|vision_start|><|image_pad|><|vision_end|>"
                  for i in 1:nimages))
    qwen_system_prefix() * "<|im_start|>user\n" * slots * prompt * "<|im_end|>\n<|im_start|>assistant\n"
end

const IMAGE_PAD_ID = 151655

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

"""
    condition_size(width, height; resolution = 1024) -> (width, height)

`calculate_dimensions(resolution^2, width / height)`: the reference resizes
every reference image to this area at its own aspect ratio, each side rounded
to a multiple of 32, and takes the output size from the last reference image
the same way when none is given.
"""
function condition_size(width::Integer, height::Integer; resolution::Integer=1024)
    ratio = Float64(width) / Float64(height)
    w = sqrt(Float64(resolution)^2 * ratio)
    h = w / ratio
    # Python's `round` is round-half-even, which is Julia's default too.
    return (Int(round(w / 32)) * 32, Int(round(h / 32)) * 32)
end

# ---------------------------------------------------------------------------
# Pillow's resampling
# ---------------------------------------------------------------------------

"""
    pil_resize(rgba, width, height) -> Array{UInt8,3}

`PIL.Image.resize(..., LANCZOS)` on an RGBA image, the resize both the vision
tower's and the VAE's copy of a reference image go through
(`VaeImageProcessor.resize`), reproduced to the byte.

`rgba` is `(4, width, height)` UInt8. Pillow resizes RGBA with the alpha
premultiplied (`RGBA -> RGBa -> resize -> RGBA`), so colour does not bleed out
of transparent pixels; resamples horizontally and then vertically through an
8-bit intermediate; and runs the filter in fixed point with 22 fractional bits
(`libImaging/Resample.c`). All three are here, because each moves the result by
more than a rounding step at the edge of an alpha matte.
"""
function pil_resize(rgba::AbstractArray{UInt8,3}, width::Integer, height::Integer)
    size(rgba, 1) == 4 || throw(ArgumentError("pil_resize wants (4, width, height) RGBA, got $(size(rgba))"))
    W, H = size(rgba, 2), size(rgba, 3)
    (W, H) == (width, height) && return copy(rgba)
    img = premultiply(rgba)
    if width != W
        img = resample_horizontal(img, width)
    end
    if height != H
        img = permutedims(resample_horizontal(permutedims(img, (1, 3, 2)), height), (1, 3, 2))
    end
    unpremultiply!(img)
end

# `MULDIV255` from `libImaging/ImPlatform.h`: a*b/255, rounded, in integers.
@inline muldiv255(a::Integer, b::Integer) = (t = UInt32(a) * UInt32(b) + 0x80; UInt8(((t >> 8) + t) >> 8))

"""`rgbA2rgba`: RGB times alpha, per channel, alpha kept."""
function premultiply(rgba::AbstractArray{UInt8,3})
    out = similar(rgba)
    @inbounds for j in axes(rgba, 3), i in axes(rgba, 2)
        a = rgba[4, i, j]
        for c in 1:3
            out[c, i, j] = muldiv255(rgba[c, i, j], a)
        end
        out[4, i, j] = a
    end
    out
end

"""`rgba2rgbA`: back out of premultiplied, truncating like the C division."""
function unpremultiply!(img::AbstractArray{UInt8,3})
    @inbounds for j in axes(img, 3), i in axes(img, 2)
        a = img[4, i, j]
        (a == 0xff || a == 0x00) && continue
        for c in 1:3
            img[c, i, j] = UInt8(min(255, (255 * Int(img[c, i, j])) ÷ Int(a)))
        end
    end
    img
end

pil_sinc(x::Float64) = x == 0.0 ? 1.0 : (y = x * pi; sin(y) / y)
pil_lanczos(x::Float64) = -3.0 <= x < 3.0 ? pil_sinc(x) * pil_sinc(x / 3) : 0.0

const PIL_PRECISION_BITS = 22

"""`precompute_coeffs` and `normalize_coeffs_8bpc` for one axis: per output
pixel, the first input pixel it reads and its fixed-point weights."""
function pil_coefficients(insize::Int, outsize::Int)
    scale = insize / outsize
    filterscale = max(scale, 1.0)
    support = 3.0 * filterscale
    bounds = Vector{Tuple{Int,Int}}(undef, outsize)
    weights = Vector{Vector{Int32}}(undef, outsize)
    for xx in 0:outsize-1
        center = (xx + 0.5) * scale
        ss = 1.0 / filterscale
        xmin = max(unsafe_trunc(Int, center - support + 0.5), 0)
        xmax = min(unsafe_trunc(Int, center + support + 0.5), insize) - xmin
        k = [pil_lanczos((x + xmin - center + 0.5) * ss) for x in 0:xmax-1]
        ww = sum(k)
        ww != 0.0 && (k ./= ww)
        weights[xx+1] = Int32[w < 0 ? unsafe_trunc(Int32, -0.5 + w * (1 << PIL_PRECISION_BITS)) :
                                      unsafe_trunc(Int32, 0.5 + w * (1 << PIL_PRECISION_BITS)) for w in k]
        bounds[xx+1] = (xmin, xmax)
    end
    bounds, weights
end

"""`ImagingResampleHorizontal_8bpc` along the second axis of `(4, W, H)`."""
function resample_horizontal(img::AbstractArray{UInt8,3}, outsize::Int)
    C, W, H = size(img)
    bounds, weights = pil_coefficients(W, outsize)
    out = Array{UInt8}(undef, C, outsize, H)
    half = Int32(1) << (PIL_PRECISION_BITS - 1)
    @inbounds for j in 1:H, xx in 1:outsize
        xmin, n = bounds[xx]
        k = weights[xx]
        for c in 1:C
            ss = half
            for x in 1:n
                ss += Int32(img[c, xmin + x, j]) * k[x]
            end
            out[c, xx, j] = UInt8(clamp(ss >> PIL_PRECISION_BITS, 0, 255))
        end
    end
    out
end

"""`white.paste(img, mask=img.getchannel("A"))`: the vision tower's copy of a
reference image is composited over white (`_get_qwen_prompt_embeds`), with
Pillow's `BLEND` rounding."""
function over_white(rgba::AbstractArray{UInt8,3})
    out = Array{UInt8}(undef, 3, size(rgba, 2), size(rgba, 3))
    @inbounds for j in axes(rgba, 3), i in axes(rgba, 2)
        a = UInt32(rgba[4, i, j])
        for c in 1:3
            t = 0x000000ff * (0x000000ff - a) + UInt32(rgba[c, i, j]) * a + 0x80
            out[c, i, j] = UInt8(((t >> 8) + t) >> 8)
        end
    end
    out
end

# ---------------------------------------------------------------------------
# The vision tower's inputs
# ---------------------------------------------------------------------------

"""
    vision_patches(rgb; patch = 16, temporal = 2, merge = 2) -> Matrix{Float32}

`Qwen2VLImageProcessorFast` on an RGB image whose sides are already multiples
of 32, which the reference's resize guarantees (`smart_resize` then leaves the
size alone). Rescale and normalise with mean and std 0.5, repeat the frame to
the temporal patch, and cut `(C, T, 16, 16)` patches in merge-block order:
row-major over 2x2 blocks, row-major inside each block. Returns
`(1536, patches)`, the transpose of the processor's `(patches, 1536)`.
"""
function vision_patches(rgb::AbstractArray{UInt8,3}; patch::Int=16, temporal::Int=2, merge::Int=2)
    C, W, H = size(rgb)
    W % (patch * merge) == 0 && H % (patch * merge) == 0 || throw(ArgumentError(
        "a reference image is resized to multiples of $(patch * merge) first, got $(W)x$(H)"))
    # The processor's `smart_resize` bounds, 65536 to 16777216 pixels. Outside
    # them it resizes again, the vision tower sees a different grid than the VAE,
    # and the prompt's image slots stop matching the latents: the reference
    # pipeline raises in `build_token_metadata`. A small `resolution` gets here
    # by rounding, e.g. 256 puts a 3:2 image at 320x192.
    65536 <= W * H <= 16777216 || throw(ArgumentError(
        "a $(W)x$(H) reference image is outside the vision processor's 65536..16777216 " *
        "pixels, where it resizes the image again and the image's slots no longer " *
        "match its latents; use a larger `resolution`"))
    gh, gw = H ÷ patch, W ÷ patch
    out = Matrix{Float32}(undef, C * temporal * patch * patch, gh * gw)
    col = 0
    for bh in 0:gh÷merge-1, bw in 0:gw÷merge-1, mh in 0:merge-1, mw in 0:merge-1
        col += 1
        ph0, pw0 = (bh * merge + mh) * patch, (bw * merge + mw) * patch
        row = 0
        # Feature order `(C, T, ph, pw)`, pw fastest: the processor's permute
        # puts channel, temporal, patch row and patch column last, in that order.
        @inbounds for c in 1:C, t in 1:temporal, y in 1:patch, x in 1:patch
            row += 1
            out[row, col] = (Float32(rgb[c, pw0 + x, ph0 + y]) - 127.5f0) / 127.5f0
        end
    end
    out, (gh, gw)
end

"""
    vision_grid_inputs(gh, gw; side = 48, merge = 2) -> (indices, weights, positions)

What `Qwen3VLVisionModel.forward` derives from the patch grid, for one image:
the bilinear taps into the learned 48x48 position table
(`get_vision_interpolation_indices_and_weights`, align_corners, merge order) and
the rotary positions (`get_vision_position_ids`). Julia order: `(4, N)`,
`(4, N)` and `(2, N)`, 0-based indices as the graph's `embedding` takes them.
"""
function vision_grid_inputs(gh::Int, gw::Int; side::Int=48, merge::Int=2)
    n = gh * gw
    indices = Matrix{Int64}(undef, 4, n)
    weights = Matrix{Float32}(undef, 4, n)
    positions = Matrix{Int64}(undef, 2, n)
    blocks_w = gw ÷ merge
    taps(index, size) = begin
        src = Float32(index) * Float32(side - 1) / Float32(max(size - 1, 1))
        f = floor(src)
        t0, t1 = clamp(Int(f), 0, side - 1), clamp(Int(f) + 1, 0, side - 1)
        w0 = max(1f0 - abs(src - f), 0f0)
        w1 = max(1f0 - abs(src - f - 1f0), 0f0)
        (t0, t1), (w0, w1)
    end
    for k in 0:n-1
        in_col = k % merge
        in_row = (k ÷ merge) % merge
        block_col = (k ÷ (merge * merge)) % blocks_w
        block_row = k ÷ (merge * merge * blocks_w)
        row = block_row * merge + in_row
        col = block_col * merge + in_col
        (h0, h1), (hw0, hw1) = taps(row, gh)
        (c0, c1), (cw0, cw1) = taps(col, gw)
        indices[:, k+1] .= (h0 * side + c0, h0 * side + c1, h1 * side + c0, h1 * side + c1)
        weights[:, k+1] .= (hw0 * cw0, hw0 * cw1, hw1 * cw0, hw1 * cw1)
        positions[:, k+1] .= (row, col)
    end
    indices, weights, positions
end

# ---------------------------------------------------------------------------
# Positions in the language model
# ---------------------------------------------------------------------------

"""
    mrope_positions(ids, grids; merge = 2) -> Matrix{Int64}

`Qwen3VLModel.get_rope_index` for one prompt: `(3, T)` (t, h, w) positions.
Text advances all three together; a run of `<|image_pad|>` takes its image's
merged grid, offset by the position the text reached, and the text after it
resumes at that position plus the grid's longer side.
"""
function mrope_positions(ids::AbstractVector{<:Integer}, grids::AbstractVector{Tuple{Int,Int}}; merge::Int=2)
    T = length(ids)
    pos = Matrix{Int64}(undef, 3, T)
    cur = 0
    i = 1
    image = 0
    while i <= T
        if ids[i] == IMAGE_PAD_ID
            image += 1
            gh, gw = grids[image] .÷ merge
            k = i
            for h in 0:gh-1, w in 0:gw-1
                pos[1, k] = cur
                pos[2, k] = cur + h
                pos[3, k] = cur + w
                k += 1
            end
            k - i == gh * gw || error("image $image has $(gh * gw) slots in its grid")
            i = k
            cur += max(gh, gw)
        else
            pos[:, i] .= cur
            cur += 1
            i += 1
        end
    end
    image == length(grids) || throw(ArgumentError(
        "the prompt has $image image slots and $(length(grids)) images were given"))
    pos
end

"""
    condition_tokens(tokenizer, prompt, grids) -> Vector{Int}

The token ids of the conditioned template with every `<|image_pad|>` widened to
one token per merged patch of its image, as `Qwen3VLProcessor` widens it.
"""
function condition_tokens(tk::QwenTokenizer, prompt::AbstractString, grids::AbstractVector{Tuple{Int,Int}};
                          merge::Int=2)
    ids = encode(tk, qwen_condition_template(isempty(prompt) ? " " : prompt, length(grids)))
    out = Int[]
    image = 0
    for id in ids
        if id == IMAGE_PAD_ID
            image += 1
            gh, gw = grids[image]
            append!(out, fill(IMAGE_PAD_ID, (gh ÷ merge) * (gw ÷ merge)))
        else
            push!(out, id)
        end
    end
    image == length(grids) || error("the template has $image image slots for $(length(grids)) images")
    out
end

# ---------------------------------------------------------------------------
# The conditioner: vision tower and language model
# ---------------------------------------------------------------------------

const VISIONGRAPH = "qwenimage21_vision"
const VLGRAPH = "qwenimage21_vl_encoder"

@inline _bf16f32(x::UInt16) = reinterpret(Float32, UInt32(x) << 16)

"""The vision tower's weights: Float32 copies of the compact checkpoint's BF16
`model.visual.*`, which the graph reads as `visual.*`."""
function compact_vision_weights(graph; compact::AbstractDict=compact_encoder(),
                                constants_dir::AbstractString=assetdir())
    constants = readsafetensors(joinpath(constants_dir, "$(VISIONGRAPH)_constants.safetensors"); mmap=false)
    out = Dict{String,Any}()
    for buffer in values(graph.buffers)
        buffer.kind === :weight || continue
        key = buffer.key
        (isempty(key) || haskey(out, key)) && continue
        if haskey(constants, key)
            out[key] = constants[key]
            continue
        end
        source = "model." * key
        haskey(compact, source) || throw(KeyError(source))
        value = compact[source]
        out[key] = eltype(value) === UInt16 ? map(_bf16f32, value) : Float32.(value)
    end
    out
end

"""
A prepared Qwen3-VL conditioner for prompts with reference images: the vision
tower and the language model, both from the compact checkpoint.
"""
struct QwenConditionEncoder{B,M,C}
    backend::B
    model::M
    tokenizer::QwenTokenizer
    drop::Int
    compact::C
end

function qwenimageconditionencoder(; backend=Mantle.defaultbackend(), dir::AbstractString=assetdir(),
                                   compact::AbstractDict=compact_encoder(),
                                   processor_dir::AbstractString=processordir())
    vision = loadgraph(joinpath(dir, "$(VISIONGRAPH).json"))
    vl = loadgraph(joinpath(dir, "$(VLGRAPH).json"))
    weights = compact_vision_weights(vision; compact, constants_dir=dir)
    merge!(weights, compact_text_encoder_weights(vl; compact, constants_dir=dir,
                                                 constants_file="$(VLGRAPH)_constants.safetensors"))
    model = Model(Dict(VISIONGRAPH => vision, VLGRAPH => vl), weights; backend)
    tokenizer = QwenTokenizer(processor_dir)
    drop = length(encode(tokenizer, qwen_system_prefix()))
    QwenConditionEncoder(model.backend, model, tokenizer, drop, compact)
end

"""
    encode_image(encoder, rgb) -> (merged, deepstack, grid)

One reference image through the vision tower: `merged` is `(4096, n)` for the
image's `n` merged patches, `deepstack` the three DeepStack features at the
same size, `grid` the patch grid `(gh, gw)`. Host arrays.
"""
function encode_image(encoder::QwenConditionEncoder, rgb::AbstractArray{UInt8,3})
    patches, (gh, gw) = vision_patches(rgb)
    indices, weights, positions = vision_grid_inputs(gh, gw)
    dev(x) = DNNKernels.toback(encoder.backend, x)
    n = (gh * gw) ÷ 4
    outs = DNNKernels.call(encoder.model, VISIONGRAPH, dev(patches), dev(indices), dev(weights),
                           dev(positions); dims=(n=n,))
    host = [Array(o) for o in outs]
    host[1], host[2:4], (gh, gw)
end

"""The encoder's plan lengths: right padding to a multiple of this. The model
is causal, so the padding changes none of the rows that are kept (see
`ENCODER_BUCKETS` for the text-only encoder's measurement of the same)."""
const VL_BUCKET = 256

"""
    encode_condition(encoder, prompt, images) -> (embeds, image_slots)

The prompt embeddings the denoiser reads for `prompt` with the reference
`images` (each `(3, width, height)` UInt8 RGB, already resized and composited
over white): the last decoder layer's output with the system turn dropped, as
`(4096, L)` Float16 on the host, and which of those `L` rows are image slots.
"""
function encode_condition(encoder::QwenConditionEncoder, prompt::AbstractString,
                          images::AbstractVector{<:AbstractArray{UInt8,3}})
    features = [encode_image(encoder, img) for img in images]
    grids = [f[3] for f in features]
    ids = condition_tokens(encoder.tokenizer, prompt, grids)
    positions = mrope_positions(ids, grids)
    T = length(ids)
    tokens = cld(T, VL_BUCKET) * VL_BUCKET
    pad = 151643
    padded = vcat(ids, fill(pad, tokens - T))
    embeds = Float32.(token_embeddings(padded; compact=encoder.compact))
    hidden = size(embeds, 1)
    deepstack = zeros(Float32, hidden, tokens, 1, 3)
    slots = findall(==(IMAGE_PAD_ID), ids)
    k = 0
    for (merged, deep, _) in features
        n = size(merged, 2)
        cols = slots[k+1:k+n]
        embeds[:, cols] .= merged
        for d in 1:3
            deepstack[:, cols, 1, d] .= deep[d]
        end
        k += n
    end
    k == length(slots) || error("the vision tower gave $k tokens for $(length(slots)) slots")
    # Padding continues the text positions; nothing kept reads it.
    last = maximum(positions) + 1
    pos = hcat(positions, repeat(reshape(last .+ (0:tokens-T-1), 1, :), 3, 1))
    dev(x) = DNNKernels.toback(encoder.backend, x)
    out = DNNKernels.call(encoder.model, VLGRAPH, dev(reshape(Float16.(embeds), hidden, tokens, 1)),
                          dev(reshape(permutedims(pos), tokens, 1, 3)),
                          dev(Float16.(deepstack)); dims=(t=tokens,))
    host = Array(first(out))
    kept = (encoder.drop + 1):T
    Float16.(host[:, kept, 1]), ids[kept] .== IMAGE_PAD_ID
end

# ---------------------------------------------------------------------------
# VAE encoder
# ---------------------------------------------------------------------------

const VAEENCODERGRAPH = "qwenimage21_vae_encoder"

"""A prepared VAE encoder: reference images to normalised latents."""
struct QwenVAEEncoder{B,M}
    backend::B
    model::M
end

function qwenimagevaeencoder(; backend=Mantle.defaultbackend(), dir::AbstractString=assetdir(),
                             weights_dir::AbstractString=vaedir())
    graph = loadgraph(joinpath(dir, "$(VAEENCODERGRAPH).json"))
    vae = readsafetensors(joinpath(weights_dir, "vae.safetensors"))
    constants = readsafetensors(joinpath(dir, "$(VAEENCODERGRAPH)_constants.safetensors"); mmap=false)
    weights = Dict{String,Any}()
    for buffer in values(graph.buffers)
        buffer.kind === :weight || continue
        key = buffer.key
        (isempty(key) || haskey(weights, key)) && continue
        weights[key] = haskey(constants, key) ? constants[key] :
                       haskey(vae, key) ? vae[key] : throw(KeyError(key))
    end
    model = Model(Dict(VAEENCODERGRAPH => graph), weights; backend)
    QwenVAEEncoder(model.backend, model)
end

"""
    encode_latents(vae, rgba) -> Matrix{Float16}

One RGBA reference image `(4, width, height)` UInt8 to its packed latents
`(64, (height ÷ 16) * (width ÷ 16))`: `VaeImageProcessor.preprocess` (all four
channels to [-1, 1]) then `_encode_vae_image`.
"""
function encode_latents(vae::QwenVAEEncoder, rgba::AbstractArray{UInt8,3})
    _, W, H = size(rgba)
    image = Array{Float32}(undef, W, H, 1, 4, 1)
    @inbounds for j in 1:H, i in 1:W, c in 1:4
        image[i, j, 1, c, 1] = Float32(rgba[c, i, j]) / 255f0 * 2f0 - 1f0
    end
    out = DNNKernels.call(vae.model, VAEENCODERGRAPH, DNNKernels.toback(vae.backend, image);
                          dims=(h=H, w=W))
    latents = Array(first(out))                     # (lw, lh, 1, 64, 1)
    lw, lh = size(latents, 1), size(latents, 2)
    Float16.(packlatents(reshape(latents, lw, lh, size(latents, 4), 1))[:, :, 1])
end

# ---------------------------------------------------------------------------
# The joint sequence
# ---------------------------------------------------------------------------

"""
    JointLayout

The denoiser's joint sequence for a conditioned prompt: text runs and image
blocks in order, the target block last, and the rotary tables over all of it.
`segments` is `(kind, range)` over the joint positions of the prefix, `kind`
`:text` or `:image`; `textrows` maps the text positions back to rows of the
prompt embeddings.
"""
struct JointLayout
    segments::Vector{Tuple{Symbol,UnitRange{Int}}}
    textrows::Vector{Int}
    prefix::Int
    total::Int
    cos::Matrix{Float32}
    sin::Matrix{Float32}
end

"""
    joint_layout(image_slots, shapes; axes, theta) -> JointLayout

`image_slots` marks which rows of the prompt embeddings are image slots;
`shapes` is the latent grid `(h, w)` of every reference image and then the
target's. A port of the reference's `forward` up to the blocks: each slot
widens to four latent tokens, the target's tokens are appended, and
`QwenImage21Rope.forward` lays out the positions: text advances the frame axis
one per token, every image block freezes it and puts its tokens on an `(h, w)`
grid centred on zero, and the next text starts past the block's longer side.
"""
function joint_layout(image_slots::AbstractVector{Bool}, shapes::AbstractVector{Tuple{Int,Int}};
                      axes::NTuple{3,Int}=QWEN_IMAGE_21.rope_axes, theta::Real=10000)
    # Joint positions of every prompt row: text rows take one, image slots four.
    isimage = Bool[]
    textrows = Int[]
    for (row, slot) in enumerate(image_slots)
        if slot
            append!(isimage, (true, true, true, true))
        else
            push!(isimage, false)
            push!(textrows, row)
        end
    end
    th, tw = last(shapes)
    append!(isimage, fill(true, th * tw))
    total = length(isimage)
    count(isimage) == sum(h * w for (h, w) in shapes) || throw(ArgumentError(
        "the prompt's image slots cover $(count(isimage) - th * tw) latent tokens and the reference " *
        "images have $(sum(h * w for (h, w) in shapes[1:end-1]))"))

    frame = Vector{Int}(undef, total)
    hidx = zeros(Int, total)
    widx = zeros(Int, total)
    segments = Tuple{Symbol,UnitRange{Int}}[]
    cursor, position = 1, 0
    for (b, (h, w)) in enumerate(shapes)
        start = findnext(isimage, cursor)
        if start > cursor
            frame[cursor:start-1] .= position .+ (0:start-cursor-1)
            position += start - cursor
            push!(segments, (:text, cursor:start-1))
        end
        block = start:start+h*w-1
        all(isimage[block]) || throw(ArgumentError(
            "image $b is a $(h)x$(w) latent grid, and its slots in the prompt do not cover it"))
        frame[block] .= position
        position += max(h, w)
        k = start
        for hh in -(h - h ÷ 2):(h ÷ 2 - 1), ww in -(w - w ÷ 2):(w ÷ 2 - 1)
            hidx[k] = hh
            widx[k] = ww
            k += 1
        end
        b < length(shapes) && push!(segments, (:image, block))
        cursor = last(block) + 1
    end
    cursor == total + 1 || error("the target block is not the end of the joint sequence")
    height = [isimage[i] ? hidx[i] : frame[i] for i in 1:total]
    width = [isimage[i] ? widx[i] : frame[i] for i in 1:total]
    co, si = rotary_from_positions((frame, height, width); axes, theta)
    JointLayout(segments, textrows, total - th * tw, total, co, si)
end

"""The three-axis rotary tables for per-token `(frame, height, width)`
positions, as `rotarytables` evaluates them."""
function rotary_from_positions(index::NTuple{3,Vector{Int}}; axes::NTuple{3,Int}, theta::Real)
    total = length(index[1])
    ncols = sum(axes) ÷ 2
    co = Matrix{Float32}(undef, ncols, total)
    si = Matrix{Float32}(undef, ncols, total)
    row = 0
    for (dim, idx) in zip(axes, index)
        half = dim ÷ 2
        for j in 1:half
            invfreq = 1f0 / Float32(theta)^(Float32(2 * (j - 1)) / Float32(dim))
            @inbounds for i in 1:total
                angle = Float32(idx[i]) * invfreq
                co[row + j, i] = cos(angle)
                si[row + j, i] = sin(angle)
            end
        end
        row += half
    end
    co, si
end

# ---------------------------------------------------------------------------
# The denoiser
# ---------------------------------------------------------------------------

const PREFIXGRAPHS = ("qwenimage21_prefix_text0", "qwenimage21_prefix_text", "qwenimage21_prefix_image")
const STEPGRAPH = "qwenimage21_step"

"""
A prepared conditioned denoiser: one `Model` holding the three prefix graphs
and the step graph, which share the compact checkpoint's weights.
"""
struct QwenConditionedTransformer{B,M}
    backend::B
    model::M
end

function qwenimageconditionedtransformer(; backend=Mantle.defaultbackend(), dir::AbstractString=assetdir(),
                                         compact::AbstractDict=compact_denoiser())
    graphs = Dict(n => loadgraph(joinpath(dir, "$n.json")) for n in (PREFIXGRAPHS..., STEPGRAPH))
    weights = Dict{String,Any}()
    for (name, g) in graphs
        merge!(weights, compact_transformer_weights(g; compact, constants_dir=dir,
                                                    constants_file="$(name)_constants.safetensors"))
    end
    model = Model(graphs, weights; backend)
    QwenConditionedTransformer(model.backend, model)
end

"""
    prefill(transformer, layout, embeds, references) -> Vector of host arrays

The prefix's keys and values: every segment of `layout` through all the blocks,
each against the keys and values of the segments before it. `embeds` is the
prompt embeddings `(4096, L)`, `references` the reference latents `(64, n)` in
order. Returns `2 * layers` host arrays `(head_dim, heads, prefix, 1)`,
`(k1, v1, k2, v2, ...)`.
"""
function prefill(t::QwenConditionedTransformer, layout::JointLayout, embeds::AbstractMatrix,
                 references::AbstractVector{<:AbstractMatrix})
    dev(x) = DNNKernels.toback(t.backend, x)
    past = nothing
    text, image = 0, 0
    for (kind, r) in layout.segments
        n = length(r)
        x = if kind === :text
            rows = layout.textrows[text+1:text+n]
            text += n
            reshape(Float16.(embeds[:, rows]), size(embeds, 1), n, 1)
        else
            image += 1
            lat = references[image]
            size(lat, 2) == n || error("reference image $image has $(size(lat, 2)) latent tokens for a $n-token block")
            reshape(Float16.(lat), size(lat, 1), n, 1)
        end
        args = (dev(x), dev(layout.cos[:, r]), dev(layout.sin[:, r]))
        outs = if past === nothing
            kind === :text || error("the prefix has to open with text")
            DNNKernels.call(t.model, PREFIXGRAPHS[1], args...; dims=(n=n,))
        else
            name = kind === :text ? PREFIXGRAPHS[2] : PREFIXGRAPHS[3]
            DNNKernels.call(t.model, name, args..., (dev(p) for p in past)...;
                            dims=(n=n, p=size(past[1], 3)))
        end
        segment = [Array(o) for o in outs]
        past = past === nothing ? segment : [cat(a, b; dims=3) for (a, b) in zip(past, segment)]
    end
    size(past[1], 3) == layout.prefix || error("the prefix came out $(size(past[1], 3)) tokens, expected $(layout.prefix)")
    # The segment plans hold their inputs, which are copies of the cache.
    DNNKernels.releaseplans!(t.model)
    past
end

"""
    sample_conditioned!(transformer, latents, layout, cache; width, height, steps)

The flow-matching loop over the target's tokens against the prefix `cache`
(from [`prefill`](@ref)). The cache is written into the step plan's own inputs
once, so no step copies it.
"""
function sample_conditioned!(t::QwenConditionedTransformer, latents, layout::JointLayout, cache;
                             width::Integer, height::Integer, steps::Integer=40)
    i = size(latents, 2)
    i == layout.total - layout.prefix || throw(DimensionMismatch(
        "latents have $i tokens and the layout's target has $(layout.total - layout.prefix)"))
    g = t.model.graphs[STEPGRAPH]
    plan = planfor(t.model.device, g, t.model.weights, (; i, p=layout.prefix))
    try
        target = layout.prefix+1:layout.total
        dev(x) = DNNKernels.toback(t.backend, x)
        rc, rs = dev(layout.cos[:, target]), dev(layout.sin[:, target])
        stored = plan.inputs[5:end]
        for (dst, src) in zip(stored, cache)
            copyto!(dst, src)
        end
        schedule = qwen_schedule(width, height; steps)
        timestep = similar(latents, eltype(latents), 1)
        for k in eachindex(schedule.timesteps)
            fill!(timestep, convert(eltype(timestep), schedule.timesteps[k] / 1000f0))
            prediction = first(replay!(plan, STEPGRAPH, (latents, timestep, rc, rs, stored...)))
            eulerstep!(latents, prediction, schedule.sigmas[k], schedule.sigmas[k + 1])
        end
    finally
        Mantle.free!(plan)
    end
    latents
end

# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------

"""
    reference_rgba(image) -> Array{UInt8,3}

A reference image as `(4, width, height)` UInt8 RGBA, from a `(4, width, height)`
or `(3, width, height)` UInt8 array. An RGB image is opaque. A matrix of colours
as FileIO loads it, indexed `[row, column]`, converts with
`reinterpret(reshape, UInt8, permutedims(RGBA{N0f8}.(img)))` on the caller's
side; this package does not depend on ColorTypes.
"""
reference_rgba(x::AbstractArray{UInt8,3}) =
    size(x, 1) == 4 ? Array(x) :
    size(x, 1) == 3 ? vcat(x, fill(0xff, 1, size(x, 2), size(x, 3))) :
    throw(ArgumentError("a reference image is (4, width, height) or (3, width, height) UInt8, got $(size(x))"))

"""
    generate(prompt, images; width, height, steps=40, seed=42, resolution=1024) -> Array{Float32,3}

Image-conditioned generation: `prompt` and up to $(MAX_REFERENCE_IMAGES) reference
`images` in, an **RGBA** image `(width, height, 4)` in [0, 1] out. This is how
Qwen-Image 2.1 edits: "Change the background to a sunset beach", "Extract the
character onto a transparent background", or a group photo from portraits.

Every reference image is resized to about `resolution^2` pixels at its own
aspect ratio. The output size defaults to the same for the LAST reference
image, as the reference pipeline does.

The four stages run one after another, each released before the next is built:
the conditioner (vision tower and Qwen3-VL), the VAE encoder, the denoiser, and
the VAE decoder.
"""
function generate(prompt::AbstractString, images::AbstractVector;
                  width::Union{Nothing,Integer}=nothing, height::Union{Nothing,Integer}=nothing,
                  steps::Integer=40, seed::Integer=42, resolution::Integer=1024,
                  backend=Mantle.defaultbackend())
    isempty(images) && return generate(prompt; width=something(width, resolution),
                                       height=something(height, resolution), steps, seed, backend)
    length(images) <= MAX_REFERENCE_IMAGES || throw(ArgumentError(
        "at most $MAX_REFERENCE_IMAGES reference images, got $(length(images)) " *
        "(the model card's limit is 10)"))
    rgba = [reference_rgba(img) for img in images]
    sizes = [condition_size(size(x, 2), size(x, 3); resolution) for x in rgba]
    resized = [pil_resize(x, s...) for (x, s) in zip(rgba, sizes)]
    cw, ch = last(sizes)
    w, h = qwen_resolution(something(width, cw), something(height, ch))
    lw, lh = w ÷ QWEN_IMAGE_21.vae_scale_factor, h ÷ QWEN_IMAGE_21.vae_scale_factor

    encoder = qwenimageconditionencoder(; backend)
    embeds, slots = encode_condition(encoder, prompt, [over_white(x) for x in resized])
    Mantle.release!(encoder)

    vaeenc = qwenimagevaeencoder(; backend)
    references = [encode_latents(vaeenc, x) for x in resized]
    Mantle.release!(vaeenc)

    shapes = vcat([(s[2] ÷ 16, s[1] ÷ 16) for s in sizes], [(lh, lw)])
    layout = joint_layout(slots, shapes)
    transformer = qwenimageconditionedtransformer(; backend)
    cache = prefill(transformer, layout, embeds, references)
    channels = QWEN_IMAGE_21.latent_channels
    noise = randn(Random.Xoshiro(seed), Float32, channels, lw * lh, 1)
    latents = DNNKernels.toback(backend, Float16.(noise))
    sample_conditioned!(transformer, latents, layout, cache; width=w, height=h, steps)
    denoised = Array(latents)
    cache = nothing
    Mantle.release!(transformer)

    vae = qwenimagevae(; backend)
    grid = unpacklatents(denoised, lw, lh)
    image = Array(decode!(vae, DNNKernels.toback(backend, reshape(Float16.(grid), lw, lh, 1, channels, 1))))
    Mantle.release!(vae)
    return Float32.(image[:, :, :, 1]) ./ 2f0 .+ 0.5f0
end

heldweights(c::Union{QwenConditionEncoder,QwenVAEEncoder,QwenConditionedTransformer}) = c.model.weights
freeplans!(c::Union{QwenConditionEncoder,QwenVAEEncoder,QwenConditionedTransformer}) =
    (DNNKernels.releaseplans!(c.model); nothing)
