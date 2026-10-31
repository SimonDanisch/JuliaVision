"""
FLUX.2-klein text-to-image on Lava: the 4B (Apache-2.0) and 9B (FLUX Non-Commercial)
distilled models, four denoising steps, through the graphs `tools/export_fluxklein.py`
writes and the weights straight from the Hugging Face checkpoint.

    using Mantle, FluxKleinRunner
    Mantle.defaultdevice!("NVIDIA")
    img = generate("a red fox in the snow"; width = 1024, height = 1024)   # (W, H, 3) in [0, 1]

Three components take turns on the device, each released before the next is
built, the way QwenImageRunner stages its own: the Qwen3 text encoder (8 GB in
fp16 for the 4B's Qwen3-4B), the transformer (7.75 GB) and the VAE decoder.

What the host does, and the graphs do not: tokenising and padding the prompt to
512 tokens, the embedding gather, the encoder's rotary tables and its causal +
padding mask, the transformer's 4-axis rotary tables and its sinusoidal timestep
projection, the flow-matching schedule and Euler steps, and the VAE's
de-normalisation and 2x2 unpatchify. Every one of those is a port of the
reference `Flux2KleinPipeline` and `Flux2Transformer2DModel`, named after what
it ports.

Weights: the checkpoint is bf16 and the graphs are exported in fp16 (the VAE in
fp32, `force_upcast`). The shards are mapped and every tensor is wrapped as a
[`BF16As`](@ref) view, which `DNNKernels.toback` materialises one tensor at a
time on its way to the device; no converted copy is written (the 9B checkpoint
alone is 32 GB) and nothing is read into anonymous memory up front: a 32 GB box
with the editor open froze doing that (2026-10-12).

Licence of the WEIGHTS: 4B Apache-2.0; 9B FLUX Non-Commercial License. The code
here is the same for both.
"""
module FluxKleinRunner

using Random
import JSON3
import Mantle
import KernelAbstractions
using DNNKernels
using DNNKernels: loadgraph, readsafetensors, Model, call, toback,
                  linspace, flowschedule, ExponentialShift, eulerstep!
using QwenImageRunner: QwenTokenizer, encode

export generate, FluxJob, assets, FluxEncoder, FluxTransformer, FluxVAE, References,
       encodeprompt, encodereferences, sample!, decode!

const REPOS = Dict("4B" => "black-forest-labs/FLUX.2-klein-4B",
                   "9B" => "black-forest-labs/FLUX.2-klein-9B")
"`tokenizer_max_length`: every prompt is padded (or truncated) to this many tokens."
const TEXT_TOKENS = 512
"`<|endoftext|>`, the tokenizer's pad token."
const PAD_TOKEN = 151643
"`text_encoder_out_layers`: the hidden states read, after these many decoder layers."
const HIDDEN_TAPS = (9, 18, 27)
"`timestep_guidance_channels`: the sinusoidal projection's width."
const TIMESTEP_CHANNELS = 256
"The chat template `_get_qwen3_prompt_embeds` wraps a prompt in (`enable_thinking=False`)."
prompttemplate(prompt::AbstractString) =
    "<|im_start|>user\n$(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# ── where things are ──────────────────────────────────────────────────────────

"""
    Assets(variant; graphs, snapshot)

Where one variant's graphs (`fluxklein_*.json`) and its Hugging Face snapshot
(tokenizer, configs, bf16 weights) live. `assets("4B")` finds the snapshot in
the Hugging Face cache (`HF_HOME`) and the graphs under `gen/fluxklein4b`.
"""
struct Assets
    variant::String
    graphs::String
    snapshot::String
end

function assets(variant::AbstractString = "4B";
                graphs = normpath(joinpath(@__DIR__, "..", "..", "gen", "fluxklein$(lowercase(variant))")),
                snapshot = hfsnapshot(variant))
    haskey(REPOS, variant) || throw(ArgumentError("variant is 4B or 9B, got $variant"))
    return Assets(String(variant), String(graphs), String(snapshot))
end

"The Hugging Face cache's snapshot directory of a variant's checkpoint."
function hfsnapshot(variant::AbstractString)
    home = get(ENV, "HF_HOME", joinpath(homedir(), ".cache", "huggingface"))
    repo = joinpath(home, "hub", "models--" * replace(REPOS[variant], "/" => "--"))
    ref = joinpath(repo, "refs", "main")
    isfile(ref) || throw(ArgumentError(
        "FLUX.2-klein $variant is not in the Hugging Face cache at $repo; download it with " *
        "`huggingface-cli download $(REPOS[variant])` (the 9B is gated: accept its licence first)"))
    return joinpath(repo, "snapshots", strip(read(ref, String)))
end

graphpath(a::Assets, part) = joinpath(a.graphs, "fluxklein_$part.json")
config(a::Assets, part) = JSON3.read(read(joinpath(a.snapshot, part, "config.json"), String))

# ── weights: the bf16 checkpoint, read as the graphs' fp16 / fp32 on upload ──

@inline bf16(x::UInt16) = reinterpret(Float32, UInt32(x) << 16)

"""
    BF16As{T}(bits)

A mapped bf16 tensor read as `T` (`Float16` for the encoder and the transformer,
`Float32` for the VAE), element by element. It is not dense, so
`DNNKernels.toback` collects it into a plain `Array{T}` right before the copy to
the device and drops it after: one tensor's worth of host memory at a time,
where a converted file would have cost the checkpoint's size again on disk.
`T(bf16(x))` is the same rounding torch's `bf16 -> fp16` cast does.
"""
struct BF16As{T,N,A<:AbstractArray{UInt16,N}} <: AbstractArray{T,N}
    bits::A
end
BF16As{T}(bits::AbstractArray{UInt16,N}) where {T,N} = BF16As{T,N,typeof(bits)}(bits)
Base.size(a::BF16As) = size(a.bits)
Base.IndexStyle(::Type{<:BF16As}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(a::BF16As{T}, i::Int) where {T} = T(bf16(a.bits[i]))
# Where the bits live, which is where `toback` asks before deciding to upload: the host.
KernelAbstractions.get_backend(a::BF16As) = KernelAbstractions.get_backend(a.bits)

"""
    checkpoint(a, part, T; rename = identity) -> Dict{String,Any}

Every tensor of one component's shards (`a.snapshot/part/*.safetensors`),
mapped, keyed `rename(key)` (`nothing` drops it), the bf16 ones as `BF16As{T}`
and anything else (the VAE's `num_batches_tracked`) as it is.
"""
function checkpoint(a::Assets, part::AbstractString, ::Type{T}; rename = identity) where {T}
    dir = joinpath(a.snapshot, part)
    out = Dict{String,Any}()
    for file in filter(endswith(".safetensors"), readdir(dir))
        for (key, x) in readsafetensors(joinpath(dir, file); mmap = true)
            name = rename(key)
            name === nothing && continue
            out[name] = x isa AbstractArray{UInt16} ? BF16As{T}(x) : x
        end
    end
    return out
end

"Hand the pool's freed blocks back to the device once nothing in flight uses them."
function trimpool!(dev)
    Mantle.waitidle(dev)
    GC.gc(true)
    Mantle.reclaim!(Mantle.pool(dev), dev; wait = true)
    Mantle.trim!(Mantle.pool(dev), dev)
    GC.gc(true)
    return nothing
end

"Free a model's plans (its activation memory), keep its weights; it rebuilds a plan on the next call."
releaseplans!(m::Model) = (DNNKernels.releaseplans!(m); trimpool!(m.device))

"Free a model's plans and weights on the device, and hand the pool's blocks back."
function release!(m::Model)
    DNNKernels.releaseplans!(m)
    DNNKernels.releaseweights!(m.weights)
    return trimpool!(m.device)
end

# ── text encoder ─────────────────────────────────────────────────────────────

"""
    FluxEncoder(a::Assets; backend)

The Qwen3 text encoder: the exported decoder layers on the device, the embedding
table mapped on the host (bf16, 151936 rows, of which a prompt reads 512), the
tokenizer, and the rotary tables and mask the graph takes as inputs.
"""
struct FluxEncoder{M,E<:AbstractMatrix{UInt16}}
    model::M
    tokenizer::QwenTokenizer
    # (hidden, vocab), bf16 bits, mapped. A type parameter and not an abstract
    # field: `encodeprompt` gathers 512 x 2560 of these, and through an abstract
    # field every read was a dynamic call.
    embed::E
    cos::Array{Float16,3}               # (head_dim, 512, 1)
    sin::Array{Float16,3}
    backend::Any
end

function FluxEncoder(a::Assets; backend = Mantle.defaultbackend(), quantize::Bool = false)
    cfg = config(a, "text_encoder")
    hidden, headdim, theta = Int(cfg["hidden_size"]), Int(cfg["head_dim"]), Float64(cfg["rope_theta"])
    tokenizer = QwenTokenizer(joinpath(a.snapshot, "tokenizer"))
    weights = checkpoint(a, "text_encoder", Float16)
    # The table stays on the host as its bf16 bits: a prompt reads 512 of its 151936 rows.
    embed = pop!(weights, "model.embed_tokens.weight").bits
    size(embed, 1) == hidden || error("embedding table is $(size(embed)), hidden size $hidden")
    filter!(kv -> startswith(first(kv), "model.layers."), weights)
    cos, sin = qwen3rope(TEXT_TOKENS, headdim, theta)
    graph = loadgraph(graphpath(a, "encoder"))
    model = Model(Dict("fluxklein_encoder" => graph), weights; backend, quantize)
    return FluxEncoder(model, tokenizer, embed, cos, sin, backend)
end

"""
Qwen3's rotary tables for positions `0..n-1`: `(headdim, n, 1)` cos and sin, the
`rotate_half` layout (`cat(freqs, freqs)`), computed in Float32 as
`Qwen3RotaryEmbedding` does before the cast.
"""
function qwen3rope(n::Integer, headdim::Integer, theta::Real)
    half = headdim ÷ 2
    invfreq = Float32[1 / theta^(2(j - 1) / headdim) for j in 1:half]
    cos = Array{Float16,3}(undef, headdim, n, 1)
    sin = similar(cos)
    for p in 1:n, j in 1:half
        angle = Float32(p - 1) * invfreq[j]
        cos[j, p, 1] = cos[j + half, p, 1] = Float16(Base.cos(angle))
        sin[j, p, 1] = sin[j + half, p, 1] = Float16(Base.sin(angle))
    end
    return cos, sin
end

"""
The pipeline's attention mask for a prompt of `real` tokens in a `n`-token
sequence, as the additive `(k, q, 1, 1)` Float16 tensor the graph reads: causal,
and the right-padding keys masked for every query, padding queries included.
"""
function encodermask(real::Integer, n::Integer = TEXT_TOKENS)
    mask = fill(Float16(-Inf), n, n, 1, 1)
    for q in 1:n, k in 1:min(q, real)
        mask[k, q, 1, 1] = zero(Float16)
    end
    return mask
end

"""
    encodeprompt(enc, prompt) -> Array{Float16,3}

`prompt_embeds` for one prompt: `(3 * hidden, 512, 1)`, the hidden states after
layers 9, 18 and 27 side by side, padding positions included (the transformer
attends to all 512 as the reference does).
"""
function encodeprompt(enc::FluxEncoder, prompt::AbstractString)
    ids = encode(enc.tokenizer, prompttemplate(prompt))
    length(ids) > TEXT_TOKENS && (ids = ids[1:TEXT_TOKENS])
    real = length(ids)
    padded = vcat(ids, fill(PAD_TOKEN, TEXT_TOKENS - real))
    hidden = size(enc.embed, 1)
    embeds = Array{Float16,3}(undef, hidden, TEXT_TOKENS, 1)
    for (p, id) in enumerate(padded)
        0 <= id < size(enc.embed, 2) || throw(ArgumentError("token id $id is outside the vocabulary"))
        @inbounds for i in 1:hidden
            embeds[i, p, 1] = Float16(bf16(enc.embed[i, id + 1]))
        end
    end
    b = enc.backend
    out = call(enc.model, "fluxklein_encoder", toback(b, embeds), toback(b, enc.cos), toback(b, enc.sin),
               toback(b, encodermask(real)); dims = (;))
    return Array(only(out))
end

# ── transformer ──────────────────────────────────────────────────────────────

"""
    FluxTransformer(a::Assets; backend)

The denoiser on the device with what its graph needs from the config: the
packed-latent channel count, the rotary axes and theta.
"""
struct FluxTransformer{M}
    model::M
    channels::Int            # `in_channels`, 128: 32 latent channels times a 2x2 patch
    axes::NTuple{4,Int}      # `axes_dims_rope`
    theta::Float64           # `rope_theta`
    backend::Any
end

function FluxTransformer(a::Assets; backend = Mantle.defaultbackend(), quantize::Bool = false)
    cfg = config(a, "transformer")
    graph = loadgraph(graphpath(a, "transformer"))
    weights = checkpoint(a, "transformer", Float16; rename = k -> "model." * k)
    model = Model(Dict("fluxklein_transformer" => graph), weights; backend, quantize)
    return FluxTransformer(model, Int(cfg["in_channels"]), Tuple(Int.(cfg["axes_dims_rope"])), Float64(cfg["rope_theta"]), backend)
end

"""
    fluxrope(text, blocks; axes, theta) -> (cos, sin)

`Flux2PosEmbed` over the joint sequence the transformer sees: the text ids
`(0, 0, 0, l)`, then one block per image, `(t, y, x, 0)` for every token of an
`lh x lw` grid in row-major order. `blocks` is `[(t, lh, lw), ...]`: the target
first with `t = 0`, then each reference image with `t = 10, 20, ...`
(`_prepare_image_ids`, `scale = 10`). Each axis is `dim` wide with `dim/2`
frequencies `theta^(-2k/dim)` and cos/sin repeated pairwise; Float64 like the
reference, cast once. Returns `(sum(axes), text + sum(lh * lw))` cos and sin.
"""
function fluxrope(text::Integer, blocks; axes::NTuple{4,Int}, theta::Real)
    total = sum(axes)
    n = text + sum(b -> b[2] * b[3], blocks)
    ids = zeros(Float64, 4, n)
    for l in 1:text
        ids[4, l] = l - 1
    end
    off = text
    for (t, lh, lw) in blocks
        for y in 1:lh, x in 1:lw
            k = off + (y - 1) * lw + x
            ids[1, k] = t
            ids[2, k] = y - 1
            ids[3, k] = x - 1
        end
        off += lh * lw
    end
    cos = Matrix{Float16}(undef, total, n)
    sin = similar(cos)
    row = 0
    for (a, dim) in enumerate(axes)
        for k in 0:(dim ÷ 2 - 1)
            f = 1 / theta^(2k / dim)
            for t in 1:n
                angle = ids[a, t] * f
                cos[row + 2k + 1, t] = cos[row + 2k + 2, t] = Float16(Base.cos(angle))
                sin[row + 2k + 1, t] = sin[row + 2k + 2, t] = Float16(Base.sin(angle))
            end
        end
        row += dim
    end
    return cos, sin
end

"""
`Timesteps(256, flip_sin_to_cos = true, downscale_freq_shift = 0)` of `t * 1000`:
`[cos(e); sin(e)]` with `e = 1000 t * exp(-log(10000) k / 128)`, as `(256, 1)` Float16.
"""
function timestepproj(t::Real; channels::Integer = TIMESTEP_CHANNELS)
    half = channels ÷ 2
    out = Matrix{Float16}(undef, channels, 1)
    for k in 0:(half - 1)
        e = Float32(1000t) * exp(-log(10000f0) * k / half)
        out[k + 1, 1] = Float16(cos(e))
        out[half + k + 1, 1] = Float16(sin(e))
    end
    return out
end

"`compute_empirical_mu`: the schedule's shift from the image token count and the step count."
function empiricalmu(tokens::Integer, steps::Integer)
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    tokens > 4300 && return a2 * tokens + b2
    m200 = a2 * tokens + b2
    m10 = a1 * tokens + b1
    a = (m200 - m10) / 190.0
    b = m200 - 200.0 * a
    return a * steps + b
end

"The pipeline's schedule: `linspace(1, 1/steps, steps)` shifted by `empiricalmu`, a 0 appended."
fluxschedule(tokens::Integer, steps::Integer) =
    flowschedule(linspace(1.0, 1 / steps, steps), ExponentialShift(empiricalmu(tokens, steps)))

"""
    References(tokens, grids)

Reference images as the transformer reads them: their packed latents side by
side, `(channels, sum(lh * lw), 1)` Float16, and each one's `(lh, lw)` token grid.
"""
struct References
    tokens::Array{Float16,3}
    grids::Vector{NTuple{2,Int}}
end
noreference(tr::FluxTransformer) = References(zeros(Float16, tr.channels, 0, 1), NTuple{2,Int}[])

"""
    sample!(tr, latents, prompt_embeds, lh, lw; steps = 4, refs = noreference(tr)) -> latents

The denoising loop on `latents`, `(channels, lh * lw, 1)` Float16 packed latents
(row-major tokens), in place: `steps` evaluations of the transformer at the
schedule's sigmas, each followed by the Euler step. `refs` are the reference
images' packed latents ([`encodereferences`](@ref)): appended after the target's
tokens at every step and dropped from the prediction, as the pipeline does
(`latent_model_input = cat([latents, image_latents])`, `noise_pred[:, :N]`).
The prompt, the reference tokens and the rotary tables are uploaded once.
"""
function sample!(tr::FluxTransformer, latents::Array{Float16,3}, prompt::Array{Float16,3}, lh::Integer, lw::Integer;
                 steps::Integer = 4, refs::References = noreference(tr))
    tokens = lh * lw
    size(latents) == (tr.channels, tokens, 1) || throw(DimensionMismatch(
        "latents are $(size(latents)), expected $((tr.channels, tokens, 1))"))
    text = size(prompt, 2)
    blocks = [(0, lh, lw); [(10k, g...) for (k, g) in enumerate(refs.grids)]]
    cos, sin = fluxrope(text, blocks; axes = tr.axes, theta = tr.theta)
    b = tr.backend
    dprompt, dcos, dsin = toback(b, prompt), toback(b, cos), toback(b, sin)
    joint = cat(latents, refs.tokens; dims = 2)
    schedule = fluxschedule(tokens, steps)
    for k in 1:steps
        sigma = schedule.sigmas[k]
        joint[:, 1:tokens, :] .= latents
        out = call(tr.model, "fluxklein_transformer", toback(b, joint), dprompt, toback(b, timestepproj(sigma)),
                   dcos, dsin; dims = (i = size(joint, 2), t = text))
        v = Array(only(out))[:, 1:tokens, :]
        eulerstep!(latents, v, sigma, schedule.sigmas[k + 1])
    end
    return latents
end


# ── VAE ──────────────────────────────────────────────────────────────────────

"""
    FluxVAE(a::Assets; backend, halfconvs = true)

The decoder and the encoder in one model (the encoder's weights are in the same
checkpoint), with the batch-norm statistics the pipeline normalises packed
latents by (`bn.running_mean`, `sqrt(bn.running_var + eps)`).

`halfconvs` (`DNNKernels.halfconvs`): the graphs stay fp32, as exported
(`force_upcast`), and the convolutions read fp16 operands scaled into range at
run time, which puts them on the matrix cores. On the 8060S (2026-10-13) an
832x1216 decode went 1.77 -> 1.41 s against Diffusers' 1.43 s in bf16, and its
output is 90.6 dB PSNR from the fp32 decode.
"""
struct FluxVAE{M}
    model::M
    mean::Vector{Float32}
    std::Vector{Float32}
    latent::Int              # `latent_channels`, 32
    backend::Any
end

function FluxVAE(a::Assets; backend = Mantle.defaultbackend(), halfconvs::Bool = true)
    cfg = config(a, "vae")
    weights = checkpoint(a, "vae", Float32; rename = k -> "vae." * k)
    mean = Float32.(weights["vae.bn.running_mean"])
    std = sqrt.(Float32.(weights["vae.bn.running_var"]) .+ Float32(cfg["batch_norm_eps"]))
    graphs = Dict(name => loadgraph(graphpath(a, part)) for (name, part) in
                  ("fluxklein_vae_decoder" => "vae_decoder", "fluxklein_vae_encoder" => "vae_encoder"))
    model = Model(graphs, weights; backend, halfconvs)
    return FluxVAE(model, mean, std, Int(cfg["latent_channels"]), backend)
end

"""
    decode!(vae, latents, lh, lw) -> Array{Float32,4}

Packed latents `(4 * latent, lh * lw, 1)` to pixels `(8 * 2lw, 8 * 2lh, 3, 1)` in
[-1, 1]: unpack by position (row-major tokens), de-normalise per channel,
unpatchify the 2x2 patch (`_unpatchify_latents`: channel `c*4 + p*2 + q` goes to
row offset `p`, column offset `q`), then the decoder graph.
"""
function decode!(vae::FluxVAE, latents::Array{Float16,3}, lh::Integer, lw::Integer)
    C = 4 * vae.latent
    size(latents) == (C, lh * lw, 1) || throw(DimensionMismatch(
        "latents are $(size(latents)), expected $((C, lh * lw, 1))"))
    grid = Array{Float32,4}(undef, 2lw, 2lh, vae.latent, 1)
    for y in 1:lh, x in 1:lw
        t = (y - 1) * lw + x
        for c in 0:(vae.latent - 1), p in 0:1, q in 0:1
            k = c * 4 + p * 2 + q
            grid[2(x - 1) + q + 1, 2(y - 1) + p + 1, c + 1, 1] =
                Float32(latents[k + 1, t, 1]) * vae.std[k + 1] + vae.mean[k + 1]
        end
    end
    out = call(vae.model, "fluxklein_vae_decoder", toback(vae.backend, grid); dims = (h = 2lh, w = 2lw))
    return Array(only(out))
end

"""
    fitreference(img; maxarea = 1024^2) -> Array{Float32,3}

A reference picture `(W, H, 3)` in [0, 1] as the pipeline sizes it: shrunk to
`maxarea` pixels when larger (keeping the aspect, sides truncated like
`int(w * scale)`), then centre-cropped to multiples of 16. The pipeline shrinks
with Lanczos and this with an area average; a picture that is already within
`maxarea` and a multiple of 16 on both sides goes through untouched in both.
"""
function fitreference(img::AbstractArray{<:Real,3}; maxarea::Integer = 1024^2)
    W, H, C = size(img)
    x = Float32.(img)
    if W * H > maxarea
        s = sqrt(maxarea / (W * H))
        x = arearesize(x, floor(Int, W * s), floor(Int, H * s))
        W, H = size(x, 1), size(x, 2)
    end
    w, h = (W ÷ 16) * 16, (H ÷ 16) * 16
    x0, y0 = (W - w) ÷ 2, (H - h) ÷ 2
    return x[x0 + 1:x0 + w, y0 + 1:y0 + h, :]
end

"A box-filter downscale of `(W, H, C)` to `(w, h, C)`: every output pixel averages the input area it covers."
function arearesize(x::Array{Float32,3}, w::Integer, h::Integer)
    W, H, C = size(x)
    out = zeros(Float32, w, h, C)
    sx, sy = W / w, H / h
    for j in 1:h, i in 1:w
        xa, xb = (i - 1) * sx, i * sx
        ya, yb = (j - 1) * sy, j * sy
        total = 0.0
        for v in floor(Int, ya):(ceil(Int, yb) - 1), u in floor(Int, xa):(ceil(Int, xb) - 1)
            wgt = (min(u + 1, xb) - max(u, xa)) * (min(v + 1, yb) - max(v, ya))
            wgt > 0 || continue
            total += wgt
            for c in 1:C
                out[i, j, c] += wgt * x[u + 1, v + 1, c]
            end
        end
        out[i, j, :] ./= total
    end
    return out
end

"""
    packlatents(mean, μ, σ, lh, lw) -> Array{Float16,3}

The encoder's posterior mean `(2lw, 2lh, latent, 1)` patchified 2x2 (channel
`c*4 + p*2 + q`), normalised by the batch-norm statistics and packed in row-major
token order. Its own function because what `call` returns is not inferred: in
line, the loop dispatched on every element and took 0.9 s of a 2.0 s reference
encode on the 8060S.
"""
function packlatents(mean::AbstractArray{<:Real,4}, μ::Vector{Float32}, σ::Vector{Float32}, lh::Int, lw::Int)
    latent = size(mean, 3)
    packed = Array{Float16,3}(undef, 4latent, lh * lw, 1)
    for y in 1:lh, xx in 1:lw
        t = (y - 1) * lw + xx
        for c in 0:(latent - 1), p in 0:1, q in 0:1
            k = c * 4 + p * 2 + q
            packed[k + 1, t, 1] = Float16((mean[2(xx - 1) + q + 1, 2(y - 1) + p + 1, c + 1, 1] - μ[k + 1]) / σ[k + 1])
        end
    end
    return packed
end

"""
    encodereferences(vae, images) -> References

Each picture `(W, H, 3)` in [0, 1]: fitted ([`fitreference`](@ref)), mapped to
[-1, 1], through the encoder graph (the posterior mean), patchified 2x2 (channel
`c*4 + p*2 + q`), normalised per channel by the batch-norm statistics, and
packed in row-major token order (`_encode_vae_image`, `_pack_latents`).
"""
function encodereferences(vae::FluxVAE, images)
    C = 4 * vae.latent
    packs = Array{Float16,3}[]
    grids = NTuple{2,Int}[]
    for img in images
        x = fitreference(img)
        W, H = size(x, 1), size(x, 2)
        pixels = reshape(x .* 2f0 .- 1f0, W, H, 3, 1)
        mean = Array(only(call(vae.model, "fluxklein_vae_encoder", toback(vae.backend, pixels);
                               dims = (h = H ÷ 8, w = W ÷ 8))))           # (W/8, H/8, latent, 1)
        lh, lw = H ÷ 16, W ÷ 16
        push!(packs, packlatents(mean, vae.mean, vae.std, lh, lw))
        push!(grids, (lh, lw))
    end
    return References(isempty(packs) ? zeros(Float16, C, 0, 1) : cat(packs...; dims = 2), grids)
end

# ── the whole pipeline ───────────────────────────────────────────────────────

"""
    FluxJob(prompt; references = (), width = 1024, height = 1024, seed = 42)

One picture of a [`generate`](@ref) batch: the prompt, the reference pictures
(`(W, H, 3)` in [0, 1]; jobs that pass the same pictures share one encoding),
the size, floored to multiples of 16 (one packed-latent token), and the seed of
the start noise.
"""
struct FluxJob
    prompt::String
    references::Vector{Array{Float32,3}}
    width::Int
    height::Int
    seed::Int
end

function FluxJob(prompt::AbstractString; references = (), width::Integer = 1024, height::Integer = 1024,
                 seed::Integer = 42)
    w, h = (width ÷ 16) * 16, (height ÷ 16) * 16
    (w, h) == (width, height) || @warn "FLUX.2-klein sizes are multiples of 16; $(width)x$(height) runs as $(w)x$(h)"
    # `convert`, not `Float32.`: a picture many jobs share stays one array, not one copy per job.
    return FluxJob(String(prompt), Array{Float32,3}[convert(Array{Float32,3}, r) for r in references], w, h, seed)
end

"The indices of `xs` grouped by `key`, groups in order of first appearance."
function groupindices(key, xs)
    groups = Dict{Any,Vector{Int}}()
    order = Any[]
    for (i, x) in enumerate(xs)
        k = key(x)
        haskey(groups, k) || push!(order, k)
        push!(get!(() -> Int[], groups, k), i)
    end
    return [groups[k] for k in order]
end

"""
    generate(jobs::AbstractVector{FluxJob}; steps = 4, variant = "4B", quantize = false,
             backend = Mantle.defaultbackend()) -> Vector{Array{Float32,3}}

Every job's picture, `(width, height, 3)` in [0, 1], with each component built
once: all prompts through the text encoder, which is then released; all
reference sets through the VAE encoder; every job's denoising on one
transformer, which is then released; all decodes. Per picture that leaves the
four transformer evaluations and a decode, where one-picture [`generate`](@ref)
calls pay for building 16 GB of weights each time. A job comes out exactly as
it would alone: its noise is its own seed's, and the components see the same
inputs. Plans (activation memory) are kept per shape, so jobs run grouped by
shape and a group's plans are freed before the next.

`quantize` stores the encoder's and the transformer's matmul weights as int8
with a per-channel scale (`DNNKernels.Model`), which is what puts the 9B's 17 GB
transformer on a 20 GB card with room to run.
"""
function generate(jobs::AbstractVector{FluxJob}; steps::Integer = 4, variant::AbstractString = "4B",
                  backend = Mantle.defaultbackend(), a::Assets = assets(variant), quantize::Bool = false)
    enc = FluxEncoder(a; backend, quantize)
    prompts = Dict(p => encodeprompt(enc, p) for p in unique(j.prompt for j in jobs))
    release!(enc.model)

    # Built before the transformer, as the references need it, and kept (0.3 GB) for the decodes.
    vae = FluxVAE(a; backend)
    refs = Dict{Vector{Array{Float32,3}},References}()
    for r in unique(j.references for j in jobs)
        refs[r] = encodereferences(vae, r)
        releaseplans!(vae.model)
    end

    tr = FluxTransformer(a; backend, quantize)
    latents = Vector{Array{Float16,3}}(undef, length(jobs))
    for group in groupindices(j -> (j.width, j.height, refs[j.references].grids), jobs)
        for i in group
            j = jobs[i]
            lh, lw = j.height ÷ 16, j.width ÷ 16
            x = Float16.(randn(Random.Xoshiro(j.seed), Float32, tr.channels, lh * lw, 1))
            latents[i] = sample!(tr, x, prompts[j.prompt], lh, lw; steps, refs = refs[j.references])
        end
        releaseplans!(tr.model)
    end
    release!(tr.model)

    images = Vector{Array{Float32,3}}(undef, length(jobs))
    for group in groupindices(j -> (j.width, j.height), jobs)
        for i in group
            j = jobs[i]
            image = decode!(vae, latents[i], j.height ÷ 16, j.width ÷ 16)
            images[i] = clamp.(image[:, :, :, 1] .* 0.5f0 .+ 0.5f0, 0f0, 1f0)
        end
        releaseplans!(vae.model)
    end
    release!(vae.model)
    return images
end

"""
    generate(prompt; references = [], width = 1024, height = 1024, steps = 4, seed = 42,
             variant = "4B", quantize = false, backend = Mantle.defaultbackend()) -> Array{Float32,3}

One generation, `(width, height, 3)` in [0, 1]: text to image, or with
`references` (pictures `(W, H, 3)` in [0, 1]) the multi-reference editing the
klein models are trained for. A batch of one [`FluxJob`](@ref); for several
pictures pass the jobs together, which builds every component once.
"""
generate(prompt::AbstractString; references = (), width::Integer = 1024, height::Integer = 1024,
         seed::Integer = 42, kw...) = only(generate([FluxJob(prompt; references, width, height, seed)]; kw...))

end # module
