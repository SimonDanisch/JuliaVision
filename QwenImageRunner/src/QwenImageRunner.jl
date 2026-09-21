"""
Qwen-Image 2.1 pipeline support for JuliaVision.

This package targets the official 32-layer, 7B visual transformer, Qwen3-VL-8B
conditioner, and 64-channel Wan-derived VAE. The host-side pipeline contract is
implemented here independently of the execution backend. Exported component
graphs are loaded through DNNKernels and can therefore run through any Mantle
backend that implements the required kernels.

The Comfy-Org compact checkpoints are not plain integer matrices. The denoiser
uses tensor-wise INT8 with ConvRot and the smallest encoder uses asymmetric
W4A8 with codebooks, per-group FP8 scales, per-channel scales, and ConvRot.
They must not be passed to the existing symmetric QInt8 path.

Upstream: https://huggingface.co/Qwen/Qwen-Image-2.1
Compact checkpoint: https://huggingface.co/Comfy-Org/Qwen-Image-2.1
License: Qwen Research License Agreement.
"""
module QwenImageRunner

using DNNKernels
using DNNKernels: loadgraph, readsafetensors, Model, planfor, replay!
# JSON3 comes through DNNKernels, which reads every exported graph with it. The
# tokenizer's `vocab.json` is the only other JSON this package reads, and a
# second copy in the manifest would be a second version to keep in step.
using DNNKernels: JSON3
using KernelAbstractions
using Lava
import Mantle

export QWEN_IMAGE_21, assetdir, ready, qwenimagegraph, qwenimageweights
export compact_transformer_weights, compact_text_encoder_weights
export QwenTokenizer, encode, decode
export QwenTextEncoder, qwenimagetextencoder, encode_prompt, token_embeddings
export qwen_prompt_template
export QwenTransformer, qwenimagetransformer, denoise!
export QwenVAEDecoder, qwenimagevae, decode!
export generate!
export packlatents, unpacklatents, image_sequence_length
export calculate_shift, qwen_schedule, euler_step!

const QWEN_IMAGE_21 = (
    transformer_layers = 32,
    hidden_size = 4096,
    attention_heads = 32,
    attention_head_dim = 128,
    context_dim = 4096,
    latent_channels = 64,
    vae_scale_factor = 16,
    patch_size = 1,
    rope_axes = (16, 56, 56),
    max_prompt_tokens = 1024,
    train_timesteps = 1000,
    base_image_seq_len = 256,
    max_image_seq_len = 8192,
    base_shift = 0.5f0,
    max_shift = 0.9f0,
    shift_terminal = 0.02f0,
)

const COMPONENT_FILES = Dict(
    :transformer => ("qwenimage21_transformer.json", "transformer.safetensors"),
    :text_encoder => ("qwenimage21_text_encoder.json", "text_encoder.safetensors"),
    :vae_decoder => ("qwenimage21_vae_decoder.json", "vae.safetensors"),
)

"""Directory containing an exported Qwen-Image 2.1 pipeline."""
function assetdir()
    dir = get(ENV, "JULIA_QWENIMAGE21_ASSETS", "")
    isempty(dir) && error(
        "QwenImageRunner: no Qwen-Image 2.1 export is installed. Set " *
        "JULIA_QWENIMAGE21_ASSETS to the directory produced by " *
        "`uv run tools/export_qwenimage21.py`.")
    dir
end

function _component(component::Symbol)
    haskey(COMPONENT_FILES, component) || throw(ArgumentError(
        "unknown Qwen-Image 2.1 component `$component`; expected " *
        join(sort!(string.(collect(keys(COMPONENT_FILES)))), ", ")))
    COMPONENT_FILES[component]
end

"""Whether all exported graphs, or one requested `component`, are installed."""
function ready(component::Union{Nothing,Symbol}=nothing;
               dir::AbstractString=get(ENV, "JULIA_QWENIMAGE21_ASSETS", ""))
    isempty(dir) && return false
    files = component === nothing ? values(COMPONENT_FILES) : (_component(component),)
    all(files) do (graph, weights)
        isfile(joinpath(dir, graph)) && isfile(joinpath(dir, weights))
    end
end

"""Load one of `:transformer`, `:text_encoder`, or `:vae_decoder`."""
function qwenimagegraph(component::Symbol; dir::AbstractString=assetdir())
    graph, _ = _component(component)
    path = joinpath(dir, graph)
    isfile(path) || throw(ArgumentError("Qwen-Image 2.1 graph not found at $path"))
    loadgraph(path)
end

"""Load the weights belonging to one exported component."""
function qwenimageweights(component::Symbol; dir::AbstractString=assetdir())
    _, weights = _component(component)
    path = joinpath(dir, weights)
    isfile(path) || throw(ArgumentError("Qwen-Image 2.1 weights not found at $path"))
    readsafetensors(path)
end

@inline _bf16(x::UInt16) = Float16(reinterpret(Float32, UInt32(x) << 16))
_bf16array(x::AbstractArray{UInt16}) = map(_bf16, x)
function _densebool(x::BitArray{N}) where {N}
    out = Array{Bool,N}(undef, size(x))
    @inbounds for i in eachindex(x)
        out[i] = x[i]
    end
    out
end

"""
    compact_transformer_weights(graph; checkpoint_dir, constants_dir)

Map Comfy-Org's compact Qwen-Image 2.1 denoiser into the exported Diffusers
graph. Quantized matrices remain packed host views and are decoded/repacked by
DNNKernels while uploading to the selected backend. Dense BF16 tensors are
converted to Float16; lifted RoPE, mask, and timestep constants come from the
allocation-free graph export.
"""
function compact_transformer_weights(graph;
        checkpoint_dir::AbstractString=get(ENV, "JULIA_QWENIMAGE21_COMPACT", ""),
        constants_dir::AbstractString=assetdir())
    isempty(checkpoint_dir) && throw(ArgumentError(
        "set JULIA_QWENIMAGE21_COMPACT to the Comfy-Org checkpoint directory"))
    checkpoint = joinpath(checkpoint_dir, "diffusion_models",
                          "qwen_image_2.1_int8_convrot.safetensors")
    constants_path = joinpath(constants_dir, "transformer_constants.safetensors")
    isfile(checkpoint) || throw(ArgumentError("compact denoiser not found at $checkpoint"))
    isfile(constants_path) || throw(ArgumentError("graph constants not found at $constants_path"))
    compact = readsafetensors(checkpoint)
    constants = readsafetensors(constants_path; mmap=false)
    out = Dict{String,Any}()

    for buffer in values(graph.buffers)
        buffer.kind === :weight || continue
        key = buffer.key
        isempty(key) && continue
        haskey(out, key) && continue
        if haskey(constants, key)
            value = constants[key]
            out[key] = value isa BitArray ? _densebool(value) : value
            continue
        end

        source = startswith(key, "model.") ? key[7:end] : key
        rows = nothing
        if occursin(".img_mlp.gate_layer.weight", source) ||
           occursin(".img_mlp.proj.weight", source)
            fused = replace(source,
                r"\.img_mlp\.(gate_layer|proj)\.weight$" => ".img_mlp.gate_up.weight")
            haskey(compact, fused) || throw(KeyError(source))
            half = size(compact[fused], 2) ÷ 2
            rows = occursin(".gate_layer.weight", source) ? (1:half) : (half+1:2half)
            source = fused
        end
        haskey(compact, source) || throw(KeyError(source))
        value = compact[source]

        quant_key = replace(source, r"\.weight$" => ".comfy_quant")
        scale_key = replace(source, r"\.weight$" => ".weight_scale")
        if eltype(value) === Int8 && haskey(compact, scale_key) && haskey(compact, quant_key)
            q = rows === nothing ? value : view(value, :, rows)
            scale = rows === nothing ? compact[scale_key] : view(compact[scale_key], :, rows)
            out[key] = DNNKernels.ConvRotQInt8HostMatrix(q, scale, 256)
        elseif eltype(value) === UInt16
            out[key] = _bf16array(rows === nothing ? value : view(value, :, rows))
        else
            out[key] = rows === nothing ? value : view(value, :, rows)
        end
    end
    out
end

"""
    QwenTransformer

A prepared, backend-independent Qwen-Image 2.1 denoiser graph. `plan` is
declared and recorded once at construction; each diffusion step is therefore a
GPU replay rather than 124+ host-side launches.
"""
struct QwenTransformer{B,G,W,P}
    backend::B
    graph::G
    weights::W
    plan::P
end

"""
    qwenimagetransformer(; backend=Mantle.LavaBackend(), dir=assetdir())

Load and prepare the exported denoiser. The graph is static in latent resolution
and prompt length; those are selected when running `tools/export_qwenimage21.py`.

`maxpasses` splits the recording into submissions of that many passes. One step
of the 32-layer model at 1024² is ~9.6 s of device time, and a single
submission that long is killed by the driver — `ring gfx_0.0.0 timeout`, a
device loss, and nothing in the Julia frame naming the cause. The completion
points cost nothing measurable and the barriers between the pieces are still
the ones the graph derived. `0` restores the single submission.
"""
function qwenimagetransformer(; backend=Mantle.LavaBackend(), dir::AbstractString=assetdir(),
                              compact_dir::Union{Nothing,AbstractString}=nothing,
                              maxpasses::Integer=64)
    graph_path = joinpath(dir, first(COMPONENT_FILES[:transformer]))
    isfile(graph_path) || throw(ArgumentError(
        "no Qwen-Image 2.1 transformer graph at $dir — run " *
        "`tools/export_qwenimage21.py --graph-only` first"))
    graph = qwenimagegraph(:transformer; dir)
    weights = compact_dir === nothing ? qwenimageweights(:transformer; dir) :
        compact_transformer_weights(graph; checkpoint_dir=compact_dir, constants_dir=dir)
    model = Model(Dict("qwenimage21_transformer" => graph), weights; backend)
    prepared = model.graphs["qwenimage21_transformer"]
    plan = planfor(model.device, prepared, model.weights, (;); maxpasses=Int(maxpasses))
    QwenTransformer(model.backend, prepared, model.weights, plan)
end

"""
    denoise!(model, latents, prompt_embeddings, timestep)

Run one transformer step. Arrays use Julia/DNNKernels order: `latents` is
`(64, image_tokens, batch)`, prompt embeddings are
`(4096, prompt_tokens, batch)`, and `timestep` is `(batch,)` scaled to `[0,1]`.
The returned device array has the same shape as `latents`.
"""
function denoise!(model::QwenTransformer, latents, prompt_embeddings, timestep)
    first(replay!(model.plan, "qwenimage21_transformer",
                  (latents, prompt_embeddings, timestep)))
end

"""A prepared Qwen-Image 2.1 VAE decoder. `plan === nothing` runs interpreted."""
struct QwenVAEDecoder{B,D,G,W,P}
    backend::B
    device::D
    graph::G
    weights::W
    plan::P
end

"""
    qwenimagevae(; backend=Mantle.LavaBackend(), dir=assetdir(), record=false)

Load and prepare the VAE decoder. Latent mean/std normalization is part of the
exported graph, so its input is directly the normalized diffusion state.

`record = false`, unlike the denoiser, because for this graph a recorded plan
costs more to build than it saves. At 1024² on an 8060S, decoding one denoised
latent: **4.79 s interpreted**, against ~50 s to build the plan and one decode
per image to amortise it.

Both were three times that — 12.8 and 16.0 s — until the convolutions stopped
falling off the tensor-core path. Eighty-two percent of this graph is
convolution, and twenty-seven of its forty-five were refused because their
im2col matrix is gigabytes; they ran at about 1 TFLOP/s where the eighteen that
fit reach 19-22. `conv_coopmat_plan` now divides the pixel axis into chunks
instead of refusing, and pads the output channels onto a column tile the staged
GEMM has. See `DNNKernels.IM2COL_CAP`. The two paths agree to
9.3e-5 rms and 4.9e-4 peak of a [-1, 1] range, so this is a speed choice and not
a correctness one.

`record = true` builds the model with `fuseattn = false`, because the fused form
cannot be recorded: the mid-block attention is a single head 1152 wide, which
`flashcm_plan` declines and `coopmat_sdpa_plan` accepts, and that plan has no
declared form. Unfusing it costs nothing measurable here (12.7 s against 12.8 s
interpreted) because it is one attention among 422 ops.

Two numbers that were in this docstring and are wrong: the decode is not 38.4 s
(that was a first call, whose extra ~6 s is shader compilation), and a recorded
submission does not have to exceed the driver's limit (`maxpasses = 8` records
and replays; 64 is what times out).
"""
function qwenimagevae(; backend=Mantle.LavaBackend(), dir::AbstractString=assetdir(),
                      record::Bool=false, maxpasses::Integer=8)
    ready(:vae_decoder; dir) || throw(ArgumentError(
        "no Qwen-Image 2.1 VAE export at $dir — run " *
        "`tools/export_qwenimage21.py --component vae` first"))
    graph = qwenimagegraph(:vae_decoder; dir)
    weights = qwenimageweights(:vae_decoder; dir)
    model = Model(Dict("qwenimage21_vae_decoder" => graph), weights; backend,
                  fuseattn = !record)
    prepared = model.graphs["qwenimage21_vae_decoder"]
    plan = record ?
        planfor(model.device, prepared, model.weights, (;); maxpasses=Int(maxpasses)) :
        nothing
    QwenVAEDecoder(model.backend, model.device, prepared, model.weights, plan)
end

"""
    decode!(model, latents)

Decode normalized latents in Julia order `(width, height, 1, channels, batch)`
to an image `(width*16, height*16, 1, 4, batch)` — the decoder's fourth channel
is alpha — on the same backend.
"""
decode!(model::QwenVAEDecoder{<:Any,<:Any,<:Any,<:Any,Nothing}, latents) =
    DNNKernels.execute!(model.graph, Dict(only(model.graph.inputs) => latents),
                        model.weights; dims=(;), backend=model.backend)[
        DNNKernels.viewroot(model.graph, only(model.graph.outputs))]

decode!(model::QwenVAEDecoder, latents) =
    first(replay!(model.plan, "qwenimage21_vae_decoder", (latents,)))

"""
    image_sequence_length(width, height) -> Int

Number of image tokens. Qwen's VAE compresses spatially by 16 and the 2.1
transformer consumes the resulting 64-channel latent grid without patching.
"""
function image_sequence_length(width::Integer, height::Integer)
    width > 0 && height > 0 || throw(ArgumentError("image dimensions must be positive"))
    width % 16 == 0 && height % 16 == 0 || throw(ArgumentError(
        "Qwen-Image 2.1 dimensions must be divisible by 16, got $(width)x$(height)"))
    (width ÷ 16) * (height ÷ 16)
end

"""
    packlatents(x) -> packed

Flatten Julia-layout VAE latents `(width, height, channels, batch)` into the
transformer's `(channels, tokens, batch)` layout. This is the byte-order
equivalent of Qwen-Image 2.1's `(B,C,H,W) -> (B,H*W,C)` transform.
"""
function packlatents(x::AbstractArray{T,4}) where {T}
    w, h, c, b = size(x)
    permutedims(reshape(x, w * h, c, b), (2, 1, 3))
end

"""Inverse of [`packlatents`](@ref), returning `(width,height,channels,batch)`."""
function unpacklatents(packed::AbstractArray{T,3}, width::Integer, height::Integer) where {T}
    width > 0 && height > 0 ||
        throw(ArgumentError("latent width and height must be positive"))
    c, tokens, b = size(packed)
    tokens == width * height || throw(DimensionMismatch(
        "got $tokens tokens, expected $(width * height)"))
    reshape(permutedims(packed, (2, 1, 3)), width, height, c, b)
end

"""Resolution-dependent shift used by the official FlowMatch scheduler."""
function calculate_shift(image_seq_len::Integer;
                         base_seq_len::Integer=QWEN_IMAGE_21.base_image_seq_len,
                         max_seq_len::Integer=QWEN_IMAGE_21.max_image_seq_len,
                         base_shift::Real=QWEN_IMAGE_21.base_shift,
                         max_shift::Real=QWEN_IMAGE_21.max_shift)
    base_seq_len < max_seq_len || throw(ArgumentError("base_seq_len must be below max_seq_len"))
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    Float32(image_seq_len * slope + base_shift - slope * base_seq_len)
end

"""
    qwen_schedule(width, height; steps=40) -> (timesteps, sigmas)

The exact Qwen-Image 2.1 dynamic-shift, exponential FlowMatch schedule. `sigmas`
has `steps + 1` entries because the final Euler update targets zero; `timesteps`
has one entry per denoiser evaluation and is expressed on the model's 0..1000
training scale.
"""
function qwen_schedule(width::Integer, height::Integer; steps::Integer=40)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    n = image_sequence_length(width, height)
    mu = calculate_shift(n)
    # The pipeline supplies `linspace(1, 1/steps, steps)` to the scheduler.
    # `1e-3` is the scheduler's default when no explicit sigmas are supplied,
    # but Qwen does supply them; confusing the two substantially changes every
    # late denoising step.
    raw = collect(range(1f0, inv(Float32(steps)); length=steps))
    emu = exp(mu)
    shifted = @. emu / (emu + (inv(raw) - 1f0))

    terminal = QWEN_IMAGE_21.shift_terminal
    scale = (1f0 - shifted[end]) / (1f0 - terminal)
    shifted = @. 1f0 - (1f0 - shifted) / scale
    timesteps = shifted .* Float32(QWEN_IMAGE_21.train_timesteps)
    sigmas = [shifted; 0f0]
    (; timesteps, sigmas, mu)
end

"""One deterministic FlowMatch Euler update, performed in place."""
function euler_step!(sample, model_output, sigma::Real, sigma_next::Real)
    axes(sample) == axes(model_output) || throw(DimensionMismatch(
        "sample and model output must have identical axes"))
    sample .+= convert(eltype(sample), sigma_next - sigma) .* model_output
    sample
end

"""
    generate!(transformer, vae, latents, prompt_embeddings; width, height, steps=40)

Run the diffusion loop from caller-supplied noise and precomputed Qwen3-VL
prompt embeddings, then decode the result. `latents` is updated in place and
has shape `(64, image_sequence_length(width,height), batch)`.

Prompt encoding is intentionally outside this method: the compact Comfy-Org
encoder is asymmetric W4A8+ConvRot and cannot be represented by the existing
symmetric INT8 weight type without silently changing the model.
"""
function generate!(transformer::QwenTransformer, vae::QwenVAEDecoder,
                   latents, prompt_embeddings;
                   width::Integer, height::Integer, steps::Integer=40)
    graph_channels = Int(last(transformer.graph.buffers["latents"].shape))
    size(latents, 1) == graph_channels || throw(DimensionMismatch(
        "latents have $(size(latents, 1)) channels, expected $graph_channels"))
    expected_tokens = image_sequence_length(width, height)
    size(latents, 2) == expected_tokens || throw(DimensionMismatch(
        "latents have $(size(latents, 2)) tokens, expected $expected_tokens"))
    size(latents, 3) == size(prompt_embeddings, 3) || throw(DimensionMismatch(
        "latent and prompt batch sizes differ"))

    schedule = qwen_schedule(width, height; steps)
    timestep = similar(latents, eltype(latents), size(latents, 3))
    for i in eachindex(schedule.timesteps)
        fill!(timestep, convert(eltype(timestep), schedule.timesteps[i] / 1000f0))
        prediction = denoise!(transformer, latents, prompt_embeddings, timestep)
        euler_step!(latents, prediction, schedule.sigmas[i], schedule.sigmas[i + 1])
    end

    lw = width ÷ QWEN_IMAGE_21.vae_scale_factor
    lh = height ÷ QWEN_IMAGE_21.vae_scale_factor
    grid = unpacklatents(latents, lw, lh)
    decode!(vae, reshape(grid, lw, lh, 1, size(grid, 3), size(grid, 4)))
end

include("tokenizer.jl")
include("textencoder.jl")

"""
    Mantle.release!(component)

Free a prepared component's plan and return its device memory to the pool.

The three components of this pipeline do not fit on an 8060S at once: the
denoiser decodes to 7.26 GB of INT8 and the Qwen3-VL conditioner to another
6.9 GB. A generation encodes its prompt, releases the encoder, denoises,
releases the denoiser, and only then decodes — which is also the order in which
each is finished with. See `examples/generate.jl`.

A method on Mantle's own `release!` rather than a second name for it: it means
the same thing here as it does for a recording, and two exported `release!`s
are an ambiguity at every call site that has both packages in scope.
"""
function Mantle.release!(component::Union{QwenTextEncoder,QwenTransformer,QwenVAEDecoder})
    component.plan === nothing || Mantle.free!(component.plan.plan)
    # The weight dict is the only reference the component holds to the device
    # arrays; the `Model` that built them is long gone.
    empty!(component.weights)
    GC.gc(true)
    Mantle.trim_gpu_pool!()
    GC.gc(true)
    nothing
end

end # module
