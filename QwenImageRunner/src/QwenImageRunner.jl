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
using LazyArtifacts
using Artifacts: artifact_hash, artifact_exists

# `@artifact_str` finds this itself; `ready` needs the path explicitly, because
# it has to answer without downloading anything.
const ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

export QWEN_IMAGE_21, assetdir, vaedir, processordir, ready, rotarytables
export qwenimagegraph, qwenimageweights, compact_denoiser, compact_encoder
export compact_transformer_weights, compact_text_encoder_weights
export QwenTokenizer, encode, decode
export QwenTextEncoder, qwenimagetextencoder, encode_prompt, token_embeddings
export qwen_prompt_template
export QwenTransformer, qwenimagetransformer, denoise!
export QwenVAEDecoder, qwenimagevae, decode!
export generate!
export packlatents, unpacklatents, image_sequence_length
export latentshape, imagetokenbounds
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

"""
    COMPONENT_ARTIFACTS

Which artifacts each component needs, for [`ready`](@ref).

The pipeline is thirteen artifacts rather than one tree, for the reason
Hunyuan3D is four: the parts are 22 MB, 644 MB, 6.8 GB and 5.9 GB, and a caller
that only wants prompt embeddings has no reason to fetch the denoiser. The two
large checkpoints are also past what a single release asset can hold, so they
arrive as shards regardless of how they are grouped here.
"""
const COMPONENT_ARTIFACTS = Dict(
    :transformer  => vcat(["qwenimage21"], ["qwenimage21-dit-w$i" for i in 1:6]),
    :text_encoder => vcat(["qwenimage21"], ["qwenimage21-enc-w$i" for i in 1:5]),
    :vae_decoder  => ["qwenimage21-vae"],
)

"""
    DIT_SHARDS, ENC_SHARDS

The compact checkpoints, split to fit a release asset. `weights-<i>of<n>` is a
byte-for-byte piece of Comfy-Org's file: [`compact_denoiser`](@ref) and
[`compact_encoder`](@ref) merge them back and nothing downstream knows they were
ever apart.
"""
const DIT_SHARDS = ntuple(i -> "weights-$(i)of6.safetensors", 6)
const ENC_SHARDS = ntuple(i -> "weights-$(i)of5.safetensors", 5)

"""
    assetdir(), vaedir(), processordir() -> String

Where the exported graphs live. `assetdir` holds the denoiser and text encoder
graphs with the constants their traces lifted out; `vaedir` holds the VAE
decoder graph beside its weights, which are small enough to travel with it;
`processordir` holds the three tokenizer tables `QwenTokenizer` reads.

These replaced `JULIA_QWENIMAGE21_ASSETS`, `JULIA_QWENIMAGE21_COMPACT` and
`JULIA_QWENIMAGE21_PROCESSOR`. Every one of those had to be set by hand against
a tree only the machine that ran the exporter had, so the package installed and
could not run anywhere else.
"""
assetdir() = @artifact_str("qwenimage21")
vaedir() = @artifact_str("qwenimage21-vae")
processordir() = joinpath(assetdir(), "processor")

function _component(component::Symbol)
    haskey(COMPONENT_FILES, component) || throw(ArgumentError(
        "unknown Qwen-Image 2.1 component `$component`; expected " *
        join(sort!(string.(collect(keys(COMPONENT_FILES)))), ", ")))
    COMPONENT_FILES[component]
end

"""Where `component`'s graph lives; only the VAE keeps its weights beside it."""
function componentdir(component::Symbol)
    _component(component)              # rejects an unknown name before downloading
    component === :vae_decoder ? vaedir() : assetdir()
end

"""
Whether the whole pipeline, or one requested `component`, is already downloaded.

Deliberately does NOT go through `@artifact_str`: that downloads, and this is
the question asked to decide whether to.
"""
function ready(component::Union{Nothing,Symbol}=nothing)
    names = component === nothing ?
        sort!(unique(reduce(vcat, values(COMPONENT_ARTIFACTS)))) :
        (haskey(COMPONENT_ARTIFACTS, component) ? COMPONENT_ARTIFACTS[component] :
         throw(ArgumentError("unknown Qwen-Image 2.1 component `$component`; expected " *
             join(sort!(string.(collect(keys(COMPONENT_ARTIFACTS)))), ", "))))
    all(names) do n
        h = artifact_hash(n, ARTIFACTS_TOML)
        h !== nothing && artifact_exists(h)
    end
end

"""Load one of `:transformer`, `:text_encoder`, or `:vae_decoder`."""
function qwenimagegraph(component::Symbol; dir::AbstractString=componentdir(component))
    graph, _ = _component(component)
    path = joinpath(dir, graph)
    isfile(path) || throw(ArgumentError("Qwen-Image 2.1 graph not found at $path"))
    loadgraph(path)
end

"""
Load the weights belonging to one exported component.

Only `:vae_decoder` ships these. The denoiser and text encoder run from
Comfy-Org's quantized checkpoints — see [`compact_denoiser`](@ref) — because the
released half-precision pair is some 60 GB and neither fits this device even
alone. `tools/export_qwenimage21.py` without `--graph-only` still writes
`transformer.safetensors` and `text_encoder.safetensors`, so this reaches them
when handed an explicit `dir`.
"""
function qwenimageweights(component::Symbol; dir::AbstractString=componentdir(component))
    _, weights = _component(component)
    path = joinpath(dir, weights)
    isfile(path) || throw(ArgumentError("Qwen-Image 2.1 weights not found at $path"))
    readsafetensors(path)
end

"""
    compact_denoiser() -> Dict

Comfy-Org's INT8 ConvRot denoiser, merged from its six shards.

`mmap = true` EXPLICITLY, and that is the whole point of this docstring.
`readsafetensors` maps a file above 4 GiB and reads one below it, on the
reasoning that a small read is cheap and a mapping that outlives its `open`
block is a subtler thing to hand back. That threshold is per FILE, and this
checkpoint is 6.92 GB in six shards of about 1.15 GB — so every one of them
took the read path and the merged dictionary was 6.92 GB of ANONYMOUS memory
before the graph was walked. This docstring used to claim the opposite.

It matters most on the machine this is built for. An 8060S has no separate
VRAM: the weights the host has read and the weights the device holds come out
of the same pool, so a checkpoint read rather than mapped is charged twice.
Measured, the six shards:

    mmap = false   1.51 s   +6920.7 MB resident, all of it anonymous
    mmap = true    0.07 s   +1.6 MB

The bytes are still read, later and one tensor at a time, as
[`compact_transformer_weights`](@ref) walks the graph — as page cache the
kernel can drop under pressure rather than anonymous memory it has to kill for.
`readsafetensors` measures that fault-in at ~1.5 GB/s against 6.0 GB/s for a
plain read, which is the price.
"""
compact_denoiser() = merge(
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w1"), DIT_SHARDS[1]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w2"), DIT_SHARDS[2]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w3"), DIT_SHARDS[3]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w4"), DIT_SHARDS[4]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w5"), DIT_SHARDS[5]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-dit-w6"), DIT_SHARDS[6]); mmap = true))

"""
    compact_encoder() -> Dict

Comfy-Org's W4A8 Qwen3-VL conditioner, merged from its five shards.

`mmap = true` for the reason [`compact_denoiser`](@ref) gives at length: five
shards, none of them over `readsafetensors`' per-file threshold, 6.9 GB read
into anonymous memory on a machine that then has to hold the device copy too.
"""
compact_encoder() = merge(
    readsafetensors(joinpath(@artifact_str("qwenimage21-enc-w1"), ENC_SHARDS[1]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-enc-w2"), ENC_SHARDS[2]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-enc-w3"), ENC_SHARDS[3]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-enc-w4"), ENC_SHARDS[4]); mmap = true),
    readsafetensors(joinpath(@artifact_str("qwenimage21-enc-w5"), ENC_SHARDS[5]); mmap = true))

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
    compact_transformer_weights(graph; compact, constants_dir)

Map Comfy-Org's compact Qwen-Image 2.1 denoiser into the exported Diffusers
graph. Quantized matrices remain packed host views and are decoded/repacked by
DNNKernels while uploading to the selected backend. Dense BF16 tensors are
converted to Float16; lifted RoPE, mask, and timestep constants come from the
allocation-free graph export.

`compact` is the checkpoint as a dictionary rather than a directory to read it
from, because it arrives as six artifacts that [`compact_denoiser`](@ref)
merges. Pass one read by hand to use a checkpoint from somewhere else.
"""
function compact_transformer_weights(graph;
        compact::AbstractDict=compact_denoiser(),
        constants_dir::AbstractString=assetdir())
    constants_path = joinpath(constants_dir, "transformer_constants.safetensors")
    isfile(constants_path) || throw(ArgumentError("graph constants not found at $constants_path"))
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
struct QwenTransformer{B,G,W,P,R}
    backend::B
    graph::G
    weights::W
    plan::P
    # The rotary tables for this prompt length, on the device. They are graph
    # inputs, so they are passed to every replay rather than uploaded as
    # weights, and they are built once here because the length does not change
    # between steps of one generation.
    rotary::R
    context_tokens::Int
    # The latent grid this plan was recorded for, and the token count it
    # implies. Both are here because `rotarytables` needs the two extents
    # separately — the image's height and width rotary axes are a grid centred
    # on zero — while the graph's `i` symbol only ever sees their product.
    latent::Tuple{Int,Int}
    image_tokens::Int
end

"""
    qwenimagetransformer(; context_tokens, dir=assetdir(),
                         latent=latentshape(dir), backend=Mantle.defaultbackend(),
                         compact=compact_denoiser())

Load and prepare the exported denoiser for a prompt of `context_tokens` tokens
at a latent grid of `latent = (height, width)`.

The graph is **generic in both**: its `t` and `i` symbols are bound here, once,
for the generation about to run. That mirrors the reference, where
`QwenImagePipeline` pads nothing for a single prompt and
`QwenImage21Transformer2DModel.forward` simply runs at whatever lengths it is
given.

`latent` defaults to the grid the export was TRACED at, which is what the old
behaviour was — that grid used to be the only one the graph could run, because
the exporter pinned the image axis to an integer while making the prompt axis a
symbol. It is now `Dim("i", ...)` and the trace grid is only a default. Use
[`image_sequence_length`](@ref) to go from pixels to tokens, or pass the latent
grid directly; a `1024x1024` picture is `(64, 64)`.

**Binding a symbol is not free, and running at a bound one is.** Each distinct
`(context_tokens, latent)` needs its own recorded plan, because a plan is a
recording of concrete dispatches — that is the cost, paid once per shape at
construction. The replay afterwards is the same work per step it always was:
the graph carries the same 3473 ops at every resolution, and nothing in the step
consults a symbol.

The rotary tables come from [`rotarytables`](@ref) rather than from the export,
because they are not independent of the prompt length: `QwenImage21Rope` freezes
the image block's frame axis at the position the text reached. An earlier export
lifted them as constants, which silently bound the whole graph to one prompt.

`maxpasses` splits the recording into submissions of that many passes. One step
of the 32-layer model at 1024² is ~9.6 s of device time, and a single
submission that long is killed by the driver — `ring gfx_0.0.0 timeout`, a
device loss, and nothing in the Julia frame naming the cause. The completion
points cost nothing measurable and the barriers between the pieces are still
the ones the graph derived. `0` restores the single submission.
"""
function qwenimagetransformer(; context_tokens::Integer,
                              dir::AbstractString=assetdir(),
                              # After `dir`, because a keyword default may only
                              # refer to keywords declared before it and this one
                              # reads the export's own trace grid.
                              latent::Tuple{Integer,Integer}=latentshape(dir),
                              backend=Mantle.defaultbackend(),
                              compact::Union{Nothing,AbstractDict}=compact_denoiser(),
                              maxpasses::Integer=64)
    graph_path = joinpath(dir, first(COMPONENT_FILES[:transformer]))
    isfile(graph_path) || throw(ArgumentError(
        "no Qwen-Image 2.1 transformer graph at $dir — run " *
        "`tools/export_qwenimage21.py --graph-only` first"))
    ctx = Int(context_tokens)
    2 <= ctx <= QWEN_IMAGE_21.max_prompt_tokens || throw(ArgumentError(
        "context_tokens must be in 2..$(QWEN_IMAGE_21.max_prompt_tokens), got $ctx"))
    graph = qwenimagegraph(:transformer; dir)
    lh, lw = Int(latent[1]), Int(latent[2])
    lh > 0 && lw > 0 || throw(ArgumentError("latent grid must be positive, got $((lh, lw))"))
    img = lh * lw
    lo, hi = imagetokenbounds(dir)
    lo <= img <= hi || throw(ArgumentError(
        "a latent grid of $(lh)x$(lw) is $img image tokens and the graph was " *
        "exported for $lo..$hi; re-export with wider `Dim(\"i\", ...)` bounds " *
        "in `tools/export_qwenimage21.py` if this resolution is wanted"))
    # `nothing` takes the half-precision route instead, which needs a `dir` that
    # has `transformer.safetensors` in it. Nothing ships one: it is 41 GB, and
    # the whole reason the compact checkpoint is the default is that it fits.
    weights = compact === nothing ? qwenimageweights(:transformer; dir) :
        compact_transformer_weights(graph; compact, constants_dir=dir)
    model = Model(Dict("qwenimage21_transformer" => graph), weights; backend)
    prepared = model.graphs["qwenimage21_transformer"]
    plan = planfor(model.device, prepared, model.weights, (; t = ctx, i = img);
                   maxpasses=Int(maxpasses))
    rc, rs = rotarytables(ctx, lh, lw)
    rotary = (DNNKernels.toback(backend, rc), DNNKernels.toback(backend, rs))
    QwenTransformer(model.backend, prepared, model.weights, plan, rotary, ctx,
                    (lh, lw), img)
end

"""
    latentshape(dir) -> (latent_height, latent_width)

The latent grid the denoiser graph was **traced** at, from the export's own
metadata. A default and no longer a constraint: the image axis is the graph's
`i` symbol, so any grid inside [`imagetokenbounds`](@ref) runs on the same
graph. `qwenimagetransformer` uses this when the caller names no grid, which
keeps the old call sites meaning what they meant.

The grid cannot be read back off the graph in any case: the graph knows only
that there are `latent_height * latent_width` image tokens, and 4096 of them is
64x64 or 128x32 with nothing to tell them apart. `rotarytables` needs the two
separately, because the image's height and width rotary axes are a grid centred
on zero — which is also why the resolution is a pair here rather than one
number.
"""
function latentshape(dir::AbstractString)
    meta = exportmeta(dir)
    Int(meta.latent_height), Int(meta.latent_width)
end

"""
    imagetokenbounds(dir) -> (min, max)

How many image tokens the exported graph was checked for — the `Dim("i", ...)`
bounds `tools/export_qwenimage21.py` declared.

Read rather than assumed, because a graph exported with different bounds is a
different graph and binding `i` outside them is undefined rather than slow. An
export predating the symbol has no bounds recorded; it is pinned to the grid it
was traced at, and says so.
"""
function imagetokenbounds(dir::AbstractString)
    meta = exportmeta(dir)
    haskey(meta, :min_image_tokens) && haskey(meta, :max_image_tokens) || begin
        lh, lw = Int(meta.latent_height), Int(meta.latent_width)
        n = lh * lw
        return (n, n)
    end
    Int(meta.min_image_tokens), Int(meta.max_image_tokens)
end

"""The denoiser export's metadata, as written by `tools/export_qwenimage21.py`."""
function exportmeta(dir::AbstractString)
    path = joinpath(dir, "qwenimage21_export.json")
    isfile(path) || throw(ArgumentError("no Qwen-Image 2.1 export metadata at $path"))
    JSON3.read(read(path, String))
end

"""
    denoise!(model, latents, prompt_embeddings, timestep)

Run one transformer step. Arrays use Julia/DNNKernels order: `latents` is
`(64, image_tokens, batch)`, prompt embeddings are
`(4096, prompt_tokens, batch)`, and `timestep` is `(batch,)` scaled to `[0,1]`.
The returned device array has the same shape as `latents`.
"""
function denoise!(model::QwenTransformer, latents, prompt_embeddings, timestep)
    size(prompt_embeddings, 2) == model.context_tokens || throw(DimensionMismatch(
        "this denoiser was prepared for $(model.context_tokens) prompt embeddings " *
        "and was given $(size(prompt_embeddings, 2)); the graph is length-generic " *
        "but a recorded plan is not, so build it with " *
        "`qwenimagetransformer(; context_tokens = $(size(prompt_embeddings, 2)))`"))
    # The same statement for the image axis, which became a symbol at the same
    # time. Without it a wrong-sized latent reaches `replay!`, whose own check
    # names the buffer rather than the resolution — true and two steps further
    # from the cause.
    size(latents, 2) == model.image_tokens || throw(DimensionMismatch(
        "this denoiser was prepared for a $(model.latent[1])x$(model.latent[2]) " *
        "latent grid ($(model.image_tokens) image tokens) and was given " *
        "$(size(latents, 2)); the graph is resolution-generic but a recorded plan " *
        "is not, so build it with `qwenimagetransformer(; context_tokens, " *
        "latent = (h, w))` for the grid you want"))
    first(replay!(model.plan, "qwenimage21_transformer",
                  (latents, prompt_embeddings, timestep, model.rotary...)))
end

"""The VAE decoder graph's name, which is also its key in the `Model`."""
const VAEGRAPH = "qwenimage21_vae_decoder"

"""
A prepared Qwen-Image 2.1 VAE decoder.

It holds its `Model` and nothing else, because a `Model` is what knows how to
plan a graph for a grid and how to keep the plan. There is no second field for
"the plan" and no field for "no plan": see [`decode!`](@ref).
"""
struct QwenVAEDecoder{B,M}
    backend::B
    model::M
end

"""
    qwenimagevae(; backend=Mantle.defaultbackend(), dir=vaedir(), maxpasses=8)

Load and prepare the VAE decoder. Latent mean/std normalization is part of the
exported graph, so its input is directly the normalized diffusion state.

The grid is not an argument. `decode!` plans for whatever latents it is handed
and the `Model` keeps that plan, so decoding two sizes builds two plans and
decoding one size twice builds one. A caller that knows its grid gains nothing
by saying so up front.

## What the interpreted path cost, and why there is no longer one

This used to take `record::Bool`, defaulting to `latent !== nothing`, and
`record = false` fell through to `DNNKernels.execute!` — which allocates one
buffer per op and frees none, so the peak is the SUM of the decoder's
activations by construction. That was never a property of interpreting a graph.
Both paths had a placer until `ea74587`, when DNNKernels' slab allocator was
deleted for the good reason that "declared, Mantle owns the arena — and a second
placer is a second answer to one question"; what it left behind was
`place(::Nothing, …) = nothing` as the only method, so every `dest` on the
unplanned path fell through to a fresh `KA.allocate`.

Measured on an 8060S, device memory for one decode plus the one-time build:

    output     interpreted   recorded   build    decode
    256x256        5 590 MB   1 014 MB    1.2 s    0.17 s
    512x512       19 889 MB   1 504 MB   14.4 s    0.64 s
    1024x1024     66 048 MB   3 880 MB   14.4 s    2.51 s

Re-measured 2026-09-23 at 256² on the same card, GTT+VRAM peak sampled at 100 Hz
while decoding: 7025 MiB for `qwenimagevae(; backend)` as it was against 1439
MiB as it is, both including the weights and the build. The two outputs are
BYTE-IDENTICAL — 1 048 576 bytes compared, old recorded against new — so this
changed what the default costs and not what it computes.

66 GB for one 1024² image, which ran here only because this APU hands out 114 GB
of GTT and is an immediate out-of-memory on any discrete card. The decode is
faster planned at every size as well. The one case the old default won was a
single 1024² decode and nothing else — 4.79 s against 16.9 s including the
build — at seventeen times the memory.

`fuseattn = false`, because the fused form cannot be recorded: the mid-block
attention is a single head 1152 wide, which `flashcm_plan` declines and
`coopmat_sdpa_plan` accepts, and that plan has no declared form. Unfusing costs
nothing measurable — 12.7 s against 12.8 s — because it is one attention among
422 ops. It used to be `!record`, which is to say it was already false whenever
a plan was built.

Eighty-two percent of this graph is convolution, and twenty-seven of its
forty-five used to be refused because their im2col matrix is gigabytes; they ran
at about 1 TFLOP/s where the eighteen that fit reach 19-22. `conv_coopmat_plan`
now chunks the pixel axis instead of refusing, and pads the output channels onto
a column tile the staged GEMM has. See `DNNKernels.IM2COL_CAP`. The two paths
agreed to 9.3e-5 rms and 4.9e-4 peak of a [-1, 1] range, so dropping the
interpreted one is a memory and speed choice and not a correctness one.

`maxpasses` is the recording's explicit submission split; 8 records and replays,
64 is what times out.
"""
function qwenimagevae(; backend=Mantle.defaultbackend(), dir::AbstractString=vaedir(),
                      maxpasses::Integer=8)
    graph = qwenimagegraph(:vae_decoder; dir)
    weights = qwenimageweights(:vae_decoder; dir)
    model = Model(Dict(VAEGRAPH => graph), weights; backend, fuseattn = false,
                  record_maxpasses = Dict(VAEGRAPH => Int(maxpasses)))
    QwenVAEDecoder(model.backend, model)
end

"""The decoder's `h` and `w` symbols, read off the latents being decoded.

`(width, height, 1, channels, batch)` is the Julia order, so `w` is the first
extent and `h` the second. Derived here rather than stored on the decoder
because the decoder has no resolution of its own: it decodes whatever it is
handed, and the grid IS the argument. `call` keys its plan cache on these, so
one decoder serves any number of grids at one plan each.
"""
vaedims(latents) = (; h = size(latents, 2), w = size(latents, 1))

"""
    decode!(model, latents)

Decode normalized latents in Julia order `(width, height, 1, channels, batch)`
to an image `(width*16, height*16, 4, batch)` — the decoder's fourth channel is
alpha — on the same backend.

The output is rank 4 and the input rank 5, which is not a typo: the graph's
declared output is `select`, torch `[1, 4, "16*h", "16*w"]`, and `select` is what
drops the depth axis the latents carry.

This said rank 5 until 2026-09-23, because the old unplanned path returned
`values[viewroot(graph, output)]` — `clamp_36`, the buffer the `select` is a VIEW
of — rather than the output the graph declares. A view has no storage of its own
on that path, so it handed back the root and an axis with it. The planned path
always returned rank 4. `examples/generate.jl` was written against the leak and
indexed `[:, :, 1, 1:3, 1]`.
"""
# One method. `call` plans for this grid on the first decode of it, keeps the
# plan on the `Model`, and replays after — so the grid does not have to be known
# when the decoder is built, and there is no unplanned path to fall down.
decode!(model::QwenVAEDecoder, latents) =
    first(DNNKernels.call(model.model, VAEGRAPH, latents; dims = vaedims(latents)))

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
    rotarytables(context_tokens, latent_height, latent_width) -> (cos, sin)

Qwen-Image 2.1's three-axis rotary tables over the joint text/image sequence, as
two `(64, context_tokens + latent_height*latent_width)` Float32 matrices.

A port of `QwenImage21Rope.forward`. Text tokens advance one shared position per
token on all three axes; the image block then freezes its frame axis at the
position the text reached and lays its tokens out on a height/width grid centred
on zero. **The image's rotary therefore depends on how long the text was**,
which is why this is computed per prompt rather than exported as a constant, and
why a graph with it baked in only works for one prompt length.

The reference precomputes a table of 8192 non-negative rows followed by 1024
negative ones so that it can gather with a signed index tensor. A row's angle is
`index * theta^(-2j/d)` either way, so this evaluates it directly; the negative
half of the reference table exists for indexing, not for different arithmetic.
"""
function rotarytables(context_tokens::Integer, latent_height::Integer,
                      latent_width::Integer;
                      axes::NTuple{3,Int}=QWEN_IMAGE_21.rope_axes, theta::Real=10000)
    nt, h, w = Int(context_tokens), Int(latent_height), Int(latent_width)
    nt >= 0 || throw(ArgumentError("context_tokens must be non-negative, got $nt"))
    h > 0 && w > 0 || throw(ArgumentError("latent dimensions must be positive"))
    total = nt + h * w

    # The frame axis: text at 0..nt-1, then the whole image block frozen at nt.
    frame = Vector{Int}(undef, total)
    for i in 1:nt
        frame[i] = i - 1
    end
    for i in (nt + 1):total
        frame[i] = nt
    end
    # Height and width follow the frame axis over the text and become the
    # centred grid over the image. Height varies slowest, as in the reference's
    # `for h in ... for _ in range(width)`.
    height, width = copy(frame), copy(frame)
    k = nt
    for hh in -(h - h ÷ 2):(h ÷ 2 - 1), ww in -(w - w ÷ 2):(w ÷ 2 - 1)
        k += 1
        height[k] = hh
        width[k] = ww
    end

    ncols = sum(axes) ÷ 2
    co = Matrix{Float32}(undef, ncols, total)
    si = Matrix{Float32}(undef, ncols, total)
    row = 0
    for (dim, index) in zip(axes, (frame, height, width))
        half = dim ÷ 2
        for j in 1:half
            invfreq = 1f0 / Float32(theta)^(Float32(2 * (j - 1)) / Float32(dim))
            @inbounds for i in 1:total
                angle = Float32(index[i]) * invfreq
                co[row + j, i] = cos(angle)
                si[row + j, i] = sin(angle)
            end
        end
        row += half
    end
    co, si
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

"""One deterministic FlowMatch Euler update, performed in place.

`Mantle.storage` on both operands, because either spelling reaches here: a
caller that uploaded its noise with `DNNKernels.toback` holds a `Mantle.Buffer`
— the pool region — and a `Buffer` is not an `AbstractArray`, so broadcasting
into one is a `MethodError` from inside `BroadcastStyle`. `storage` is the
identity on anything that is already an array.
"""
function euler_step!(sample, model_output, sigma::Real, sigma_next::Real)
    x, d = Mantle.storage(sample), Mantle.storage(model_output)
    axes(x) == axes(d) || throw(DimensionMismatch(
        "sample and model output must have identical axes"))
    x .+= convert(eltype(x), sigma_next - sigma) .* d
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
# The denoiser holds one hand-rolled plan and its weights directly; the encoder
# and the decoder hold a `Model`, which holds its weights and the plans `call`
# has built. Two things to free either way, reached differently, so two methods
# rather than a branch inside one.
# `free!(c.plan)` and NOT `free!(c.plan.plan)`. The second frees the Mantle
# `Plan` and stops there; `DNNKernels.free!(::RecordedPlan)` frees that AND the
# buffers the emit owns, which is where `residentweights` put the weights.
#
# It was the inner one until 2026-09-23, so `release!` returned the transients
# and left ~6.9 GB of INT8 encoder resident. Measured over a 256x256 generation,
# GTT+VRAM: 7977 MiB with the encoder up, 7531 after releasing it — 446 MiB of
# the 6.9 GB — and then 15119 once the denoiser loaded on top of what should
# have been gone. The staging this file's docstring describes did not work.
freeplans!(c::QwenTransformer) =
    (c.plan === nothing || Mantle.free!(c.plan); nothing)
freeplans!(c::Union{QwenTextEncoder,QwenVAEDecoder}) =
    (DNNKernels.releaseplans!(c.model); nothing)

heldweights(c::QwenTransformer) = c.weights
heldweights(c::Union{QwenTextEncoder,QwenVAEDecoder}) = c.model.weights

function Mantle.release!(component::Union{QwenTextEncoder,QwenTransformer,QwenVAEDecoder})
    dev = Mantle.todevice(component.backend)
    freeplans!(component)
    # The weight dict is the only reference the component holds to the device
    # arrays; the `Model` that built them is long gone. `releaseweights!` and not
    # `empty!`: dropping the last reference to a device array returns nothing,
    # because Mantle frees on a verb and never from a finalizer. This emptied the
    # dict and left the memory for months.
    DNNKernels.releaseweights!(heldweights(component))
    # …but dropping the last Julia reference frees nothing while the DEVICE still
    # names them: a command buffer retains every resource it references until it
    # completes, so the upload that wrote these weights holds them until it does.
    # Measured without this: 15.1 GiB still allocated after releasing the denoiser.
    Mantle.waitidle(dev)
    GC.gc(true)
    # `reclaim!` then `trim!`, both core's pool verbs, rather than Vulkan's
    # `trim_gpu_pool!`: that name lives in Mantle's Vulkan tree and does not exist
    # on another backend, and this is the one thing in the pipeline that has to
    # work on every one of them — the denoiser is 7.26 GB and the conditioner 6.9,
    # and they fit only because each is gone before the next arrives.
    #
    # `reclaim!(…; wait = true)` first: a block is handed back only when nothing is
    # out on loan, and a region retired this frame is still pinned until the device
    # is past it. Trimming without the wait finds every block still live and frees
    # nothing.
    Mantle.reclaim!(Mantle.pool(dev), dev; wait = true)
    Mantle.trim!(Mantle.pool(dev), dev)
    GC.gc(true)
    nothing
end

end # module
