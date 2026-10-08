"""
    Normalization(dev, j)

A latent's per-channel statistics from `pipeline.json`, as `Buffer`s a stage's
graph reads. The flow models sample the normalised latent; the decoders read the
raw one.
"""
struct Normalization
    mean::Mantle.Buffer
    std::Mantle.Buffer
end

Normalization(dev::Mantle.Device, j) =
    Normalization(Mantle.Buffer(dev, Float32.(j.mean)), Mantle.Buffer(dev, Float32.(j.std)))

"""`y = x * std + mean`, per channel over `(C, N, 1)`."""
function denormalize!(y, x, mean, std)
    i = KI.get_global_id().x
    i <= length(x) || return nothing
    c = (i - 1) % length(mean) + 1
    @inbounds y[i] = x[i] * std[c] + mean[c]
    return nothing
end

"""`y = (x - mean) / std`, per channel over `(C, N, 1)`."""
function normalize!(y, x, mean, std)
    i = KI.get_global_id().x
    i <= length(x) || return nothing
    c = (i - 1) % length(mean) + 1
    @inbounds y[i] = (x[i] - mean[c]) / std[c]
    return nothing
end

FlowSampler(j::JSON3.Object) =
    FlowSampler(; steps = j.steps, strength = j.guidance_strength,
                rescale = j.guidance_rescale, interval = Tuple(j.guidance_interval),
                rescale_t = j.rescale_t, sigma_min = j.sigma_min)

"""
    FlowStage

One flow graph with what `sampling.json` beside it says about running it: the
sampler, the normalisation its output is undone with (`nothing` for the sparse
structure, whose latent is decoded as sampled) and, for the texture models, the
one the shape latent they are conditioned on is normalised with.
"""
struct FlowStage{N,C}
    model::Model
    name::String
    sampler::FlowSampler
    norm::N
    concat::C
end

"""
    loadmodel(part; backend) -> Model

`part`'s graph `trellis2_<part>.json` and its weights (see `assets.jl`) as one
`Model`.

DINOv3 at 1024 px in fp32 as ONE submission runs long enough that RADV cancels
it, and the call returned whatever the output held (NaN) while only the next
submission reported the lost device. Mantle cuts every recording by the
device's submission budget now, so nothing is asked for here.
"""
function loadmodel(part::AbstractString; backend)
    name = "trellis2_$part"
    return Model(Dict(name => loadgraph(graphfile(graphdir(part), name))), readweights(part); backend)
end

"""The exported graph `name.json` in `dir`."""
function graphfile(dir::AbstractString, name::AbstractString)
    p = joinpath(dir, "$name.json")
    isfile(p) || throw(ArgumentError(
        "TRELLIS.2: no $name.json in $dir. Export it with tools/export_trellis2.py and bind it with tools/make_artifacts.jl."))
    return p
end

"""
    conditioner(; cascade, backend) -> Model

DINOv3 as one `Model` with a graph per resolution the pipeline encodes at:
`trellis2_cond` (512 px) and, for the cascade, `trellis2_cond_1024` (part
`"cond-1024"`). One set of weights for both, 1.13 GB: the two exports'
`weights.safetensors` are the same file, which is why only part `"cond"` has
weights. The 1024 px graph is renamed, since both exports call themselves
`trellis2_cond` and a `Model`'s graphs need distinct names.
"""
function conditioner(; cascade::Bool, backend)
    gs = Dict("trellis2_cond" => loadgraph(graphfile(graphdir("cond"), "trellis2_cond")))
    cascade && (gs["trellis2_cond_1024"] =
                    DNNKernels.Graph(loadgraph(graphfile(graphdir("cond-1024"), "trellis2_cond")),
                                     "trellis2_cond_1024"))
    return Model(gs, readweights("cond"); backend)
end

"""
    FlowStage(part; backend)

The flow model `part` (one of [`FLOWS`](@ref)) with the sampler and the latent
normalisation its `sampling.json` describes.
"""
function FlowStage(part::AbstractString; backend)
    cfg = JSON3.read(read(joinpath(graphdir(part), "sampling.json"), String))
    model = loadmodel(part; backend)
    norm = haskey(cfg, :normalization) ? Normalization(model.device, cfg.normalization) : nothing
    concat = haskey(cfg, :concat_normalization) ?
        Normalization(model.device, cfg.concat_normalization) : nothing
    return FlowStage(model, "trellis2_$part", FlowSampler(cfg.sampler), norm, concat)
end

"""
    encodeimage(m, image; graph = "trellis2_cond") -> cond

DINOv3 over one normalised image, `(W, H, 3, 1)` (torch's `(1, 3, H, W)`
reversed), giving `(1024, tokens, 1)`: 1029 tokens at 512 px through
`trellis2_cond`, 4101 at 1024 through `trellis2_cond_1024` (see
[`conditioner`](@ref)).
"""
encodeimage(m::Model, image; graph::AbstractString = "trellis2_cond") =
    copy(Mantle.storage(only(call(m, graph, image; dims = (;)))))

"""
    outputbuffer(model, name, id; dims) -> Mantle.Buffer

A `Buffer` of the shape and element type `model`'s graph `name` declares for `id`,
to bind a composed module's output to: a transient is arena memory, valid only
inside the run, and a stage reads its result after it.
"""
function outputbuffer(model::Model, name::AbstractString, id::AbstractString;
                      dims::NamedTuple = (;))
    b = model.graphs[name].buffers[id]
    return fill!(Mantle.Buffer(model.device, b.dtype, DNNKernels.evalshape(b.shape, dims)), 0)
end

"""
    sparsestructure(ss, ssdec, cond, noise; resolution) -> coords

`sample_sparse_structure`: the dense flow over the `(16, 16, 16, 8, 1)` latent
from `noise`, guided against a zero condition, then the decoder, as ONE graph;
then [`occupiedcoords`](@ref) at `resolution` (32 for the 512 and cascade
pipelines) on the host, because how many voxels there are is what the next
stage is sized by.
"""
function sparsestructure(ss::FlowStage, ssdec::Model, cond::AbstractArray,
                         noise::AbstractArray; resolution::Integer = 32)
    dev = ss.model.device
    g = Mantle.Graph(dev)
    x = Mantle.Buffer(dev, noise)
    sampleflow!(g, ss.model, ss.name, ss.sampler, x, Mantle.Buffer(dev, cond),
                Mantle.Buffer(dev, zero(cond)))
    dec = ssdec.graphs["trellis2_ssdec"]
    logits = outputbuffer(ssdec, "trellis2_ssdec", only(dec.outputs))
    emitgraph!(g, ssdec, "trellis2_ssdec"; inputs = Dict(only(dec.inputs) => x),
               outputs = Dict(only(dec.outputs) => logits))
    runonce!(g)
    return occupiedcoords(Array(Mantle.storage(logits)), resolution)
end

"""
    shapeslat(stage, cond, coords, noise) -> slat::Mantle.Buffer

`sample_shape_slat`: the shape flow over the `N` voxels of `coords` from `noise`
(`(32, N, 1)`), then denormalised, as one graph.
"""
function shapeslat(stage::FlowStage, cond::AbstractArray, coords::AbstractMatrix{Int32},
                   noise::AbstractArray)
    dev = stage.model.device
    g = Mantle.Graph(dev)
    x = Mantle.Buffer(dev, noise)
    rope = Mantle.Buffer(dev, ropetable(coords))
    sampleflow!(g, stage.model, stage.name, stage.sampler, x, Mantle.Buffer(dev, cond),
                Mantle.Buffer(dev, zero(cond)), (rope,); dims = (; n = size(coords, 2)))
    slat = fill!(Mantle.Buffer(dev, Float32, size(noise)), 0)
    Mantle.dispatch!(g, denormalize!, (slat, x, stage.norm.mean, stage.norm.std),
                     length(slat); name = "shape/denormalize")
    runonce!(g)
    return slat
end

"""
    texslat(stage, cond, coords, shape, noise) -> slat::Mantle.Buffer

`sample_tex_slat`: the texture flow conditioned on the normalised shape latent
as `concat_cond`, unguided (strength 1), then denormalised with the texture
statistics, as one graph. `shape` is [`shapeslat`](@ref)'s result.
"""
function texslat(stage::FlowStage, cond::AbstractArray, coords::AbstractMatrix{Int32},
                 shape::Mantle.Buffer, noise::AbstractArray)
    dev = stage.model.device
    g = Mantle.Graph(dev)
    x = Mantle.Buffer(dev, noise)
    rope = Mantle.Buffer(dev, ropetable(coords))
    concat = Mantle.Transient.Buffer(g, Float32, size(shape)...)
    Mantle.dispatch!(g, normalize!, (concat, shape, stage.concat.mean, stage.concat.std),
                     length(concat); name = "tex/normalize shape")
    sampleflow!(g, stage.model, stage.name, stage.sampler, x, Mantle.Buffer(dev, cond),
                Mantle.Buffer(dev, zero(cond)), (rope, concat); dims = (; n = size(coords, 2)))
    slat = fill!(Mantle.Buffer(dev, Float32, size(noise)), 0)
    Mantle.dispatch!(g, denormalize!, (slat, x, stage.norm.mean, stage.norm.std),
                     length(slat); name = "tex/denormalize")
    runonce!(g)
    return slat
end
