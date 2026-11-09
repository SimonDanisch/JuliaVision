# The networks, loaded from an export (`tools/export_indextts25.py`): one
# DNNKernels model per graph, and the GPT's prefill and decode in one model so
# they share its weights.

using DNNKernels: loadgraph, readsafetensors, Model, call, emitgraph!

"""The exported graphs every voice and every utterance runs, in `Networks`' order."""
const PARTS = ("w2v", "campplus", "emo", "regulator", "codec", "dit", "bigvgan")

"""
    Networks

The seven non-GPT networks as DNNKernels models.
"""
struct Networks{M}
    w2v::M
    campplus::M
    emo::M
    regulator::M
    codec::M
    dit::M
    bigvgan::M
end

"""
Which export of each network a precision loads. `:exact` is fp32 throughout, the
configuration that reproduces upstream's audio. `:fast` runs the DiT and the
codec under fp16 autocast (matmuls and convolutions in fp16, norms in fp32; ~1e-4
of the output's range from fp32); w2v-BERT, CAMPPlus and the emotion encoder run
once per clip and stay fp32, and so does BigVGAN, as upstream keeps it: its
fp16 export is 0.17 of range off at its worst sample.
"""
const PRECISIONS = Dict(
    :exact => Dict(p => p for p in PARTS),
    :fast => Dict(p => (p in ("dit", "codec") ? "$(p)_fp16" : p) for p in PARTS))

"""One exported graph `dir/<export>/` and its weights, as graph `part` of a model."""
function readpartweights(dir::AbstractString)
    path = joinpath(dir, "weights.safetensors")
    isfile(path) && return readsafetensors(path)
    manifest = JSON3.read(read(joinpath(dir, "weights.json"), String))
    weights = Dict{String,Any}()
    for shard in manifest
        w = readsafetensors(joinpath(assetdir(String(shard.artifact)), String(shard.file)))
        isempty(intersect(keys(weights), keys(w))) || error("weight shards overlap in $dir")
        merge!(weights, w)
    end
    return weights
end

function loadpart(dir::AbstractString, part::AbstractString, export_::AbstractString = part; device)
    g = loadgraph(joinpath(dir, export_, "$export_.json"))
    w = readpartweights(joinpath(dir, export_))
    return Model(Dict(part => g), w; device)
end

function Networks(dir::AbstractString; device, precision::Symbol = :fast)
    exports = PRECISIONS[precision]
    return Networks((loadpart(dir, p, exports[p]; device) for p in PARTS)...)
end

"""
    GPT

The GPT: its prefill as an exported graph (`<name>_prefill`), the decode step's
weights, and one [`Decoder`](@ref) per beam count, made on first use.
"""
struct GPT{M,W}
    prefill::M
    weights::W
    decoders::Dict{Int,Any}
end

"""
    loadgpt(dir, name, T; device) -> GPT

The base GPT is `gpt`, the German checkpoint `gpt_de`; `T` is the decode
step's weight type.
"""
function loadgpt(dir::AbstractString, name::AbstractString, ::Type{T} = Float16; device) where {T}
    gp = loadgraph(joinpath(dir, "$(name)_prefill", "$(name)_prefill.json"))
    w = readpartweights(joinpath(dir, "$(name)_prefill"))
    weights = GPTWeights(device, w, T)
    prefill = Model(Dict("prefill" => DNNKernels.Graph(gp, "$(name)_prefill")), w; device)
    return GPT(prefill, weights, Dict{Int,Any}())
end

decoder(g::GPT, beams::Int) = get!(() -> Decoder(g.prefill.device, g.weights; beams), g.decoders, beams)

function Base.close(g::GPT)
    Mantle.waitidle(g.prefill.device)
    foreach(close, values(g.decoders)); empty!(g.decoders)
    close(g.prefill)
    close(g.weights)
    return nothing
end

"""The single graph of a one-graph model."""
onlygraph(m::Model) = only(values(m.graphs))
graphname(m::Model) = only(keys(m.graphs))

"""
    outputbuffer(m, dims) -> Mantle.Buffer

A buffer shaped like the only output of `m`'s only graph at `dims`, to bind it
to: what a graph declares as output is arena memory inside the run unless bound.
"""
function outputbuffer(m::Model, dims::NamedTuple)
    g = onlygraph(m)
    b = g.buffers[only(g.outputs)]
    return Mantle.Buffer(m.device, b.dtype, DNNKernels.evalshape(b.shape, dims))
end

"""
    emit!(g, m, dims, inputs...; out = nothing) -> output resource

Declare `m`'s only graph into `g` with `inputs` in the graph's input order, the
output bound to `out` when given.
"""
function emit!(g, m::Model, dims::NamedTuple, inputs...; out = nothing)
    aten = onlygraph(m)
    length(inputs) == length(aten.inputs) ||
        throw(ArgumentError("$(graphname(m)) takes $(length(aten.inputs)) inputs, got $(length(inputs))"))
    outputs = out === nothing ? Dict{String,Any}() : Dict{String,Any}(only(aten.outputs) => out)
    res = emitgraph!(g, m, graphname(m); dims, inputs = Dict(zip(aten.inputs, inputs)), outputs)
    return res[only(aten.outputs)]
end
