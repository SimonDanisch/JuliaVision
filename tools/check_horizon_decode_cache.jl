# Diagnostic only: reuse the loaded full model's weights to compare cache sizes.
# Run after check_horizon_generation_replay.jl. Supply the original short-cache
# export explicitly; verify the shape substitution against that export first.
using HorizonRunner, DNNKernels, Mantle, Statistics, Test

function check_decode_cache(reference, short_export)
    DK = DNNKernels
    remap(x::AbstractDict) = Dict(k => remap(v) for (k, v) in x)
    remap(x::AbstractVector) = map(remap, x)
    remap(x::Tuple) = map(remap, x)
    remap(x) = x isa Integer && x == 4096 ? 64 : x
    full_export = joinpath(@__DIR__, "..", "gen", "graphs", "horizon32b",
                          "horizon32b_decode.json")
    readgraph(p) = DK.JSON3.read(read(p, String), Dict{String,Any})
    @assert remap(readgraph(full_export)) == readgraph(short_export)
    g = reference.model.graphs["horizon32b_decode"]
    buffers = Dict(k => DK.Buffer(b.id, b.kind, remap(b.shape), b.dtype, b.key,
        b.live, b.of, b.viewop, remap(b.attrs)) for (k, b) in g.buffers)
    ops = [DK.Op(o.id, o.aten, o.ins, o.out, remap(o.attrs)) for o in g.ops]
    shortgraph = DK.Graph(g.name, g.symbols, g.inputs, g.outputs, buffers,
                          g.order, ops, g.fusion)
    shortmodel = DK.Model(Dict(g.name => shortgraph), reference.model.weights,
                          reference.backend, 1, 1, 1; record=true)
    short = Horizon32B(reference.backend, shortmodel, 64, 1, reference.vocab)
    k, v = newcache(short)
    args = (DK.toback(short.backend, ones(Int64, 1, 1)), k, v,
            DK.toback(short.backend, Int64[31]), causalmask(short, Int64[31]))
    expected = Array(first(DK.call(shortmodel, g.name, args...; dims=(;))))
    fill!(k, 0); fill!(v, 0)
    actual = Array(first(DK.call(shortmodel, g.name, args...; dims=(;))))
    @test actual == expected
    recording = only(v for (k, v) in shortmodel.scratch if k isa Tuple &&
                     !isempty(k) && first(k) === :mantleplan)
    timings = Float64[]
    for _ in 1:15
        t = @elapsed begin
            Mantle.run!(recording.plan)
            Mantle.waitidle(Mantle.vk_context(short.backend))
        end
        push!(timings, 1000t)
    end
    println("CACHE64 replay median_ms=", median(timings), " samples=", timings)
    short
end
