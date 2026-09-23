"""
Every op of every exported graph on this machine can be DECLARED.

This is the precondition for deleting the eager op library. `runop!` and
`emitop!` are two implementations of the same 80-odd ops, and the eager half
goes when nothing needs it — but "nothing needs it" is a claim about the
graphs that actually exist, not about the arm counts. There are eager arms for
ops no exporter has ever produced (`view_as_real`, `pow.Scalar`,
`fused.maskedattention`, …), and an arm count would keep scoring them.

So the question this asks is the one that matters: take every graph the depot
has, and check `emitgraph` could place all of it.

## What it covers, and what it cannot

It walks installed artifacts, so it is only as complete as this machine's
depot. That is a floor and not a ceiling, and it is stated in the output: the
count of graphs and of distinct ops is printed, so a run that suddenly covers
three graphs instead of twenty-six is visible rather than quietly green. It
asserts a MINIMUM graph count for the same reason — see
[[a-skipping-test-is-a-green-test]]: a scan that finds nothing passes every
assertion about what it found.

It does not run anything. A declarable op can still be a wrong op; that is what
the parity gates are for.
"""

using Test
using DNNKernels
using DNNKernels: coverage, loadgraph
import JSON3

const ARTIFACTS = joinpath(homedir(), ".julia", "artifacts")

"""
Is this JSON an exported ATen graph?

`ops` and `buffers` are not enough on their own: Kokoro's `vocab.json` has both
and its `buffers` are `Int`s, so `loadgraph` dies inside `Buffer`. Look at what
a buffer actually is.
"""
function isexportedgraph(path)
    o = JSON3.read(read(path, String))
    o isa JSON3.Object && haskey(o, :ops) && haskey(o, :buffers) || return false
    vals = o.buffers isa JSON3.Array ? o.buffers : collect(values(o.buffers))
    isempty(vals) && return false
    return vals[1] isa JSON3.Object && haskey(vals[1], :kind)
end

function exportedgraphs()
    out = String[]
    isdir(ARTIFACTS) || return out
    for (root, _, files) in walkdir(ARTIFACTS), f in files
        endswith(f, ".json") || continue
        p = joinpath(root, f)
        # Whole-file JSON parses of every .json in the depot would read weight
        # manifests and vocabularies too; the cheap substring check first.
        filesize(p) > 200_000_000 && continue
        txt = read(p, String)
        (occursin("\"ops\"", txt) && occursin("\"buffers\"", txt)) || continue
        isexportedgraph(p) && push!(out, p)
    end
    return out
end

"""A one-op graph, which is all `coverage` reads: `g.ops` and nothing else."""
probegraph(aten) = DNNKernels.Graph(
    "probe", String[], String[], String[],
    Dict{String,DNNKernels.Buffer}(), String[],
    [DNNKernels.Op("0", aten, String[], "0", Dict{String,Any}())],
    Vector{Vector{String}}())

@testset "every exported graph is declarable" begin
    paths = exportedgraphs()
    names = sort(unique(basename.(paths)))
    ops = Set{String}()
    bad = Dict{String,Vector{String}}()
    for p in paths
        g = loadgraph(p)
        for o in g.ops
            push!(ops, o.aten)
        end
        _, miss = coverage(g)
        isempty(miss) || (bad[basename(p)] = miss)
    end

    @info "declared coverage" graphs = length(paths) distinct = length(names) ops = length(ops)

    # A scan that found nothing passes everything below it. These runners are
    # installed here; if the depot is emptier than that, say so rather than
    # reporting a vacuous pass.
    @test length(names) >= 20
    @test length(ops) >= 60

    # The assertion. `bad` is the pair so a failure names the graph and the op
    # rather than saying `false`.
    @test bad == Dict{String,Vector{String}}()
end

@testset "coverage asks the declared path" begin
    # The thing the rewrite fixed: it used to ask `runop!`. An op with an eager
    # arm and no declared one must read as MISSING, and the reverse must not.
    #
    # `_assert_tensor_metadata.default` is exactly that op today — a `runop!`
    # arm, no `emitop!` arm, and no exporter emits it. If it ever gains a
    # declared arm this assertion is the thing to update, not to delete.
    eageronly = "_assert_tensor_metadata.default"
    hasrun = isconcretetype(Base.unwrap_unionall(
        which(DNNKernels.runop!, Tuple{DNNKernels.Ctx, DNNKernels.Op,
                                       Val{Symbol(eageronly)}}).sig).parameters[4])
    hasemit = isconcretetype(Base.unwrap_unionall(
        which(DNNKernels.emitop!, Tuple{DNNKernels.EmitCtx, DNNKernels.Op,
                                        Val{Symbol(eageronly)}}).sig).parameters[4])
    @test hasrun          # it is in the eager library
    @test !hasemit        # and not in the declared one

    _, miss = coverage(probegraph(eageronly))
    @test miss == [eageronly]
end

@testset "the noise ops count as declared" begin
    # `emitgraph` handles these itself and they have no `emitop!` arm, so a
    # `coverage` that only asked `emitop!` would call every stochastic graph
    # incomplete. Kokoro's vocoder is one.
    for a in DNNKernels.NOISEOPS
        @test !isconcretetype(Base.unwrap_unionall(
            which(DNNKernels.emitop!, Tuple{DNNKernels.EmitCtx, DNNKernels.Op,
                                            Val{Symbol(a)}}).sig).parameters[4])
        impl, miss = coverage(probegraph(a))
        @test miss == String[]
        @test impl == [a]
    end
end
