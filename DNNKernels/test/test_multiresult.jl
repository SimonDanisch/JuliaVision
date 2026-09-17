# Where a multi-output op's result shapes come from.
#
# A composite op declares `shapes`/`dtypes` on its output buffer, one entry per
# element of torch's schema, and each element is read through a `getitem` view.
# The two describe the same bytes, so they are one fact written twice, and in
# Kokoro's exports they disagree: 119 buffers between kokorotext and kokorovoc
# against 1317 that agree across every other model installed here. The VIEW is
# symbolic and the metadata holds that symbol evaluated at the trace:
# `[1, "t", 512]` against `[1, 30, 512]`, `[1, 128, "120*f + 1"]` against
# `[1, 128, 11761]`. It is not one op's quirk: in those graphs it is every
# `native_layer_norm`, every `_native_batch_norm_legit`, every sdpa and every
# `lstm.input`.
#
# What made it worth a rule rather than a per-op check is that the metadata is
# CONCRETE. Declared from it, a graph plans the traced length for whatever
# length it was called with, and nothing downstream has a symbol left to
# disagree with. `_scaled_dot_product_efficient_attention` happens to compare
# its result against its operands and refuse, but an op that trusts its
# destination would simply write 30 columns of a 17-column answer.
#
# So the view wins, and the two ways it could be wrong are refused rather than
# ordered: two views of one element that disagree, and a view whose RANK differs
# from the metadata's, which is a disagreement about what the tensor is.

using Test
import DNNKernels
include("fixtures.jl")
const DK = DNNKernels

view0(id, of, i, shape) =
    DK.Buffer(id, :view, shape, Float32, "", (0, 2), of,
              "<built-in function getitem>", Dict{String,Any}("arg1" => i))

function multigraph(bufs...)
    d = Dict{String,DK.Buffer}(b.id => b for b in bufs)
    return DK.Graph("g", ["t"], String[], String[], d,
                    collect(keys(d)), DK.Op[])
end

@testset "a multi-output element takes its reader's shape" begin
    parent = DK.Buffer("ls", :transient, Any[], Float32, "", (0, 2), "", "",
                       Dict{String,Any}(
                           "shapes" => Any[Any[1, 30, 512], Any[2, 1, 256]],
                           "dtypes" => Any[Float32, Float32]))
    g = multigraph(parent, view0("y", "ls", 0, Any[1, "t", 512]))
    shapes = DK.resultshapes(g)
    @test shapes == Dict{String,Any}("ls#0" => Any[1, "t", 512])
    # The element with a reader takes the reader's shape…
    @test DK.resultshape(shapes, "ls#0", Any[1, 30, 512]) == Any[1, "t", 512]
    # …and one with none keeps the metadata's, which is all there is.
    @test DK.resultshape(shapes, "ls#1", Any[2, 1, 256]) == Any[2, 1, 256]
    # The shape that reaches `make` is then the symbolic one, resolved per call.
    @test DK.evalshape(DK.resultshape(shapes, "ls#0", Any[1, 30, 512]),
                       (; t = 17)) == (512, 17, 1)
    @test DK.evalshape(DK.resultshape(shapes, "ls#0", Any[1, 30, 512]),
                       (; t = 30)) == (512, 30, 1)
end

@testset "two readers of one element that disagree are refused" begin
    parent = DK.Buffer("ls", :transient, Any[], Float32, "", (0, 2), "", "",
                       Dict{String,Any}("shapes" => Any[Any[1, 30, 512]],
                                        "dtypes" => Any[Float32]))
    # Same element, two shapes. They name the same bytes, so one is wrong and
    # nothing here can tell which.
    g = multigraph(parent,
                   view0("y1", "ls", 0, Any[1, "t", 512]),
                   view0("y2", "ls", 0, Any[1, "t", 256]))
    @test_throws "ls#0" DK.resultshapes(g)
    # Two readers that AGREE are ordinary. A result with several consumers is
    # the common case, and it must not be refused.
    g2 = multigraph(parent,
                    view0("y1", "ls", 0, Any[1, "t", 512]),
                    view0("y2", "ls", 0, Any[1, "t", 512]))
    @test DK.resultshapes(g2)["ls#0"] == Any[1, "t", 512]
end

@testset "a reader of a different rank is refused" begin
    shapes = Dict{String,Any}("ls#0" => Any[1, "t", 512])
    # Both substrings, so the refusal has to name the element AND say what is
    # wrong with it.
    @test_throws ["ls#0", "rank"] DK.resultshape(shapes, "ls#0", Any[30, 512])
end

# The same statement over the real exports this package binds: all ten of them
# agree with their metadata, so the rule is INERT for them. Asserted because a
# rule that quietly changed what the working graphs declare would be the wrong
# fix. The two Kokoro graphs it does change belong to `KokoroRunner`'s
# artifact, and `tools/declared_coverage.jl` is what sweeps those.
@testset "the rule is inert for the exports that agree" begin
    graphs = [Fixtures.matanyone(n) for n in Fixtures.matanyonenames()]
    for n in ("sam2_encoder", "sam2_decoder")
        push!(graphs, Fixtures.sam2(n))
    end
    nagree, ndisagree = 0, 0
    for g in graphs
        for (key, s) in DK.resultshapes(g)
            of, i = split(key, '#')
            meta = get(g.buffers[of].attrs, "shapes", nothing)
            meta === nothing && continue
            k = parse(Int, i) + 1
            (k <= length(meta) && meta[k] !== nothing) || continue
            collect(meta[k]) == collect(s) ? (nagree += 1) : (ndisagree += 1)
        end
    end
    @test nagree > 0            # the comparison has something to compare
    @test ndisagree == 0
end
