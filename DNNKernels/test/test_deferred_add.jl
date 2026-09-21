"""
An `add` deferred into the layer norm that reads it, and when it must not be.

`add.Tensor` whose result feeds a `native_layer_norm` emits NO pass of its own:
the fused norm kernel computes the sum as it reads. That is worth a dispatch on
the 91 transformer residuals in this tree.

It is only legal when the norm is the sum's ONLY reader, and that clause was
missing. The fused kernel does store the sum, but it stores it where the NORM
sits in the order, which can be a long way after another reader. MatAnyone's
`readout_query` has `add_398` read by `add_403` and -- through the view
`permute_37` -- by `_to_copy_55`, both ~50 passes before the norm that would
compute it. They read a buffer nothing had written yet.

Zeros make that look like an answer. It cost nothing visible for as long as
freshly pooled memory was zero, and surfaced only where the placer had given
those bytes to another transient first: MatAnyone's whole matte came back NaN,
and the graph was correct with `keepall` or with aliasing off, which is what made
it look like a placement bug for a long time. It was never one -- the sum was
simply never computed in time, on every backend.

Asked of the SHIPPED graphs, because the case that broke is in the data.
"""

using Test
import DNNKernels
include(joinpath(@__DIR__, "fixtures.jl"))

const DKD = DNNKernels

vbuf(id, of) = DKD.Buffer(id, :view, Any[2, 2], Float32, "", (0, 2), of,
                          "view.default", Dict{String,Any}())
tbuf(id) = DKD.Buffer(id, :transient, Any[2, 2], Float32, "", (0, 2), "", "",
                      Dict{String,Any}())

@testset "bufferuses counts every read, not every direct one" begin
    # One op input.
    g = DKD.Graph("g", String[], String[], String[],
                  Dict("a" => tbuf("a"), "b" => tbuf("b")), ["a", "b"],
                  [DKD.Op("b", "relu.default", ["a"], "b", Dict{String,Any}())])
    @test DKD.bufferuses(g, "a") == 1
    @test DKD.bufferuses(g, "b") == 0

    # A VIEW over it is a read even though no op names it. This is the term the
    # deferral was missing.
    g2 = DKD.Graph("g", String[], String[], String[],
                   Dict("a" => tbuf("a"), "v" => vbuf("v", "a")), ["a", "v"],
                   DKD.Op[])
    @test DKD.bufferuses(g2, "a") == 1

    # …and being a graph output is a read, by the caller.
    g3 = DKD.Graph("g", String[], String[], ["a"],
                   Dict("a" => tbuf("a")), ["a"], DKD.Op[])
    @test DKD.bufferuses(g3, "a") == 1
end

if Fixtures.have("readout_query")
    @testset "the deferral refuses a sum with other readers" begin
        g = Fixtures.matanyone("readout_query")
        adds = Dict(op.out => op for op in g.ops if op.aten == "add.Tensor")

        # `add_398`: the norm is `native_layer_norm_3`, and `add_403` and the view
        # `permute_37` read it too. Three reads, so no deferral.
        @test haskey(adds, "add_398")
        @test DKD.bufferuses(g, "add_398") == 3
        @test DKD.deferrednorm(g, adds["add_398"]) === nothing

        # `add_395`: the norm is its only reader, so the fusion still applies --
        # the fix must not have turned the optimisation off.
        @test haskey(adds, "add_395")
        @test DKD.bufferuses(g, "add_395") == 1
        n = DKD.deferrednorm(g, adds["add_395"])
        @test n !== nothing && n.aten == "native_layer_norm.default"

        # And that both cases are live across the shipped graphs: a deferral that
        # never fires is not an optimisation, and one that always fires is the bug.
        yes = nos = 0
        for name in Fixtures.matanyonenames()
            gg = Fixtures.matanyone(name)
            for op in gg.ops
                op.aten == "add.Tensor" || continue
                norms = count(x -> x.aten == "native_layer_norm.default" &&
                                   !isempty(x.ins) && x.ins[1] == op.out, gg.ops)
                norms == 1 || continue
                DKD.deferrednorm(gg, op) === nothing ? (nos += 1) : (yes += 1)
            end
        end
        @test yes > 0
        @test nos > 0
    end
else
    @info "no matanyone graphs bound — the deferral's real-data case is skipped"
end
