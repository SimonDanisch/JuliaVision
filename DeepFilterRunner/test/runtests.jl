"""
DeepFilterNet3 is **not ported yet**, so this asserts the contract that state has: the
package loads and precompiles on a machine with no assets, `ready()` answers
`false` rather than throwing, and `assetdir()` refuses with a message naming the
three steps that would fix it — export, bind, switch to `@artifact_str`.

Assets come from the artifact and from nowhere else, so there is deliberately no
path here to assert; see `DNNKernels/src/assets.jl`. When this model is ported,
replace this with the shape in `NeuralLUTRunner/test`: a subprocess asserting
`Lava.no_pipeline_compilation` reports 0 refusals, **with** a negative control
whose kernel body is novel per run.
"""

using Test, DeepFilterRunner

@testset "DeepFilterRunner" begin
    # `ready` must answer, not throw: the workload branches on it and has to
    # precompile to nothing rather than fail on a machine with no assets.
    @test DeepFilterRunner.ready() === false

    # And the refusal has to be legible — a caller who has not ported this model
    # should be told what to do, not handed a path that will never exist.
    err = try; DeepFilterRunner.assetdir(); nothing; catch e; e; end
    @test err isa ErrorException
    @test occursin("not ported yet", err.msg)
    @test occursin("make_artifacts.jl", err.msg)

    # The graph loader defaults `dir` to `assetdir()`, so it refuses for the same
    # reason rather than failing later with something unreadable.
    @test_throws ErrorException DeepFilterRunner.deepfilternetgraph()
end
