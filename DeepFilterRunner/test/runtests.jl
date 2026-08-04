"""
DeepFilterRunner is **not ported** — nothing has been traced into a `DNNKernels` graph.
What this file pins is the half that IS solved: the upstream checkpoint is bound
as an artifact, so the port can start on any machine without fetching by hand.

The assertions are split on purpose. `assetdir()` must resolve and carry the
checkpoint; `ready()` must stay **false** and the graph accessor must still
throw. A test that only checked "the artifact resolves" would go green the moment
the fetch was solved and stay green forever, which is not the claim.
"""

using Test, DeepFilterRunner

@testset "DeepFilterRunner" begin
    @testset "the upstream checkpoint is bound" begin
        dir = DeepFilterRunner.assetdir()
        @test isdir(dir)
        @test isfile(joinpath(dir, "DeepFilterNet3.zip"))
        @test !isempty(DeepFilterRunner.checkpoints())
    end

    @testset "…and that is not the same as being ported" begin
        @test DeepFilterRunner.ready() == false
        @test_throws ArgumentError DeepFilterRunner.deepfilternetgraph()
    end
end
