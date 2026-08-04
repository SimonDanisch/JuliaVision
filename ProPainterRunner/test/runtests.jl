"""
ProPainter is **not ported** — nothing here has been traced into a `DNNKernels`
graph. What this file pins is the half that IS solved: the upstream checkpoints
are bound as an artifact, so the port can start on any machine without
re-fetching them by hand.

So the assertions are deliberately split. `assetdir()` must resolve and must
contain the checkpoints; `ready()` must stay **false** and the graph accessor must
still throw. A test that only checked "the artifact resolves" would go green the
moment the fetch was solved and stay green forever after, which is exactly the
claim not being made.
"""

using Test, ProPainterRunner

@testset "ProPainterRunner" begin
    @testset "the upstream checkpoints are bound" begin
        dir = ProPainterRunner.assetdir()
        @test isdir(dir)
        for f in ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")
            @test isfile(joinpath(dir, f))
        end
        @test !isempty(ProPainterRunner.checkpoints())
    end

    @testset "…and that is not the same as being ported" begin
        # The distinction this package exists to keep honest. Weights on disk, no
        # graph — `ready()` answers the second question, not the first.
        @test ProPainterRunner.ready() == false
        # The error has to name the path, so whoever picks up the port is told
        # where the graph belongs rather than handed a MethodError later.
        @test_throws ArgumentError ProPainterRunner.propaintergraph()
    end
end
