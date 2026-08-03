"""
Until the port runs, this asserts the two things that are true now and must stay
true: the package loads on a machine with no assets, and the asset lookup names
a real place rather than throwing something unreadable.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the workload drives the real call. See SAM2Runner/test for
the shape it should take; it has to run in a subprocess because Julia's
compile-time counter is per-process.
"""

using Test, ProPainterRunner

@testset "ProPainterRunner" begin
    dir = ProPainterRunner.assetdir()
    @test dir isa AbstractString
    @test !isempty(dir)

    if ProPainterRunner.ready()
        @info "ProPainter: export present" dir
        g = ProPainterRunner.propaintergraph()
        @test g !== nothing
        w = ProPainterRunner.propainterweights()
        @test !isempty(w)
    else
        @info "ProPainter: no export; run tools/export_propainter.py" dir
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError ProPainterRunner.propaintergraph()
    end
end
