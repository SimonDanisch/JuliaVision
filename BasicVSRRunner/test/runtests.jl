"""
Until the port runs, this asserts the two things that are true now and must stay
true: the package loads on a machine with no assets, and the asset lookup names
a real place rather than throwing something unreadable.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the workload drives the real call. See SAM2Runner/test for
the shape it should take; it has to run in a subprocess because Julia's
compile-time counter is per-process.
"""

using Test, BasicVSRRunner

@testset "BasicVSRRunner" begin
    dir = BasicVSRRunner.assetdir()
    @test dir isa AbstractString
    @test !isempty(dir)

    if BasicVSRRunner.ready()
        @info "BasicVSR++: export present" dir
        g = BasicVSRRunner.basicvsrppgraph()
        @test g !== nothing
        w = BasicVSRRunner.basicvsrppweights()
        @test !isempty(w)
    else
        @info "BasicVSR++: no export; run tools/export_basicvsrpp.py" dir
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError BasicVSRRunner.basicvsrppgraph()
    end
end
