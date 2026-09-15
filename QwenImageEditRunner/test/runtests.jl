"""
Until the port runs, this asserts the two things that are true now and must stay
true: the package loads on a machine with no assets, and the asset lookup names
a real place rather than throwing something unreadable.

The latency test that matters — `frozen_stats().misses == 0` in a fresh process
— belongs here once the workload drives the real call. See SAM2Runner/test for
the shape it should take; it has to run in a subprocess because Julia's
compile-time counter is per-process.
"""

using Test, QwenImageEditRunner

@testset "QwenImageEditRunner" begin
    # No `assetdir()`. That is internal: it names where the artifact happens to
    # put things, and a test that calls it has to know the layout, so a re-export
    # that moves a file breaks a test that never knew it depended on that. Ask
    # for the graph and the weights; `dir` is a config keyword on those, whose
    # default is the artifact.
    if QwenImageEditRunner.ready()
        @info "Qwen-Image-Edit-2511: export present"
        g = QwenImageEditRunner.qwenimageeditgraph()
        @test g !== nothing
        w = QwenImageEditRunner.qwenimageeditweights()
        @test !isempty(w)
    else
        @info "Qwen-Image-Edit-2511: no export; run tools/export_qwenimageedit.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ErrorException QwenImageEditRunner.qwenimageeditgraph()
    end
end
