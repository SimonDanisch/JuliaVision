"""
BasicVSR++ REDS4, 4x video upscaling — the graph is bound and it **runs**, but it
is **not verified against PyTorch**.

That distinction is the whole point of this file. `coverage` reports none of its
2290 ops missing, the model builds, and one 5-frame clip comes back the right
shape and finite. What has never been measured is whether the numbers are
*right*: there is no `tools/verify_basicvsrpp.jl`, and the artifact carries no
references (they are in `gen/basicvsrpp/refs.safetensors`, developer-side). So
every assertion below is about dispatch and coverage, and none of them is about
accuracy — do not read a green run here as parity.

The coverage assertion is the load-bearing one. An op that writes part of its
output leaves the rest as whatever the scratch slab held; the model poisons the
slab first, so a partial write reads as NaN rather than as a plausible number.
"""

using Test, BasicVSRRunner, KernelAbstractions, Lava
const KA = KernelAbstractions

@testset "BasicVSRRunner" begin
    @test BasicVSRRunner.ready()
    g = BasicVSRRunner.basicvsrppgraph()
    @test g !== nothing
    @test length(g.ops) == 2290

    backend = try
        b = LavaBackend(); KA.synchronize(b); b
    catch err
        # `LavaError` only. A bare `catch` here would eat a typo in this file and
        # report the skip as a pass, which is how a suite goes green while
        # testing nothing.
        err isa Lava.LavaError || rethrow()
        @info "no working device; skipping the upscale" exception = err
        nothing
    end

    if backend !== nothing
        m = BasicVSRRunner.basicvsrppmodel(; backend)
        # The export baked (T = 5, 64 x 64); a different shape is a re-export.
        lqs = KA.allocate(backend, Float32, 64, 64, 3, 5, 1)
        fill!(lqs, 0.5f0)
        out = BasicVSRRunner.upscale(m, lqs)
        KA.synchronize(backend)
        got = Array(out)
        @test size(got) == (256, 256, 3, 5, 1)      # 4x, every frame
        @test count(isnan, got) == 0                # nothing left unwritten
        @test all(isfinite, got)
        @test any(!iszero, got)                     # and it is not a dead graph
    end
end
