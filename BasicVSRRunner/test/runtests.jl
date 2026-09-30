"""
BasicVSR++ REDS4, 4x video upscaling: coverage, dispatch, and parity with PyTorch.

Parity was first measured on 2026-09-30, against upstream's model files run
with the artifact's own weights: **1.1e-5** max abs error over all five 256x256
frames of a real clip (the fox in `docs/examples/basicvsr.jl`). The testset
below pins it on a synthetic clip, sampled at 512 points written by
`tools/verify_basicvsrpp.py`.

The coverage assertions still matter on their own. An op that writes part of its
output leaves the rest as whatever the scratch slab held; the model poisons the
slab first, so a partial write reads as NaN rather than as a plausible number.
"""

using Test, BasicVSRRunner, KernelAbstractions, Lava
using Mantle: LavaBackend
const KA = KernelAbstractions

tri(x) = abs(mod(x, 64) - 32) * 4

"The clip `tools/verify_basicvsrpp.py` ran: integer arithmetic, so both sides build it exactly."
parityclip() = Float32[min(tri(2i + 3(t - 1) + 11(c - 1)) + tri(3j + 7(c - 1)), 255) / 255f0
                       for i in 1:64, j in 1:64, c in 1:3, t in 1:5, _ in 1:1]

function parityreference(path = joinpath(@__DIR__, "fixtures", "parity.txt"))
    rows = [split(l) for l in eachline(path) if !startswith(l, "#")]
    return [(parse.(Int, r[1:4])..., parse(Float32, r[5])) for r in rows]
end

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

        @testset "parity with PyTorch" begin
            ref = parityreference()
            @test length(ref) == 512
            hr = Array(BasicVSRRunner.upscale(m, parityclip()))
            KA.synchronize(backend)
            err = maximum(abs(hr[x, y, c, t, 1] - v) for (x, y, c, t, v) in ref)
            @test err < 1f-4
        end
    end
end
