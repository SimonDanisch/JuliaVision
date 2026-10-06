using Test, DNNKernels, Mantle, Random

@testset "INT8 decode reduction epilogues" begin
    dev = Mantle.Device()
    rng = MersenneTwister(86)
    relu(x) = max(x, zero(x))
    silu(x) = x / (one(x) + exp(-x))
    for T in (Float16, Float32), splits in (1, 3), hasbias in (false, true), epi in (identity, relu, silu)
        m, mp = 37, 40
        ph = randn(rng, Float32, mp, 1, splits)
        sc = rand(rng, Float32, m)
        bh = randn(rng, Float32, m)
        p, s, b = Mantle.Buffer(dev, ph), Mantle.Buffer(dev, sc), Mantle.Buffer(dev, bh)
        outs = [Mantle.Buffer(dev, T, (m, 1)) for _ in 1:2]
        plans = Any[]
        for (i, fused) in enumerate((false, true))
            g = Mantle.Graph(dev)
            args = (outs[i], p, s, b, Int32(m), Int32(mp), Int32(splits), Val(hasbias))
            Mantle.dispatch!(g, DNNKernels.q8reduce_kernel!, fused ? (args..., epi) : args,
                             m; group=256)
            fused || Mantle.dispatch!(g, DNNKernels.denseew!,
                (outs[i], (outs[i],), epi, Int32(m)), m)
            push!(plans, Mantle.record!(Mantle.Plan(g)))
        end
        try
            for _ in 1:2
                foreach(Mantle.run!, plans)
                old, got = Array.(outs)
                @test got == old
                sums = copy(ph[1:m, 1, 1])
                for j in 2:splits; sums .+= ph[1:m, 1, j]; end
                want = T.(epi.(T.(sums .* sc .+ (hasbias ? bh : zeros(Float32, m)))))
                @test isapprox(vec(got), want; rtol=T === Float16 ? 0.002 : 2e-6, atol=2e-6)
                ph .*= -0.75f0
                copyto!(p, ph)
            end
        finally
            foreach(Mantle.free!, plans)
            foreach(Mantle.free!, (outs..., p, s, b))
        end
    end
end

@testset "INT8 decode GEMM epilogue" begin
    dev = Mantle.Device()
    rng = MersenneTwister(87)
    m, k = 37, 131
    a = DNNKernels.QInt8Matrix(Mantle.Buffer(dev, rand(rng, UInt32, cld(m,4), k)),
                               Mantle.Buffer(dev, fill(0.002f0, m)), m)
    b = Mantle.Buffer(dev, randn(rng, Float16, k, 1))
    outs = [Mantle.Buffer(dev, Float16, (m,1)) for _ in 1:2]
    epi(x) = max(x, zero(x))
    plans = Any[]
    for (i, fused) in enumerate((false, true))
        g = Mantle.Graph(dev)
        aten = DNNKernels.Graph("q8epi", String[], String[], String[],
            Dict{String,DNNKernels.Buffer}(), String[], DNNKernels.Op[])
        ctx = DNNKernels.EmitCtx(aten, g, dev, (;), Dict{String,Any}(), Set{String}(),
            Ref("out"), Any[], Dict{String,Any}())
        op = DNNKernels.Op("mm", "mm.default", String[], "out", Dict{String,Any}())
        DNNKernels.gemm!(ctx, op, outs[i], a, b; epi=fused ? epi : identity)
        fused || DNNKernels.ewdispatch!(ctx, outs[i], (m,1), (outs[i],),
            (DNNKernels.bcstrides((m,1),(m,1)),), epi; name="relu")
        push!(plans, Mantle.record!(Mantle.Plan(g)))
    end
    try
        @test length(plans[1].passes) == length(plans[2].passes) + 1
        foreach(Mantle.run!, plans)
        @test Array(outs[1]) == Array(outs[2])
    finally
        foreach(Mantle.free!, plans)
        foreach(Mantle.free!, (outs..., a.q, a.scale, b))
    end
end
