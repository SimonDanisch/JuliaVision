using Test, DNNKernels, Mantle, KernelAbstractions, Random

# A branch that picks a tile which does not divide its shape makes `q8gemm!`
# throw from inside a graph, and the tile choice grew a branch per column count
# — so sweep the shapes rather than trusting the reading. Host-only.
@testset "int8 tile divides every shape it is asked about" begin
    for m in (64,128,192,256,320,1024,5120,10240,53248),
        k in (32,64,96,160,2048,5120,8192,16384,26624),
        n in (16,32,48,64,96,128,192,256,512,1024)
        cfg = DNNKernels.q8gemm_tile(m,k,n)
        stm,stn,wm,wn,bk,pad = cfg
        @test haskey(DNNKernels.Q8_GEMM_KERNELS, cfg)
        @test m % (16stm*wm) == 0
        @test n % (16stn*wn) == 0
        @test k % bk == 0
    end
    # The measured picks for Horizon's four layer shapes; see `q8gemm_tile`.
    @test DNNKernels.q8gemm_tile(53248,5120,128) == (4,2,2,4,32,8)
    @test DNNKernels.q8gemm_tile(53248,5120,512) == (2,4,2,2,32,8)
    @test DNNKernels.q8gemm_tile(10240,5120,64) == (4,2,4,2,32,8)
    @test DNNKernels.q8gemm_tile(5120,8192,64) == (2,1,2,2,64,8)
    @test DNNKernels.q8gemm_tile(5120,26624,32) == (2,1,2,1,32,8)
    @test DNNKernels.q8gemm_tile(5120,26624,128) == (4,2,4,2,32,8)
    @test DNNKernels.q8gemm_tile(5120,26624,512) == (4,2,2,2,32,8)
    # Qwen-Image 2.1's stacked gate+proj at 1024², which is `m = 6k` exactly and
    # used to miss the stacked-projection branch by falling through to
    # `m % 256 == 0`. 42.7 ms that way against 37.9 here.
    @test DNNKernels.q8gemm_tile(24576,4096,4224) == (2,4,2,2,32,8)
    # Its three neighbours in the same layer are NOT this branch and keep what
    # they had: the bound moved, it did not become a different rule.
    @test DNNKernels.q8gemm_tile(12288,4096,4224) == (4,2,4,2,32,8)
    @test DNNKernels.q8gemm_tile(4096,12288,4224) == (4,2,2,2,32,8)
    @test DNNKernels.q8gemm_tile(4096,4096,4224) == (4,2,4,2,32,8)
end

@testset "direct int8 cooperative GEMM" begin
    backend = Mantle.defaultbackend()
    dev = DNNKernels.caps(backend)
    if dev.coopmat && dev.coopmatsubgroup == 32 && dev.tile == 16
        rng = MersenneTwister(73)
        a = DNNKernels.quantizeint8(backend,
            DNNKernels.toback(backend, randn(rng, Float16, 256, 256)))
        q, scale = Array(a.q), Array(a.scale)
        weights = Float32[Float16(DNNKernels.q8byte(q[cld(i,4),j], (i-1)%4)*scale[i])
                          for i in 1:256, j in 1:256]
        bh = randn(rng, Float16, 256, 128)
        b = DNNKernels.toback(backend, bh)
        biash = randn(rng, Float32, 256)
        bias = DNNKernels.toback(backend, biash)
        c = KernelAbstractions.allocate(backend, Float16, 256, 128)
        reference = Float16.(weights*Float32.(bh) .+ biash)
        for cfg in keys(DNNKernels.Q8_GEMM_KERNELS)
            DNNKernels.q8gemm!(c,a,b; tiling=cfg,bias)
            got = Array(c)
            @test all(isfinite, got)
            @test maximum(abs, Float32.(got).-Float32.(reference)) <=
                  0.002maximum(abs,Float32.(reference))
        end
        b16=DNNKernels.toback(backend,bh[:,1:16]); c16=similar(c,256,16)
        tile16=DNNKernels.q8gemm_tiling(dev,a,b16,c16)
        @test tile16 !== nothing
        DNNKernels.q8gemm!(c16,a,b16;tiling=tile16,bias)
        @test maximum(abs,Float32.(Array(c16)).-Float32.(reference[:,1:16])) <=
              0.002maximum(abs,Float32.(reference[:,1:16]))
        @test DNNKernels.q8gemm_tiling(DNNKernels.caps(KernelAbstractions.CPU()),a,b,c) === nothing
        # `Mantle.storage(b)`: `toback` hands back the pool region, and these two
        # want a badly-shaped ARRAY operand to check the refusals with. `view`
        # is deliberately not defined on a `Buffer` — Mantle already means
        # something else by a view of a resource.
        bv = view(Mantle.storage(b), :, 1:17)
        @test_throws DimensionMismatch DNNKernels.q8gemm!(c, a, bv)
        @test_throws ArgumentError DNNKernels.q8gemm!(similar(c, 256, 17), a, bv)
    else
        @test_skip false
    end
end

@testset "pipelined int8 GEMM for wide products" begin
    backend = Mantle.defaultbackend()
    dev = DNNKernels.caps(backend)
    # Which products take it: wide ones, measured at Qwen-Image 2.1's four.
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 24576, 4096, 4224) == (2, 4, 2, 2, 32)
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 12288, 4096, 4224) == (2, 4, 2, 2, 32)
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 4096, 4096, 4224) == (2, 4, 2, 2, 32)
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 4096, 12288, 4224) == (4, 2, 2, 2, 32)
    # Narrow ones keep the tiles measured for them.
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 53248, 5120, 512) === nothing
    @test DNNKernels.q8gemm_pipelined_tile(dev, Float16, 4096, 4096, 4118) === nothing
    @test DNNKernels.q8gemm_pipelined_tile(dev, Int8, 4096, 4096, 4224) === nothing

    if dev.coopmat && dev.coopmatsubgroup == 32 && dev.tile == 16
        rng = MersenneTwister(19)
        # Both tiles, and a k deep enough to run the prefetch through many blocks.
        for (m, k, n) in ((256, 512, 1024), (256, 1024, 1024))
            a = DNNKernels.quantizeint8(backend,
                DNNKernels.toback(backend, Float16.(0.05f0 .* randn(rng, Float32, m, k))))
            q, scale = Array(a.q), Array(a.scale)
            # The exact product this kernel computes: scale[m] * (q B), in fp32.
            qf = Float32[DNNKernels.q8byte(q[cld(i, 4), j], (i - 1) % 4) for i in 1:m, j in 1:k]
            bh = Float16.(randn(rng, Float32, k, n))
            biash = randn(rng, Float32, m)
            exact = scale .* (qf * Float32.(bh)) .+ biash
            cfg = DNNKernels.q8gemm_pipelined_tile(dev, Float16, m, k, n)
            @test cfg !== nothing
            b = DNNKernels.toback(backend, bh)
            bias = DNNKernels.toback(backend, biash)
            for T in (Float32, Float16)
                c = KernelAbstractions.allocate(backend, T, m, n)
                DNNKernels.q8gemm_pipelined!(c, a, b; cfg, bias)
                # fp32: the accumulation order is the only difference. fp16: plus
                # the output's own rounding.
                tol = T === Float32 ? 1e-4 : 2e-3
                @test maximum(abs, Float32.(Array(c)) .- exact) <= tol * maximum(abs, exact)
            end
            # The other tile on the same operands, and an epilogue.
            other = cfg == (2, 4, 2, 2, 32) ? (4, 2, 2, 2, 32) : (2, 4, 2, 2, 32)
            c = KernelAbstractions.allocate(backend, Float32, m, n)
            DNNKernels.q8gemm_pipelined!(c, a, b; cfg = other, epilogue = x -> max(x, 0f0))
            @test maximum(abs, Array(c) .- max.(exact .- biash, 0f0)) <= 1e-4 * maximum(abs, exact)
        end
    end
end
