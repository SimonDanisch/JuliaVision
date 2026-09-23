"""
The int8-by-int8 product, which is NOT on any model's path — see `w8a8.jl` for
the measurement that kept it off one.

It exists because the int8 cooperative-matrix units are 3.1x the fp16 ones on
an 8060S (74.7 TOP/s against 24.4, staging taken out of the loop), and the
question of whether a model can reach them is worth being able to answer
without writing the kernel again. This file is what keeps the answer honest:
a kernel that is wrong and unused is worse than no kernel, because it looks
available.

Three things are checked:

  * the quantiser against the host, to its own last step, including the
    padding columns that make a ragged product fit the tile;
  * the product against the exact scaled integer product, to fp16 rounding;
  * **four k blocks**, which is where the double buffering first reuses a
    buffer for the second time. Guarding the prefetch and the store on
    `kb + 1 < nb` instead of clamping the index makes that case — and only that
    case — a wrong answer, because the two barriers are then at non-uniform
    control flow.
"""

import KernelInterface as KI
using Test, DNNKernels, Mantle, KernelAbstractions, Random

const DKW = DNNKernels

@testset "int8 x int8 product" begin
    backend = Mantle.LavaBackend()
    caps = DNNKernels.caps(backend)
    if caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16
        rng = MersenneTwister(19)

        @testset "the quantiser" begin
            K, N = 256, 300
            np = DKW.w8a8columns(N)
            @test np == 384
            xh = Float16.(randn(rng, Float32, K, N) .* 0.4f0)
            q, s = DKW.w8a8quantize(backend, DNNKernels.toback(backend, xh), np)
            KernelAbstractions.synchronize(backend)
            sh = Array(s)
            values = reshape(reinterpret(Int8, Array(q)), K, np)
            for n in 1:N
                a = maximum(abs, Float32.(xh[:, n]))
                @test sh[n] ≈ a / 127f0 rtol=1f-6
                # The kernel multiplies by `127/a` rather than dividing by the
                # scale, so the host reference does too. One step of slack and
                # not equality: the product is a float multiply on two
                # different machines, and a value that lands on `x.5` on one of
                # them lands just under it on the other. Five columns of 300
                # differ by exactly one step, which is the quantiser's own
                # resolution.
                want = clamp.(round.(Float32.(xh[:, n]) .* (127f0 / a)), -127f0, 127f0)
                @test maximum(abs, Float32.(values[:, n]) .- want) <= 1
            end
            # The columns past `N` are the product's padding and contribute
            # nothing: zero values, and a scale that cannot divide by zero.
            @test all(values[:, (N+1):np] .== 0)
            @test all(sh[(N+1):np] .== 1f0)
        end

        @testset "the product, and four k blocks" begin
            # `K = 256` is four 64-deep blocks, which is the shape the guarded
            # form got wrong. The others bracket it.
            for (M, K, N) in ((128, 64, 128), (128, 128, 128), (128, 256, 128),
                              (256, 1024, 256))
                np = DKW.w8a8columns(N)
                @test DKW.w8a8ok(caps, Float16, M, K, np)
                wh = Float16.(randn(rng, Float32, M, K) .* 0.2f0)
                A = DNNKernels.quantizeint8(backend, DNNKernels.toback(backend, wh))
                xh = Float16.(randn(rng, Float32, K, N) .* 0.4f0)
                # `w8a8quantize` declares into a graph, so its results are
                # `Mantle.Buffer`s — pool regions — as a packed weight's fields
                # are. A BARE launch has no graph to resolve one against, so
                # every operand goes through `Mantle.storage` here; `dispatch!`
                # does that itself and needs none of this.
                qbuf, sbuf = DKW.w8a8quantize(backend, DNNKernels.toback(backend, xh), np)
                qb, sb = Mantle.storage(qbuf), Mantle.storage(sbuf)
                C = KernelAbstractions.allocate(backend, Float16, M, np)
                KI.Kernel(backend, DKW.w8a8_gemm_kernel!)(
                    C, DKW.w8a8weight(A), Mantle.storage(A.scale), qb, sb,
                    Val(M), Val(np), Val(K);
                    ndrange = (M ÷ DKW.W8A8_BM) * (np ÷ DKW.W8A8_BN) * DKW.W8A8_WG, workgroupsize = DKW.W8A8_WG)
                KernelAbstractions.synchronize(backend)
                qa = Float32.(Array(DKW.w8a8weight(A))[1:M, :])
                qbh = Float32.(reshape(reinterpret(Int8, Array(qb)), K, np))
                want = (qa .* Array(Mantle.storage(A.scale))) * (qbh .* reshape(Array(sb), 1, :))
                got = Float32.(Array(C))
                # The integer product is exact; what is left is the fp16 store.
                @test maximum(abs, got .- want) <= 0.002maximum(abs, want)
                @test all(isfinite, got)
            end
        end

        @testset "a shape the tile does not divide is refused" begin
            @test !DKW.w8a8ok(caps, Float16, 100, 64, 128)    # M
            @test !DKW.w8a8ok(caps, Float16, 128, 64, 100)    # N
            @test !DKW.w8a8ok(caps, Float16, 128, 100, 128)   # K
            @test !DKW.w8a8ok(caps, Int32, 128, 64, 128)      # output type
        end

        @testset "the weight is the packed weight's own bytes" begin
            # No repacking: four output rows to a `UInt32` little-endian IS
            # int8 `(M, K)` with M contiguous.
            wh = Float16.(randn(rng, Float32, 128, 64) .* 0.2f0)
            A = DNNKernels.quantizeint8(backend, DNNKernels.toback(backend, wh))
            bytes = Array(DKW.w8a8weight(A))
            words = Array(A.q)
            @test size(bytes) == (128, 64)
            for k in 1:64, g in 1:32
                w = words[g, k]
                for r in 0:3
                    @test bytes[(g-1)*4 + r + 1, k] ==
                          reinterpret(Int8, UInt8((w >> (8r)) & 0xff))
                end
            end
        end
    else
        @test_skip false
    end
end
