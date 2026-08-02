"""
The non-overlapping transposed convolution, against the gather it replaces.

`ConvTranspose2d` with stride equal to the kernel and no padding is not really a
convolution: the receptive fields do not overlap, so each output pixel comes from
exactly one input pixel and one weight slice, and the whole op is a GEMM plus a
depth-to-space interleave. SAM 2's mask decoder upsamples with two of them.

Checked against `convtranspose2d` — the gather — rather than against a host
reference, because that kernel is already the thing `verify_sam2.jl` validates
node by node. What is at stake here is that a *different route to the same
answer* agrees, and the gather is the answer.

Both element types, because the two paths diverge underneath: fp16 reaches the
cooperative-matrix GEMM and fp32 the strided one, and only the first has ever
been exercised by the encoder.
"""

using Test, DNNKernels, Lava, KernelAbstractions
const KA = KernelAbstractions
const DK = DNNKernels

@testset "transposed convolution via GEMM" begin
    back = LavaBackend()
    ws = DK.Workspace(back)

    @testset "agrees with the gather" begin
        # The decoder's two shapes, plus a small one whose channel counts are
        # not multiples of the GEMM tile.
        for (Ci, Hi, Co, T) in ((256, 64, 64, Float16), (64, 128, 32, Float16),
                                (256, 64, 64, Float32), (64, 128, 32, Float32),
                                (32, 16, 8, Float16), (24, 8, 12, Float32))
            hx = T.(randn(Float32, Hi, Hi, Ci, 1) .* 0.1f0)
            hw = T.(randn(Float32, 2, 2, Co, Ci) .* 0.1f0)
            hb = T.(randn(Float32, Co) .* 0.1f0)
            x, w, b = DK.toback(back, hx), DK.toback(back, hw), DK.toback(back, hb)
            ref = KA.allocate(back, T, 2Hi, 2Hi, Co, 1); fill!(ref, zero(T))
            got = KA.allocate(back, T, 2Hi, 2Hi, Co, 1); fill!(got, zero(T))
            DK.convolutiontranspose!(ref, x, w, b, (2,2), (0,0), (1,1), (0,0), 1)
            DK.reset!(ws)
            DK.convolutiontranspose!(got, x, w, b, (2,2), (0,0), (1,1), (0,0), 1; ws)
            KA.synchronize(back)
            r, g = Float32.(Array(ref)), Float32.(Array(got))
            # A shuffle that drops a sub-pixel phase leaves an exact lattice of
            # zeros, and a relative-error check averages straight over it. So
            # assert per phase — not "everything is nonzero", which fails
            # honestly: with random data a handful of fp16 outputs round to zero
            # (109 of 1 048 576 on the first shape here).
            for dx in 1:2, dy in 1:2
                @test any(!iszero, @view g[dx:2:end, dy:2:end, :, :])
            end
            @test maximum(abs, g .- r) / maximum(abs, r) < (T === Float16 ? 5e-3 : 1e-5)
            x = w = b = ref = got = nothing; GC.gc()
        end
    end

    @testset "only the non-overlapping case is taken" begin
        w22 = KA.allocate(back, Float32, 2, 2, 8, 8)
        w33 = KA.allocate(back, Float32, 3, 3, 8, 8)
        # stride == kernel, nothing else set: the case this exists for.
        @test DK.shufflecase(w22, (2,2), (0,0), (1,1), (0,0), 1)
        # A kernel wider than the stride overlaps, so the identity fails.
        @test !DK.shufflecase(w33, (2,2), (0,0), (1,1), (0,0), 1)
        # Padding, dilation, output padding and groups each break it.
        @test !DK.shufflecase(w22, (2,2), (1,1), (1,1), (0,0), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (2,2), (0,0), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (1,1), (1,1), 1)
        @test !DK.shufflecase(w22, (2,2), (0,0), (1,1), (0,0), 2)
        w22 = w33 = nothing; GC.gc()
    end

    @testset "an overlapping kernel still goes through the gather" begin
        # 3x3 stride 2: the fields overlap, so the GEMM route must not fire and
        # the answer must still be right.
        Ci, Hi, Co = 16, 8, 8
        hx = randn(Float32, Hi, Hi, Ci, 1) .* 0.1f0
        hw = randn(Float32, 3, 3, Co, Ci) .* 0.1f0
        x, w = DK.toback(back, hx), DK.toback(back, hw)
        ox = DK.convtransposesize(Hi, 3, 2, 0, 1, 0)
        ref = KA.allocate(back, Float32, ox, ox, Co, 1); fill!(ref, 0f0)
        got = KA.allocate(back, Float32, ox, ox, Co, 1); fill!(got, 0f0)
        DK.convolutiontranspose!(ref, x, w, nothing, (2,2), (0,0), (1,1), (0,0), 1)
        DK.reset!(ws)
        DK.convolutiontranspose!(got, x, w, nothing, (2,2), (0,0), (1,1), (0,0), 1; ws)
        KA.synchronize(back)
        @test Array(ref) == Array(got)
        x = w = ref = got = nothing; GC.gc()
    end
end

@testset "1x1 convolution as a plain GEMM" begin
    back = LavaBackend()
    ws = DK.Workspace(back)

    @testset "agrees with the im2col path" begin
        # SAM 2's own 1x1 shapes plus one whose channel counts miss the GEMM
        # tile, since `matmul!` has to fall back there rather than compute
        # something else.
        for (Wi, Hi, Cin, Cout) in ((256, 256, 144, 256), (128, 128, 256, 64),
                                    (64, 64, 576, 256), (32, 32, 32, 48))
            T = Float16
            x = DK.toback(back, T.(randn(Float32, Wi, Hi, Cin, 1) .* 0.1f0))
            w = DK.toback(back, T.(randn(Float32, 1, 1, Cin, Cout) .* 0.1f0))
            b = DK.toback(back, T.(randn(Float32, Cout) .* 0.1f0))
            ref = KA.allocate(back, T, Wi, Hi, Cout, 1); fill!(ref, zero(T))
            got = KA.allocate(back, T, Wi, Hi, Cout, 1); fill!(got, zero(T))
            # The routing is half the test: `convolution!` must recognise the
            # shape and take the GEMM. Asserted rather than assumed, because
            # `onebyone` returning `false` would silently compare the im2col path
            # against itself. (It used to be switched off with `CONV_1X1_GEMM`,
            # which is exactly the predicate-that-answers-configuration the
            # review's finding 7 names; the switch is gone.)
            @test DK.onebyone(w, (1,1), (0,0), (1,1), 1)
            DK.reset!(ws)
            DK.convolution_coopmat!(ref, x, w, b, (1,1), (0,0), (1,1); ws)
            DK.reset!(ws)
            DK.convolution!(got, x, w, b, (1,1), (0,0), (1,1), 1; ws)
            KA.synchronize(back)
            r, g = Float32.(Array(ref)), Float32.(Array(got))
            @test any(!iszero, g)
            # fp16, and the two paths accumulate in a different order, so this is
            # the format's tolerance rather than the algorithm's.
            @test maximum(abs, g .- r) / maximum(abs, r) < 5e-3
            x = w = b = ref = got = nothing; GC.gc()
        end
    end

    @testset "only the 1x1 case is taken" begin
        w11 = KA.allocate(back, Float16, 1, 1, 8, 8)
        w33 = KA.allocate(back, Float16, 3, 3, 8, 8)
        @test DK.onebyone(w11, (1,1), (0,0), (1,1), 1)
        @test !DK.onebyone(w33, (1,1), (0,0), (1,1), 1)   # a real receptive field
        @test !DK.onebyone(w11, (2,2), (0,0), (1,1), 1)   # stride skips pixels
        @test !DK.onebyone(w11, (1,1), (1,1), (1,1), 1)   # padding adds rows
        @test !DK.onebyone(w11, (1,1), (0,0), (1,1), 2)   # grouped
        # Dilation is meaningless at 1x1 but an exported graph may still carry
        # it; refuse rather than assume.
        @test !DK.onebyone(w11, (1,1), (0,0), (2,2), 1)
        w11 = w33 = nothing; GC.gc()
    end
end

@testset "a reduction axis off the tile is padded onto the tensor cores" begin
    # `CRS = Cin*KH*KW` is the weight's own extent, so it used to be a flat
    # refusal: a convolution whose channel count did not land on 16 stayed on the
    # implicit-GEMM kernel. SAM 2's stem is `7x7x3`, `CRS = 147`, and it ran at
    # 0.99 TFLOP/s there. Padding both halves with zeros — im2col writes zero
    # columns, the weight gets a zeroed copy — makes the padded product identical
    # to the real one and takes it to a tensor-core GEMM (2.800 -> 1.147 ms).
    #
    # The reference is `convolution_igemm!`, the path it replaces. Not a host
    # computation: what is at stake is that a *different route to the same
    # answer* agrees, and both accumulate in fp16, so the tolerance is fp16's.
    back = LavaBackend()
    function pair(Wi, Hi, Cin, Cout, KW, KH, s, p)
        OW = (Wi + 2p - KW) ÷ s + 1
        OH = (Hi + 2p - KH) ÷ s + 1
        x = KA.allocate(back, Float16, Wi, Hi, Cin, 1)
        copyto!(x, Float16.(reshape(0.4 .* sin.(range(0, 9, Wi * Hi * Cin)), Wi, Hi, Cin, 1)))
        w = KA.allocate(back, Float16, KW, KH, Cin, Cout)
        copyto!(w, Float16.(reshape(0.3 .* cos.(range(0, 7, KW * KH * Cin * Cout)),
                                    KW, KH, Cin, Cout)))
        o1 = KA.allocate(back, Float16, OW, OH, Cout, 1); fill!(o1, Float16(0))
        o2 = KA.allocate(back, Float16, OW, OH, Cout, 1); fill!(o2, Float16(0))
        ws = DK.Workspace(back)
        DK.reset!(ws)
        DK.convolution_coopmat!(o1, x, w, nothing, (s, s), (p, p), (1, 1); ws)
        DK.convolution_igemm!(o2, x, w, nothing, (s, s), (p, p), (1, 1))
        KA.synchronize(back)
        r = (Float64.(Array(o1)), Float64.(Array(o2)), DK.conv_coopmat_applicable(o1, x, w))
        x = w = o1 = o2 = nothing; GC.gc()
        r
    end

    @testset "CRS $(cin*kw*kh) ($(cin)x$(kw)x$(kh))" for (wi, hi, cin, cout, kw, kh, s, p) in
            [(256, 256, 3, 144, 7, 7, 4, 3),   # SAM 2's stem, 147 -> 160
             (64, 64, 5, 32, 3, 3, 1, 1),      #  45 ->  48
             (48, 48, 7, 48, 5, 5, 2, 2),      # 175 -> 176
             (32, 32, 32, 64, 3, 3, 1, 1)]     # 288, already on the tile
        got, want, applicable = pair(wi, hi, cin, cout, kw, kh, s, p)
        @test applicable
        @test size(got) == size(want)
        @test all(isfinite, got)
        # fp16 accumulation in a different order; the on-tile cases in this same
        # list sit at the same 1e-3, so a padding bug would have to hide under
        # the noise floor of the path that was never padded.
        @test maximum(abs, got .- want) / maximum(abs, want) < 5e-3
    end

    @testset "the pad is refused when the waste is large" begin
        # A concatenated scalar channel gives MatAnyone `Cin = 17`, which would
        # round to 32 and pay 88% waste to reach the tensor cores. `CONV_CRS_PAD`
        # is where that line sits.
        x = KA.allocate(back, Float16, 32, 32, 17, 1); fill!(x, Float16(0.1))
        w = KA.allocate(back, Float16, 1, 1, 17, 32); fill!(w, Float16(0.1))
        o = KA.allocate(back, Float16, 32, 32, 32, 1); fill!(o, Float16(0))
        @test !DK.conv_coopmat_applicable(o, x, w)          # 17 -> 32, refused
        old = DK.CONV_CRS_PAD[]
        try
            DK.CONV_CRS_PAD[] = 2.0
            @test DK.conv_coopmat_applicable(o, x, w)       # ...only by policy
            DK.CONV_CRS_PAD[] = 1.0
            # 1.0 is the old behaviour exactly: nothing off the tile gets in.
            w2 = KA.allocate(back, Float16, 7, 7, 3, 144); fill!(w2, Float16(0.1))
            x2 = KA.allocate(back, Float16, 256, 256, 3, 1); fill!(x2, Float16(0.1))
            o2 = KA.allocate(back, Float16, 64, 64, 144, 1); fill!(o2, Float16(0))
            @test !DK.conv_coopmat_applicable(o2, x2, w2)
            w2 = x2 = o2 = nothing
        finally
            DK.CONV_CRS_PAD[] = old
        end
        x = w = o = nothing; GC.gc()
    end
end
