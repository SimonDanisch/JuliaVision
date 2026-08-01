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
