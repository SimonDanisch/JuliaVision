"""
`batchedmatmul!` routes each batch plane through `matmul!`.

`aten::bmm` had no capability dispatch of its own: it went straight to
`launch!(ctx, mm3, ...)`, one thread per output element walking K in global
memory. On Depth Anything V2 Small that one line was **79.6% of the forward
pass** — 826 ms of 972 — and routing the planes through `matmul!` took the frame
to 233 ms.

What made it survive so long is that it does not look like a matmul in a
profile. Both `bmm` shapes launch as `gpu_ndmap_flat!`, the generic elementwise
launcher, so a by-kernel-name profile reports them as elementwise work; this was
once written up as "83.2% elementwise, 4.7% convolution".

It is also very unevenly distributed. The two attention products do *identical*
arithmetic, 720.7 M MACs each, and measured 10.1x apart in situ:

    attn*V  out (64, 1370, 6)   K=1370    64.47 ms/call
    Q*K'    out (1370, 1370, 6) K=64       6.39 ms/call

so a spot check on the wrong one of the two finds nothing much wrong.

This file pins the three things the fix rests on: the planes are contiguous
`LavaArray{T,2}` slices (so the routing is free and the operand type stays
visible to `mm_coopmat_plan`), the routed result equals the naive kernel, and a
qualifying fp16 batch really does reach the tensor-core plan.
"""

using Test, Lava, DNNKernels, KernelAbstractions
using DNNKernels: batchedmatmul!, mm3, mm_coopmat_plan, MMCoopMatPlan, Decline,
                  Ctx, Workspace, Device, launch!
const KA = KernelAbstractions
const DK = DNNKernels

back = LavaBackend()
ctx = DK.Ctx(back; ws = Workspace(back))
dev = Device(back)

"`out[i,j,b] = Σ_k A[i,k,b] B[k,j,b]`, on the host in Float64."
function ref3(A, B)
    n, k, nb = size(A); m = size(B, 2)
    out = zeros(Float64, n, m, nb)
    for b in 1:nb, j in 1:m, i in 1:n
        out[i, j, b] = sum(Float64(A[i, t, b]) * Float64(B[t, j, b]) for t in 1:k)
    end
    out
end

@testset "batchedmatmul!" begin

    @testset "against a Float64 host reference" begin
        # Deliberately awkward extents: none is a multiple of the 16-wide tile,
        # so these exercise the decline-and-fall-back route as well.
        for (n, k, m, nb) in ((8, 5, 7, 3), (17, 33, 9, 2), (64, 1370, 1, 1),
                              (1, 64, 130, 4))
            Ah = rand(Float32, n, k, nb) .- 0.5f0
            Bh = rand(Float32, k, m, nb) .- 0.5f0
            A = KA.allocate(back, Float32, n, k, nb); copyto!(A, Ah)
            B = KA.allocate(back, Float32, k, m, nb); copyto!(B, Bh)
            out = KA.allocate(back, Float32, n, m, nb)
            batchedmatmul!(ctx, out, A, B)
            KA.synchronize(back)
            r = ref3(Ah, Bh)
            # fp32 accumulation over K terms; scale the tolerance with K.
            @test maximum(abs.(Array(out) .- r)) < 1e-4 * max(k, 1)
        end
    end

    # The equivalence the whole change rests on: same answer as the kernel it
    # replaced, on the exact shapes DINOv2's attention produces. `mm3` is the
    # trusted implementation here — obviously correct, just slow.
    @testset "equals the naive mm3 kernel on the DINOv2 attention shapes" begin
        for (n, k, m, nb) in ((64, 1370, 1370, 6),      # attn * V
                              (1370, 64, 1370, 6))      # Q * K'
            A = KA.allocate(back, Float32, n, k, nb); copyto!(A, rand(Float32, n, k, nb) .- 0.5f0)
            B = KA.allocate(back, Float32, k, m, nb); copyto!(B, rand(Float32, k, m, nb) .- 0.5f0)
            routed = KA.allocate(back, Float32, n, m, nb)
            naive  = KA.allocate(back, Float32, n, m, nb)
            batchedmatmul!(ctx, routed, A, B)
            launch!(ctx, mm3, naive, A, B)
            KA.synchronize(back)
            r, v = Array(routed), Array(naive)
            # Different summation order, same arithmetic: fp32 reassociation only.
            @test maximum(abs.(r .- v)) < 1e-3 * maximum(abs.(v))
        end
    end

    # Why the routing costs nothing, and why it does not blind the plan.
    @testset "a batch plane is a contiguous 2-D LavaArray" begin
        A = KA.allocate(back, Float32, 64, 1370, 6)
        for b in (1, 3, 6)
            v = view(A, :, :, b)
            @test v isa Lava.LavaArray{Float32,2}     # not a SubArray: no copy,
            @test stride(v, 1) == 1                   # and `mm_coopmat_plan`
            @test stride(v, 2) == size(A, 1)          # still sees the operand type
            @test size(v) == (64, 1370)
        end
    end

    # The point of routing through `matmul!` rather than hardcoding `mul!`: an
    # fp16 batch whose extents land on the tile must reach the tensor-core plan.
    # Depth Anything cannot — it is fp32 throughout — which is exactly why the
    # assertion is written against a purpose-built operand and not against a model.
    @testset "an on-tile fp16 plane reaches the cooperative-matrix plan" begin
        if dev.coopmat
            t = dev.tile
            A = KA.allocate(back, Float16, 4t, 2t, 3)
            B = KA.allocate(back, Float16, 2t, 4t, 3)
            out = KA.allocate(back, Float16, 4t, 4t, 3)
            plan = mm_coopmat_plan(dev, view(out, :, :, 1),
                                   view(A, :, :, 1), view(B, :, :, 1))
            @test plan isa MMCoopMatPlan
            # And the fp32 case this model actually hits declines, on operands.
            A32 = KA.allocate(back, Float32, 4t, 2t, 3)
            B32 = KA.allocate(back, Float32, 2t, 4t, 3)
            d = mm_coopmat_plan(dev, view(out, :, :, 1),
                                view(A32, :, :, 1), view(B32, :, :, 1))
            @test d isa Decline && d.reason === :operands
        else
            @test_broken dev.coopmat   # no cooperative matrices on this device
        end
    end

    # Half of Depth Anything's `bmm` arrive with a non-dense `A` — a
    # `ReshapedArray` of a `SubArray` of a `PermutedDimsArray` — so the plane is a
    # nested `SubArray`, not a `LavaArray{T,2}`. Those are the *expensive* twelve,
    # and a `DenseArray` guard that excluded them cost the whole win (4.15x ->
    # 1.02x). They must go through `matmul!` too, and still be right.
    @testset "a permuted operand is routed and still correct" begin
        n, k, m, nb = 6, 5, 7, 3
        Ah = rand(Float32, k, n, nb) .- 0.5f0        # stored transposed
        Bh = rand(Float32, k, m, nb) .- 0.5f0
        Araw = KA.allocate(back, Float32, k, n, nb); copyto!(Araw, Ah)
        B = KA.allocate(back, Float32, k, m, nb); copyto!(B, Bh)
        A = PermutedDimsArray(Araw, (2, 1, 3))       # (n, k, nb), not dense
        @test !(view(A, :, :, 1) isa Lava.LavaArray{Float32,2})
        out = KA.allocate(back, Float32, n, m, nb)
        batchedmatmul!(ctx, out, A, B)
        KA.synchronize(back)
        @test maximum(abs.(Array(out) .- ref3(permutedims(Ah, (2, 1, 3)), Bh))) < 1e-4 * k
    end

    # The routing gate. Per-plane costs `nbatch - 1` extra dispatches, so a batch
    # of small planes must NOT take it: routing every `bmm` per plane cost
    # MatAnyone 14.5% (173.8 ms against 151.8) before this existed. The shapes
    # below are the real ones, and the fp16 pair is the trap — `mm_coopmat_plan`
    # accepts a 16x32 fp16 plane happily, so "can use a good kernel" is not a
    # sufficient test and the tile-fill clause is what actually decides it.
    @testset "per-plane routing is gated on plane size" begin
        d16(a, b, c) = KA.allocate(back, Float16, a, b, c)
        d32(a, b, c) = KA.allocate(back, Float32, a, b, c)

        # MatAnyone readout_query: batch 8, planes 16x32 and 32x16. One flat launch.
        @test !DNNKernels.planewise_worth(ctx, d16(16,16,8), d16(16,32,8), d16(32,16,8))
        @test !DNNKernels.planewise_worth(ctx, d16(32,16,8), d16(32,16,8), d16(16,16,8))

        # Depth Anything attention: batch 6, planes that dwarf a tile.
        @test DNNKernels.planewise_worth(ctx, d32(64,1370,6), d32(64,1370,6), d32(1370,1370,6))
        @test DNNKernels.planewise_worth(ctx, d32(1370,1370,6), d32(1370,64,6), d32(64,1370,6))

        # A single plane pays no extra dispatch, so it always routes through
        # `matmul!` and gets a real kernel instead of `mm3` for free.
        @test DNNKernels.planewise_worth(ctx, d32(65536,4,1), d32(65536,32,1), d32(32,4,1))

        # ...and the results agree whichever way the gate falls.
        for (n, k, m, nb) in ((16, 32, 16, 8), (64, 130, 96, 6))
            Ah = rand(Float32, n, k, nb) .- 0.5f0
            Bh = rand(Float32, k, m, nb) .- 0.5f0
            A = KA.allocate(back, Float32, n, k, nb); copyto!(A, Ah)
            B = KA.allocate(back, Float32, k, m, nb); copyto!(B, Bh)
            out = KA.allocate(back, Float32, n, m, nb)
            batchedmatmul!(ctx, out, A, B); KA.synchronize(back)
            @test maximum(abs.(Array(out) .- ref3(Ah, Bh))) < 1e-4 * k
        end
    end

    # `mm3` is still the path for anything not sliceable plane-wise, and it has
    # to stay correct: a mismatched batch extent must not silently take the loop.
    @testset "non-sliceable shapes still work" begin
        A = KA.allocate(back, Float32, 6, 4); B = KA.allocate(back, Float32, 4, 5)
        Ah = rand(Float32, 6, 4); Bh = rand(Float32, 4, 5)
        copyto!(A, Ah); copyto!(B, Bh)
        out = KA.allocate(back, Float32, 6, 5, 1)
        # ndims(A) == 2, so the plane loop must decline and `mm3` must run.
        A3 = reshape(A, 6, 4, 1); B3 = reshape(B, 4, 5, 1)
        batchedmatmul!(ctx, out, A3, B3)
        KA.synchronize(back)
        @test maximum(abs.(Array(out)[:, :, 1] .- Ah * Bh)) < 1e-4
    end
end
