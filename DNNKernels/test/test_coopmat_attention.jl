"""
Attention through two batched cooperative-matrix GEMMs.

Both halves of attention are matrix products — `S = qT k` and `O = P vT`, one per
(head, batch) — so `Lava.coopmat_gemm!`'s `nbatch` runs all of them in one
dispatch each. The three-pass scalar kernels run at 2.3 TFLOP/s on SAM 2's
global attention against the 13 the same device's GEMM sustains.

It is **gated on sequence length**, and this file pins both sides of that gate:
below `COOPMAT_MINL` the E-padding copies and the wider fp32 score matrix cost
more than the tensor cores save. Measured, whole op, same total token count:

    L=256   0.81x (loses)   L=512  1.10x   L=1024 1.48x
    L=2048  1.76x           L=4096 2.00x

On SAM 2's encoder that is -28.5 ms of a 338.5 ms GPU total: `attn_scores`
34.9 -> 14.9 and `attn_apply` 42.9 -> 17.5, against +12.5 in the GEMM and +4.8
in the padding copies.
"""

using Test, Lava, DNNKernels, KernelAbstractions
using DNNKernels: sdpa, coopmat_sdpa_applicable, COOPMAT_MINL, Workspace
const KA = KernelAbstractions
const LD = DNNKernels

"Run `sdpa` with the coopmat path forced on and forced off, and compare."
function bothpaths(E, L, H, B)
    back = LavaBackend()
    mk(f, s) = (a = KA.allocate(back, Float16, E, L, H, B);
                copyto!(a, Float16.(reshape(0.4 .* f.(range(0, s, E * L * H * B)), E, L, H, B))); a)
    q, k, v = mk(sin, 9), mk(cos, 7), mk(sin, 5)
    scale = 1 / sqrt(E)
    old = COOPMAT_MINL[]
    outs = map((typemax(Int), 1)) do minl        # off, then on
        COOPMAT_MINL[] = minl
        o = sdpa(q, k, v, nothing, scale; backend = back, ws = Workspace(back))
        KA.synchronize(back)
        Array(o)
    end
    COOPMAT_MINL[] = old
    q = k = v = nothing
    GC.gc()
    ref, got = outs
    return maximum(abs, ref .- got) / max(maximum(abs, ref), eps(Float32))
end

@testset "cooperative-matrix attention" begin
    if !Lava.coopmat_gemm_available()
        @info "no cooperative-matrix support on this device; skipping"
    else
        @testset "agrees with the three-pass path" begin
            # E = 72 is SAM 2's head dimension and is NOT a multiple of the
            # 16-wide tile — it pads to 80, which is the case most likely to be
            # wrong. E = 64 needs no padding and is the control.
            @test bothpaths(72, 64, 2, 1) < 5e-3
            @test bothpaths(72, 128, 2, 2) < 5e-3
            @test bothpaths(72, 256, 8, 1) < 5e-3
            @test bothpaths(64, 128, 4, 1) < 5e-3
        end

        @testset "the gate refuses what it cannot compute" begin
            back = LavaBackend()
            f16(dims...) = KA.allocate(back, Float16, dims...)
            f32(dims...) = KA.allocate(back, Float32, dims...)
            q = f16(72, 1024, 4, 1)
            # A bias has to be added inside the score pass, and a GEMM has no
            # epilogue to add it in.
            @test !coopmat_sdpa_applicable(q, q, q, f16(1024, 1024, 4, 1), 1024, 1024)
            @test coopmat_sdpa_applicable(q, q, q, nothing, 1024, 1024)
            # Extents off the tile: the mask decoder's 23-token prompt.
            @test !coopmat_sdpa_applicable(q, q, q, nothing, 23, 4096)
            # Short sequences lose. Written against `COOPMAT_MINL` rather than a
            # literal, because the threshold is a measured property of the GEMM
            # underneath and it has already moved once: 512 -> 256 when the staged
            # `vec2` GEMM took that path from 20.6 to 35.3 TFLOP/s. A test that
            # hardcoded 256 as "refused" is what caught the change, which is
            # useful once and an obstacle every time after.
            let L = COOPMAT_MINL[]
                @test coopmat_sdpa_applicable(q, q, q, nothing, L, L)
                below = L - Lava.GEMM_TILE          # still tile-aligned
                @test !coopmat_sdpa_applicable(q, q, q, nothing, below, below)
            end
            # fp32 operands have no cooperative-matrix load.
            @test !coopmat_sdpa_applicable(f32(72, 1024, 4, 1), q, q, nothing, 1024, 1024)
            q = nothing
            GC.gc()
        end
    end
end
