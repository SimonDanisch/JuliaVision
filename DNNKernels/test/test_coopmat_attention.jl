"""
Attention through two batched cooperative-matrix GEMMs.

Both halves of attention are matrix products — `S = qT k` and `O = P vT`, one per
(head, batch) — so `Lava.coopmat_gemm!`'s `nbatch` runs all of them in one
dispatch each. The three-pass scalar kernels run at 2.3 TFLOP/s on SAM 2's
global attention against the 13 the same device's GEMM sustains.

It is **gated on sequence length**, and this file pins both sides of that gate:
below the plan's `minl` the E-padding copies and the wider fp32 score matrix cost
more than the tensor cores save. Measured, whole op, same total token count:

    L=256   0.81x (loses)   L=512  1.10x   L=1024 1.48x
    L=2048  1.76x           L=4096 2.00x

On SAM 2's encoder that is -28.5 ms of a 338.5 ms GPU total: `attn_scores`
34.9 -> 14.9 and `attn_apply` 42.9 -> 17.5, against +12.5 in the GEMM and +4.8
in the padding copies.
"""

using Test, Lava, DNNKernels, KernelAbstractions
using DNNKernels: sdpa, sdpa_coopmat!, coopmat_sdpa_plan, CoopMatSDPAPlan, Decline,
                  Workspace, Ctx, Device
const KA = KernelAbstractions
const LD = DNNKernels

"The three passes, on the host, in fp32 over the fp16 operands."
function threepass(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float32, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(qh[e,lq,h,b] * kh[e,lk,h,b] for e in 1:E) * scale for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e,lq,h,b] = sum(s[lk] * vh[e,lk,h,b] for lk in 1:Lk)
        end
    end
    out
end

"""
Run the cooperative-matrix path **by name** and compare it to the reference.

This used to call `sdpa` twice with `COOPMAT_MINL` flipped, which is how a test
stops testing anything: `sdpa` tries the **fused** path first, and the fused path
takes all four shapes below, so both halves of the A/B ran the same kernel and
the comparison was of a result with itself. Vacuous since flash landed, and it
could not have failed. Calling `sdpa_coopmat!` with a plan asserts the routing as
well as the arithmetic — the same lesson as the two toggles deleted in step 1.
"""
function coopmatpath(E, L, H, B)
    back = LavaBackend()
    ctx = Ctx(back; ws = Workspace(back))
    host(f, s) = Float16.(reshape(0.4 .* f.(range(0, s, E * L * H * B)), E, L, H, B))
    qh, kh, vh = host(sin, 9), host(cos, 7), host(sin, 5)
    mk(x) = (a = KA.allocate(back, Float16, E, L, H, B); copyto!(a, x); a)
    q, k, v = mk(qh), mk(kh), mk(vh)
    scale = 1 / sqrt(E)

    # `minl = 1` is what `COOPMAT_MINL[] = 1` used to mean: these shapes are
    # shorter than the gate, and the point here is the arithmetic, not the gate.
    plan = coopmat_sdpa_plan(ctx.dev, q, k, v, nothing; minl = 1)
    plan isa CoopMatSDPAPlan || return (nothing, plan)
    out = KA.allocate(back, Float32, E, L, H, B); fill!(out, 0f0)
    LD.reset!(ctx.ws)
    sdpa_coopmat!(ctx, out, plan, q, k, v, scale)
    KA.synchronize(back)
    got = Array(out)
    ref = threepass(Float32.(qh), Float32.(kh), Float32.(vh), Float32(scale))
    q = k = v = out = nothing
    GC.gc()
    # A kernel that writes nothing also "matches" a zero reference.
    maximum(abs, got) > 1e-3 || return (Inf, plan)
    (maximum(abs, ref .- got) / max(maximum(abs, ref), eps(Float32)), plan)
end
bothpaths(E, L, H, B) = coopmatpath(E, L, H, B)[1]

@testset "cooperative-matrix attention" begin
    if !Lava.coopmat_gemm_available()
        @info "no cooperative-matrix support on this device; skipping"
    else
        @testset "agrees with the three-pass reference" begin
            # E = 72 is SAM 2's head dimension and is NOT a multiple of the
            # 16-wide tile — it pads to 80, which is the case most likely to be
            # wrong. E = 64 needs no padding and is the control.
            @test bothpaths(72, 64, 2, 1) < 5e-3
            @test bothpaths(72, 128, 2, 2) < 5e-3
            @test bothpaths(72, 256, 8, 1) < 5e-3
            @test bothpaths(64, 128, 4, 1) < 5e-3
        end

        @testset "the gate refuses what it cannot compute, and says why" begin
            back = LavaBackend()
            dev = Device(back)
            f16(dims...) = KA.allocate(back, Float16, dims...)
            f32(dims...) = KA.allocate(back, Float32, dims...)
            q = f16(72, 1024, 4, 1)
            # The lengths come off the operands now instead of being passed
            # alongside them, so a caller can no longer describe a shape the
            # arrays do not have — which is the same class of drift the plan
            # objects exist to remove.
            #
            # A bias has to be added inside the score pass, and a GEMM has no
            # epilogue to add it in.
            @test coopmat_sdpa_plan(dev, q, q, q, f16(1024, 1024, 4, 1)).reason === :bias
            @test coopmat_sdpa_plan(dev, q, q, q, nothing) isa CoopMatSDPAPlan
            # Extents off the tile: the mask decoder's 23-token prompt.
            # `minl = 1` so the length gate does not fire first and mask what is
            # under test — 23 is both off the tile AND short.
            @test coopmat_sdpa_plan(dev, f16(72, 23, 4, 1), f16(72, 4096, 4, 1),
                                    f16(72, 4096, 4, 1), nothing;
                                    minl = 1).reason === :extent
            # Short sequences lose. Written against the plan's own default rather
            # than a literal, because the threshold is a measured property of the
            # GEMM underneath and it has already moved once: 512 -> 256 when the
            # staged `vec2` GEMM took that path from 20.6 to 35.3 TFLOP/s. A test
            # that hardcoded 256 as "refused" is what caught the change, which is
            # useful once and an obstacle every time after.
            let L = 256
                s = f16(72, L, 4, 1)
                @test coopmat_sdpa_plan(dev, s, s, s, nothing; minl = L) isa CoopMatSDPAPlan
                b = L - dev.tile                    # still tile-aligned
                sb = f16(72, b, 4, 1)
                @test coopmat_sdpa_plan(dev, sb, sb, sb, nothing; minl = L).reason === :short
                s = sb = nothing
            end
            # fp32 operands have no cooperative-matrix load.
            @test coopmat_sdpa_plan(dev, f32(72, 1024, 4, 1), q, q, nothing).reason === :eltype
            # And a device without the feature refuses regardless of shape.
            nocm = Device(false, dev.tile, dev.subgroup, dev.coopmatsubgroup,
                          dev.sharedbudget, dev.workgrouplimit, dev.cores,
                          dev.launchgroup)
            @test coopmat_sdpa_plan(nocm, q, q, q, nothing).reason === :nocoopmat
            q = nothing
            GC.gc()
        end
    end
end
