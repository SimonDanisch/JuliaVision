"""
The tensor-core convolution when its im2col matrix does not fit, and when its
output channel count is not a tile.

Both used to end the same way — on the implicit-GEMM kernel at about one
TFLOP/s. The size was a refusal (`Decline(:im2colsize)`) and the channel count
was silently slow, because the staged GEMM's rate is decided by which `bn` its
tiling gets and there is a factor of thirteen between having one and not. The
Qwen-Image 2.1 VAE runs into both: twenty-seven of its forty-five convolutions
were refused on size and its channel counts are 144, 288 and 576.

Now the pixel axis is divided into `plan.rows`-sized chunks through one buffer
and the weight is zero-padded to `plan.CoutP` columns, so what has to hold is
that a chunked, padded convolution computes the same numbers as the definition.
Against a host reference rather than against the other kernel: the two do not
accumulate in the same order and never did.
"""

import KernelInterface
using Test, DNNKernels, Mantle, KernelAbstractions, Random

const DKC = DNNKernels

"""`out[ow, oh, co, n]` from the definition, in the reversed layout."""
function convref(x, w, bias, stride, pad, dil)
    KW, KH, Cin, Cout = size(w)
    Wid, Hei, _, N = size(x)
    OW = (Wid + 2pad[1] - dil[1] * (KW - 1) - 1) ÷ stride[1] + 1
    OH = (Hei + 2pad[2] - dil[2] * (KH - 1) - 1) ÷ stride[2] + 1
    out = zeros(Float32, OW, OH, Cout, N)
    for n in 1:N, co in 1:Cout, oh in 1:OH, ow in 1:OW
        acc = bias === nothing ? 0f0 : Float32(bias[co])
        for ci in 1:Cin, kh in 1:KH, kw in 1:KW
            ix = (ow - 1) * stride[1] - pad[1] + (kw - 1) * dil[1] + 1
            iy = (oh - 1) * stride[2] - pad[2] + (kh - 1) * dil[2] + 1
            (1 <= ix <= Wid && 1 <= iy <= Hei) || continue
            acc += Float32(x[ix, iy, ci, n]) * Float32(w[kw, kh, ci, co])
        end
        out[ow, oh, co, n] = acc
    end
    out
end

@testset "a chunked, channel-padded convolution is the same convolution" begin
    # Whatever backend is loaded, not a named one: `Mantle.LavaBackend` exists only
    # where Lava does, so naming it made this file ERROR on a machine with a
    # different GPU rather than take the capability skip below that was written
    # for exactly this case.
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    # `coopmatkernels`, not `caps.coopmat` alone: Metal reports cooperative
    # matrices (simdgroup matrices are real) while the staged GEMM these tests
    # exercise -- `GEMM_TILINGS`, `q8gemm_tiling`, `conv_coopmat_plan` -- is
    # compiled in only where Lava is. Asking the narrower question is what makes
    # the skip below fire instead of a `Decline(:host)` failing an assertion.
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        ctx = DKC.Ctx(backend)
        rng = MersenneTwister(19)
        dev = KernelAbstractions.allocate
        # `Cout` on the tile but not on a column tile (144, 288), on a 64-tile
        # but not a 128 (576), and already on one (128). `Cin` chosen so `CRS`
        # both does and does not land on `bk`.
        for (Wid, Hei, Cin, Cout, KW, KH, st, pd, dl, bias) in
                ((24, 20, 16, 144, 3, 3, (1, 1), (1, 1), (1, 1), true),
                 (24, 20, 16, 288, 3, 3, (1, 1), (1, 1), (1, 1), false),
                 (16, 16, 32, 576, 3, 3, (1, 1), (1, 1), (1, 1), true),
                 (17, 13, 16, 128, 3, 3, (2, 2), (1, 1), (1, 1), true),
                 (16, 16, 48, 128, 1, 1, (1, 1), (0, 0), (1, 1), false))
            xh = Float16.(randn(rng, Float32, Wid, Hei, Cin, 1) .* 0.3f0)
            wh = Float16.(randn(rng, Float32, KW, KH, Cin, Cout) .* 0.1f0)
            bh = bias ? Float16.(randn(rng, Float32, Cout) .* 0.2f0) : nothing
            ref = convref(xh, wh, bh, st, pd, dl)
            OW, OH = size(ref, 1), size(ref, 2)

            x = DKC.toback(backend, xh)
            w = DKC.toback(backend, wh)
            b = bh === nothing ? nothing : DKC.toback(backend, bh)
            out = dev(backend, Float16, OW, OH, Cout, 1)

            # Every chunk count from "all at once" down to several, by shrinking
            # the budget rather than by a parameter the caller does not have.
            for cap in (1 << 30, 1 << 16, 1 << 14)
                plan = DKC.conv_coopmat_plan(caps, Float16, Float16,
                                             size(out), size(w); im2colcap = cap)
                plan isa DKC.ConvCoopMatPlan || continue
                @test plan.CoutP >= Cout
                @test plan.CoutP % Mantle.GEMM_TILE == 0
                @test plan.rows >= DKC.GEMM_BLOCK || plan.rows == plan.NPQ
                fill!(out, Float16(0))
                DKC.convolution_coopmat!(ctx, out, plan, x, w, b, st, pd, dl)
                KernelAbstractions.synchronize(backend)
                got = Float32.(Array(out))
                @test maximum(abs, got .- ref) / max(1f-3, maximum(abs, ref)) < 5e-3
            end

            # The fused activation rides on the same epilogue.
            plan = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w);
                                         im2colcap = 1 << 14)
            if plan isa DKC.ConvCoopMatPlan
                fill!(out, Float16(0))
                DKC.convolution_coopmat!(ctx, out, plan, x, w, b, st, pd, dl; act = :relu)
                KernelAbstractions.synchronize(backend)
                got = Float32.(Array(out))
                @test maximum(abs, got .- max.(ref, 0f0)) / max(1f-3, maximum(abs, ref)) < 5e-3
            end
        end
    else
        @test_skip false
    end
end

# The gathering convolution computes what the materialising one computes.
#
# `ConvGather` replaces the im2col matrix with an address rule, so the property
# that matters is not "close to the definition" but "the same numbers": both paths
# hand the identical operand to the identical GEMM schedule, so they agree
# BIT-FOR-BIT and the assertion below is `==` rather than a tolerance. One host
# comparison is carried as well, because two paths agreeing says nothing if the
# shared half is wrong.
#
# **The shapes are chosen so that `plan.gathertiling` is set, and the test
# asserts that first.** Nothing in this file's other testsets reaches the gathering
# path — every shape there is either too small for a staged tiling or too wide in
# `Cout` — so a rule change that quietly stopped gathering would leave the whole
# kernel untested while every assertion still passed.
@testset "the gathered convolution is the materialised one" begin
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        ctx = DKC.Ctx(backend)
        rng = MersenneTwister(23)
        dev = KernelAbstractions.allocate
        # `Cout = 128` is the rule: one column block. The 1x1 cases are the
        # cheap ones; the 3x3 at 48x128 is the smallest spatial shape whose
        # plan still chooses a single GEMM plane, which is what gathering needs.
        for (Wid, Hei, Cin, Cout, KW, KH, st, pd, dl, bias, act) in
                ((24, 20, 16, 128, 1, 1, (1, 1), (0, 0), (1, 1), true,  :none),
                 (24, 20, 32, 128, 1, 1, (1, 1), (0, 0), (1, 1), false, :relu),
                 (48, 128,  8, 128, 3, 3, (1, 1), (1, 1), (1, 1), true,  :none),
                 (80, 80,   8, 128, 3, 3, (1, 1), (1, 1), (1, 1), false, :relu))
            OW = (Wid + 2pd[1] - dl[1] * (KW - 1) - 1) ÷ st[1] + 1
            OH = (Hei + 2pd[2] - dl[2] * (KH - 1) - 1) ÷ st[2] + 1
            xh = Float16.(randn(rng, Float32, Wid, Hei, Cin, 1) .* 0.3f0)
            wh = Float16.(randn(rng, Float32, KW, KH, Cin, Cout) .* 0.05f0)
            bh = bias ? Float16.(randn(rng, Float32, Cout) .* 0.2f0) : nothing

            x = DKC.toback(backend, xh)
            w = DKC.toback(backend, wh)
            b = bh === nothing ? nothing : DKC.toback(backend, bh)
            out = dev(backend, Float16, OW, OH, Cout, 1)

            pg = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w))
            pm = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w);
                                       gather = false)
            @test pg isa DKC.ConvCoopMatPlan
            # Without this the rest of the testset can pass vacuously.
            @test pg.gathertiling !== nothing
            @test pm.gathertiling === nothing

            fill!(out, Float16(0))
            DKC.convolution_coopmat!(ctx, out, pg, x, w, b, st, pd, dl; act)
            KernelAbstractions.synchronize(backend)
            gathered = Array(out)

            fill!(out, Float16(0))
            DKC.convolution_coopmat!(ctx, out, pm, x, w, b, st, pd, dl; act)
            KernelAbstractions.synchronize(backend)
            materialised = Array(out)

            @test gathered == materialised
        end

        # And that the shared half is right, on the cheapest gathering shape.
        Wid, Hei, Cin, Cout = 24, 20, 16, 128
        xh = Float16.(randn(rng, Float32, Wid, Hei, Cin, 1) .* 0.3f0)
        wh = Float16.(randn(rng, Float32, 1, 1, Cin, Cout) .* 0.05f0)
        bh = Float16.(randn(rng, Float32, Cout) .* 0.2f0)
        ref = convref(xh, wh, bh, (1, 1), (0, 0), (1, 1))
        x = DKC.toback(backend, xh); w = DKC.toback(backend, wh)
        b = DKC.toback(backend, bh)
        out = dev(backend, Float16, Wid, Hei, Cout, 1)
        plan = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w))
        @test plan.gathertiling !== nothing
        fill!(out, Float16(0))
        DKC.convolution_coopmat!(ctx, out, plan, x, w, b, (1, 1), (0, 0), (1, 1))
        KernelAbstractions.synchronize(backend)
        got = Float32.(Array(out))
        @test maximum(abs, got .- ref) / max(1f-3, maximum(abs, ref)) < 5e-3
    else
        @test_skip false
    end
end

# The channel pad is chosen by tile WIDTH and not by least padding, which is the
# one place this differs from `Mantle.gemm_padn`.
# The WIDE gather: one column block of 144 or 288 output channels
# (`Mantle.CONV_GATHER_TILINGS`), so no im2col and no padded channel. The
# materialised arm pads 144 to 256 and 288 to 384; at the VAE's sizes the two
# agree bit for bit (the table on `CONV_GATHER_TILINGS`), but a shape this small
# can put the materialised GEMM on split-K, which sums in another order, so the
# comparison is to fp32 reassociation and not `==`. A wrong address is O(1).
# Forced here: these shapes are far below the workgroup count the automatic
# rule wants, which is pinned separately below. Each is large enough that the
# gathering GEMM runs one plane, which it has to.
@testset "the wide gathered convolution is the materialised one" begin
    backend = first(Mantle.eachbackend())
    caps = DNNKernels.caps(backend)
    if DNNKernels.coopmatkernels(caps) && caps.coopmatsubgroup == 32
        ctx = DKC.Ctx(backend)
        rng = MersenneTwister(29)
        for (Wid, Hei, Cin, Cout, bias, cap) in
                ((40, 24, 16, 144, true, 1 << 30),
                 (64, 60, 16, 144, false, 1152 * 144 * 4),   # four chunks of 960
                 (50, 22, 32, 288, true, 1 << 30),           # 1100 pixels in 1152 rows
                 (72, 48, 32, 576, true, 1728 * 576 * 4),    # two column blocks, two chunks
                 (48, 24, 64, 1152, false, 1 << 30))         # four column blocks
            xh = Float16.(randn(rng, Float32, Wid, Hei, Cin, 1) .* 0.3f0)
            wh = Float16.(randn(rng, Float32, 3, 3, Cin, Cout) .* 0.1f0)
            bh = bias ? randn(rng, Float32, Cout) .* 0.2f0 : nothing
            ref = convref(xh, wh, bh, (1, 1), (1, 1), (1, 1))
            x = DKC.toback(backend, xh); w = DKC.toback(backend, wh)
            b = bh === nothing ? nothing : DKC.toback(backend, bh)
            out = KernelAbstractions.allocate(backend, Float32, Wid, Hei, Cout, 1)

            pg = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w);
                                       gather = true, im2colcap = cap)
            pm = DKC.conv_coopmat_plan(caps, Float16, Float16, size(out), size(w);
                                       gather = false)
            @test pg.gathertiling in Mantle.CONV_GATHER_TILINGS
            @test pg.CoutP == Cout                      # no padded channel
            @test Cout % Mantle.gemm_bn(pg.gathertiling) == 0
            @test DKC.planmp(pg) % Mantle.gemm_bm(pg.gathertiling) == 0
            @test pm.gathertiling === nothing
            cap < 1 << 30 && @test pg.rows < pg.NPQ

            fill!(out, 0f0)
            DKC.convolution_coopmat!(ctx, out, pg, x, w, b, (1, 1), (1, 1), (1, 1))
            KernelAbstractions.synchronize(backend)
            gathered = Array(out)
            fill!(out, 0f0)
            DKC.convolution_coopmat!(ctx, out, pm, x, w, b, (1, 1), (1, 1), (1, 1))
            KernelAbstractions.synchronize(backend)
            materialised = Array(out)
            @test maximum(abs, gathered .- materialised) / maximum(abs, materialised) < 1e-5
            @test maximum(abs, gathered .- ref) / maximum(abs, ref) < 5e-3
        end
    else
        @test_skip false
    end
end

# Twelve subgroups a workgroup: a chunk that launches too few of them per core
# loses to im2col (1152 -> 1152 over 64x64 on 40 cores: 172 workgroups, 6.21 ms
# against 5.55), and one that launches enough wins. Relative to the device's own
# core count, so it holds on any card.
@testset "the wide gather wants enough workgroups" begin
    caps = DNNKernels.caps(first(Mantle.eachbackend()))
    if DNNKernels.coopmatkernels(caps)
        # 1152 output channels at 96 x 288: four column blocks per 96 rows.
        enough = 2 * caps.cores * 96              # 8 per core exactly
        wide(npq) = DKC.convgather_wide(caps, npq, 1152 * 9, 1152 * 9, 1152, 1 << 30)
        @test wide(enough) isa DKC.ConvCoopMatPlan
        @test wide(enough).gathertiling == (3, 3, 2, 6, 32, 8)
        @test wide(enough - 96) === nothing
        @test DKC.convgather_wide(caps, enough ÷ 2, 1152 * 9, 1152 * 9, 1152, 1 << 30;
                                  force = true) isa DKC.ConvCoopMatPlan
        # 144 takes the 144-wide block and 576 the 288-wide one; 256 has neither.
        @test DKC.convgather_widetiling(144, 1312) == (3, 3, 4, 3, 32, 8)
        @test DKC.convgather_widetiling(576, 5184) == (3, 3, 2, 6, 32, 8)
        @test DKC.convgather_widetiling(256, 2304) === nothing
        # A reduction padded only to 16 has no 32-deep tiling.
        @test DKC.convgather_widetiling(144, 1296 + 8) === nothing
    else
        @test_skip false
    end
end

@testset "the channel pad reaches for the wider column tile" begin
    # `GEMM_TILINGS` and `gemm_bn` are the cooperative-matrix tilings, compiled
    # in only where that path is — the same thing `coopmatkernels` asks
    # `Mantle.staged_gemm_tile` about. `convcoutpad` is not arithmetic that
    # stands apart from them: it READS the table to find the widest admissible
    # column tile, so the whole testset needs the guard and not just the claim
    # about `bns`.
    if Mantle.staged_gemm_tile() !== nothing
        bns = sort(unique(Mantle.gemm_bn.(Mantle.GEMM_TILINGS)))
        @test 128 in bns
        # `MP` and `CRSP` chosen so every staged tiling is admissible.
        MP, CRSP = 65536, 2592
        @test DKC.convcoutpad(MP, 288, CRSP) == 384      # 1.33x at 17.3 TFLOP/s
        @test DKC.convcoutpad(MP, 144, CRSP) == 256      # 1.78x at 16.2, over 192 at 9.3
        @test DKC.convcoutpad(MP, 576, CRSP) == 640
        @test DKC.convcoutpad(MP, 1152, CRSP) == 1152    # already a column tile
        # Nothing within the budget, so the shape keeps its own extent.
        @test DKC.convcoutpad(MP, 100, CRSP) == 128
        @test DKC.convcoutpad(MP, 130, CRSP) == 192 ||
              DKC.convcoutpad(MP, 130, CRSP) == 256
        # A reduction axis no staged tiling divides admits nothing to pad onto.
        @test DKC.convcoutpad(MP, 288, 2590) == 288
    else
        @test_skip false
    end
end

# The chunks are even, so the last one is not mostly padding.
#
# `im2col` writes every row of its buffer and the GEMM multiplies every row of
# it, because `MP` is the leading dimension and there is nowhere to say "only
# this many rows are real". Taking the largest chunk the budget allows therefore
# puts the whole remainder in the last chunk AT FULL COST: the VAE's
# `144 -> 144 @ 1024²` ran ten chunks of 102144 and a last of 27136 doing
# 102144 rows of work, and `1152 -> 1152 @ 128²` ran two chunks of 12864 for
# 16384 pixels — 57% padding.
@testset "the im2col chunks are even" begin
    # See the note above: the capability check below is the one that decides.
    back = first(Mantle.eachbackend())
    dev = DNNKernels.caps(back)
    # `coopmatkernels`, not `caps.coopmat` alone: Metal reports cooperative
    # matrices (simdgroup matrices are real) while the staged GEMM these tests
    # exercise -- `GEMM_TILINGS`, `q8gemm_tiling`, `conv_coopmat_plan` -- is
    # compiled in only where Lava is. Asking the narrower question is what makes
    # the skip below fire instead of a `Decline(:host)` failing an assertion.
    if DNNKernels.coopmatkernels(dev) && dev.coopmatsubgroup == 32
        # `(Cin, Cout, H, W)` from the Qwen-Image 2.1 VAE decoder, the shapes
        # `tools/gap_vs_rocm.jl` tracks. Each is planned twice: materialised,
        # whose chunk is an im2col matrix of `GEMM_BLOCK`-row blocks, and as the
        # wide gather it takes by default, whose chunk is the fp32 result of the
        # tiling's own blocks. Both chunk under the same budget.
        for (Cin, Cout, H, W) in ((288, 288, 1024, 1024), (144, 144, 1024, 1024),
                                  (576, 576, 512, 512), (1152, 1152, 256, 256),
                                  (1152, 1152, 128, 128))
            x = KernelAbstractions.allocate(back, Float16, W, H, Cin, 1)
            w = KernelAbstractions.allocate(back, Float16, 3, 3, Cin, Cout)
            o = KernelAbstractions.allocate(back, Float16, W, H, Cout, 1)
            cap = DNNKernels.im2colbudget(x)
            pm = DNNKernels.conv_coopmat_plan(dev, Float16, Float16, size(o), size(w);
                                              im2colcap = cap, gather = false)
            pg = DNNKernels.conv_coopmat_plan(dev, o, x, w)
            @test pm isa DNNKernels.ConvCoopMatPlan && pm.gathertiling === nothing
            @test pg isa DNNKernels.ConvCoopMatPlan && pg.gathertiling !== nothing
            for (p, block, rowbytes) in
                    ((pm, DKC.GEMM_BLOCK, pm.CRSP * sizeof(Float16)),
                     (pg, Mantle.gemm_bm(pg.gathertiling), pg.CoutP * sizeof(Float32)))
                nchunk = cld(p.NPQ, p.rows)
                # Still a whole number of blocks (or every pixel in one chunk,
                # which `planmp` pads), still inside the budget, and still the
                # same number of chunks the budget forces.
                @test p.rows % block == 0 || p.rows == p.NPQ
                @test DNNKernels.planmp(p) * rowbytes <= cap
                @test nchunk == cld(p.NPQ, min(p.NPQ, (cap ÷ rowbytes ÷ block) * block))
                # And the last chunk is nearly full rather than nearly empty. The
                # deficit is at most one block per earlier chunk — rounding the
                # even size up to a block is the only slack — so the bound is
                # stated as a fraction, generously: every shape here measures
                # 92% or better, and the rule this replaced gave 27% on two.
                last = p.NPQ - (nchunk - 1) * p.rows
                @test 0 < last <= p.rows
                @test 4 * last >= 3 * p.rows
                @test last > p.rows - nchunk * block
            end
            x = w = o = nothing; GC.gc()
        end
    else
        @test_skip false
    end
end

# `col` is `Float16`, so one element a thread is a 2-byte store — half of what a
# lane can retire — and the column decomposition above it (four divisions and
# two remainders) is paid per element rather than per column. `IM2COL_VEC`
# threads write a PAIR. That is only sound while the pair stays inside one
# column, which is `MP % 2 == 0`; `padgemm` rounds to `GEMM_BLOCK = 192`, so it
# always is, and the launcher asks rather than relying on it.
@testset "the paired im2col store writes what the scalar one writes" begin
    # See the note at the top of this file.
    back = first(Mantle.eachbackend())
    dev = DNNKernels.caps(back)
    if dev.coopmat && dev.coopmatsubgroup == 32
        @test DKC.IM2COL_VEC == 2
        rng = MersenneTwister(0x1520c0)
        # Small enough to run whole, and chosen so the pad cases are all here:
        # a `Cin` the tile does not divide, a stem-shaped 3, and an odd spatial
        # extent so `NPQ` is not a multiple of anything.
        for (Cin, Cout, H, W) in ((144, 144, 12, 12), (288, 144, 9, 11),
                                  (3, 32, 13, 13), (40, 24, 8, 8))
            x = KernelAbstractions.allocate(back, Float16, W, H, Cin, 1)
            w = KernelAbstractions.allocate(back, Float16, 3, 3, Cin, Cout)
            o = KernelAbstractions.allocate(back, Float16, W, H, Cout, 1)
            plan = DNNKernels.conv_coopmat_plan(dev, o, x, w)
            plan isa DNNKernels.ConvCoopMatPlan || (x = w = o = nothing; continue)
            MP = DNNKernels.padgemm(plan.rows)
            @test MP % DKC.IM2COL_VEC == 0
            copyto!(x, Float16.(randn(rng, Float32, W, H, Cin, 1) .* 0.3f0))
            ntot = MP * plan.CRSP
            scalar = KernelAbstractions.allocate(back, Float16, MP, plan.CRSP)
            paired = KernelAbstractions.allocate(back, Float16, MP, plan.CRSP)
            for (dst, vec) in ((scalar, 1), (paired, DKC.IM2COL_VEC))
                fill!(dst, Float16(0))
                KernelInterface.Kernel(back, DNNKernels.im2col_kernel!)(dst, x, Val(MP), Val(vec),
                    Val(3), Val(3), Val(1), Val(1), Val(1), Val(1), Val(1), Val(1),
                    Val(W), Val(H), W, H, plan.rows, ntot, Cin, 0;
                    ndrange = cld(ntot, vec))
            end
            KernelAbstractions.synchronize(back)
            # BIT-identical, not close: the pair changes which thread writes a
            # cell, never what it writes.
            @test Array(scalar) == Array(paired)
            x = w = o = scalar = paired = nothing; GC.gc()
        end
    else
        @test_skip false
    end
end
