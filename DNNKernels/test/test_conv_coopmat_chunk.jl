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

# The channel pad is chosen by tile WIDTH and not by least padding, which is the
# one place this differs from `Mantle.gemm_padn`.
@testset "the channel pad reaches for the wider column tile" begin
    # `GEMM_TILINGS` and `gemm_bn` are the cooperative-matrix tilings, compiled
    # in only where that path is — the same thing `coopmatkernels` asks with
    # `isdefined(M, :GEMM_TILINGS)`. `convcoutpad` is not arithmetic that stands
    # apart from them: it READS the table to find the widest admissible column
    # tile, so the whole testset needs the guard and not just the claim about
    # `bns`.
    if isdefined(Mantle, :GEMM_TILINGS)
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
        GB = DKC.GEMM_BLOCK
        # `(Cin, Cout, H, W)` from the Qwen-Image 2.1 VAE decoder, the shapes
        # `tools/gap_vs_rocm.jl` tracks.
        for (Cin, Cout, H, W) in ((288, 288, 1024, 1024), (144, 144, 1024, 1024),
                                  (576, 576, 512, 512), (1152, 1152, 256, 256),
                                  (1152, 1152, 128, 128))
            x = KA.allocate(back, Float16, W, H, Cin, 1)
            w = KA.allocate(back, Float16, 3, 3, Cin, Cout)
            o = KA.allocate(back, Float16, W, H, Cout, 1)
            p = DNNKernels.conv_coopmat_plan(dev, o, x, w)
            @test p isa DNNKernels.ConvCoopMatPlan
            cap = DNNKernels.im2colbudget(x)
            nchunk = cld(p.NPQ, p.rows)
            # Still a whole number of GEMM blocks, still inside the budget, and
            # still the same number of chunks the budget forces.
            @test p.rows % GB == 0
            @test DNNKernels.padgemm(p.rows) * p.CRSP * sizeof(Float16) <= cap
            @test nchunk == cld(p.NPQ, min(p.NPQ, (cap ÷ (p.CRSP * 2) ÷ GB) * GB))
            # And the last chunk is nearly full rather than nearly empty. The
            # deficit is at most one `GEMM_BLOCK` per earlier chunk — rounding
            # the even size up to a block is the only slack — so the bound is
            # stated as a fraction, generously: every shape here measures 92% or
            # better, and the rule this replaced gave 27% on two of them.
            last = p.NPQ - (nchunk - 1) * p.rows
            @test 0 < last <= p.rows
            @test 4 * last >= 3 * p.rows
            @test last > p.rows - nchunk * GB
            x = w = o = nothing; GC.gc()
        end
    else
        @test_skip false
    end
end
