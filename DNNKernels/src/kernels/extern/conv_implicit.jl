"""
Implicit-GEMM convolution.

Ported from ggml's Vulkan `conv2d_mm.comp` (dev/ggml-vulkan/ggml/src/ggml-vulkan/
vulkan-shaders/conv2d_mm.comp), which is the fastest portable-Vulkan 2-D
convolution we have a reference for. The direct kernel in `conv.jl` reads every
operand from global memory once per multiply-add and shares nothing between
threads; at 3x3 256->256 that is 2304 uncached loads per output element, and it
measures 18 GFLOP/s.

The convolution is one GEMM:

    C[K, NPQ] = A[K, CRS] * B[CRS, NPQ]

    K   = Cout                 output channels   (rows)
    CRS = Cin * KH * KW        reduction extent
    NPQ = N * OH * OW          output pixels     (columns)

`B` is never materialised — that is the "implicit" part. Each element is fetched
straight from `x` with the im2col index computed on the fly, so we pay none of
the KH*KW memory blow-up that an explicit im2col costs.

Reuse comes from staging both operands in shared memory and giving each thread a
TS_K x TS_NPQ register tile: one pass of the inner loop does TS_K + TS_NPQ shared
loads and TS_K * TS_NPQ multiply-adds.

Layout is the reversed one used throughout (`x` is `(W,H,Cin,N)`, `w` is
`(KW,KH,Cin,Cout)`, `out` is `(OW,OH,Cout,N)`), which happens to match ggml's
own memory order exactly, so the index arithmetic carries over unchanged apart
from 1-based offsets.

Tile sizes are `Val` parameters rather than constants because the right choice is
shape-dependent: ggml's 128x128 default would put this model's dominant
convolution (NPQ=120, K=256) on two workgroups. `convtiles` picks them.
"""

"""
    convtiles(K, NPQ) -> (BS_K, BS_NPQ, TS_K, TS_NPQ, WG)

Block and thread tile for a `K x NPQ` output. Chosen to keep enough workgroups in
flight to fill the device: a 128x128 block is only worth it when the output is
big enough to still produce many blocks, and this model's feature maps are small
(NPQ as low as 120), so we step down rather than launch two workgroups.

`WG == (BS_K / TS_K) * (BS_NPQ / TS_NPQ)` is required — one thread per output
element of the block tile.
"""
# The four block shapes ggml ships, as (BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ);
# see vk_conv_block_sizes and the WG_SIZE/TS_K rules at ggml-vulkan.cpp:511,5668.
# TS_NPQ is not free — it is BS_K*BS_NPQ/WG/TS_K, so every row here satisfies
# WG == (BS_K/TS_K) * (BS_NPQ/TS_NPQ). Note 64x32 is the odd one out: 128 threads
# and a deeper CRS block, not 256.
const CONV_SHAPE_128x128 = (128, 128, 16, 256, 8, 8)
const CONV_SHAPE_64x32   = ( 64,  32, 32, 128, 4, 4)
const CONV_SHAPE_32x256  = ( 32, 256, 16, 256, 8, 4)
const CONV_SHAPE_64x128  = ( 64, 128, 16, 256, 8, 4)

"""
    convtiles(K, NPQ; cores) -> (BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ)

Block shape for a `K x NPQ` output, ported from `ggml_vk_conv_select_shape`
(ggml-vulkan.cpp:10768). Take the largest tile that still yields at least two
tiles per shader core, gated on `K` so a narrow output does not get a block wider
than it is; otherwise fall back to 64x32.

`cores` is the shader-core count — 48 on the RTX 4000 Ada this was tuned against,
and `ctx.dev.cores` at the call site. ggml uses 32 as its placeholder when the
count cannot be queried, and so does the default here: `Lava.DeviceCaps` reports 0 when
the device will not say, and 0 would make every `>= 2cores` test trivially true
and always pick the widest block.
"""
function convtiles(K::Int, NPQ::Int; cores::Int = 32)
    cores <= 0 && (cores = 32)
    ntiles(s) = cld(K, s[1]) * cld(NPQ, s[2])
    if K > 64 && ntiles(CONV_SHAPE_128x128) >= 2cores
        CONV_SHAPE_128x128
    elseif K <= 32 && ntiles(CONV_SHAPE_32x256) >= 2cores
        CONV_SHAPE_32x256
    elseif K <= 64 && ntiles(CONV_SHAPE_64x128) >= 2cores
        CONV_SHAPE_64x128
    else
        CONV_SHAPE_64x32
    end
end

# `ACC` is the accumulator and shared-memory element type, and it is passed as a
# static parameter rather than computed in the body as `accum(eltype(x))`.
# `@localmem` with a type that is a *local variable* rather than a static
# parameter compiles without complaint and then silently writes nothing — the
# kernel runs and every output stays untouched. Worth remembering: it fails
# quietly, so it looks like an indexing bug.
@kernel function conv2d_igemm!(out, @Const(x), @Const(w), @Const(bias),
                               ::Val{ACC}, ::Val{SPLITK}, ::Val{ACT},
                               ::Val{BS_K}, ::Val{BS_CRS}, ::Val{BS_NPQ},
                               ::Val{TS_K}, ::Val{TS_NPQ},
                               ::Val{KW}, ::Val{KH},
                               ::Val{SX}, ::Val{SY}, ::Val{PX}, ::Val{PY},
                               ::Val{DX}, ::Val{DY},
                               Cin, Cout, Wid, Hei, OW, OH,
                               NPQ, CRS, NBN) where {ACC,SPLITK,ACT,BS_K,BS_CRS,BS_NPQ,
                                                     TS_K,TS_NPQ,KW,KH,SX,SY,PX,PY,DX,DY}
    # `@uniform`, not plain locals: these are read after a `@synchronize`, and
    # only `@uniform`/`@private` storage survives a barrier on the CPU backend.
    @uniform T = eltype(out)
    @uniform A = ACC

    # Workgroup-uniform: derived only from the block-shape parameters, so it is
    # the same for every item and the CPU backend can hoist it out of its
    # per-workitem loop.
    @uniform NT_K = BS_K ÷ TS_K
    @uniform NT_NPQ = BS_NPQ ÷ TS_NPQ
    @uniform WG = NT_K * NT_NPQ
    @uniform ArpWg = WG ÷ BS_CRS
    @uniform BrpWg = max(WG ÷ BS_NPQ, 1)
    @uniform Ash_stride = BS_CRS + 4
    @uniform Bsh_stride = BS_NPQ + 4
    @uniform nblk = cld(CRS, BS_CRS)
    # Split-K: the reduction is divided over SPLITK workgroups, which is the only
    # way to get both a large thread tile (arithmetic intensity) and enough
    # workgroups to fill the device. This model's dominant convolution is
    # 256 output channels over 120 pixels with a 2304-deep reduction — output
    # tiling alone yields 16 workgroups on 48 SMs, and shrinking the tile to get
    # more drops the MAC:load ratio from 16:8 to 2:3.
    @uniform blkper = cld(nblk, SPLITK)

    # +4 on the minor extent staggers rows across banks; without it every thread
    # in a pass hits the same bank on the A load. The 4 is written out literally
    # in both calls rather than shared via a local: `@localmem` miscompiles
    # silently — the kernel runs and writes nothing — if either its type or its
    # size expression involves a local variable rather than static parameters.
    Ash = @localmem ACC (BS_K * (BS_CRS + 4),)
    Bsh = @localmem ACC (BS_CRS * (BS_NPQ + 4),)

    # The accumulators live across `@synchronize`, so they must be `@private`;
    # a plain local is not carried over a barrier on the CPU backend. Sized
    # exactly TS_K x TS_NPQ so the unused lanes cost nothing.
    acc = @private ACC (TS_K, TS_NPQ)
    @inbounds Base.Cartesian.@nexprs 8 j -> Base.Cartesian.@nexprs 8 i -> begin
        if i <= TS_K && j <= TS_NPQ
            acc[i, j] = zero(A)
        end
    end

    for bb in 0:(blkper - 1)
        # Indices are recomputed in every barrier-separated region rather than
        # once at the top. KA's own tiled-matmul example does the same ("get
        # global values again"): a *derived* local does not survive a
        # `@synchronize`, because the CPU backend runs each region as its own
        # loop over workitems.
        # `@index(Local, NTuple)`, not `Linear`: on KA's CPU backend the Linear
        # form has no method inside a kernel containing `@synchronize` (only the
        # Cartesian/NTuple forms are defined there), which is why KA's own tiled
        # matmul example uses NTuple too. The workgroup is (WG, 1), so the first
        # component is the linear id.
        ltid, = @index(Local, NTuple)
        tid = ltid - 1
        bk, gn = @index(Group, NTuple)
        # the second grid axis carries both the NPQ block and the split index
        bnpq = (gn - 1) % NBN + 1
        ksplit = (gn - 1) ÷ NBN
        B_idx_K = (bk - 1) * BS_K
        B_idx_NPQ = (bnpq - 1) * BS_NPQ
        Ar = tid ÷ BS_CRS
        Ac = tid % BS_CRS
        Br = tid ÷ BS_NPQ
        Bc = tid % BS_NPQ

        # Blocks past the end make `crs_a`/`crs_b` exceed CRS, so the existing
        # bounds guards already stage zeros — the trip count stays uniform, which
        # `@synchronize` requires.
        b_crs = ksplit * blkper + bb

        # ── stage A (the kernel), BS_K x BS_CRS ───────────────────────────
        @inbounds begin
            crs_a = b_crs * BS_CRS + Ac
            cin_a = crs_a ÷ (KW * KH)
            rem_a = crs_a % (KW * KH)
            kh_a = rem_a ÷ KW
            kw_a = rem_a % KW
            r = 0
            while r < BS_K
                ky = r + Ar
                kidx = B_idx_K + ky
                Ash[ky * Ash_stride + Ac + 1] = (kidx < Cout && crs_a < CRS) ?
                    A(w[kw_a + 1, kh_a + 1, cin_a + 1, kidx + 1]) : zero(A)
                r += ArpWg
            end

            # ── stage B (the input), BS_CRS x BS_NPQ, gathered by im2col ───
            r = 0
            while r < BS_CRS
                by = r + Br
                npq = B_idx_NPQ + Bc
                n_idx = npq ÷ (OH * OW)
                npqr = npq - n_idx * OH * OW
                oh = npqr ÷ OW
                ow = npqr - oh * OW

                crs_b = b_crs * BS_CRS + by
                cin_b = crs_b ÷ (KW * KH)
                rem_b = crs_b % (KW * KH)
                kh_b = rem_b ÷ KW
                kw_b = rem_b % KW

                ix = ow * SX - PX + kw_b * DX
                iy = oh * SY - PY + kh_b * DY
                inb = (0 <= ix < Wid) && (0 <= iy < Hei) && npq < NPQ && crs_b < CRS
                Bsh[by * Bsh_stride + Bc + 1] =
                    inb ? A(x[ix + 1, iy + 1, cin_b + 1, n_idx + 1]) : zero(A)
                r += BrpWg
            end
        end

        @synchronize

        # ── the only hot loop: TS_K + TS_NPQ shared loads per TS_K*TS_NPQ MACs
        #
        # The register tile is *strided*, not blocked: thread T_x owns columns
        # T_x, T_x+NT_NPQ, ... rather than a contiguous run of TS_NPQ. Blocked
        # ownership makes neighbouring threads read TS_NPQ apart, which is a
        # TS_NPQ-way shared-memory bank conflict — measurably fatal at TS_NPQ=8.
        ctid, = @index(Local, NTuple)
        T_y = (ctid - 1) ÷ NT_NPQ
        T_x = (ctid - 1) % NT_NPQ
        @inbounds for k in 0:(BS_CRS - 1)
            Base.Cartesian.@nexprs 8 i -> a_i = i <= TS_K ?
                Ash[(T_y + (i - 1) * NT_K) * Ash_stride + k + 1] : zero(A)
            Base.Cartesian.@nexprs 8 j -> b_j = j <= TS_NPQ ?
                Bsh[k * Bsh_stride + T_x + (j - 1) * NT_NPQ + 1] : zero(A)
            # The guard belongs on the multiply-add, not just the operand loads.
            # Zeroing a_i/b_j past the tile gives the right answer but still
            # issues all 64 FMAs; TS_K/TS_NPQ are static, so this `if` folds.
            Base.Cartesian.@nexprs 8 j -> Base.Cartesian.@nexprs 8 i -> begin
                if i <= TS_K && j <= TS_NPQ
                    acc[i, j] = muladd(a_i, b_j, acc[i, j])
                end
            end
        end

        @synchronize
    end

    # ── write back ────────────────────────────────────────────────────────
    wtid, = @index(Local, NTuple)
    wbk, wgn = @index(Group, NTuple)
    W_y = (wtid - 1) ÷ NT_NPQ
    W_x = (wtid - 1) % NT_NPQ
    WB_K = (wbk - 1) * BS_K
    WB_NPQ = ((wgn - 1) % NBN) * BS_NPQ
    @inbounds Base.Cartesian.@nexprs 8 i -> begin
        if i <= TS_K
            kidx = WB_K + W_y + (i - 1) * NT_K
            if kidx < Cout
                bv = bias === nothing ? zero(A) : A(bias[kidx + 1])
                Base.Cartesian.@nexprs 8 j -> begin
                    if j <= TS_NPQ
                        npq = WB_NPQ + W_x + (j - 1) * NT_NPQ
                        if npq < NPQ
                            n_idx = npq ÷ (OH * OW)
                            npqr = npq - n_idx * OH * OW
                            oh = npqr ÷ OW
                            ow = npqr - oh * OW
                            if SPLITK == 1
                                v = acc[i, j] + bv
                                ACT === :relu && (v = max(v, zero(v)))
                                out[ow + 1, oh + 1, kidx + 1, n_idx + 1] = T(v)
                            else
                                # partial sums from each split accumulate; the
                                # host pre-fills the bias so it is added once
                                Atomix.@atomic out[ow + 1, oh + 1, kidx + 1,
                                                   n_idx + 1] += T(acc[i, j])
                            end
                        end
                    end
                end
            end
        end
    end
end

"""
    convsplit(nbk, nbn, nblk; cores) -> SPLITK

How many ways to split the reduction. Output tiling alone gives `nbk*nbn`
workgroups; split enough to reach roughly two per shader core, but never finer
than one CRS block per split (below that the splits have nothing to do) and
capped at 8, past which the atomic traffic on the output outweighs the extra
parallelism.
"""
function convsplit(nbk::Int, nbn::Int, nblk::Int; cores::Int = 48)
    # Already enough workgroups to fill the device: splitting then only buys the
    # cost of pre-filling the output and the atomic traffic. Measured on
    # 3x3 64->64 at 60x32 (60 workgroups), a 2-way split was 2.8x *slower*, while
    # 3x3 256->256 at 15x8 (16 workgroups) gained 4.3x from a 6-way split.
    nbk * nbn >= cores && return 1
    want = cld(2cores, max(nbk * nbn, 1))
    clamp(min(want, nblk), 1, 8)
end

"""
    convolution_igemm!(ctx, out, x, w, bias, stride, padding, dilation) -> out

Implicit-GEMM path. Dense (`groups == 1`) convolutions only; the grouped case
still goes through the direct kernel, and this model has none.
"""
function convolution_igemm!(ctx, out, x, w, bias, stride, padding, dilation; act::Symbol=:none)
    KWk, KHk, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = Cin * KHk * KWk
    NPQ = N * OH * OW
    # The shader-core count comes off the context, not from the literal 48 this
    # was tuned against: the whole rule is "at least two tiles per core", so on a
    # part with a different count it picks the wrong block outright.
    BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ = convtiles(Cout, NPQ; cores = ctx.dev.cores)
    nbk = cld(Cout, BS_K)
    nbn = cld(NPQ, BS_NPQ)
    splitk = convsplit(nbk, nbn, cld(CRS, BS_CRS))
    backend = ctx.backend
    # Split-K accumulates with `Atomix.@atomic +=` on the destination, and
    # Vulkan 1.3 has no fp16 atomic add (`AtomicFloat16AddEXT` is not in the
    # allowed capability set), so an fp16 output accumulates into an fp32 scratch
    # and is converted once at the end. Two extra dispatches against a 4-8x win
    # from the split, so it stays worth it.
    #
    # A fused activation forces the same detour even for an fp32 output: each
    # split contributes a *partial* sum, so applying `relu` inside the write-back
    # would clamp partial sums independently and silently give the wrong answer.
    # It has to be applied once, after the splits have been summed.
    acc = (splitk > 1 && (eltype(out) !== Float32 || act !== :none)) ?
          KernelAbstractions.allocate(backend, Float32, size(out)...) : out
    if splitk > 1
        # every split accumulates atomically, so the destination has to start at
        # the bias (or zero) rather than being overwritten
        bias === nothing ? fill!(acc, zero(eltype(acc))) :
            (acc .= reshape(bias, ntuple(i -> i == 3 ? length(bias) : 1, ndims(acc))))
    end
    # Only fold the activation into the write-back when there is a single split;
    # otherwise it is applied in the conversion pass below.
    kact = (splitk == 1 && act === :relu) ? :relu : :none
    conv2d_igemm!(backend, (WG, 1))(
        acc, x, w, bias, Val(accum(eltype(x))), Val(splitk), Val(kact),
        Val(BS_K), Val(BS_CRS), Val(BS_NPQ), Val(TS_K), Val(TS_NPQ),
        Val(KWk), Val(KHk),
        Val(stride[1]), Val(stride[2]), Val(padding[1]), Val(padding[2]),
        Val(dilation[1]), Val(dilation[2]),
        Cin, Cout, Wid, Hei, OW, OH, NPQ, CRS, nbn;
        ndrange = (nbk * WG, nbn * splitk))
    if acc !== out
        act === :relu ? (out .= max.(acc, zero(eltype(acc)))) : (out .= acc)
    elseif splitk > 1 && act === :relu
        out .= max.(out, zero(eltype(out)))
    end
    out
end
