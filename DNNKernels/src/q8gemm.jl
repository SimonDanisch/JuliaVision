# Dequantize packed weights directly into a cooperative-matrix tile.
const Q8_GEMM_KERNELS = Dict{NTuple{6,Int},Any}()
for cfg in ((2, 1, 2, 2, 32, 8), (2, 2, 2, 2, 32, 8), (2, 2, 2, 2, 64, 8),
            (2, 1, 2, 2, 64, 8), (2, 4, 2, 2, 32, 8), (4, 2, 2, 2, 32, 8),
            (4, 2, 2, 4, 32, 8), (4, 2, 2, 4, 64, 8),
            (4, 2, 4, 2, 32, 8), (4, 2, 2, 4, 32, 16),
            (2, 1, 2, 1, 32, 8), (2, 1, 4, 1, 32, 8))
    STM, STN, WM, WN, BK, PAD = cfg
    BM, BN, WG = 16STM*WM, 16STN*WN, 32WM*WN
    LDA2, LDB2 = (BM+PAD)÷2, (BK+PAD)÷2
    AR, BR = BM*BK÷4÷WG, BK*BN÷2÷WG
    name = Symbol("q8gemm_", join(cfg, "_"), "!")
    @eval begin
        function $name(
                C, Q, S, B, bias, epi,
                ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
            sA = KI.localmemory(Mantle.GemmV2, Val(($LDA2*$BK,)), Val(1))
            sB = KI.localmemory(Mantle.GemmV2, Val(($LDB2*$BN,)), Val(2))
            tid = Int32(KI.get_local_id().x) - Int32(1)
            blk = Int32(KI.get_group_id().x) - Int32(1)
            tm = (blk % Int32(M÷$BM)) * Int32($BM)
            tn = (blk ÷ Int32(M÷$BM)) * Int32($BN)
            sg = tid ÷ Int32(32)
            sm, sn = (sg % Int32($WM))*Int32($STM), (sg ÷ Int32($WM))*Int32($STN)
            sr = tm + Int32(4)*(tid % Int32($(BM÷4)))
            @inbounds begin
                scale0 = S[sr+1]
                scale1 = S[sr+2]
                scale2 = S[sr+3]
                scale3 = S[sr+4]
            end
            bp = bias === nothing ? bias : pointer(bias)
            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                c_mt_nt = Mantle.accinit(bias, bp, Int32(1)+tm+(sm+Int32(mt-1))*Int32(16))
            for kb in Int32(0):Int32(K÷$BK-1)
                k0 = kb*Int32($BK)
                @inbounds for r in Int32(0):Int32($AR-1)
                    idx = tid + r*Int32($WG)
                    p, kk = Mantle.splitidx(idx, Val($(BM÷4)))
                    row = tm+4p
                    w = Q[Int32(1)+row÷Int32(4)+(k0+kk)*Int32(M÷4)]
                    a0 = Float16(q8byte(w,0)*scale0)
                    a1 = Float16(q8byte(w,1)*scale1)
                    a2 = Float16(q8byte(w,2)*scale2)
                    a3 = Float16(q8byte(w,3)*scale3)
                    sA[Int32(1)+2p+kk*Int32($LDA2)] = (VecElement(a0),VecElement(a1))
                    sA[Int32(2)+2p+kk*Int32($LDA2)] = (VecElement(a2),VecElement(a3))
                end
                @inbounds for r in Int32(0):Int32($BR-1)
                    idx = tid+r*Int32($WG)
                    p,j = Mantle.splitidx(idx,Val($(BK÷2)))
                    g = Int32(1)+k0+2p+(tn+j)*Int32(K)
                    sB[Int32(1)+p+j*Int32($LDB2)] = (VecElement(B[g]),VecElement(B[g+1]))
                end
                KI.barrier()
                Base.Cartesian.@nexprs $(BK÷16) u -> begin
                    kt = Int32((u-1)*16)
                    Base.Cartesian.@nexprs $STM mt -> begin
                        a = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(
                            sA, Int32(1)+(sm+Int32(mt-1))*Int32(8)+kt*Int32($LDA2),Int32($LDA2))
                        Base.Cartesian.@nexprs $STN nt -> begin
                            b = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(
                                sB,Int32(1)+kt÷Int32(2)+(sn+Int32(nt-1))*Int32(16*$LDB2),Int32($LDB2))
                            c_mt_nt = muladd(a,b,c_mt_nt)
                        end
                    end
                end
                KI.barrier()
            end
            Base.Cartesian.@nexprs $STN nt -> Base.Cartesian.@nexprs $STM mt ->
                Mantle.accstore!(C,Int32(1)+tm+(sm+Int32(mt-1))*Int32(16)+
                    (tn+(sn+Int32(nt-1))*Int32(16))*Int32(M),Int32(M),c_mt_nt,epi)
            return nothing
        end
        Q8_GEMM_KERNELS[$cfg] = $name
    end
end

function q8gemm!(C, A::QInt8Matrix, B; tiling=(2,1,2,2,32,8), bias=nothing, epilogue=identity)
    m,k = size(A); n = size(B,2)
    size(B,1) == k && size(C) == (m,n) || throw(DimensionMismatch("q8gemm operands"))
    eltype(C) in (Float16,Float32) || throw(ArgumentError("q8gemm requires floating-point output"))
    stm,stn,wm,wn,bk,pad = tiling
    bm,bn,wg = 16stm*wm,16stn*wn,32wm*wn
    m%bm == 0 && n%bn == 0 && k%bk == 0 || throw(ArgumentError("q8gemm tile does not divide operands"))
    # `Mantle.storage` on the weight's fields: they are `Mantle.Buffer`s, and a
    # bare launch packs its arguments with no graph to resolve them against.
    KI.Kernel(KernelAbstractions.get_backend(C), Q8_GEMM_KERNELS[tiling])(
        C,Mantle.storage(A.q),Mantle.storage(A.scale),B,bias,epilogue,Val(m),Val(n),Val(k);
        ndrange=(m÷bm)*(n÷bn)*wg, workgroupsize=wg)
    C
end

"""
    q8gemm_tile(m, k, n) -> cfg

Which registered tile runs an `(m, k) x (k, n)` int8 product. Shape in, tile
out, nothing else: the device caps and operand types are
[`q8gemm_tiling`](@ref)'s business, and keeping this pure is what lets a test
sweep every shape the branches distinguish.

`n`, the number of activation columns, splits the decision before the weight's
aspect ratio does: narrow n cannot fill a 128-column tile at all, and wide n
changes which direction is worth blocking. Measured with `tools/bench_q8_tiles.jl`
(recorded replay -- an immediate launch hides the differences behind its own
dispatch cost) on the four shapes one Horizon layer runs, in ms:

                     m, k  |  n=16   32     64    128    256    512
    gate+up   53248,  5120 | 1.67  2.04   1.90   2.98   6.52  12.37
    qkv       10240,  5120 | 0.27  0.31   0.37   0.54   1.09   2.29
    attn out   5120,  8192 | 0.26  0.32   0.43   0.57   0.91   1.83
    ffn down   5120, 26624 | 1.19  1.04   1.31   2.22   3.29   6.78

Those are this function's picks, and they are the per-shape optimum everywhere
except three points it gives up 2-6% on rather than grow another branch (qkv at
16 and 256, attn out at 128). What it replaced ran 3.38, 3.90, 5.59, 7.33,
13.03 and 28.20 ms summed per layer against 3.38, 3.71, 4.01, 6.31, 11.80 and
23.27 here. The previous rule sent everything with `n % 128 == 0` to a
128x128 tile padded by 16; no measured shape wanted that padding.

Every branch must pick a tile that divides `m` and `n` -- `q8gemm!` throws
otherwise, and this is the only place that knows the shape.
"""
function q8gemm_tile(m::Integer, k::Integer, n::Integer)
    deep = k >= 2m          # a long reduction, like the FFN down projection
    # A stacked projection, like SwiGLU's gate+up. The bound is `6k` and not
    # `8k` because Qwen-Image 2.1's stacked gate+proj is `24576 x 4096`, exactly
    # `6k`, and fell through to `m % 256 == 0`'s `(4,2,4,2)` — the slowest of
    # the three plausible tiles at that shape in each of three sweeps
    # (42.86, 43.28, 42.74 ms), against `(2,4,2,2)`'s 39.59, 39.36 and **37.92**
    # interleaved. Horizon's own picks are unchanged: its gate+up is `10.4k` and
    # was already over the bound, and nothing else the chooser is pinned on
    # reaches this branch.
    #
    # Measured on the ISOLATED product rather than in a layer, because the
    # session that found it could no longer place a layer-sized plan. The
    # ranking between the two challengers moved between sweeps; the gap to the
    # tile being replaced did not.
    tall = m >= 6k
    if n%32 != 0
        m%128 == 0 ? (2,1,4,1,32,8) : (2,1,2,1,32,8)
    elseif n%64 != 0
        # 32 columns: one 16-wide tile per subgroup pair, and a long reduction
        # would rather have more blocks than wider ones (1.04 against 1.23).
        deep ? (2,1,2,1,32,8) : (2,1,2,2,k >= 16384 && k%64 == 0 ? 64 : 32,8)
    elseif n%128 != 0
        # 64 columns is the crossover: anything deeper than it is wide still
        # wants the narrow tile with the deep k-block (ffn down 1.31 against
        # 2.08), while a wide weight already wants the 256-row one (1.90
        # against 2.48).
        if k > m
            (2,1,2,2,k%64 == 0 ? 64 : 32,8)
        elseif m%256 == 0
            (4,2,4,2,32,8)
        elseif m%128 == 0
            (4,2,2,2,32,8)
        else
            (2,2,2,2,32,8)
        end
    elseif tall
        # 128 columns exactly is one tile wide, so the rows carry the blocking;
        # past that the 64-row/128-column tile wins (6.52 against 7.83).
        n == 128 && m%128 == 0 ? (4,2,2,4,32,8) : (2,4,2,2,32,8)
    elseif deep && n >= 512 && m%128 == 0
        (4,2,2,2,32,8)
    elseif m%256 == 0
        (4,2,4,2,32,8)
    elseif m%128 == 0
        (4,2,2,2,32,8)
    else
        (2,2,2,2,32,8)
    end
end

"""The tile [`q8gemm_tile`](@ref) picks, or `nothing` to leave the int8 weight
to the dequantise-and-reuse path: wrong operand types, no cooperative matrix, an
extent the tiles cannot divide, or a tile this device has no room for."""
function q8gemm_tiling(dev, A, B, C)
    # `islavaarray` and not `isa Mantle.LavaArray`: the type exists only where that
    # backend is loaded, so naming it here made this refuse every operand on any
    # other one. The shape and element type are asked separately for the same reason.
    # `islavaarray` resolves a `Mantle.Buffer` itself; `eltype`, `ndims` and
    # `size` answer on one directly.
    islavaarray(B) && eltype(B) === Float16 && ndims(B) == 2 && islavaarray(C) ||
        return nothing
    q8gemm_tiling(dev, eltype(C), size(A)..., size(B, 2))
end

"""
    q8gemm_tiling(caps, Tout, m, k, n) -> cfg | nothing

The same decision from extents alone, which is all a DECLARATION has: the
declared form holds transients rather than arrays, so it cannot ask an operand
what type it is, and the two paths must not answer differently.
"""
function q8gemm_tiling(caps, ::Type{Tout}, m::Integer, k::Integer, n::Integer) where {Tout}
    caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16 || return nothing
    Tout in (Float16, Float32) || return nothing
    m%64 == 0 && k%32 == 0 && n%16 == 0 || return nothing
    max(m*k, k*n, m*n) <= typemax(Int32) || return nothing
    cfg = q8gemm_tile(m, k, n)
    stm,stn,wm,wn,bk,pad = cfg
    wg = 32wm*wn
    shared = 2*((16stm*wm+pad)*bk+(bk+pad)*16stn*wn)
    wg <= caps.workgrouplimit && shared <= caps.sharedbudget || return nothing
    cfg
end

"""
    q8gemm_columns(n) -> np

The column count to run a packed int8 product at: `n`, padded to the widest
column tile once the product is wide enough to want one.

Every tile has to divide `n`, and a wide product's column count is whatever the
caller's sequence happens to be. Qwen-Image 2.1 at 1024² over a 22-token prompt
runs `n = 4118`, which is `2 x 29 x 71`: nothing divides it, so the GEMM fell
out of this path entirely and dequantised 100 MB of weights per product instead.
Even `n = 4128` only admits a 32-column tile. Measured at `12288 x 4096`:

    n = 4118   11.27 TOP/s   (dequantise, then the fp16 GEMM)
    n = 4128    8.60 TOP/s   (32-column tile, the widest that divides it)
    n = 4096   21.61 TOP/s   (128-column tile)

so padding 4118 up to 4224 buys 2x and costs 2.6% of the arithmetic plus two
copies of the activation. Below 256 columns the wide tiles cannot fill anyway
and the padding would be the whole cost, so a narrow product keeps the tile its
own extent admits.
"""
q8gemm_columns(n::Integer) = n >= 256 ? cld(n, 128) * 128 : cld(n, 16) * 16
