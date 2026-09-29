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
                # Global indices in Int32, as Mantle's narrow kernel forms them:
                # `splitidx` returns `Int`, and with an Int64 index Lava loads
                # only the first half of the pair `(B[g], B[g+1])`. The second
                # read as 0, every product lost half its k terms (and faulted
                # the device at 6144 x 4096 x 256), and Qwen-Image's
                # conditioner came out NaN.
                @inbounds for r in Int32(0):Int32($AR-1)
                    idx = tid + r*Int32($WG)
                    p, kk = map(Int32, Mantle.splitidx(idx, Val($(BM÷4))))
                    row = tm+Int32(4)*p
                    w = Q[Int32(1)+row÷Int32(4)+(k0+kk)*Int32(M÷4)]
                    a0 = Float16(q8byte(w,0)*scale0)
                    a1 = Float16(q8byte(w,1)*scale1)
                    a2 = Float16(q8byte(w,2)*scale2)
                    a3 = Float16(q8byte(w,3)*scale3)
                    sA[Int32(1)+Int32(2)*p+kk*Int32($LDA2)] = (VecElement(a0),VecElement(a1))
                    sA[Int32(2)+Int32(2)*p+kk*Int32($LDA2)] = (VecElement(a2),VecElement(a3))
                end
                @inbounds for r in Int32(0):Int32($BR-1)
                    idx = tid+r*Int32($WG)
                    p,j = map(Int32, Mantle.splitidx(idx,Val($(BK÷2))))
                    g = Int32(1)+k0+Int32(2)*p+(tn+j)*Int32(K)
                    sB[Int32(1)+p+j*Int32($LDB2)] = (VecElement(B[g]),VecElement(B[g+Int32(1)]))
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
    harnesslaunch!(KernelAbstractions.get_backend(C), Q8_GEMM_KERNELS[tiling],
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

# ── The pipelined kernel for wide products ─────────────────────────────────
#
# `q8gemm_*!` above dequantises `q * scale` to fp16 as it stages, loads the
# activations two bytes at a time, and stages the weight tile m-contiguous, so a
# weight fragment is sixteen 2-byte shared-memory reads. At Qwen-Image 2.1's
# products (n = 4224) that is 21-22 TFLOP/s against the vendor fp16 GEMM's 35:
# its k-loop issued 466 instructions per 16 WMMAs, the vendor's 7 per WMMA.
#
# This one differs in four ways, each measured at the 24576 x 4096 x 4224
# stacked gate+up product (ms, interleaved):
#
#     shipped q8gemm_2_4_2_2_32_8                     39.2
#     weights staged k-contiguous, 16-byte
#       activation loads, next block prefetched        33.5
#     exact int8 in staging, scale in the epilogue     31.9
#     prefetch carried as <4 x i32>                    31.5
#
# **The scale moves to the epilogue**, `C = scale[m] * (q B) + bias[m]`, which is
# both cheaper and more exact: an int8 is exact in fp16, and 0x6400|u is 1024+u,
# so a byte converts with an or and one packed subtract (`q8magicpair`) instead
# of a convert, a multiply and a pack per element; and `q * scale` is no longer
# rounded to fp16 before the product. Max error against the exact fp32 result
# 9.9e-4 against the shipped kernel's 1.5e-3, on outputs up to 3.5.
#
# **What bounds it now is not the matrix unit.** With every WMMA and fragment
# load removed the kernel still takes 31.7 ms of its 32.0; without the next
# block's global loads 24.5. Bigger tiles (128x128 on four or eight subgroups)
# and grouped launch orders were tried and are not faster.

const Q8_H2 = NTuple{2,VecElement{Float16}}
const Q8_H4 = NTuple{4,VecElement{Float16}}
const Q8_U4 = NTuple{4,VecElement{UInt32}}

@inline q8asH2(x::UInt32) = Base.llvmcall("""
    %r = bitcast i32 %0 to <2 x half>
    ret <2 x half> %r""", Q8_H2, Tuple{UInt32}, x)
"""`x .- 1152` on a packed pair: one `v_pk_add_f16`."""
@inline q8sub1152(x::Q8_H2) = Base.llvmcall("""
    %r = fsub <2 x half> %0, <half 0xH6480, half 0xH6480>
    ret <2 x half> %r""", Q8_H2, Tuple{Q8_H2}, x)
"""
Bytes at bit offsets `s0` of `lo` and `s1` of `hi`, already `xor 0x80`, as the
exact fp16 pair of their int8 values: `0x6400 | u` is `1024 + u` and `u` is the
int8 plus 128.
"""
@inline q8magicpair(lo::UInt32, s0, hi::UInt32, s1) =
    q8sub1152(q8asH2(((lo >> s0) & 0xff) | (((hi >> s1) & 0xff) << 16) | 0x64006400))
@inline q8catpairs(a::Q8_H2, b::Q8_H2) = (a[1], a[2], b[1], b[2])
"""The low and high 8 bytes of a 16-byte activation chunk, as staged halves."""
@inline q8h4lo(x::Q8_U4) = Base.llvmcall("""
    %s = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 0, i32 1>
    %r = bitcast <2 x i32> %s to <4 x half>
    ret <4 x half> %r""", Q8_H4, Tuple{Q8_U4}, x)
@inline q8h4hi(x::Q8_U4) = Base.llvmcall("""
    %s = shufflevector <4 x i32> %0, <4 x i32> poison, <2 x i32> <i32 2, i32 3>
    %r = bitcast <2 x i32> %s to <4 x half>
    ret <4 x half> %r""", Q8_H4, Tuple{Q8_U4}, x)

"""
    q8gemm_pipelined_kernel!(C, Q, S, B, bias, epi, Val(M), Val(N), Val(K), Val(cfg))

`C = epi(scale .* (q * B) .+ bias)` for packed int8 `Q` (`M/4 x K` words, four
rows a word) and fp16 `B` (`K x N`, `K` contiguous and 16-byte aligned). `cfg =
(STM, STN, WM, WN, BK)`: `WM x WN` subgroups of `16STM x 16STN` outputs each.
"""
function q8gemm_pipelined_kernel!(C, Q, S, B, bias, epi, ::Val{M}, ::Val{N}, ::Val{K},
                                  ::Val{CFG}) where {M,N,K,CFG}
    # (m, k) at m*(BK+8) + k and (k, n) at n*(BK+8) + k, in 8-byte vectors. Sizes
    # from the type parameters, never a local: see `flash.jl` on `@localmem`.
    sA = KI.localmemory(Q8_H4, Val((16 * CFG[1] * CFG[3] * (CFG[5] + 8) ÷ 4,)), Val(1))
    sB = KI.localmemory(Q8_H4, Val((16 * CFG[2] * CFG[4] * (CFG[5] + 8) ÷ 4,)), Val(2))
    STM, STN, WM, WN, BK = CFG
    BM, BN, WG = 16STM * WM, 16STN * WN, 32WM * WN
    RS4 = Int32((BK + 8) ÷ 4)
    RG = BM ÷ 4                        # row groups: one word is four rows at one k
    APT = RG * (BK ÷ 4) ÷ WG           # 4-row x 4-k weight blocks a thread stages
    CB = BK ÷ 8                        # 16-byte activation chunks in a column
    BPT = BN * CB ÷ WG                 # activation chunks a thread stages

    tid = Int32(KI.get_local_id().x) - Int32(1)
    blk = Int32(KI.get_group_id().x) - Int32(1)
    tm = (blk % Int32(M ÷ BM)) * Int32(BM)
    tn = (blk ÷ Int32(M ÷ BM)) * Int32(BN)
    sg = tid ÷ Int32(32)
    sm, sn = (sg % Int32(WM)) * Int32(STM), (sg ÷ Int32(WM)) * Int32(STN)
    # `WG` is a multiple of `RG`, so a thread's row group is the same in every block.
    rg = tid % Int32(RG)
    @inbounds begin
        Base.Cartesian.@nexprs 4 nt -> Base.Cartesian.@nexprs 4 mt -> if mt <= STM && nt <= STN
            c_mt_nt = zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
        end
        # Per-thread pointers, fixed for the loop; a block adds a uniform offset.
        Base.Cartesian.@nexprs 4 a -> if a <= APT
            kga_a = (tid + Int32((a - 1) * WG)) ÷ Int32(RG)
            pq_a = pointer(Q) + 4 * (Int(tm ÷ Int32(4) + rg) + Int(Int32(4) * kga_a) * (M ÷ 4))
        end
        Base.Cartesian.@nexprs 8 c -> if c <= BPT
            kc_c = (tid + Int32((c - 1) * WG)) % Int32(CB)
            col_c = (tid + Int32((c - 1) * WG)) ÷ Int32(CB)
            pb_c = reinterpret(Ptr{Q8_U4}, pointer(B) + 2 * (Int(tn + col_c) * K + Int(kc_c) * 8))
        end
        QSTEP = 4 * (M ÷ 4)            # bytes between k-columns of Q
        Base.Cartesian.@nexprs 4 a -> if a <= APT
            Base.Cartesian.@nexprs 4 j -> w_a_j = Core.Intrinsics.pointerref(pq_a + (j - 1) * QSTEP, 1, 4) ⊻ 0x80808080
        end
        Base.Cartesian.@nexprs 8 c -> if c <= BPT
            b_c = Core.Intrinsics.pointerref(pb_c, 1, 16)
        end
        for kb in Int32(0):Int32(K ÷ BK - 1)
            # Block kb, from registers into shared.
            Base.Cartesian.@nexprs 4 a -> if a <= APT
                Base.Cartesian.@nexprs 4 i -> sA[1 + (Int32(4) * rg + Int32(i - 1)) * RS4 + kga_a] =
                    q8catpairs(q8magicpair(w_a_1, 8(i - 1), w_a_2, 8(i - 1)),
                               q8magicpair(w_a_3, 8(i - 1), w_a_4, 8(i - 1)))
            end
            Base.Cartesian.@nexprs 8 c -> if c <= BPT
                sB[1 + col_c * RS4 + Int32(2) * kc_c] = q8h4lo(b_c)
                sB[2 + col_c * RS4 + Int32(2) * kc_c] = q8h4hi(b_c)
            end
            KI.barrier()
            # Block kb+1 loads while kb's products run.
            if kb + Int32(1) < Int32(K ÷ BK)
                k1 = Int(kb + Int32(1)) * BK
                Base.Cartesian.@nexprs 4 a -> if a <= APT
                    Base.Cartesian.@nexprs 4 j -> w_a_j = Core.Intrinsics.pointerref(pq_a + (k1 + j - 1) * QSTEP, 1, 4) ⊻ 0x80808080
                end
                Base.Cartesian.@nexprs 8 c -> if c <= BPT
                    b_c = Core.Intrinsics.pointerref(pb_c + 2k1, 1, 16)
                end
            end
            Base.Cartesian.@nexprs 4 u -> if u <= BK ÷ 16
                Base.Cartesian.@nexprs 4 mt -> if mt <= STM
                    a_mt = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}(sA,
                        1 + (sm + Int32(mt - 1)) * Int32(16) * RS4 + Int32((u - 1) * 4), RS4, Val(true))
                end
                Base.Cartesian.@nexprs 4 nt -> if nt <= STN
                    bm = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}(sB,
                        1 + (sn + Int32(nt - 1)) * Int32(16) * RS4 + Int32((u - 1) * 4), RS4, Val(false))
                    Base.Cartesian.@nexprs 4 mt -> if mt <= STM
                        c_mt_nt = muladd(a_mt, bm, c_mt_nt)
                    end
                end
            end
            KI.barrier()
        end
        # The row scale and the bias, as stride-0 matrices over the rows.
        Base.Cartesian.@nexprs 4 mt -> if mt <= STM
            r0_mt = Int32(1) + tm + (sm + Int32(mt - 1)) * Int32(16)
            fs_mt = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(pointer(S), r0_mt, 0, Val(false))
        end
        Base.Cartesian.@nexprs 4 nt -> Base.Cartesian.@nexprs 4 mt -> if mt <= STM && nt <= STN
            acc = Mantle.coopmat_mul(c_mt_nt, fs_mt)
            if bias !== nothing
                acc = Mantle.coopmat_add(acc, Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}(
                    pointer(bias), r0_mt, 0, Val(false)))
            end
            Mantle.accstore!(C, r0_mt + (tn + (sn + Int32(nt - 1)) * Int32(16)) * Int32(M), Int32(M), acc, epi)
        end
    end
    return nothing
end

"""
    q8gemm_pipelined_tile(caps, Tout, m, k, n) -> cfg | nothing

The pipelined kernel's tile for an `(m, k) x (k, n)` product, or `nothing` where
it is not the one to run. Wide products only: it was measured at `n = 4224`
against the tiles [`q8gemm_tile`](@ref) picks (ms):

    24576 x 4096 (gate+up)   39.2 -> 31.5   (2,4,2,2,32)
    12288 x 4096 (qkv)       22.0 -> 17.9   (2,4,2,2,32)
     4096 x 4096 (out)        6.3 ->  5.9   (2,4,2,2,32)
     4096 x 12288 (down)     19.5 -> 17.6   (4,2,2,2,32)

A narrow product (LLM decode, `n` of 16 to 512) keeps the older kernels, whose
tiles were measured there and which this was not.
"""
function q8gemm_pipelined_tile(caps, ::Type{Tout}, m::Integer, k::Integer, n::Integer) where {Tout}
    caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16 || return nothing
    Tout in (Float16, Float32) || return nothing
    n >= 1024 && n % 128 == 0 && k % 32 == 0 || return nothing
    max(m * k, k * n, m * n) <= typemax(Int32) || return nothing
    cfg = k >= 2m && m % 128 == 0 ? (4, 2, 2, 2, 32) : m % 64 == 0 ? (2, 4, 2, 2, 32) : nothing
    cfg === nothing && return nothing
    STM, STN, WM, WN, BK = cfg
    shared = 2 * (16STM * WM + 16STN * WN) * (BK + 8)
    32WM * WN <= caps.workgrouplimit && shared <= caps.sharedbudget || return nothing
    cfg
end

"""The launch for `q8gemm_pipelined_kernel!` at `cfg`."""
function q8gemm_pipelined_launch(C, A, B, bias, epi, m, n, k, cfg)
    STM, STN, WM, WN, BK = cfg
    BM, BN, WG = 16STM * WM, 16STN * WN, 32WM * WN
    (kern = q8gemm_pipelined_kernel!,
     args = (C, A.q, A.scale, B, bias, epi, Val(m), Val(n), Val(k), Val(cfg)),
     ndrange = (m ÷ BM) * (n ÷ BN) * WG, group = WG)
end

"""Run the pipelined product now; `cfg` from [`q8gemm_pipelined_tile`](@ref)."""
function q8gemm_pipelined!(C, A::QInt8Matrix, B; cfg, bias = nothing, epilogue = identity)
    m, k = size(A); n = size(B, 2)
    size(B, 1) == k && size(C) == (m, n) || throw(DimensionMismatch("q8gemm operands"))
    l = q8gemm_pipelined_launch(C, (q = Mantle.storage(A.q), scale = Mantle.storage(A.scale)),
                                B, bias, epilogue, m, n, k, cfg)
    harnesslaunch!(KernelAbstractions.get_backend(C), l.kern, l.args...;
                   ndrange = l.ndrange, workgroupsize = l.group)
    C
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
