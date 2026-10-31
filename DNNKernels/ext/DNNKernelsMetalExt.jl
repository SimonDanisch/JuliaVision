module DNNKernelsMetalExt

# PTQ1 times Float16 columns on Metal's tensor units, decoding each weight once per
# column tile. A workgroup is one 64-row PTQ1 tile and TN columns: per 128-wide block
# it decodes the tile's weights into threadgroup memory as fp16 (a trit times its fp16
# scale is exact), and `matmul2d` multiplies them against the block's rows of `x`, read
# straight from device memory, accumulating in a cooperative tensor (registers) that
# reaches `out` once. Against the expansion route (decode the matrix to fp16 in device
# memory, then the native GEMM over it), on an M5 and Bonsai's 17408x5120 FFN gate: 0.88
# against 3.89 ms at 32 columns, 2.36 against 4.56 at 128, 9.35 against 10.15 at 512;
# from about 768 the expansion's one-shot product is ahead (18.7 against 18.0 at 1024).

import DNNKernels, Mantle
import KernelInterface as KI
import Metal
using Metal: MtlInlineTensor, TensorOpsMatmul2D, MtlCooperativeTensor, matmul2d_descriptor,
             matmul2d_multiply_accumulate

const TM = DNNKernels.PTQ1_TILE
const NSIMD = 4
const WG = 32 * NSIMD

DNNKernels.ptq1_tensor_available(::Mantle.MetalDevice) = Metal.tensor_matmul_capable()

# Thread t owns row t & 63 of the tile and words q0 = t >> 6, q0 + 2 and q0 + 4 of each
# block, and word 6 too when q0 == 0; word 6 also carries the block's scale, which is
# why every thread loads it.
@inline _words(data32, base::Int32, q0::Int32) =
    @inbounds (data32[base + Int32(TM) * q0], data32[base + Int32(TM) * (q0 + Int32(2))],
               data32[base + Int32(TM) * (q0 + Int32(4))], data32[base + Int32(6 * TM)])

# The tile is K-contiguous, row r at r * LDA halves, and `matmul2d` reads it transposed.
# Bytes 0-15 of a block hold elements byte + 16n, bytes 16-23 hold 80 + (byte - 16) + 8n,
# bytes 24-25 four each at 120 + (byte - 24) + 2n; so digit n of a word's four bytes is
# four CONSECUTIVE elements, one 8-byte store, where a column-major tile took four
# 2-byte ones: 10.5 against 9.7 ms at 512 columns, 2.74 against 2.43 at 128. The row
# stride is padded off a multiple of 128: unpadded, the 32 rows a store instruction
# covers fall in one bank, and 48 columns took 1.70 ms against 1.21.
const LDA = 136
const H4 = NTuple{4, VecElement{Float16}}
const H2 = NTuple{2, VecElement{Float16}}

# Word q's trits, scaled, into row r. The digit chain is fp16: y = 3f - 1, floor is the
# trit, a trit times its fp16 scale is exact.
@inline function _decode_word!(s4, s2, r::Int32, q::Int32, word::UInt32, scale::Float16)
    if q < Int32(6)
        f = ntuple(j -> Float16((word >> UInt32(8 * (j - 1))) & 0xff) * Float16(1 / 256), Val(4))
        e0 = q < Int32(4) ? Int32(4) * q : Int32(64) + Int32(4) * q
        st = q < Int32(4) ? Int32(16) : Int32(8)
        Base.Cartesian.@nexprs 5 n -> begin
            y = map(v -> muladd(v, Float16(3), Float16(-1)), f)
            t = map(floor, y)
            f = map(-, y, t)
            @inbounds s4[(r * Int32(LDA) + e0 + st * Int32(n - 1)) >> Int32(2) + Int32(1)] =
                map(v -> VecElement(v * scale), t)
        end
    else
        f = ntuple(j -> Float16((word >> UInt32(8 * (j - 1))) & 0xff) * Float16(1 / 256), Val(2))
        Base.Cartesian.@nexprs 4 n -> begin
            y = map(v -> muladd(v, Float16(3), Float16(-1)), f)
            t = map(floor, y)
            f = map(-, y, t)
            @inbounds s2[(r * Int32(LDA) + Int32(120 + 2 * (n - 1))) >> Int32(1) + Int32(1)] =
                map(v -> VecElement(v * scale), t)
        end
    end
    return nothing
end

@inline function _decode_block!(s4, s2, r::Int32, q0::Int32, w)
    scale = reinterpret(Float16, UInt16(w[4] >> 16))
    _decode_word!(s4, s2, r, q0, w[1], scale)
    _decode_word!(s4, s2, r, q0 + Int32(2), w[2], scale)
    _decode_word!(s4, s2, r, q0 + Int32(4), w[3], scale)
    q0 == Int32(0) && _decode_word!(s4, s2, r, Int32(6), w[4], scale)
    return nothing
end

"""
    ptq1_tensor_kernel!(out, data, x, M, K, N, c0, Val(TN), Val(NA))

`out = A * x` for tiled PTQ1 `A` (`M x K`) and `N` Float16 columns `x` (`K x N`), into
`out` (`M x N`), columns `c0` (zero-based) onward. One workgroup of 128 a 64 x (NA TN)
tile of `out`, in NA accumulators of TN columns that share each decoded block.

Two of 64 against one of 128, on the FFN gate: 2.33 against 2.46 ms at 128 columns,
9.27 against 9.72 at 512; four of 64, or of 32, lose (11.3 at 512, 2.92 at 128).
Decoding into alternating halves of a 2 x 64 x 128 buffer, one barrier a block instead
of two, was slower at every width: 1.12 against 0.77 ms at 8 columns, 12.7 against
10.6 at 512.
"""
function ptq1_tensor_kernel!(out, data, x, M::Int32, K::Int32, N::Int32, c0::Int32,
                             ::Val{TN}, ::Val{NA}) where {TN,NA}
    data32 = reinterpret(UInt32, data)
    sA = KI.localmemory(Float16, Val((TM * LDA,)), Val(1))
    s4 = reinterpret(H4, sA)
    s2 = reinterpret(H2, sA)
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    tiles = (M + Int32(TM - 1)) ÷ Int32(TM)
    tile = wg % tiles
    col0 = c0 + (wg ÷ tiles) * Int32(NA * TN)
    blocks = K >> Int32(7)
    r = t & Int32(TM - 1)
    q0 = t >> Int32(6)
    base = tile * blocks * Int32(7 * TM) + r + Int32(1)
    QK = Int32(DNNKernels.PTQ1_QK)
    tA = MtlInlineTensor(sA, (QK, Int32(TM)), (Int32(1), Int32(LDA)))
    tB = MtlInlineTensor(x, (K, N))
    op = TensorOpsMatmul2D{matmul2d_descriptor(TM, TN, DNNKernels.PTQ1_QK; transpose_left = true,
                                               mode = matmul2d_multiply_accumulate), Int32(NSIMD)}()
    accs = ntuple(_ -> MtlCooperativeTensor{Float32}(op, Float16, Float16), Val(NA))
    w = _words(data32, base, q0)
    for b in Int32(0):(blocks - Int32(1))
        _decode_block!(s4, s2, r, q0, w)
        b + Int32(1) < blocks && (w = _words(data32, base + (b + Int32(1)) * Int32(7 * TM), q0))
        KI.barrier()
        _products!(op, tA, tB, accs, b * QK + Int32(1), col0, N, Val(TN))
        KI.barrier()
    end
    row0 = tile * Int32(TM)
    Base.Cartesian.@nexprs 2 j -> if j <= NA
        c = col0 + Int32((j - 1) * TN)
        c < N && _store!(out, accs[j], M, N, row0, c, min(Int32(TN), N - c))
    end
    return nothing
end

# Block k0's product into each accumulator whose columns start inside `x`; the last
# workgroup's second one may not.
@inline function _products!(op, tA, tB, accs::NTuple{NA}, k0::Int32, col0::Int32, N::Int32,
                            ::Val{TN}) where {NA,TN}
    QK = Int32(DNNKernels.PTQ1_QK)
    Base.Cartesian.@nexprs 2 j -> if j <= NA
        c = col0 + Int32((j - 1) * TN)
        c < N && op(tA, view(tB, (k0, c + Int32(1)), (QK, min(Int32(TN), N - c))), accs[j])
    end
    return nothing
end

# A Float32 `out` takes the tile in one store, clipped to it; any other element type goes
# value by value, converted.
@inline function _store!(out::AbstractArray{Float32}, acc, M, N, row0, col0, ncols)
    tO = MtlInlineTensor(out, (M, N))
    copyto!(view(tO, (row0 + Int32(1), col0 + Int32(1)), (min(Int32(TM), M - row0), ncols)), acc)
    return nothing
end

@inline function _store!(out::AbstractArray{T}, acc, M, N, row0, col0, ncols) where {T}
    for i in 1:length(acc)
        if isvalid(acc, i)
            rr, cc = Metal.coordinate(acc, i)
            row = row0 + rr - Int32(1)
            col = col0 + cc - Int32(1)
            (row < M && cc <= ncols) && (@inbounds out[row + col * M + Int32(1)] = T(acc[i]))
        end
    end
    return nothing
end

function DNNKernels.ptq1_mm_tensor!(::Mantle.MetalDevice, g, out, A::DNNKernels.PTQ1Matrix, x,
                                    first::Integer, N::Integer; name::AbstractString)
    M, K = size(A)
    TN, NA = DNNKernels.ptq1_tensor_tile(N - first)
    groups = cld(M, TM) * cld(N - first, NA * TN)
    Mantle.dispatch!(g, ptq1_tensor_kernel!,
        (out, A.data, x, Int32(M), Int32(K), Int32(N), Int32(first), Val(TN), Val(NA)),
        groups * WG; group = WG, name)
    return nothing
end

end
