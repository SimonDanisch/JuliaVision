"""
    PTQ1Matrix

Prism PTQ1_0 weights, still packed. Each 128-value block is 28 bytes: 26 bytes of
base-3 digits and an FP16 scale, seven 32-bit words. The checkpoint stores a row's
blocks one after another; on the device the rows are in tiles of `PTQ1_TILE`, and
word `q` of block `b` of a tile's 64 rows is 64 consecutive words (see
[`ptq1_word0`](@ref)). Rows are padded to a whole tile with zeros, and `m` is the
logical row count.
"""
struct PTQ1Matrix{A}
    data::A
    m::Int
    k::Int
end

Base.size(A::PTQ1Matrix) = (A.m, A.k)
Base.size(A::PTQ1Matrix, d::Integer) = d == 1 ? A.m : d == 2 ? A.k : 1
Base.eltype(::PTQ1Matrix) = Float16
Base.ndims(::PTQ1Matrix) = 2

const PTQ1_QK = 128
const PTQ1_BLOCK_BYTES = 28
const PTQ1_WG = 256

# Rows per tile of the device layout. A decode work-item is one row of a tile, so
# the 64 work-items reading word q of a block read 256 consecutive bytes, while all
# of them read the same `x`. In the checkpoint's order the rows sit a whole row
# apart, which is why the decode column used to give a row eight work-items on
# eight different blocks, and eight different `x` addresses per load: on a Radeon
# 8060S, whose loads retire in order, every wait for `x` then waited for weight
# words still coming from memory, and a 17408x5120 projection took 157 us where its
# loads alone take 91.
const PTQ1_TILE = 64

"""
    ptq1_word0(row, block, blocks) -> Int32

The one-based index, in 32-bit words, of word 0 of `block` of `row` in the tiled
layout. The block's word `q` is `64q` words further on.
"""
@inline ptq1_word0(row::Int32, b::Int32, blocks::Int32) =
    ((row >> Int32(6)) * blocks + b) * Int32(7 * PTQ1_TILE) + (row & Int32(PTQ1_TILE - 1)) + Int32(1)

# The checkpoint's order into tiles, one word per work-item; rows past M are zero.
function ptq1_tile_kernel!(tiled, raw, M::Int32, K::Int32, n::Int32)
    i = Int32(KI.get_global_id().x - 1)
    if i < n
        raw32 = reinterpret(UInt32, raw)
        tiled32 = reinterpret(UInt32, tiled)
        blocks = K >> Int32(7)
        r = i & Int32(PTQ1_TILE - 1)
        rest = i >> Int32(6)
        q = rest % Int32(7)
        rest = rest ÷ Int32(7)
        b = rest % blocks
        row = (rest ÷ blocks) * Int32(PTQ1_TILE) + r
        @inbounds tiled32[i + Int32(1)] =
            row < M ? raw32[(row * blocks + b) * Int32(7) + q + Int32(1)] : UInt32(0)
    end
    return nothing
end

"""
    ptq1matrix(backend, file, tensor) -> PTQ1Matrix
    ptq1matrix(backend, bytes, m, k) -> PTQ1Matrix

Upload one packed matrix, in the checkpoint's order (a row's blocks contiguous),
and tile it on the device. Nothing is dequantised.
"""
function ptq1matrix(backend, file::GGUFFile, tensor::Union{GGUFTensor,AbstractString})
    t = tensor isa GGUFTensor ? tensor : file[tensor]
    t.typeid == GGML_TYPE_PTQ1_0 || throw(ArgumentError("$(t.name) is GGML type $(t.typeid), not PTQ1_0"))
    length(t.dims) == 2 || throw(ArgumentError("$(t.name) is rank $(length(t.dims)), expected a matrix"))
    k, m = t.dims
    ptq1matrix(backend, ggufbytes(file, t), m, k)
end

function ptq1matrix(backend, bytes::AbstractVector{UInt8}, m::Integer, k::Integer)
    k % PTQ1_QK == 0 || throw(ArgumentError("PTQ1 input width $k is not divisible by $PTQ1_QK"))
    blocks = k ÷ PTQ1_QK
    length(bytes) == m * blocks * PTQ1_BLOCK_BYTES || throw(DimensionMismatch(
        "$(length(bytes)) bytes are not $m rows of $blocks PTQ1 blocks"))
    words = cld(m, PTQ1_TILE) * PTQ1_TILE * blocks * 7
    words <= typemax(Int32) || throw(ArgumentError("a $(m)x$(k) PTQ1 matrix exceeds Int32 word indexing"))
    dev = Mantle.todevice(backend)
    raw = Mantle.Buffer(dev, bytes)
    tiled = Mantle.Buffer(dev, UInt8, (4 * words,))
    g = Mantle.Graph(dev)
    Mantle.dispatch!(g, ptq1_tile_kernel!, (tiled, raw, Int32(m), Int32(k), Int32(words)),
                     words; group = 256, name = "ptq1_tile")
    runonce!(g)
    Mantle.free!(raw)
    PTQ1Matrix(tiled, Int(m), Int(k))
end

@inline ptq1_scale(data32, wb::Int32) =
    Float32(reinterpret(Float16, UInt16(data32[wb + Int32(6 * PTQ1_TILE)] >> 16)))

# Element `e` of the block whose word 0 is `wb`, as a trit in -1:1.
@inline function ptq1_value(data32, wb::Int32, e::Int32)
    j = Int32(0)
    n = Int32(0)
    if e < Int32(80)
        j = e & Int32(15)
        n = e >> 4
    elseif e < Int32(120)
        t = e - Int32(80)
        j = Int32(16) + (t & Int32(7))
        n = t >> 3
    else
        t = e - Int32(120)
        j = Int32(24) + (t & Int32(1))
        n = t >> 1
    end
    b = (data32[wb + Int32(PTQ1_TILE) * (j >> 2)] >> (UInt32(8) * UInt32(j & Int32(3)))) & UInt32(0xff)
    # n is in 0:4. Multiplying by 3^n modulo 256 drops the n leading trits: the
    # recurrence `b = (3b) & 0xff`, n times, without a loop.
    p = n == Int32(0) ? UInt32(1) :
        n == Int32(1) ? UInt32(3) :
        n == Int32(2) ? UInt32(9) :
        n == Int32(3) ? UInt32(27) : UInt32(81)
    b = (b * p) & UInt32(0xff)
    Int32((b * UInt32(3)) >> 8) - Int32(1)
end

# K-parts per workgroup of the column kernel, which reads `x` from device memory: a
# tile's 64 rows are S parts of 64 work-items, part s taking blocks s, s + S, ...
# Enough parts that a projection has some 768 parts in flight, between 4 and 16;
# measured unstaged on a Radeon 8060S and an M5 for every Bonsai shape: fewer
# starves the narrow ones (5120 rows want 8), more costs the reduction.
ptq1_parts(M::Integer) = clamp(prevpow(2, max(1, 768 ÷ cld(M, PTQ1_TILE))), 4, 16)

# The work-item's place in its workgroup through the sub-group ids, so that a part
# is whole sub-groups whatever order KI gives them, and its `x` address is uniform.
@inline _ptq1_slot() = Int32(KI.get_sub_group_id() - 1) * Int32(KI.get_sub_group_size()) +
                       Int32(KI.get_sub_group_local_id() - 1)

# One trit of a PTQ1 byte, decoded and multiplied in. `f` is the byte's remaining
# digits as a fraction of 256, so `3f - 1` has the leading trit minus one as its
# integer part and the rest as its fraction: the integer recurrence
# `(b * 3) >> 8`, `b = (b * 3) & 0xff`, in float form.
#
# In HALF precision, and exactly: every value is a multiple of 1/256 with |y| < 2,
# nine bits of significand, and half has eleven. So this is bit-identical to the
# single-precision chain and 6-8% faster on an M5. It needs no device feature the
# kernel did not already use, since the block scale is a Float16 too.
@inline function ptq1_trit(f::Float16, acc::Float32, x::Float32)
    y = muladd(f, Float16(3), Float16(-1))
    t = floor(y)
    return y - t, muladd(Float32(t), x, acc)
end

# The decode column's parts, and from how many blocks it double-buffers: see
# `ptq1_mul_rows_kernel!`.
const PTQ1_DECODE_PARTS = 4
const PTQ1_DECODE_DB_BLOCKS = 64

@inline _ptq1_x4(x4, i::Int32, n::Int32) = i < n ? @inbounds(x4[i + Int32(1)]) :
    (VecElement(0f0), VecElement(0f0), VecElement(0f0), VecElement(0f0))

# A block's seven words, the last apart since it holds the scale. A block past the
# end reads block 0 instead, so a part with nothing left still loads in bounds.
@inline function _ptq1_words(data32, row::Int32, blk::Int32, blocks::Int32)
    w0 = ptq1_word0(row, blk < blocks ? blk : Int32(0), blocks)
    @inbounds ((data32[w0], data32[w0 + Int32(PTQ1_TILE)], data32[w0 + Int32(2 * PTQ1_TILE)],
                data32[w0 + Int32(3 * PTQ1_TILE)], data32[w0 + Int32(4 * PTQ1_TILE)],
                data32[w0 + Int32(5 * PTQ1_TILE)]), data32[w0 + Int32(6 * PTQ1_TILE)])
end

# One block of one row against the block's `x` in shared memory from float4 `xb`.
# Words 0-3 hold bytes 0-15, whose five trits are e = j + 16n; words 4-5 hold bytes
# 16-23, e = 80 + j + 8n. Either way a word's four bytes at one n are four
# consecutive elements: one float4. Word 6's two bytes hold four trits each, e = 120
# + j + 2n, and the scale is its high half. Unrolled, so every `x` read is a
# constant offset from `xb`.
@inline function _ptq1_block(acc::Float32, ws, last::UInt32, xs, xb::Int32)
    scale = Float32(reinterpret(Float16, UInt16(last >> 16)))
    a0 = 0f0; a1 = 0f0; a2 = 0f0; a3 = 0f0
    Base.Cartesian.@nexprs 6 q -> begin
        word = ws[q]
        f0 = Float16(word & 0xff) * Float16(1 / 256)
        f1 = Float16((word >> 8) & 0xff) * Float16(1 / 256)
        f2 = Float16((word >> 16) & 0xff) * Float16(1 / 256)
        f3 = Float16(word >> 24) * Float16(1 / 256)
        Base.Cartesian.@nexprs 5 n -> begin
            # float4 q - 1 + 4(n - 1) for words 0-3, 16 + q - 1 + 2(n - 1) for 4-5
            v = @inbounds xs[xb + Int32(q <= 4 ? q - 1 + 4 * (n - 1) : 15 + q + 2 * (n - 1))]
            f0, a0 = ptq1_trit(f0, a0, v[1].value)
            f1, a1 = ptq1_trit(f1, a1, v[2].value)
            f2, a2 = ptq1_trit(f2, a2, v[3].value)
            f3, a3 = ptq1_trit(f3, a3, v[4].value)
        end
    end
    g0 = Float16(last & 0xff) * Float16(1 / 256)
    g1 = Float16((last >> 8) & 0xff) * Float16(1 / 256)
    va = @inbounds xs[xb + Int32(30)]
    vb = @inbounds xs[xb + Int32(31)]
    g0, a0 = ptq1_trit(g0, a0, va[1].value); g1, a1 = ptq1_trit(g1, a1, va[2].value)
    g0, a0 = ptq1_trit(g0, a0, va[3].value); g1, a1 = ptq1_trit(g1, a1, va[4].value)
    g0, a0 = ptq1_trit(g0, a0, vb[1].value); g1, a1 = ptq1_trit(g1, a1, vb[2].value)
    g0, a0 = ptq1_trit(g0, a0, vb[3].value); g1, a1 = ptq1_trit(g1, a1, vb[4].value)
    return muladd(scale, (a0 + a1) + (a2 + a3), acc)
end

"""
    ptq1_mul_rows_kernel!(out, data, x, bias, M, K, Val(HASBIAS), Val(S), Val(DB))

The decode column, `N == 1`: `out = A * x (+ bias)` for tiled PTQ1 `A` and a
Float32 `x`. Most of a Bonsai token is this kernel.

A workgroup is one 64-row tile in `S` parts of 64 work-items; a work-item is a row
of the tile. The blocks go in rounds of `S`, part `s` taking the round's `s`-th, so
the 64 loads of a block's word are 256 consecutive bytes and the parts' sums meet
in shared memory at the end. A round's `x`, `S` blocks of it, is staged in shared
memory first, and each part reads it there at one address. With `DB` the staging
is double-buffered: the next round's `x` and words load while this one decodes,
one barrier a round instead of two.

µs per launch, each of the 64 matrices read once per run (the 8060S's cache holds
one of them):

    shape (M x K)      Radeon 8060S                M5
                       checkpoint  tiled  staged   checkpoint  tiled  staged
    17408 x  5120         157       114     97        178      174    174
     5120 x 17408         149       100     95 (DB)   184      175    176 (DB)
    10240 x  5120          89        72     56        108      106    105
     5120 x  6144          63        46     36         68       65     67
     1024 x  5120          10        11     13         16       16     16

"Checkpoint" read the checkpoint's order, eight work-items to a row on eight
consecutive blocks, so the eight read eight different `x`. Its decode was free on
the 8060S (without the `x` loads it ran at the speed of its weight loads, 91 us),
its `x` loads were not: the Radeon retires loads in order, and a wait for `x` waited
for the weight words issued before it. "Tiled" read `x` from device memory at one
address per part; the same wait remained, for 24 us of the 114. Staged, `x` comes
through shared memory, which has its own counter. On an M5 each costs about the
same; there the old kernel was issue-bound, and `x` as float4 at a few addresses,
`ptq1_trit`'s three exact float operations per trit, the scale applied once per
block and the loops unrolled by hand took it from 838 to 180 us. Two barriers a
round cost the M5 5% at 136 blocks, which `DB` takes back, and double-buffering
costs it 4% at 40, hence `PTQ1_DECODE_DB_BLOCKS`. More parts than four was
slower staged on the M5 at every shape.

`x` is read as float4, so it must be 16-byte aligned. Every buffer and transient
Mantle places is.
"""
function ptq1_mul_rows_kernel!(out, data, x, bias, M::Int32, K::Int32,
                               ::Val{HASBIAS}, ::Val{S}, ::Val{DB}) where {HASBIAS,S,DB}
    V4 = NTuple{4,VecElement{Float32}}
    xs = KI.localmemory(V4, Val((32 * S * (DB ? 2 : 1),)), Val(1))
    sh = KI.localmemory(Float32, Val((PTQ1_TILE * S,)), Val(2))
    _ptq1_rows_tile!(out, reinterpret(UInt32, data), reinterpret(V4, x), xs, sh, bias, M, K,
                     Int32(KI.get_group_id().x - 1), Val(HASBIAS), Val(S), Val(DB))
    return nothing
end

"""
    ptq1_mul_rows2_kernel!(out1, data1, out2, data2, x, M1, M2, K, Val(S), Val(DB))

Two decode columns over the same `x` in one launch, `A1`'s tiles and then `A2`'s.
Bonsai's FFN gate and up as one launch instead of two: 196 to 184 us on a Radeon
8060S, 348 to 340 on an M5.
"""
function ptq1_mul_rows2_kernel!(out1, data1, out2, data2, x, M1::Int32, M2::Int32, K::Int32,
                                ::Val{S}, ::Val{DB}) where {S,DB}
    V4 = NTuple{4,VecElement{Float32}}
    xs = KI.localmemory(V4, Val((32 * S * (DB ? 2 : 1),)), Val(1))
    sh = KI.localmemory(Float32, Val((PTQ1_TILE * S,)), Val(2))
    x4 = reinterpret(V4, x)
    tile = Int32(KI.get_group_id().x - 1)
    tiles1 = (M1 + Int32(PTQ1_TILE - 1)) >> Int32(6)
    # Uniform across the workgroup, so each branch's barriers are too.
    if tile < tiles1
        _ptq1_rows_tile!(out1, reinterpret(UInt32, data1), x4, xs, sh, out1, M1, K, tile,
                         Val(false), Val(S), Val(DB))
    else
        _ptq1_rows_tile!(out2, reinterpret(UInt32, data2), x4, xs, sh, out2, M2, K, tile - tiles1,
                         Val(false), Val(S), Val(DB))
    end
    return nothing
end

# One workgroup's tile of the decode column; see `ptq1_mul_rows_kernel!`.
@inline function _ptq1_rows_tile!(out, data32, x4, xs, sh, bias, M::Int32, K::Int32, tile::Int32,
                                  ::Val{HASBIAS}, ::Val{S}, ::Val{DB}) where {HASBIAS,S,DB}
    slot = _ptq1_slot()
    part = slot >> Int32(6)
    r = slot & Int32(PTQ1_TILE - 1)
    row = tile * Int32(PTQ1_TILE) + r
    blocks = K >> Int32(trailing_zeros(PTQ1_QK))
    nx4 = K >> Int32(2)
    # A round's x is 32 S float4: one each for the first half of the work-items.
    stager = slot < Int32(32 * S)
    acc = 0f0
    b0 = Int32(0)
    if DB
        stager && (@inbounds xs[slot + Int32(1)] = _ptq1_x4(x4, slot, nx4))
        ws, last = _ptq1_words(data32, row, part, blocks)
        KI.barrier()
        buf = Int32(0)
        while b0 < blocks
            nb = b0 + Int32(S)
            xn = _ptq1_x4(x4, nb * Int32(32) + slot, nx4)
            wsn, lastn = _ptq1_words(data32, row, nb + part, blocks)
            b0 + part < blocks &&
                (acc = _ptq1_block(acc, ws, last, xs, (buf * Int32(S) + part) * Int32(32) + Int32(1)))
            buf = Int32(1) - buf
            stager && (@inbounds xs[buf * Int32(32 * S) + slot + Int32(1)] = xn)
            KI.barrier()
            ws, last = wsn, lastn
            b0 = nb
        end
    else
        while b0 < blocks
            stager && (@inbounds xs[slot + Int32(1)] = _ptq1_x4(x4, b0 * Int32(32) + slot, nx4))
            # The words before the barrier, so they are in flight while it waits.
            ws, last = _ptq1_words(data32, row, b0 + part, blocks)
            KI.barrier()
            b0 + part < blocks && (acc = _ptq1_block(acc, ws, last, xs, part * Int32(32) + Int32(1)))
            KI.barrier()
            b0 += Int32(S)
        end
    end
    @inbounds sh[slot + Int32(1)] = acc
    KI.barrier()
    if part == Int32(0) && row < M
        @inbounds begin
            t = sh[r + Int32(1)]
            for j in Int32(1):Int32(S - 1)
                t += sh[j * Int32(PTQ1_TILE) + r + Int32(1)]
            end
            HASBIAS && (t += Float32(bias[row + Int32(1)]))
            out[row + Int32(1)] = eltype(out)(t)
        end
    end
    return nothing
end

# The decode column's launch: (ndrange, group, Val(S), Val(DB)).
function ptq1_rows_launch(M::Integer, K::Integer)
    S = PTQ1_DECODE_PARTS
    (cld(M, PTQ1_TILE) * PTQ1_TILE * S, PTQ1_TILE * S, Val(S),
     Val(K ÷ PTQ1_QK >= PTQ1_DECODE_DB_BLOCKS))
end

"""
    ptq1_rows2!(g, out1, A1::PTQ1Matrix, out2, A2::PTQ1Matrix, x; name)

Declare `out1 = A1 * x` and `out2 = A2 * x` into graph `g` as one launch of
`ptq1_mul_rows2_kernel!`. The two have the same width.
"""
function ptq1_rows2!(g, out1, A1::PTQ1Matrix, out2, A2::PTQ1Matrix, x; name::AbstractString)
    (M1, K), (M2, K2) = size(A1), size(A2)
    K == K2 || throw(DimensionMismatch("ptq1_rows2!: widths $K and $K2 differ"))
    nd1, group, S, DB = ptq1_rows_launch(M1, K)
    nd2, _, _, _ = ptq1_rows_launch(M2, K)
    Mantle.dispatch!(g, ptq1_mul_rows2_kernel!,
        (out1, A1.data, out2, A2.data, x, Int32(M1), Int32(M2), Int32(K), S, DB), nd1 + nd2; group, name)
    return nothing
end

"""
    ptq1_rows!(g, out, A::PTQ1Matrix, x; name)

Declare the decode column `out = A * x` into graph `g` through
`ptq1_mul_rows_kernel!`.
"""
function ptq1_rows!(g, out, A::PTQ1Matrix, x; name::AbstractString)
    M, K = size(A)
    nd, group, S, DB = ptq1_rows_launch(M, K)
    Mantle.dispatch!(g, ptq1_mul_rows_kernel!,
        (out, A.data, x, A.data, Int32(M), Int32(K), Val(false), S, DB), nd; group, name)
    return nothing
end

# Wide-prefill tile.  The small-N kernels give each output row its own lanes,
# which is the right shape for decode but leaves almost all matrix reuse on
# the table during prompt processing.  This kernel stages a 64x32x32 matrix
# tile: every decoded weight is shared by 32 prompt columns and every activation
# is shared by 64 output rows.
const PTQ1_MM_BM = 64
const PTQ1_MM_BN = 32
const PTQ1_MM_BK = 32

# A few activation columns at once: `ptq1_mul_rows_kernel!`'s tile and parts, each
# decoded trit applied to NC columns. Decoding is most of a column's cost, so sharing it
# is worth more than any reuse of the weights: one 17408x5120 projection on an M5
# costs 177 us for one column, 127 a column for two and 112 for four. Past four
# the accumulators spill (145 for eight, 260 for sixteen), so wider batches are
# taken four columns at a time, or by the native GEMM (`ptq1_native_gemm!`).
#
# `x` is any float type and read four elements at a time, so a column must start
# 16-byte aligned for Float32 and 8 for Float16; `K` a multiple of 128 does it.
const PTQ1_COLS = 4

@inline _ptq1_acc(acc::NTuple{N,Float32}, t::Float32, xs::NTuple{N,Float32}) where {N} =
    ntuple(c -> muladd(t, xs[c], acc[c]), Val(N))
@inline function _ptq1_tritn(f::Float16, acc::NTuple{N,Float32}, xs::NTuple{N,Float32}) where {N}
    y = muladd(f, Float16(3), Float16(-1))
    t = floor(y)
    return y - t, _ptq1_acc(acc, Float32(t), xs)
end
# Column c's four (or two) elements at vector index i, and one lane of each.
@inline _ptq1_cols(xv, i::Int32, stride::Int32, ::Val{N}) where {N} =
    ntuple(c -> @inbounds(xv[i + Int32(c - 1) * stride]), Val(N))
@inline _ptq1_lane(v::NTuple{N}, l::Int) where {N} = ntuple(c -> Float32(v[c][l].value), Val(N))
@inline _ptq1_scaled(acc::NTuple{N,Float32}, scale::Float32, a::NTuple{N,Float32}) where {N} =
    ntuple(c -> muladd(scale, a[c], acc[c]), Val(N))

function ptq1_mul_cols_kernel!(out, data, x, bias, M::Int32, K::Int32,
                               ::Val{NC}, ::Val{HASBIAS}, ::Val{S}) where {NC,HASBIAS,S}
    data32 = reinterpret(UInt32, data)
    T = eltype(x)
    x4 = reinterpret(NTuple{4,VecElement{T}}, x)
    x2 = reinterpret(NTuple{2,VecElement{T}}, x)
    sh = KI.localmemory(Float32, Val((PTQ1_TILE * S * NC,)), Val(1))
    slot = _ptq1_slot()
    part = slot >> Int32(6)
    r = slot & Int32(PTQ1_TILE - 1)
    row = Int32(KI.get_group_id().x - 1) * Int32(PTQ1_TILE) + r
    blocks = K >> Int32(trailing_zeros(PTQ1_QK))
    acc = ntuple(_ -> 0f0, Val(NC))
    blk = part
    @inbounds while blk < blocks
        w0 = ptq1_word0(row, blk, blocks)
        last = data32[w0 + Int32(6 * PTQ1_TILE)]
        scale = Float32(reinterpret(Float16, UInt16(last >> 16)))
        a = ntuple(_ -> 0f0, Val(NC))
        x4b = blk * Int32(PTQ1_QK ÷ 4) + Int32(1)
        Base.Cartesian.@nexprs 6 q -> begin
            word = data32[w0 + Int32((q - 1) * PTQ1_TILE)]
            f0 = Float16(word & 0xff) * Float16(1 / 256)
            f1 = Float16((word >> 8) & 0xff) * Float16(1 / 256)
            f2 = Float16((word >> 16) & 0xff) * Float16(1 / 256)
            f3 = Float16(word >> 24) * Float16(1 / 256)
            Base.Cartesian.@nexprs 5 n -> begin
                v = _ptq1_cols(x4, x4b + Int32(q <= 4 ? q - 1 + 4 * (n - 1) : 15 + q + 2 * (n - 1)),
                               K >> Int32(2), Val(NC))
                f0, a = _ptq1_tritn(f0, a, _ptq1_lane(v, 1))
                f1, a = _ptq1_tritn(f1, a, _ptq1_lane(v, 2))
                f2, a = _ptq1_tritn(f2, a, _ptq1_lane(v, 3))
                f3, a = _ptq1_tritn(f3, a, _ptq1_lane(v, 4))
            end
        end
        g0 = Float16(last & 0xff) * Float16(1 / 256)
        g1 = Float16((last >> 8) & 0xff) * Float16(1 / 256)
        x2b = blk * Int32(PTQ1_QK ÷ 2) + Int32(60 + 1)
        Base.Cartesian.@nexprs 4 n -> begin
            v = _ptq1_cols(x2, x2b + Int32(n - 1), K >> Int32(1), Val(NC))
            g0, a = _ptq1_tritn(g0, a, _ptq1_lane(v, 1))
            g1, a = _ptq1_tritn(g1, a, _ptq1_lane(v, 2))
        end
        acc = _ptq1_scaled(acc, scale, a)
        blk += Int32(S)
    end
    Base.Cartesian.@nexprs 4 c -> (c <= NC &&
        (@inbounds sh[(c - 1) * PTQ1_TILE * S + slot + Int32(1)] = acc[c]))
    KI.barrier()
    if part == Int32(0) && row < M
        b = HASBIAS ? Float32(@inbounds bias[row + Int32(1)]) : 0f0
        Base.Cartesian.@nexprs 4 c -> if c <= NC
            @inbounds begin
                base = Int32((c - 1) * PTQ1_TILE * S) + r + Int32(1)
                t = sh[base]
                for j in Int32(1):Int32(S - 1)
                    t += sh[base + j * Int32(PTQ1_TILE)]
                end
                out[row + Int32(1) + Int32(c - 1) * M] = eltype(out)(t + b)
            end
        end
    end
    return nothing
end
function ptq1_mul_mm_kernel!(out, data, x,
                                                M::Int32, K::Int32, N::Int32,
                                                mtiles::Int32)
    data32 = reinterpret(UInt32, data)
    atile = KI.localmemory(Float32, Val((PTQ1_MM_BM * PTQ1_MM_BK,)), Val(1))
    btile = KI.localmemory(Float32, Val((PTQ1_MM_BN * PTQ1_MM_BK,)), Val(2))

    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    mt = wg % mtiles
    nt = wg ÷ mtiles
    m0 = mt * Int32(PTQ1_MM_BM)
    n0 = nt * Int32(PTQ1_MM_BN)

    # A 16x16 thread grid computes a 4x2 microtile per thread.
    lr0 = (t & Int32(15)) * Int32(4)
    lc0 = (t >> 4) * Int32(2)
    ar0 = 0f0; ar1 = 0f0; ar2 = 0f0; ar3 = 0f0
    br0 = 0f0; br1 = 0f0; br2 = 0f0; br3 = 0f0
    blocks = K ÷ Int32(PTQ1_QK)

    k0 = Int32(0)
    while k0 < K
        ai = t
        while ai < Int32(PTQ1_MM_BM * PTQ1_MM_BK)
            lr = ai ÷ Int32(PTQ1_MM_BK)
            lk = ai % Int32(PTQ1_MM_BK)
            row = m0 + lr
            k = k0 + lk
            av = 0f0
            if row < M && k < K
                wb = ptq1_word0(row, k ÷ Int32(PTQ1_QK), blocks)
                av = ptq1_scale(data32, wb) * Float32(ptq1_value(data32, wb, k % Int32(PTQ1_QK)))
            end
            @inbounds atile[ai + Int32(1)] = av
            ai += Int32(PTQ1_WG)
        end

        bi = t
        while bi < Int32(PTQ1_MM_BN * PTQ1_MM_BK)
            lc = bi ÷ Int32(PTQ1_MM_BK)
            lk = bi % Int32(PTQ1_MM_BK)
            col = n0 + lc
            k = k0 + lk
            bv = 0f0
            if col < N && k < K
                @inbounds bv = Float32(x[col * K + k + Int32(1)])
            end
            @inbounds btile[bi + Int32(1)] = bv
            bi += Int32(PTQ1_WG)
        end
        KI.barrier()

        for lk in Int32(0):Int32(PTQ1_MM_BK - 1)
            a0 = atile[(lr0 + Int32(0)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            a1 = atile[(lr0 + Int32(1)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            a2 = atile[(lr0 + Int32(2)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            a3 = atile[(lr0 + Int32(3)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            b0 = btile[(lc0 + Int32(0)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            b1 = btile[(lc0 + Int32(1)) * Int32(PTQ1_MM_BK) + lk + Int32(1)]
            ar0 = muladd(a0, b0, ar0); br0 = muladd(a0, b1, br0)
            ar1 = muladd(a1, b0, ar1); br1 = muladd(a1, b1, br1)
            ar2 = muladd(a2, b0, ar2); br2 = muladd(a2, b1, br2)
            ar3 = muladd(a3, b0, ar3); br3 = muladd(a3, b1, br3)
        end
        KI.barrier()
        k0 += Int32(PTQ1_MM_BK)
    end

    row0 = m0 + lr0
    col0 = n0 + lc0
    if col0 < N
        row0 + Int32(0) < M && (@inbounds out[row0 + Int32(1) + col0 * M] = eltype(out)(ar0))
        row0 + Int32(1) < M && (@inbounds out[row0 + Int32(2) + col0 * M] = eltype(out)(ar1))
        row0 + Int32(2) < M && (@inbounds out[row0 + Int32(3) + col0 * M] = eltype(out)(ar2))
        row0 + Int32(3) < M && (@inbounds out[row0 + Int32(4) + col0 * M] = eltype(out)(ar3))
    end
    if col0 + Int32(1) < N
        col1 = col0 + Int32(1)
        row0 + Int32(0) < M && (@inbounds out[row0 + Int32(1) + col1 * M] = eltype(out)(br0))
        row0 + Int32(1) < M && (@inbounds out[row0 + Int32(2) + col1 * M] = eltype(out)(br1))
        row0 + Int32(2) < M && (@inbounds out[row0 + Int32(3) + col1 * M] = eltype(out)(br2))
        row0 + Int32(3) < M && (@inbounds out[row0 + Int32(4) + col1 * M] = eltype(out)(br3))
    end
    return nothing
end

"""
    ptq1mul!(ctx, out, A, x[, bias]) -> out

Multiply packed PTQ1 weights by one or more activation columns. `x` is `K×N`
and `out` is `M×N`; vectors are the `N == 1` case.
"""
function ptq1mul!(ctx, out, A::PTQ1Matrix, x, bias=nothing)
    M, K = size(A)
    length(x) % K == 0 || throw(DimensionMismatch("activation length $(length(x)) is not divisible by $K"))
    N = length(x) ÷ K
    length(out) == M * N || throw(DimensionMismatch("output has $(length(out)) values, expected $(M*N)"))
    S = ptq1_parts(M)
    tiles = cld(M, PTQ1_TILE) * PTQ1_TILE * S
    if N >= 8 && bias === nothing
        mtiles = cld(M, PTQ1_MM_BM)
        ntiles = cld(N, PTQ1_MM_BN)
        harnesslaunch!(ctx.backend, ptq1_mul_mm_kernel!,
            out, Mantle.storage(A.data), x, Int32(M), Int32(K), Int32(N), Int32(mtiles);
            ndrange = mtiles * ntiles * PTQ1_WG, workgroupsize = PTQ1_WG)
    elseif N == 1
        eltype(x) === Float32 || throw(ArgumentError(
            "the PTQ1 decode column reads Float32 activations, got $(eltype(x))"))
        nd, group, SD, DB = ptq1_rows_launch(M, K)
        harnesslaunch!(ctx.backend, ptq1_mul_rows_kernel!,
            out, Mantle.storage(A.data), x,
            bias === nothing ? Mantle.storage(A.data) : bias,
            Int32(M), Int32(K), Val(bias !== nothing), SD, DB;
            ndrange = nd, workgroupsize = group)
    else
        xs, os = Mantle.storage(x), Mantle.storage(out)
        for j in 0:PTQ1_COLS:N-1
            nc = min(PTQ1_COLS, N - j)
            harnesslaunch!(ctx.backend, ptq1_mul_cols_kernel!,
                view(os, M * j + 1:M * (j + nc)), Mantle.storage(A.data),
                view(xs, K * j + 1:K * (j + nc)),
                bias === nothing ? Mantle.storage(A.data) : bias,
                Int32(M), Int32(K), Val(nc), Val(bias !== nothing), Val(S);
                ndrange = tiles, workgroupsize = PTQ1_TILE * S)
        end
    end
    out
end

"""
    ptq1_cols!(g, out, A::PTQ1Matrix, x, first, N; name)

Declare columns `first+1:N` of `out = A * x` into graph `g`, `PTQ1_COLS` columns a
launch through `ptq1_mul_cols_kernel!`. `out` and `x` are the flat `M×N` and
`K×N` resources; each launch gets views of its own columns.
"""
function ptq1_cols!(g, out, A::PTQ1Matrix, x, first::Integer, N::Integer; name::AbstractString)
    M, K = size(A)
    S = ptq1_parts(M)
    j = Int(first)
    while j < N
        nc = min(PTQ1_COLS, N - j)
        Mantle.dispatch!(g, ptq1_mul_cols_kernel!,
            (Mantle.viewof(out, (M * nc,); offset = M * j), A.data,
             Mantle.viewof(x, (K * nc,); offset = K * j), A.data,
             Int32(M), Int32(K), Val(nc), Val(false), Val(S)),
            cld(M, PTQ1_TILE) * PTQ1_TILE * S; group = PTQ1_TILE * S,
            name = string(name, ".cols", j))
        j += nc
    end
    return nothing
end

function ptq1_getrows_kernel!(out, data, rows,
                                                M::Int32, K::Int32, N::Int32)
    data32 = reinterpret(UInt32, data)
    i = Int32(KI.get_global_id().x - 1)
    if i < K * N
        k = i % K
        n = i ÷ K
        row = Int32(rows[n + Int32(1)])
        if row >= Int32(0) && row < M
            wb = ptq1_word0(row, k ÷ Int32(PTQ1_QK), K ÷ Int32(PTQ1_QK))
            @inbounds out[i + Int32(1)] = eltype(out)(
                ptq1_scale(data32, wb) * Float32(ptq1_value(data32, wb, k % Int32(PTQ1_QK))))
        end
    end
    return nothing
end

"""
    ptq1_getrows!(ctx, out, table, rows) -> out

Decode zero-based vocabulary rows into consecutive `K`-element columns.
"""
function ptq1_getrows!(ctx, out, A::PTQ1Matrix, rows)
    N = length(rows)
    length(out) == A.k * N || throw(DimensionMismatch("embedding output must contain $(A.k*N) values"))
    harnesslaunch!(ctx.backend, ptq1_getrows_kernel!, out, Mantle.storage(A.data), rows,
                                                 Int32(A.m), Int32(A.k), Int32(N);
                                                ndrange=A.k * N, workgroupsize = 256)
    out
end

# One work-item per (row, block), consecutive work-items on consecutive rows, so
# each of a block's 128 stores lands beside its neighbours' in the column-major
# `out`, and in the tiled layout its word loads too. One element per work-item, each decoding its own trit through
# `ptq1_value`'s byte loads, was 4.4x slower: 10.2 ms against 2.3 to expand
# Bonsai's 17408x5120 FFN gate to Float16 on an M5.
function ptq1_dequant_kernel!(out, data, M::Int32, K::Int32)
    data32 = reinterpret(UInt32, data)
    i = Int32(KI.get_global_id().x - 1)
    blocks = K ÷ Int32(PTQ1_QK)
    if i < M * blocks
        m = i % M
        blk = i ÷ M
        w0 = ptq1_word0(m, blk, blocks)
        last = data32[w0 + Int32(6 * PTQ1_TILE)]
        scale = Float32(reinterpret(Float16, UInt16(last >> 16)))
        col = blk * Int32(PTQ1_QK)
        # Byte j's trits: e = j + 16n for bytes 0-15, 80 + (j - 16) + 8n for 16-23,
        # and four of them at 120 + (j - 24) + 2n for 24-25.
        @inbounds for j in Int32(0):Int32(25)
            word = j < Int32(24) ? data32[w0 + Int32(PTQ1_TILE) * (j >> 2)] : last
            bp = (word >> (8 * (j & Int32(3)))) & UInt32(0xff)
            e = j < Int32(16) ? j : j < Int32(24) ? Int32(64) + j : Int32(96) + j
            stride = j < Int32(16) ? Int32(16) : j < Int32(24) ? Int32(8) : Int32(2)
            for _ in Int32(1):(j < Int32(24) ? Int32(5) : Int32(4))
                tr = Int32((bp * UInt32(3)) >> 8) - Int32(1)
                bp = (bp * UInt32(3)) & UInt32(0xff)
                out[m + (col + e) * M + Int32(1)] = eltype(out)(scale * Float32(tr))
                e += stride
            end
        end
    end
    return nothing
end

"""Expand a PTQ1 matrix to a device Float32 matrix, for verification only."""
function ptq1_dequant(ctx, A::PTQ1Matrix)
    dev = Mantle.todevice(ctx.backend)
    g = Mantle.Graph(dev)
    out = Mantle.Buffer(dev, Float32, size(A))
    Mantle.dispatch!(g, ptq1_dequant_kernel!, (out, A.data, Int32(A.m), Int32(A.k)),
                     A.m * (A.k ÷ PTQ1_QK); group = 256, name = "ptq1_dequant")
    runonce!(g)
    out
end

"""
    ptq1_native_gemm!(g, out, A::PTQ1Matrix, x, N; name) -> Int

Declare the leading columns of `out = A * x` (`N` Float16 activation columns,
`out` and `x` flat) into graph `g` as `A` expanded to a Float16 transient, then the
device's own GEMM through `Mantle.native_gemm_dispatch!`. Returns how many columns
it covered, the largest multiple of 32, or 0 having declared nothing where the
device has no native GEMM for Float16 operands into `eltype(out)`. The caller
takes the rest, e.g. through [`ptq1_cols!`](@ref).

Multiples of 32 because a GEMM tiles its columns by the largest of 64, 32, 16 or 8
that divides them, and a narrower tile is slow; and one product, because each
reads the whole expansion at least once. A second product for a remainder of 8 to
24 columns cost more on an M5 than taking those columns four at a time.

The prefill route for a device without the staged cooperative-matrix kernels that
`ptq1_coopmat_kernel!` decodes into. The expansion is what that kernel avoids: it
costs a write and a read of 2 bytes per weight. It is still the faster route by
far where the alternative is `ptq1_mul_mm_kernel!`'s scalar product. On an M5, one
17408x5120 FFN projection over 512 tokens took 339 ms there and 10.4 here (2.3 to
expand, 8.1 for Metal's tensor-op GEMM at 11 TFLOP/s).
"""
function ptq1_native_gemm!(g, out, A::PTQ1Matrix, x, N::Integer; name::AbstractString)
    M, K = size(A)
    cover = 32 * (Int(N) ÷ 32)
    (eltype(x) === Float16 && cover > 0) || return 0
    Mantle.native_gemm_available(g.dev, Float16, Float16, eltype(out)) || return 0
    W = Mantle.Transient.Buffer(g, Float16, (M, K))
    Mantle.dispatch!(g, ptq1_dequant_kernel!, (W, A.data, Int32(M), Int32(K)),
                     M * (K ÷ PTQ1_QK); group = 256, name = string(name, ".expand"))
    fused = Mantle.native_gemm_dispatch!(g.dev, g, Mantle.viewof(out, (M, cover)), W,
                                         Mantle.viewof(x, (K, cover)); name)
    fused === nothing && error(
        "ptq1_native_gemm!: `native_gemm_available` admitted Float16 x Float16 -> " *
        "$(eltype(out)) on $(typeof(g.dev)) and `native_gemm_dispatch!` then " *
        "declined $(M)x$(cover)x$(K) for `$name`. The two answers have to agree.")
    return cover
end

# Wide PTQ1 matrix multiply for devices with 16x16x16 fp16 cooperative
# matrices.  A 128x128 output block is owned by eight wave32 subgroups.  Each
# packed weight is decoded directly into the 128x32 shared A tile and consumed
# immediately, avoiding the 54 GB full-model fp16 expansion that a 512-token
# Bonsai prefill would otherwise write and read back.
const PTQ1_COOP_BM = 128
const PTQ1_COOP_BN = 128
const PTQ1_COOP_BK = 32
const PTQ1_COOP_PAD = 8
const PTQ1_COOP_WG = 256

function ptq1_coopmat_kernel!(
        C, data, B, ::Val{M}, ::Val{N}, ::Val{K}) where {M,N,K}
    lda = Int32(PTQ1_COOP_BM + PTQ1_COOP_PAD)
    ldb = Int32(PTQ1_COOP_BK + PTQ1_COOP_PAD)
    sA = KI.localmemory(Float16, Val(((PTQ1_COOP_BM + PTQ1_COOP_PAD) * PTQ1_COOP_BK,)), Val(1))
    sB = KI.localmemory(Float16, Val(((PTQ1_COOP_BK + PTQ1_COOP_PAD) * PTQ1_COOP_BN,)), Val(2))

    data32 = reinterpret(UInt32, data)
    tid = Int32(KI.get_local_id().x - 1)
    blk = Int32(KI.get_group_id().x - 1)
    nblocks_m = Int32(M ÷ PTQ1_COOP_BM)
    tm = (blk % nblocks_m) * Int32(PTQ1_COOP_BM)
    tn = (blk ÷ nblocks_m) * Int32(PTQ1_COOP_BN)

    sg = tid ÷ Int32(32)
    sm = (sg % Int32(2)) * Int32(4)
    sn = (sg ÷ Int32(2)) * Int32(2)
    AM = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixA}
    BM = Mantle.AcceleratedMatrix{Float16,16,16,Mantle.MatrixB}
    CM = Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator}
    c00 = zero(CM); c10 = zero(CM); c20 = zero(CM); c30 = zero(CM)
    c01 = zero(CM); c11 = zero(CM); c21 = zero(CM); c31 = zero(CM)
    blocks = Int32(K ÷ PTQ1_QK)

    for kb in Int32(0):Int32(K ÷ PTQ1_COOP_BK - 1)
        k0 = kb * Int32(PTQ1_COOP_BK)

        # Two groups per lane cover the 512 eight-trit groups in a 128x32 tile.
        @inbounds for r in Int32(0):Int32(1)
            idx = tid + r * Int32(PTQ1_COOP_WG)
            row = idx ÷ Int32(4)
            e0 = (idx & Int32(3)) * Int32(8)
            k = k0 + e0
            qe = k % Int32(PTQ1_QK)
            wb = ptq1_word0(tm + row, k ÷ Int32(PTQ1_QK), blocks)
            scale = ptq1_scale(data32, wb)
            Base.Cartesian.@nexprs 8 l -> begin
                sA[row + (e0 + Int32(l - 1)) * lda + Int32(1)] =
                    Float16(scale * Float32(ptq1_value(
                        data32, wb, qe + Int32(l - 1))))
            end
        end

        # B is already fp16: transformed activations are kept in that precision
        # because that is the cooperative instruction's input precision.
        @inbounds for r in Int32(0):Int32(15)
            idx = tid + r * Int32(PTQ1_COOP_WG)
            kk = idx % Int32(PTQ1_COOP_BK)
            j = idx ÷ Int32(PTQ1_COOP_BK)
            sB[kk + j * ldb + Int32(1)] =
                B[k0 + kk + (tn + j) * Int32(K) + Int32(1)]
        end
        KI.barrier()

        Base.Cartesian.@nexprs 2 u -> begin
            kt = Int32(u - 1) * Int32(16)
            a0 = AM(sA, Int32(1) + sm * Int32(16) + kt * lda, lda)
            a1 = AM(sA, Int32(1) + (sm + Int32(1)) * Int32(16) + kt * lda, lda)
            a2 = AM(sA, Int32(1) + (sm + Int32(2)) * Int32(16) + kt * lda, lda)
            a3 = AM(sA, Int32(1) + (sm + Int32(3)) * Int32(16) + kt * lda, lda)
            b0 = BM(sB, Int32(1) + kt + sn * Int32(16) * ldb, ldb)
            b1 = BM(sB, Int32(1) + kt + (sn + Int32(1)) * Int32(16) * ldb, ldb)
            c00 = muladd(a0, b0, c00)
            c10 = muladd(a1, b0, c10)
            c20 = muladd(a2, b0, c20)
            c30 = muladd(a3, b0, c30)
            c01 = muladd(a0, b1, c01)
            c11 = muladd(a1, b1, c11)
            c21 = muladd(a2, b1, c21)
            c31 = muladd(a3, b1, c31)
        end
        KI.barrier()
    end

    Mantle.accstore!(C, Int32(1) + tm + sm * Int32(16) +
        (tn + sn * Int32(16)) * Int32(M), Int32(M), c00)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(1)) * Int32(16) +
        (tn + sn * Int32(16)) * Int32(M), Int32(M), c10)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(2)) * Int32(16) +
        (tn + sn * Int32(16)) * Int32(M), Int32(M), c20)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(3)) * Int32(16) +
        (tn + sn * Int32(16)) * Int32(M), Int32(M), c30)
    Mantle.accstore!(C, Int32(1) + tm + sm * Int32(16) +
        (tn + (sn + Int32(1)) * Int32(16)) * Int32(M), Int32(M), c01)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(1)) * Int32(16) +
        (tn + (sn + Int32(1)) * Int32(16)) * Int32(M), Int32(M), c11)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(2)) * Int32(16) +
        (tn + (sn + Int32(1)) * Int32(16)) * Int32(M), Int32(M), c21)
    Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(3)) * Int32(16) +
        (tn + (sn + Int32(1)) * Int32(16)) * Int32(M), Int32(M), c31)
    return nothing
end

# Normalized 1024-wide FWHT. 256 threads process the 512 butterfly pairs in
# two passes per stage. `signs` spans the full input width; the transform itself
# is block diagonal. Forward folded matmuls apply S then H. Embedding lookup is
# the inverse and applies H then S.
function hadamard1024_kernel!(x, signs, width::Int32,
                                               blocks::Int32,
                                               ::Val{HASSIGNS},
                                               ::Val{INVERSE}) where {HASSIGNS,INVERSE}
    sh = KI.localmemory(Float32, Val((1024,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    block = wg % blocks
    col = wg ÷ blocks
    base = col * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        v = Float32(x[base + i + Int32(1)])
        if HASSIGNS && !INVERSE
            v *= Float32(signs[block * Int32(1024) + i + Int32(1)])
        end
        sh[i + Int32(1)] = v
    end
    KI.barrier()
    # `stride` doubles from one, so it is a power of two at every stage and the
    # butterfly's index split is a shift and a mask. Written as `÷` and `%` on a
    # runtime value it was two integer divisions per pair per stage — forty per
    # thread — on a device with no integer divide. Same defect, and same fix, as
    # `im2col_kernel!`'s `OW`; see the note there for what it was worth in a
    # pass that is bandwidth-bound rather than this one.
    stride = Int32(1)
    lg = Int32(0)
    while stride < Int32(1024)
        @inbounds for pass in Int32(0):Int32(1)
            p = t + pass * Int32(256)
            group = p >> lg
            j = p & (stride - Int32(1))
            aidx = group * (stride << 1) + j
            bidx = aidx + stride
            a = sh[aidx + Int32(1)]
            b = sh[bidx + Int32(1)]
            sh[aidx + Int32(1)] = a + b
            sh[bidx + Int32(1)] = a - b
        end
        KI.barrier()
        stride <<= 1
        lg += Int32(1)
    end
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        v = sh[i + Int32(1)] * 0.03125f0
        if HASSIGNS && INVERSE
            v *= Float32(signs[block * Int32(1024) + i + Int32(1)])
        end
        x[base + i + Int32(1)] = eltype(x)(v)
    end
    return nothing
end

"""
    hadamard!(ctx, x, signs=nothing; inverse=false, width=size(x,1))

Apply Prism's normalized blockwise 1024-point Walsh-Hadamard transform in
place. `x` contains one or more contiguous columns of `width` elements.
"""
function hadamard!(ctx, x, signs=nothing; inverse::Bool=false, width::Integer=size(x, 1))
    width % 1024 == 0 || throw(ArgumentError("Hadamard width $width is not divisible by 1024"))
    length(x) % width == 0 || throw(DimensionMismatch("array length is not divisible by Hadamard width"))
    signs !== nothing && length(signs) != width && throw(DimensionMismatch("sign vector has length $(length(signs)), expected $width"))
    blocks = width ÷ 1024
    columns = length(x) ÷ width
    harnesslaunch!(ctx.backend, hadamard1024_kernel!, x, signs === nothing ? x : signs,
        Int32(width), Int32(blocks), Val(signs !== nothing), Val(inverse);
        ndrange=blocks * columns * 256, workgroupsize = 256)
    x
end
