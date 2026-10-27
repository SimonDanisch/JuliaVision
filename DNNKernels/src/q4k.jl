"""
Q4_K weights: llama.cpp's 4.5-bit "K-quant", multiplied against int8 activations
with the GPU's packed dot product ([`KI.dot4x8`](@ref KernelInterface.dot4x8)).

Decode reads every weight once per token, so its floor is bytes over bandwidth
(`quant.jl`). Q4_K is 144 bytes per 256 weights against int8's 256: K2 Horizon 32B
reads 18.8 GB a token instead of 33.5, and decodes in 105 ms instead of 167 (8060S).

**The block is ggml's `block_q4_K`, byte for byte**, so a GGUF tensor can be used as
it is: `d` and `dmin` (fp16), twelve bytes packing eight 6-bit scales and eight
6-bit minimums, then 128 bytes of nibbles. Weight `i` of sub-block `j` (32 weights
each) is `d * scale[j] * q - dmin * min[j]`. Each 64 weights share 32 bytes of
nibbles: the low nibbles are the first 32 weights, the high nibbles the next 32. As
16-byte vectors a block is nine: the header and eight of nibbles. A row's blocks
follow each other, rows follow rows.

**Activations are quantized too**, to int8 in blocks of 32 with a float scale each
(ggml's `q8_1`), once per product. A sub-block's sum is then `dot4x8` over words
of nibbles and words of activations, scaled once; the minimum's term needs the sum
of the activation block, which the quantizer stores beside its scale.
"""

const QK_K = 256
const Q4K_VECS = 9                  # a block as 16-byte vectors
const UVec4 = NTuple{4,VecElement{UInt32}}

"""
    KQuantMatrix

A weight in one of llama.cpp's K-quant formats: [`Q4KMatrix`](@ref) or
[`Q6KMatrix`](@ref). They share the activation quantizer, the tensor-core GEMM and
the dequantizing fallback, each through its own `kq_stage`; the GEMV and the
quantizer are each format's own.
"""
abstract type KQuantMatrix end

"""
    Q4KMatrix

An `(m, k)` weight in Q4_K: `q` holds `9 * cld(k, 256)` vectors per row (see the
file's docstring). `k` is a multiple of 256.
"""
struct Q4KMatrix{Q} <: KQuantMatrix
    q::Q
    m::Int
    k::Int
end
Base.size(A::Q4KMatrix) = (A.m, A.k)
Base.size(A::Q4KMatrix, d::Integer) = d == 1 ? A.m : d == 2 ? A.k : 1
# Rank 2, but no float element type: a library GEMM must not take it for a matrix
# it can read.
Base.ndims(::Q4KMatrix) = 2
# `residentweights` passes every weight through `toback`; see `QInt8Matrix`'s.
toback(backend, A::Q4KMatrix) = Q4KMatrix(toback(backend, A.q), A.m, A.k)

# ── quantization ─────────────────────────────────────────────────────────────
#
# A port of ggml's `quantize_row_q4_K_ref`, which is what `llama-quantize` runs
# without an importance matrix. Written once for both sides: the kernel calls
# `q4k_block` per block, and so does the host reference the tests compare against.

# ggml's `nearest_int`: round half to even, in float.
@inline q4k_round(x::Float32) = unsafe_trunc(Int32, round(x))

@inline q4k_level(x::Float32, iscale::Float32, lo::Float32) =
    Float32(clamp(q4k_round(iscale * (x - lo)), Int32(0), Int32(15)))

"""
    q4k_fit(x) -> (scale, min)

One sub-block's scale and minimum: ggml's `make_qkx2_quants(32, 15, x, weights,
…, -1, 0.1, 20, false)`, a weighted least-squares search over 21 candidate
scales. The weight of a value is the sub-block's RMS plus its magnitude. As in
ggml, an accepted candidate's minimum is the origin the following candidates are
measured from.
"""
@inline function q4k_fit(x::NTuple{32,Float32})
    s2 = 0f0
    Base.Cartesian.@nexprs 32 i -> (s2 += x[i] * x[i])
    av = sqrt(s2 / 32f0)
    hi = x[1]
    lo = x[1]
    sw = av + abs(x[1])
    sx = sw * x[1]
    Base.Cartesian.@nexprs 31 j -> begin
        xi = x[j + 1]
        lo = xi < lo ? xi : lo
        hi = xi > hi ? xi : hi
        wi = av + abs(xi)
        sw += wi
        sx += wi * xi
    end
    lo = lo > 0f0 ? 0f0 : lo
    hi == lo && return (0f0, -lo)
    iscale = 15f0 / (hi - lo)
    scale = 1f0 / iscale
    best = 0f0
    Base.Cartesian.@nexprs 32 i -> begin
        diff = scale * q4k_level(x[i], iscale, lo) + lo - x[i]
        best += (av + abs(x[i])) * (diff * diff)
    end
    for is in Int32(0):Int32(20)
        iscale = (-1f0 + 0.1f0 * Float32(is) + 15f0) / (hi - lo)
        sl = 0f0
        sl2 = 0f0
        sxl = 0f0
        Base.Cartesian.@nexprs 32 i -> begin
            l = q4k_level(x[i], iscale, lo)
            wi = av + abs(x[i])
            sl += wi * l
            sl2 += wi * l * l
            sxl += wi * l * x[i]
        end
        D = sw * sl2 - sl * sl
        if D > 0f0
            ts = (sw * sxl - sx * sl) / D
            tm = (sl2 * sx - sl * sxl) / D
            if tm > 0f0
                tm = 0f0
                ts = sxl / sl2
            end
            cur = 0f0
            Base.Cartesian.@nexprs 32 i -> begin
                diff = ts * q4k_level(x[i], iscale, lo) + tm - x[i]
                cur += (av + abs(x[i])) * (diff * diff)
            end
            if cur < best
                best = cur
                scale = ts
                lo = tm
            end
        end
    end
    return (scale, -lo)
end

@inline q4k_bytes(b0, b1, b2, b3) =
    UInt32(b0) | (UInt32(b1) << 8) | (UInt32(b2) << 16) | (UInt32(b3) << 24)

@inline function q4k_nibble(x::Float32, d::Float32, dm::Float32)
    # A zero scale makes every value of the sub-block `-dm` whatever its nibble.
    # ggml leaves the search's levels there; zero dequantizes the same.
    d == 0f0 && return UInt32(0)
    return UInt32(clamp(q4k_round((x + dm) / d), Int32(0), Int32(15)))
end

"""
    q4k_block(W, row, b) -> NTuple{9,UVec4}

Block `b` (0-based) of row `row` (0-based) of the float matrix `W`.
"""
@inline function q4k_block(W, row::Int32, b::Int32)
    k0 = b * Int32(QK_K)
    fits = ntuple(Val(8)) do j
        q4k_fit(ntuple(i -> Float32(@inbounds W[row + Int32(1), k0 + Int32((j - 1) * 32 + i)]), Val(32)))
    end
    maxscale = 0f0
    maxmin = 0f0
    Base.Cartesian.@nexprs 8 j -> begin
        maxscale = fits[j][1] > maxscale ? fits[j][1] : maxscale
        maxmin = fits[j][2] > maxmin ? fits[j][2] : maxmin
    end
    iscale = maxscale > 0f0 ? 63f0 / maxscale : 0f0
    imin = maxmin > 0f0 ? 63f0 / maxmin : 0f0
    ls = ntuple(j -> min(UInt8(63), q4k_round(iscale * fits[j][1]) % UInt8), Val(8))
    lm = ntuple(j -> min(UInt8(63), q4k_round(imin * fits[j][2]) % UInt8), Val(8))
    dh = Float16(maxscale / 63f0)
    mh = Float16(maxmin / 63f0)
    # ggml's packing of eight 6-bit pairs into twelve bytes (`get_scale_min_k4`
    # reverses it): bytes 0-3 and 4-7 hold the first four scales and minimums,
    # with the top two bits of the last four above them; bytes 8-11 the low
    # four bits of the last four.
    sb(j) = ls[j] | ((ls[j + 4] >> 4) << 6)
    mb(j) = lm[j] | ((lm[j + 4] >> 4) << 6)
    lb(j) = (ls[j + 4] & 0x0f) | ((lm[j + 4] & 0x0f) << 4)
    header = (VecElement(UInt32(reinterpret(UInt16, dh)) | (UInt32(reinterpret(UInt16, mh)) << 16)),
              VecElement(q4k_bytes(sb(1), sb(2), sb(3), sb(4))),
              VecElement(q4k_bytes(mb(1), mb(2), mb(3), mb(4))),
              VecElement(q4k_bytes(lb(1), lb(2), lb(3), lb(4))))
    d = ntuple(j -> Float32(dh) * Float32(ls[j]), Val(8))
    dm = ntuple(j -> Float32(mh) * Float32(lm[j]), Val(8))
    # Vector `1 + 2g + h` holds bytes `16h:16h+15` of group `g`'s 32: the low
    # nibbles are weights `64g + 16h + i` (sub-block 2g), the high ones 32 later
    # (sub-block 2g + 1).
    nibbles = ntuple(Val(8)) do v
        g, h = (v - 1) ÷ 2, (v - 1) % 2
        ntuple(Val(4)) do w
            word = UInt32(0)
            for byte in 0:3
                i = k0 + Int32(64g + 16h + 4(w - 1) + byte + 1)
                lo = q4k_nibble(Float32(@inbounds W[row + Int32(1), i]), d[2g + 1], dm[2g + 1])
                hi = q4k_nibble(Float32(@inbounds W[row + Int32(1), i + Int32(32)]), d[2g + 2], dm[2g + 2])
                word |= (lo | (hi << 4)) << (8byte)
            end
            VecElement(word)
        end
    end
    return (header, nibbles...)
end

function q4k_quantize_kernel!(Q, W, rows::Int32, nb::Int32)
    t = Int32(KI.get_global_id().x) - Int32(1)
    if t < rows * nb
        # Consecutive work-items take consecutive rows, so the column-major
        # weight is read along its contiguous axis.
        row, b = t % rows, t ÷ rows
        blk = q4k_block(W, row, b)
        base = (row * nb + b) * Int32(Q4K_VECS)
        Base.Cartesian.@nexprs 9 v -> (@inbounds Q[base + Int32(v)] = blk[v])
    end
    return nothing
end

"""
    quantizeq4k(backend, W) -> Q4KMatrix

Quantize a device-resident `(M, K)` float weight on the device. `K` must be a
multiple of 256. `W` is not modified.
"""
function quantizeq4k(backend, W::Union{AbstractMatrix,Mantle.Buffer{<:Any,2}})
    rows, K = size(W)
    K % QK_K == 0 || throw(ArgumentError("Q4_K needs a reduction extent that is a multiple of $QK_K, got $K"))
    nb = K ÷ QK_K
    dev = Mantle.todevice(backend)
    g = Mantle.Graph(dev)
    q = Mantle.Buffer(dev, UVec4, (Q4K_VECS * nb, rows))
    Mantle.dispatch!(g, q4k_quantize_kernel!, (q, W, Int32(rows), Int32(nb)), rows * nb;
                     group = 256, name = "q4kquant")
    runonce!(g)
    return Q4KMatrix(q, rows, K)
end

"""
    q4k_quantize(W::AbstractMatrix) -> Vector{UVec4}

The host reference of [`quantizeq4k`](@ref), from the same `q4k_block`.
"""
function q4k_quantize(W::AbstractMatrix)
    rows, K = size(W)
    nb = K ÷ QK_K
    Q = Vector{UVec4}(undef, Q4K_VECS * nb * rows)
    for row in 0:(rows - 1), b in 0:(nb - 1)
        blk = q4k_block(W, Int32(row), Int32(b))
        base = (row * nb + b) * Q4K_VECS
        for v in 1:Q4K_VECS
            Q[base + v] = blk[v]
        end
    end
    return Q
end

# ── dequantization ───────────────────────────────────────────────────────────

# Word `i` (0-based) of the twelve scale bytes, selected rather than indexed: a
# tuple indexed by a run-time value is spilled to memory on the device.
@inline q4k_word(hd::UVec4, i) = ifelse(i == 0, hd[2].value, ifelse(i == 1, hd[3].value, hd[4].value))
@inline q4k_byte(hd::UVec4, b) = (q4k_word(hd, b ÷ 4) >> (8 * (b % 4))) & 0xff

"""
    q4k_scalemin(hd, j) -> (scale, min)

The 6-bit scale and minimum of sub-block `j` (0-based) from a block's header
vector: ggml's `get_scale_min_k4`.
"""
@inline function q4k_scalemin(hd::UVec4, j)
    if j < 4
        return (q4k_byte(hd, j) & 0x3f, q4k_byte(hd, j + 4) & 0x3f)
    else
        return ((q4k_byte(hd, j + 4) & 0x0f) | ((q4k_byte(hd, j - 4) >> 6) << 4),
                (q4k_byte(hd, j + 4) >> 4) | ((q4k_byte(hd, j) >> 6) << 4))
    end
end

@inline q4k_half(w::UInt32, i) = Float32(reinterpret(Float16, UInt16((w >> (16 * i)) & 0xffff)))

"""
    q4k_dequantize(Q::AbstractVector{UVec4}, M, K) -> Matrix{Float32}

Host dequantization, ggml's `dequantize_row_q4_K` per row.
"""
function q4k_dequantize(Q::AbstractVector{UVec4}, rows::Integer, K::Integer)
    nb = K ÷ QK_K
    W = Matrix{Float32}(undef, rows, K)
    for row in 0:(rows - 1), b in 0:(nb - 1)
        base = (row * nb + b) * Q4K_VECS
        hd = Q[base + 1]
        d, dmin = q4k_half(hd[1].value, 0), q4k_half(hd[1].value, 1)
        for j in 0:7
            sc, m = q4k_scalemin(hd, j)
            g, high = j ÷ 2, j % 2
            for i in 0:31
                byte = 32g + i
                word = Q[base + 2 + byte ÷ 16][(byte % 16) ÷ 4 + 1].value
                q = (word >> (8 * (byte % 4) + 4high)) & 0x0f
                W[row + 1, b * QK_K + 32j + i + 1] = d * Float32(sc) * Float32(q) - dmin * Float32(m)
            end
        end
    end
    return W
end

"""
    kquant_dequant!(W, Ws, kind, rows, nb)

The fp16 matrix of a K-quant weight, for products the GEMM cannot take: one
work-item per row and staging step (`kq_stage`), 32 weights. Consecutive
work-items take consecutive rows, so the column-major `W` is written along its
contiguous axis.
"""
@inline function kquant_dequant!(W, Ws, kind, rows::Int32, nb::Int32)
    t = Int32(KI.get_global_id().x) - Int32(1)
    if t < rows * nb * Int32(8)
        row, rest = t % rows, t ÷ rows
        b, v = rest ÷ Int32(8), rest % Int32(8)
        g, h = v ÷ Int32(2), v % Int32(2)
        x = kq_stage(kind, Ws, row, b, g, h, nb)
        k0 = b * Int32(QK_K) + Int32(64) * g + Int32(16) * h      # 0-based column of the first sixteen
        Base.Cartesian.@nexprs 4 w -> Base.Cartesian.@nexprs 4 e -> begin
            i = k0 + Int32(4 * (w - 1) + e - 1)
            @inbounds W[row + Int32(1) + i * rows] = eltype(W)(x[w][e].value)
            @inbounds W[row + Int32(1) + (i + Int32(32)) * rows] = eltype(W)(x[w + 4][e].value)
        end
    end
    return nothing
end

q4k_dequant_kernel!(W, Q, rows::Int32, nb::Int32) = kquant_dequant!(W, (Q,), Val(:q4k), rows, nb)

# ── activations ──────────────────────────────────────────────────────────────

"""
    q8act_kernel!(xq, xds, B, K, n)

Each 32 values of each column of `B` (`K x N`) as int8 with one scale: ggml's
`quantize_row_q8_1`, with the sum of the block kept as `d * sum` and both in
fp32. `xq` holds a column as `K ÷ 16` vectors of four words, `xds` the `(d, d *
sum)` pairs, `K ÷ 32` per column. One work-item per block.
"""
function q8act_kernel!(xq, xds, B, K::Int32, n::Int32)
    t = Int32(KI.get_global_id().x) - Int32(1)
    if t < n
        nk = K ÷ Int32(32)
        blk, col = t % nk, t ÷ nk
        k0 = blk * Int32(32) + col * K
        amax = 0f0
        Base.Cartesian.@nexprs 32 i -> begin
            x_i = Float32(@inbounds B[k0 + Int32(i)])
            amax = max(amax, abs(x_i))
        end
        d = amax / 127f0
        id = d == 0f0 ? 0f0 : 1f0 / d
        Base.Cartesian.@nexprs 32 i -> (q_i = q4k_round(x_i * id))
        s = Int32(0)
        Base.Cartesian.@nexprs 32 i -> (s += q_i)
        word(a, b, c, e) = q4k_bytes(a % UInt8, b % UInt8, c % UInt8, e % UInt8)
        base = (blk + col * nk) * Int32(2)
        @inbounds begin
            xq[base + Int32(1)] = (VecElement(word(q_1, q_2, q_3, q_4)), VecElement(word(q_5, q_6, q_7, q_8)),
                                   VecElement(word(q_9, q_10, q_11, q_12)), VecElement(word(q_13, q_14, q_15, q_16)))
            xq[base + Int32(2)] = (VecElement(word(q_17, q_18, q_19, q_20)), VecElement(word(q_21, q_22, q_23, q_24)),
                                   VecElement(word(q_25, q_26, q_27, q_28)), VecElement(word(q_29, q_30, q_31, q_32)))
            xds[base + Int32(1)] = d
            xds[base + Int32(2)] = d * Float32(s)
        end
    end
    return nothing
end

# ── the product ──────────────────────────────────────────────────────────────
#
# A group of 32 lanes per `R` rows, its lanes along the reduction: eight lanes a
# block, so a group reads four consecutive blocks of a row (576 contiguous bytes)
# per step. Lane `p` of a block's eight takes nibble vector `p`: half `h`
# of group `j`, sixteen weights of sub-block `2j` and the sixteen of `2j + 1`.
# Its activations are two vectors per column, read once per step and used for
# all `R` rows; that reuse is what `R` is for.
#
# The minimum's term belongs to a whole sub-block, so of the two lanes sharing
# one only `h == 0` adds it. Columns are `N` up to eight: a token, or a token and
# drafts checked in the same pass, which read every weight once for all of them.

@inline q4k_lo(w::UInt32) = w & 0x0f0f0f0f
@inline q4k_hi(w::UInt32) = (w >> 4) & 0x0f0f0f0f

@inline q4k_dot(q::UVec4, x::UVec4, f) =
    KI.dot4x8(f(q[1].value), x[1].value) + KI.dot4x8(f(q[2].value), x[2].value) +
    KI.dot4x8(f(q[3].value), x[3].value) + KI.dot4x8(f(q[4].value), x[4].value)

const Q4K_WG = 256

# The sum over a group of 32 lanes, in lane 0: a `shfl_down` tree, right in a
# 32- or a 64-wide subgroup alike, since no step reads past its own group.
@inline function q4k_sum32(v::Float32)
    v += KI.shfl_down(v, 16)
    v += KI.shfl_down(v, 8)
    v += KI.shfl_down(v, 4)
    v += KI.shfl_down(v, 2)
    v += KI.shfl_down(v, 1)
    return v
end

q4krows(n) = n >= 4 ? 2 : 1

"""
The widest product that runs as GEMV passes of eight columns, each a read of the
weight, where the K-quant GEMM (`kquant_gemm!`) could run instead. Measured on the
8060S at Horizon's qkv and down projections, in ms:

    columns            16     32     64    128    512
    qkv   GEMV passes  0.25   0.46   0.92   1.83   7.30
          GEMM         0.33   0.36   0.49   0.71   2.79
    down  GEMV passes  0.84   1.66   3.30   6.58  26.34
          GEMM         1.39   1.13   1.45   1.96   9.54
"""
const Q4K_GEMV_COLUMNS = Ref(16)

"""Where the GEMM has no tile, GEMV passes up to this many columns and the fp16
GEMM on a dequantized copy of the weight past it."""
const Q4K_GEMV_FALLBACK = 128

const Q4K_GEMV_KERNELS = Dict{Tuple{Int,Int},Any}()
for N in 1:8, R in (1, 2, 4)
    name = Symbol("q4k_gemv_", N, "_", R, "!")
    @eval function $name(out, Q, xq, xds, bias, epi, rows::Int32, nb::Int32, ::Val{HASBIAS}) where {HASBIAS}
        # Groups of 32 lanes from the local id rather than the hardware's
        # subgroups, which may be 64 wide; the sum at the end stays within them.
        lid = Int32(KI.get_local_id().x) - Int32(1)
        lane = lid % Int32(32)
        wave = (Int32(KI.get_group_id().x) - Int32(1)) * Int32(Q4K_WG ÷ 32) + lid ÷ Int32(32)
        r0 = wave * Int32($R)
        p = lane % Int32(8)
        j, h = p ÷ Int32(2), p % Int32(2)
        cols = nb * Int32(16)           # activation vectors, and (d, d * sum) floats, a column
        keepmin = h == Int32(0) ? 1f0 : 0f0
        Base.Cartesian.@nexprs $N c -> Base.Cartesian.@nexprs $R r -> (acc_c_r = 0f0)
        b = lane ÷ Int32(8)
        while b < nb
            Base.Cartesian.@nexprs $R r -> begin
                vb_r = (min(r0 + Int32(r - 1), rows - Int32(1)) * nb + b) * Int32(Q4K_VECS)
                hd_r = @inbounds Q[vb_r + Int32(1)]
                qv_r = @inbounds Q[vb_r + Int32(2) + Int32(2) * j + h]
                d_r = q4k_half(hd_r[1].value, 0)
                dmin_r = q4k_half(hd_r[1].value, 1)
                sl_r, ml_r = q4k_scalemin(hd_r, Int32(2) * j)
                sh_r, mh_r = q4k_scalemin(hd_r, Int32(2) * j + Int32(1))
            end
            xv = b * Int32(16) + Int32(4) * j + h       # 0-based, the low half's vector
            xs = (b * Int32(8) + Int32(2) * j) * Int32(2)
            Base.Cartesian.@nexprs $N c -> begin
                xlo = @inbounds xq[xv + Int32(1) + Int32(c - 1) * cols]
                xhi = @inbounds xq[xv + Int32(3) + Int32(c - 1) * cols]
                o = xs + Int32(c - 1) * cols
                dlo, slo = @inbounds xds[o + Int32(1)], xds[o + Int32(2)]
                dhi, shi = @inbounds xds[o + Int32(3)], xds[o + Int32(4)]
                Base.Cartesian.@nexprs $R r -> begin
                    il = q4k_dot(qv_r, xlo, q4k_lo)
                    ih = q4k_dot(qv_r, xhi, q4k_hi)
                    acc_c_r += d_r * (Float32(sl_r) * dlo * Float32(il) + Float32(sh_r) * dhi * Float32(ih)) -
                               keepmin * dmin_r * (Float32(ml_r) * slo + Float32(mh_r) * shi)
                end
            end
            b += Int32(4)
        end
        Base.Cartesian.@nexprs $N c -> Base.Cartesian.@nexprs $R r -> begin
            v_c_r = q4k_sum32(acc_c_r)
            row_c_r = r0 + Int32(r - 1)
            if lane == Int32(0) && row_c_r < rows
                HASBIAS && (v_c_r += Float32(@inbounds bias[row_c_r + Int32(1)]))
                @inbounds out[row_c_r + Int32(1) + Int32(c - 1) * rows] = eltype(out)(epi(v_c_r))
            end
        end
        return nothing
    end
    Q4K_GEMV_KERNELS[(N, R)] = getfield(@__MODULE__, name)
end

"""
    kqgemv!(backend, out, A::KQuantMatrix, B; bias, epilogue, R) -> out

`out = epilogue.(A * B .+ bias)` for a `B` of one to eight columns: the
activations quantized to int8, then one pass over the weight for all columns. `R`
is the rows a group of 32 lanes takes; two from four columns on, where reusing the
activations across rows was measured faster on the 8060S.
"""
function kqgemv!(backend, out, A::KQuantMatrix, B; bias = nothing, epilogue = identity,
                  R::Int = q4krows(size(B, 2)))
    m, k = size(A)
    n = size(B, 2)
    size(B, 1) == k && size(out) == (m, n) || throw(DimensionMismatch("kqgemv! operands"))
    1 <= n <= 8 || throw(ArgumentError("kqgemv! takes one to eight columns, got $n"))
    dev = Mantle.todevice(backend)
    g = Mantle.Graph(dev)
    xq = Mantle.Buffer(dev, UVec4, (k ÷ 16, n))
    xds = Mantle.Buffer(dev, Float32, (k ÷ 16, n))
    Mantle.dispatch!(g, q8act_kernel!, (xq, xds, B, Int32(k), Int32(n * (k ÷ 32))), n * (k ÷ 32);
                     group = 256, name = "q8act")
    Mantle.dispatch!(g, kqgemv_kernels(A)[(n, R)],
                     (out, kqarrays(A)..., xq, xds, bias === nothing ? out : bias, epilogue,
                      Int32(m), Int32(k ÷ QK_K), Val(bias !== nothing)),
                     cld(cld(m, R), Q4K_WG ÷ 32) * Q4K_WG; group = Q4K_WG, name = "q4kgemv")
    runonce!(g)
    return out
end

# ── the product for prompts ──────────────────────────────────────────────────
#
# Past a few hundred columns a product is compute-bound, and the GEMV's int8 dot
# products run on the vector ALU while the matrix units sit idle. This is the
# staged cooperative-matrix GEMM of `q8gemm_pipelined_kernel!`, reading Q4_K: the
# weight tile is dequantized to fp16 as it is staged, and never exists in fp16
# anywhere else, where the fallback below writes and reads 2 bytes a weight of
# the whole matrix per product.
#
# A step of the reduction is one group of 64 weights (`K` is a multiple of 256).
# A row's group is two 16-byte vectors of nibbles, each sixteen weights of the
# group's two sub-blocks, so a thread stages one (row, vector) pair: one header
# and one vector load, 32 weights. The tiles are k-contiguous, a row (weights) or
# a column (activations) of `64 + 8` halves apiece, read as row- and column-major
# fragments.

const Q4K_GEMM_BK = 64
const Q4K_GEMM_RS4 = (Q4K_GEMM_BK + 8) ÷ 4   # a staged row, in 4-half vectors

@inline q4k_h4(a, b, c, d) = (VecElement(Float16(a)), VecElement(Float16(b)),
                              VecElement(Float16(c)), VecElement(Float16(d)))
# The four weights of byte lane `f` (low or high nibbles) of word `w`.
@inline q4k_word4(w::UInt32, f, s::Float32, o::Float32) =
    q4k_h4(s * Float32(f(w) & 0x0f) - o, s * Float32((f(w) >> 8) & 0x0f) - o,
           s * Float32((f(w) >> 16) & 0x0f) - o, s * Float32((f(w) >> 24) & 0x0f) - o)

"""
    kq_stage(Val(kind), Ws, row, b, g, h, nb) -> NTuple{8,NTuple{4,VecElement{Float16}}}

The staging step of a K-quant format: weights `16h + i` and `32 + 16h + i` (`i`
0 to 15) of the `g`-th 64 (`g` 0 to 3) of block `b` of row `row`, dequantized,
four to a vector, the first sixteen then the second. All 0-based; `Ws` are the
format's arrays, `nb` the blocks in a row.
"""
function kq_stage end

# A Q4_K group is one vector of nibbles per half: low nibbles the first 32
# weights, high the next.
@inline function kq_stage(::Val{:q4k}, Ws, row::Int32, b::Int32, g::Int32, h::Int32, nb::Int32)
    Q = Ws[1]
    vb = (row * nb + b) * Int32(Q4K_VECS)
    hd = @inbounds Q[vb + Int32(1)]
    qv = @inbounds Q[vb + Int32(2) + Int32(2) * g + h]
    d, dmin = q4k_half(hd[1].value, 0), q4k_half(hd[1].value, 1)
    sl, ml = q4k_scalemin(hd, Int32(2) * g)
    sh, mh = q4k_scalemin(hd, Int32(2) * g + Int32(1))
    dl, ol = d * Float32(sl), dmin * Float32(ml)
    dh, oh = d * Float32(sh), dmin * Float32(mh)
    lo = ntuple(w -> q4k_word4(qv[w].value, identity, dl, ol), Val(4))
    hi = ntuple(w -> q4k_word4(qv[w].value, x -> x >> 4, dh, oh), Val(4))
    return (lo..., hi...)
end

# The same weights as `kq_stage`, staged for the GEMM's matrix units. `kq_stage`
# narrows `s * q - o`, a float product, with `Float16`, which Lava's
# exact-narrowing pass spells in ~20 integer operations a weight so that a
# driver cannot fuse the product into the conversion (RADV would: one rounding
# of the exact value instead of Julia's two). For an operand staged for the
# matrix units either rounding is right, and `stagehalf` lets the conversion be
# the hardware's: the select it narrows is not arithmetic the pass looks
# through. It differs from `Float16` only for NaN, which no weight is. GPU time
# at 512 columns: Q4_K 15-19% less (8192x2048: 1.07 -> 0.87 ms), Q6_K 8-16%.
# Computing the weights in Float16 from exact halves was a little faster still,
# but 4-10x less accurate: a Q4_K weight is a scale and a minimum that cancel.
function kq_stage_mma end

@inline stagehalf(x::Float32) = Float16(ifelse(x == x, x, 0f0))
@inline q4k_h4_mma(a, b, c, d) = (VecElement(stagehalf(a)), VecElement(stagehalf(b)),
                                  VecElement(stagehalf(c)), VecElement(stagehalf(d)))
@inline q4k_word4_mma(w::UInt32, f, s::Float32, o::Float32) =
    q4k_h4_mma(s * Float32(f(w) & 0x0f) - o, s * Float32((f(w) >> 8) & 0x0f) - o,
               s * Float32((f(w) >> 16) & 0x0f) - o, s * Float32((f(w) >> 24) & 0x0f) - o)

@inline function kq_stage_mma(::Val{:q4k}, Ws, row::Int32, b::Int32, g::Int32, h::Int32, nb::Int32)
    Q = Ws[1]
    vb = (row * nb + b) * Int32(Q4K_VECS)
    hd = @inbounds Q[vb + Int32(1)]
    qv = @inbounds Q[vb + Int32(2) + Int32(2) * g + h]
    d, dmin = q4k_half(hd[1].value, 0), q4k_half(hd[1].value, 1)
    sl, ml = q4k_scalemin(hd, Int32(2) * g)
    sh, mh = q4k_scalemin(hd, Int32(2) * g + Int32(1))
    dl, ol = d * Float32(sl), dmin * Float32(ml)
    dh, oh = d * Float32(sh), dmin * Float32(mh)
    lo = ntuple(w -> q4k_word4_mma(qv[w].value, identity, dl, ol), Val(4))
    hi = ntuple(w -> q4k_word4_mma(qv[w].value, x -> x >> 4, dh, oh), Val(4))
    return (lo..., hi...)
end

"""
    kquant_gemm!(C, Ws, B, epi, Val(kind), Val(M), Val(N), Val(K), Val(cfg))

`C = epi(A * B)` for a K-quant weight `A` (`M x K`, its arrays `Ws`) and fp16 `B`
(`K x N`, `K` contiguous and 16-byte aligned). `cfg = (STM, STN, WM, WN)`: `WM x
WN` subgroups of `16STM x 16STN` outputs each. A thread stages one row's
`kq_stage` per step of 64.
"""
@inline function kquant_gemm!(C, Ws, B, epi, kind, ::Val{M}, ::Val{N}, ::Val{K},
                              ::Val{CFG}) where {M,N,K,CFG}
    # Sizes from the type parameters, never a local: see `flash.jl` on `@localmem`.
    sA = KI.localmemory(Q8_H4, Val((16 * CFG[1] * CFG[3] * Q4K_GEMM_RS4,)), Val(1))
    sB = KI.localmemory(Q8_H4, Val((16 * CFG[2] * CFG[4] * Q4K_GEMM_RS4,)), Val(2))
    STM, STN, WM, WN = CFG
    BM, BN, WG = 16STM * WM, 16STN * WN, 32WM * WN
    RS4 = Int32(Q4K_GEMM_RS4)
    NB = Int32(K ÷ QK_K)
    APT = 2BM ÷ WG                     # (row, half) pairs a thread stages
    BPT = BN * (Q4K_GEMM_BK ÷ 8) ÷ WG  # 16-byte activation chunks a thread stages

    tid = Int32(KI.get_local_id().x) - Int32(1)
    blk = Int32(KI.get_group_id().x) - Int32(1)
    tm = (blk % Int32(M ÷ BM)) * Int32(BM)
    tn = (blk ÷ Int32(M ÷ BM)) * Int32(BN)
    sg = tid ÷ Int32(32)
    sm, sn = (sg % Int32(WM)) * Int32(STM), (sg ÷ Int32(WM)) * Int32(STN)
    @inbounds begin
        Base.Cartesian.@nexprs 4 nt -> Base.Cartesian.@nexprs 4 mt -> if mt <= STM && nt <= STN
            c_mt_nt = zero(Mantle.AcceleratedMatrix{Float32,16,16,Mantle.Accumulator})
        end
        Base.Cartesian.@nexprs 4 a -> if a <= APT
            row_a = (tid + Int32((a - 1) * WG)) ÷ Int32(2)
            h_a = (tid + Int32((a - 1) * WG)) % Int32(2)
        end
        Base.Cartesian.@nexprs 8 c -> if c <= BPT
            kc_c = (tid + Int32((c - 1) * WG)) % Int32(Q4K_GEMM_BK ÷ 8)
            col_c = (tid + Int32((c - 1) * WG)) ÷ Int32(Q4K_GEMM_BK ÷ 8)
            pb_c = reinterpret(Ptr{Q8_U4}, pointer(B) + 2 * (Int(tn + col_c) * K + Int(kc_c) * 8))
        end
        for kb in Int32(0):Int32(K ÷ Q4K_GEMM_BK - 1)
            b, g = kb ÷ Int32(4), kb % Int32(4)
            Base.Cartesian.@nexprs 4 a -> if a <= APT
                x = kq_stage_mma(kind, Ws, tm + row_a, b, g, h_a, NB)
                base = Int32(1) + row_a * RS4 + Int32(4) * h_a
                Base.Cartesian.@nexprs 4 w -> begin
                    sA[base + Int32(w - 1)] = x[w]
                    sA[base + Int32(8) + Int32(w - 1)] = x[w + 4]
                end
            end
            k0 = Int(kb) * Q4K_GEMM_BK
            Base.Cartesian.@nexprs 8 c -> if c <= BPT
                v = Core.Intrinsics.pointerref(pb_c + 2k0, 1, 16)
                sB[Int32(1) + col_c * RS4 + Int32(2) * kc_c] = q8h4lo(v)
                sB[Int32(2) + col_c * RS4 + Int32(2) * kc_c] = q8h4hi(v)
            end
            KI.barrier()
            Base.Cartesian.@nexprs 4 u -> begin
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
        Base.Cartesian.@nexprs 4 nt -> Base.Cartesian.@nexprs 4 mt -> if mt <= STM && nt <= STN
            Mantle.accstore!(C, Int32(1) + tm + (sm + Int32(mt - 1)) * Int32(16) +
                (tn + (sn + Int32(nt - 1)) * Int32(16)) * Int32(M), Int32(M), c_mt_nt, epi)
        end
    end
    return nothing
end

q4kgemm_kernel!(C, Q, B, epi, m::Val, n::Val, k::Val, cfg::Val) =
    kquant_gemm!(C, (Q,), B, epi, Val(:q4k), m, n, k, cfg)

kqarrays(A::Q4KMatrix) = (A.q,)
kqgemm_kernel(::Q4KMatrix) = q4kgemm_kernel!
kqdequant_kernel(::Q4KMatrix) = q4k_dequant_kernel!
kqgemv_kernels(::Q4KMatrix) = Q4K_GEMV_KERNELS

"""
    kqgemm_tile(caps, Tout, m, k, n) -> cfg | nothing

The tile `kquant_gemm!` runs an `(m, k) x (k, n)` K-quant product at, or `nothing`
where it cannot: no 16 x 16 cooperative matrices on 32-lane subgroups, or extents
no tile divides.
"""
function kqgemm_tile(caps, ::Type{Tout}, m::Integer, k::Integer, n::Integer) where {Tout}
    caps.coopmat && caps.coopmatsubgroup == 32 && caps.tile == 16 || return nothing
    Tout in (Float16, Float32) || return nothing
    k % QK_K == 0 && m % 64 == 0 && n % 16 == 0 || return nothing
    max(m * k, k * n, m * n) <= typemax(Int32) || return nothing
    cfg = n % 128 == 0 ? (2, 4, 2, 2) : n % 64 == 0 ? (2, 2, 2, 2) : n % 32 == 0 ? (2, 1, 2, 2) : (2, 1, 2, 1)
    STM, STN, WM, WN = cfg
    shared = 8 * (16STM * WM + 16STN * WN) * Q4K_GEMM_RS4
    32WM * WN <= caps.workgrouplimit && shared <= caps.sharedbudget || return nothing
    return cfg
end

"""The launch for the K-quant GEMM at `cfg`."""
function kqgemm_launch(C, A::KQuantMatrix, B, epi, cfg)
    m, k = size(A)
    n = size(B, 2)
    STM, STN, WM, WN = cfg
    BM, BN, WG = 16STM * WM, 16STN * WN, 32WM * WN
    (kern = kqgemm_kernel(A),
     args = (C, kqarrays(A)..., B, epi, Val(m), Val(n), Val(k), Val(cfg)),
     ndrange = (m ÷ BM) * (n ÷ BN) * WG, group = WG)
end

"""Run the K-quant GEMM now; `cfg` from [`kqgemm_tile`](@ref)."""
function kqgemm!(C, A::KQuantMatrix, B; cfg, epilogue = identity)
    m, k = size(A)
    size(B, 1) == k && size(C) == (m, size(B, 2)) || throw(DimensionMismatch("kqgemm operands"))
    l = kqgemm_launch(C, A, B, epilogue, cfg)
    l = (; l..., args = map(Mantle.storage, l.args))
    harnesslaunch!(KernelAbstractions.get_backend(C), l.kern, l.args...;
                   ndrange = l.ndrange, workgroupsize = l.group)
    return C
end
