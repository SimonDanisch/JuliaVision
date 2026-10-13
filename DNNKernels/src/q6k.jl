"""
Q6_K weights: llama.cpp's 6.5625-bit K-quant, for the weights a mixed policy
gives more bits (llama.cpp's Q4_K_M puts `lm_head` and half the down projections
in it). The GEMM and the dequantizing fallback are the K-quant ones in `q4k.jl`,
through this format's `kq_stage`; the quantizer and the GEMV are here.

Weight `i` of a 256-weight block is `d * sc[i ÷ 16] * (q - 32)`: six-bit `q`, an
int8 scale per sixteen weights, one fp16 `d`. ggml's order within each half of
128: weight `l + 32Q` (`l` 0 to 31, quadrant `Q` 0 to 3) keeps its low four bits
in byte `l` of the half's 64 `ql` bytes (`Q` 0 and 2) or byte `32 + l` (`Q` 1 and
3), the low nibble for `Q` 0 and 1, and its high two bits at `2Q` in byte `l` of
the half's 32 `qh` bytes.

**Not GGUF's byte layout, the same bytes.** ggml's `block_q6_K` is 210 bytes,
which no 16-byte load can walk. Here a matrix is four planes: `ql` (128 bytes a
block), `qh` (64), `sc` (the sixteen scales) and `d`, each row's blocks in order.
A GGUF tensor converts by moving bytes, and a weight is still 6.5625 bits.

The GEMV gets `q - 32` as a signed int8 without a minimum term: for six bits,
`y = q ⊻ 0x20` is `q - 32` when bit 5 of `y` is clear and `q - 32 + 64`
otherwise, so `y | (y & 0x20) * 6` sign-extends it, a byte at a time.
"""

"""
    Q6KMatrix

An `(m, k)` weight in Q6_K, `k` a multiple of 256: per block, eight vectors of
`ql`, four of `qh`, one of scales and an fp16 `d` (see the file's docstring).
"""
struct Q6KMatrix{L,H,S,D} <: KQuantMatrix
    ql::L       # (8nb, m) UVec4
    qh::H       # (4nb, m) UVec4
    sc::S       # (nb, m) UVec4, sixteen int8
    d::D        # (nb, m) Float16
    m::Int
    k::Int
end
Base.size(A::Q6KMatrix) = (A.m, A.k)
Base.size(A::Q6KMatrix, d::Integer) = d == 1 ? A.m : d == 2 ? A.k : 1
Base.ndims(::Q6KMatrix) = 2
toback(backend, A::Q6KMatrix) =
    Q6KMatrix(toback(backend, A.ql), toback(backend, A.qh), toback(backend, A.sc),
              toback(backend, A.d), A.m, A.k)

# ── quantization ─────────────────────────────────────────────────────────────
#
# A port of ggml's `quantize_row_q6_K_ref`, shared by the kernel and the host
# reference like `q4k_block`.

@inline q6k_level(x::Float32, iscale::Float32) = clamp(q4k_round(iscale * x), Int32(-32), Int32(31))

"""
    q6k_fit(x) -> (scale, iscale)

Sixteen weights' scale: ggml's `make_qx_quants(16, 32, x, L, 1, NULL)`, a search
over 19 candidates weighting each weight by its square. Also the inverse scale
its levels were taken at, which ggml keeps for a sub-block whose quantized scale
comes out zero.
"""
@inline function q6k_fit(x::NTuple{16,Float32})
    mx = 0f0
    amax = 0f0
    Base.Cartesian.@nexprs 16 i -> begin
        ax_i = abs(x[i])
        if ax_i > amax
            amax = ax_i
            mx = x[i]
        end
    end
    amax < 1f-15 && return (0f0, 0f0)
    iscale = -32f0 / mx
    sumlx = 0f0
    suml2 = 0f0
    Base.Cartesian.@nexprs 16 i -> begin
        l = Float32(q6k_level(x[i], iscale))
        w = x[i] * x[i]
        sumlx += w * x[i] * l
        suml2 += w * l * l
    end
    scale = suml2 != 0f0 ? sumlx / suml2 : 0f0
    best = scale * sumlx
    bestiscale = iscale
    for is in Int32(-9):Int32(9)
        is == Int32(0) && continue
        iscale = -(32f0 + 0.1f0 * Float32(is)) / mx
        sumlx = 0f0
        suml2 = 0f0
        Base.Cartesian.@nexprs 16 i -> begin
            l = Float32(q6k_level(x[i], iscale))
            w = x[i] * x[i]
            sumlx += w * x[i] * l
            suml2 += w * l * l
        end
        if suml2 > 0f0 && sumlx * sumlx > best * suml2
            scale = sumlx / suml2
            best = scale * sumlx
            bestiscale = iscale
        end
    end
    return (scale, bestiscale)
end

"""
    q6k_block(W, row, b) -> (ql, qh, sc, d)

Block `b` (0-based) of row `row` (0-based) of the float matrix `W`: eight `ql`
vectors, four `qh` vectors, the scale vector and `d`.
"""
@inline function q6k_block(W, row::Int32, b::Int32)
    k0 = b * Int32(QK_K)
    fits = ntuple(Val(16)) do j
        q6k_fit(ntuple(i -> Float32(@inbounds W[row + Int32(1), k0 + Int32(16 * (j - 1) + i)]), Val(16)))
    end
    maxscale = 0f0
    maxabs = 0f0
    Base.Cartesian.@nexprs 16 j -> begin
        a_j = abs(fits[j][1])
        if a_j > maxabs
            maxabs = a_j
            maxscale = fits[j][1]
        end
    end
    zero4 = ntuple(_ -> VecElement(UInt32(0)), Val(4))
    maxabs < 1f-15 && return (ntuple(_ -> zero4, Val(8)), ntuple(_ -> zero4, Val(4)), zero4, Float16(0))
    iscale = -128f0 / maxscale
    dh = Float16(1f0 / iscale)
    sc = ntuple(j -> min(Int32(127), q4k_round(iscale * fits[j][1])) % Int8, Val(16))
    dsub = ntuple(j -> Float32(dh) * Float32(sc[j]), Val(16))
    # The six-bit level of weight `i` (0-based, known when this inlines). A
    # sub-block whose scale rounded to zero keeps the levels of its fit, as in
    # ggml, and an all-zero one ggml's zeros; they dequantize to zero either way.
    @inline function level(i)
        j = i ÷ 16 + 1
        x = Float32(@inbounds W[row + Int32(1), k0 + Int32(i + 1)])
        dsub[j] != 0f0 && return UInt32(clamp(q4k_round(x / dsub[j]), Int32(-32), Int32(31)) + Int32(32))
        return fits[j][2] == 0f0 ? UInt32(0) : UInt32(q6k_level(x, fits[j][2]) + Int32(32))
    end
    # `ql` byte `c` of half `n` (`r = c % 64`) holds weights `128n + r` (low nibble)
    # and `128n + r + 64` (high).
    ql = ntuple(Val(8)) do v
        ntuple(Val(4)) do w
            c0 = 16 * (v - 1) + 4 * (w - 1)
            i0 = 128 * (c0 ÷ 64) + c0 % 64
            word = UInt32(0)
            Base.Cartesian.@nexprs 4 e -> begin
                word |= ((level(i0 + e - 1) & 0x0f) | ((level(i0 + e + 63) & 0x0f) << 4)) << (8 * (e - 1))
            end
            VecElement(word)
        end
    end
    # `qh` byte `c` of half `n` (`l = c % 32`): the high bits of `128n + l + 32Q`.
    qh = ntuple(Val(4)) do u
        ntuple(Val(4)) do w
            c0 = 16 * (u - 1) + 4 * (w - 1)
            i0 = 128 * (c0 ÷ 32) + c0 % 32
            word = UInt32(0)
            Base.Cartesian.@nexprs 4 e -> begin
                i = i0 + e - 1
                word |= ((level(i) >> 4) | ((level(i + 32) >> 4) << 2) | ((level(i + 64) >> 4) << 4) |
                         ((level(i + 96) >> 4) << 6)) << (8 * (e - 1))
            end
            VecElement(word)
        end
    end
    scv = ntuple(w -> VecElement(q4k_bytes(sc[4w - 3] % UInt8, sc[4w - 2] % UInt8,
                                           sc[4w - 1] % UInt8, sc[4w] % UInt8)), Val(4))
    return (ql, qh, scv, dh)
end

function q6k_quantize_kernel!(QL, QH, SC, D, W, rows::Int32, nb::Int32)
    t = Int32(KI.get_global_id().x) - Int32(1)
    if t < rows * nb
        row, b = t % rows, t ÷ rows
        ql, qh, scv, dh = q6k_block(W, row, b)
        blk = row * nb + b
        @inbounds begin
            Base.Cartesian.@nexprs 8 v -> (QL[blk * Int32(8) + Int32(v)] = ql[v])
            Base.Cartesian.@nexprs 4 u -> (QH[blk * Int32(4) + Int32(u)] = qh[u])
            SC[blk + Int32(1)] = scv
            D[blk + Int32(1)] = dh
        end
    end
    return nothing
end

"""
    quantizeq6k(backend, W) -> Q6KMatrix

Quantize a device-resident `(M, K)` float weight on the device. `K` must be a
multiple of 256. `W` is not modified.
"""
function quantizeq6k(backend, W::Union{AbstractMatrix,Mantle.Buffer{<:Any,2}})
    rows, K = size(W)
    K % QK_K == 0 || throw(ArgumentError("Q6_K needs a reduction extent that is a multiple of $QK_K, got $K"))
    nb = K ÷ QK_K
    dev = Mantle.todevice(backend)
    g = Mantle.Graph(dev)
    ql = Mantle.Buffer(dev, UVec4, (8nb, rows))
    qh = Mantle.Buffer(dev, UVec4, (4nb, rows))
    sc = Mantle.Buffer(dev, UVec4, (nb, rows))
    d = Mantle.Buffer(dev, Float16, (nb, rows))
    Mantle.dispatch!(g, q6k_quantize_kernel!, (ql, qh, sc, d, W, Int32(rows), Int32(nb)), rows * nb;
                     group = 256, name = "q6kquant")
    runonce!(g)
    return Q6KMatrix(ql, qh, sc, d, rows, K)
end

"""
    q6k_quantize(W::AbstractMatrix) -> (ql, qh, sc, d)

The host reference of [`quantizeq6k`](@ref), from the same `q6k_block`.
"""
function q6k_quantize(W::AbstractMatrix)
    rows, K = size(W)
    nb = K ÷ QK_K
    ql = Vector{UVec4}(undef, 8nb * rows)
    qh = Vector{UVec4}(undef, 4nb * rows)
    sc = Vector{UVec4}(undef, nb * rows)
    d = Vector{Float16}(undef, nb * rows)
    for row in 0:(rows - 1), b in 0:(nb - 1)
        l, h, s, dh = q6k_block(W, Int32(row), Int32(b))
        blk = row * nb + b
        ql[8blk .+ (1:8)] .= l
        qh[4blk .+ (1:4)] .= h
        sc[blk + 1] = s
        d[blk + 1] = dh
    end
    return ql, qh, sc, d
end

"""
    q6k_ggml(ql, qh, sc, d) -> Vector{UInt8}

The same blocks in ggml's `block_q6_K` layout, 210 bytes each, for comparison
with llama.cpp.
"""
function q6k_ggml(ql, qh, sc, d)
    nblk = length(d)
    out = UInt8[]
    for blk in 0:(nblk - 1)
        append!(out, reinterpret(UInt8, ql[8blk .+ (1:8)]))
        append!(out, reinterpret(UInt8, qh[4blk .+ (1:4)]))
        append!(out, reinterpret(UInt8, [sc[blk + 1]]))
        append!(out, reinterpret(UInt8, [d[blk + 1]]))
    end
    return out
end

# Scale `i` (0-based) of a block's scale vector, signed; the word selected rather
# than indexed, as `q4k_word` does.
@inline q6k_word(v::UVec4, i) =
    ifelse(i == 0, v[1].value, ifelse(i == 1, v[2].value, ifelse(i == 2, v[3].value, v[4].value)))
@inline q6k_scale(scv::UVec4, i) = ((q6k_word(scv, i ÷ 4) >> (8 * (i % 4))) % UInt8) % Int8

"""
    q6k_dequantize(ql, qh, sc, d, M, K) -> Matrix{Float32}

Host dequantization, ggml's `dequantize_row_q6_K` per row.
"""
function q6k_dequantize(ql, qh, sc, d, rows::Integer, K::Integer)
    nb = K ÷ QK_K
    W = Matrix{Float32}(undef, rows, K)
    for row in 0:(rows - 1), b in 0:(nb - 1)
        blk = row * nb + b
        lb = reinterpret(UInt8, ql[8blk .+ (1:8)])
        hb = reinterpret(UInt8, qh[4blk .+ (1:4)])
        sb = reinterpret(Int8, [sc[blk + 1]])
        db = Float32(d[blk + 1])
        for n in 0:1, l in 0:31, Q in 0:3
            low = (lb[64n + (Q % 2) * 32 + l + 1] >> (4 * (Q ÷ 2))) & 0x0f
            q = Int(low | (((hb[32n + l + 1] >> (2Q)) & 0x03) << 4))
            i = 128n + 32Q + l
            W[row + 1, b * QK_K + i + 1] = db * Float32(sb[i ÷ 16 + 1]) * Float32(q - 32)
        end
    end
    return W
end

# ── the K-quant interfaces ───────────────────────────────────────────────────

# Step `g` of a Q6_K block is quadrants `2(g % 2)` and `2(g % 2) + 1` of half
# `g ÷ 2`: the low nibbles of the half's two `ql` vectors for `h` and the qh bits
# 0 to 3, or their high nibbles and bits 4 to 7.
@inline q6k_q(L::UInt32, H::UInt32, ls, hs) = ((L >> ls) & 0x0f0f0f0f) | (((H >> hs) & 0x03030303) << 4)
@inline q6k_word4(q::UInt32, s::Float32) =
    q4k_h4(s * Float32(Int32(q & 0xff) - Int32(32)), s * Float32(Int32((q >> 8) & 0xff) - Int32(32)),
           s * Float32(Int32((q >> 16) & 0xff) - Int32(32)), s * Float32(Int32(q >> 24) - Int32(32)))

@inline function kq_stage(::Val{:q6k}, Ws, row::Int32, b::Int32, g::Int32, h::Int32, nb::Int32)
    QL, QH, SC, D = Ws
    blk = row * nb + b
    n, qs = g ÷ Int32(2), g % Int32(2)
    A = @inbounds QL[blk * Int32(8) + Int32(4) * n + h + Int32(1)]
    B = @inbounds QL[blk * Int32(8) + Int32(4) * n + Int32(2) + h + Int32(1)]
    H = @inbounds QH[blk * Int32(4) + Int32(2) * n + h + Int32(1)]
    scv = @inbounds SC[blk + Int32(1)]
    d = Float32(@inbounds D[blk + Int32(1)])
    sa = d * Float32(q6k_scale(scv, Int32(8) * n + Int32(4) * qs + h))
    sb = d * Float32(q6k_scale(scv, Int32(8) * n + Int32(4) * qs + Int32(2) + h))
    ls = Int32(4) * qs
    first = ntuple(w -> q6k_word4(q6k_q(A[w].value, H[w].value, ls, ls), sa), Val(4))
    second = ntuple(w -> q6k_word4(q6k_q(B[w].value, H[w].value, ls, ls + Int32(2)), sb), Val(4))
    return (first..., second...)
end

q6kgemm_kernel!(C, QL, QH, SC, D, B, epi, m::Val, n::Val, k::Val, cfg::Val) =
    kquant_gemm!(C, (QL, QH, SC, D), B, epi, Val(:q6k), m, n, k, cfg)
q6k_dequant_kernel!(W, QL, QH, SC, D, rows::Int32, nb::Int32) =
    kquant_dequant!(W, (QL, QH, SC, D), Val(:q6k), rows, nb)

kqarrays(A::Q6KMatrix) = (A.ql, A.qh, A.sc, A.d)
kqgemm_kernel(::Q6KMatrix) = q6kgemm_kernel!
kqdequant_kernel(::Q6KMatrix) = q6k_dequant_kernel!
kqgemv_kernels(::Q6KMatrix) = Q6K_GEMV_KERNELS

# ── the product ──────────────────────────────────────────────────────────────
#
# The layout of the Q4_K GEMV with four lanes a block instead of eight: lane `p`
# takes half `p ÷ 2` and its sixteen `l` at `16(p % 2)`, which is three vectors
# (two of `ql`, one of `qh`) and 64 weights, sixteen in each quadrant. A group of
# 32 lanes reads eight blocks a step.

# `q - 32` as a signed int8, four at a time (see the file's docstring).
@inline function q6k_signed(q::UInt32)
    y = q ⊻ 0x20202020
    return y | ((y & 0x20202020) * UInt32(6))
end

@inline q6k_dot(L::UVec4, H::UVec4, x::UVec4, ls, hs) =
    KI.dot4x8(q6k_signed(q6k_q(L[1].value, H[1].value, ls, hs)), x[1].value) +
    KI.dot4x8(q6k_signed(q6k_q(L[2].value, H[2].value, ls, hs)), x[2].value) +
    KI.dot4x8(q6k_signed(q6k_q(L[3].value, H[3].value, ls, hs)), x[3].value) +
    KI.dot4x8(q6k_signed(q6k_q(L[4].value, H[4].value, ls, hs)), x[4].value)

const Q6K_GEMV_KERNELS = Dict{Tuple{Int,Int},Any}()
for N in 1:8, R in (1, 2, 4)
    name = Symbol("q6k_gemv_", N, "_", R, "!")
    @eval function $name(out, QL, QH, SC, D, xq, xds, bias, epi, rows::Int32, nb::Int32,
                         ::Val{HASBIAS}) where {HASBIAS}
        lid = Int32(KI.get_local_id().x) - Int32(1)
        lane = lid % Int32(32)
        wave = (Int32(KI.get_group_id().x) - Int32(1)) * Int32(Q4K_WG ÷ 32) + lid ÷ Int32(32)
        r0 = wave * Int32($R)
        p = lane % Int32(4)
        n, h = p ÷ Int32(2), p % Int32(2)
        cols = nb * Int32(16)
        Base.Cartesian.@nexprs $N c -> Base.Cartesian.@nexprs $R r -> (acc_c_r = 0f0)
        b = lane ÷ Int32(4)
        while b < nb
            Base.Cartesian.@nexprs $R r -> begin
                blk_r = min(r0 + Int32(r - 1), rows - Int32(1)) * nb + b
                A_r = @inbounds QL[blk_r * Int32(8) + Int32(4) * n + h + Int32(1)]
                B_r = @inbounds QL[blk_r * Int32(8) + Int32(4) * n + Int32(2) + h + Int32(1)]
                H_r = @inbounds QH[blk_r * Int32(4) + Int32(2) * n + h + Int32(1)]
                scv_r = @inbounds SC[blk_r + Int32(1)]
                d_r = Float32(@inbounds D[blk_r + Int32(1)])
                Base.Cartesian.@nexprs 4 Q -> (s_Q_r = Float32(q6k_scale(scv_r, Int32(8) * n + Int32(2 * (Q - 1)) + h)))
            end
            xv = b * Int32(16) + Int32(8) * n + h       # 0-based vector of quadrant 0; quadrant Q at 2Q on
            xd = (b * Int32(8) + Int32(4) * n) * Int32(2)
            Base.Cartesian.@nexprs $N c -> begin
                Base.Cartesian.@nexprs 4 Q -> begin
                    x_Q = @inbounds xq[xv + Int32(2 * (Q - 1) + 1) + Int32(c - 1) * cols]
                    dx_Q = @inbounds xds[xd + Int32(2 * (Q - 1) + 1) + Int32(c - 1) * cols]
                end
                Base.Cartesian.@nexprs $R r -> begin
                    i_1 = q6k_dot(A_r, H_r, x_1, Int32(0), Int32(0))
                    i_2 = q6k_dot(B_r, H_r, x_2, Int32(0), Int32(2))
                    i_3 = q6k_dot(A_r, H_r, x_3, Int32(4), Int32(4))
                    i_4 = q6k_dot(B_r, H_r, x_4, Int32(4), Int32(6))
                    acc_c_r += d_r * (s_1_r * dx_1 * Float32(i_1) + s_2_r * dx_2 * Float32(i_2) +
                                      s_3_r * dx_3 * Float32(i_3) + s_4_r * dx_4 * Float32(i_4))
                end
            end
            b += Int32(8)
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
    Q6K_GEMV_KERNELS[(N, R)] = getfield(@__MODULE__, name)
end
