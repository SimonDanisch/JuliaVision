"""
Int8 weights: W8A16, per-output-channel symmetric.

Autoregressive decode is one token against the whole parameter set, so it reads
every weight once per token and does one multiply-add with each. It is therefore
bandwidth-bound outright, and the only way to make it faster is to move fewer
bytes. K2 Horizon 32B at fp16 is 69.6 GB per token, which on a device that reads
at ~220 GB/s is a 306 ms floor no kernel can beat. At int8 it is 34.8 GB and the
floor is 158 ms.

**Per output channel, symmetric, no zero point.** `scale[m] = max_k |W[m,k]|/127`
and `q[m,k] = round(W[m,k]/scale[m])`, so a row's largest weight lands exactly on
±127 and the dequantised value is `q * scale`. Per-channel rather than per-tensor
because one outlier row would otherwise crush the resolution of every other row;
symmetric rather than affine because a zero point costs a second read in the
inner loop and weight distributions here are centred.

Activations stay fp16 (W8A16). Quantising them too would halve the activation
traffic as well, but activations are kilobytes against the weights' gigabytes,
and an fp16 activation keeps the accumulation exact against the same reference.

**Quantised on the DEVICE, at load.** The host loop is 33.5 G elements and takes
minutes; three passes on the GPU take about a second, and the fp16 copy is freed
as soon as its rows are packed, so peak memory is the largest single weight above
the int8 total rather than both sets at once.

**Stored as `UInt32`, four rows to a word, and that is not a detail.** What this
device gives you depends on how many bytes each LANE loads, not on how many the
wave does. Measured with a plain streaming read, 2 GiB:

    element   elem/thread     GB/s
    Int8               64    143.9
    Int8              256    114.9
    Float16            64    166.3
    Float32            64    225.1
    UInt32             64    221.8
    UInt32            256    226.9

A one-byte load per lane reaches 144 GB/s and a four-byte load reaches 227 — and
227 GB/s is also exactly what llama.cpp gets out of this GPU. So the packed word
holds four CONSECUTIVE OUTPUT ROWS, one thread unpacks it into four accumulators,
and `x[k]` is a broadcast load every lane shares.
"""

struct QInt8Matrix{Q,S}
    q::Q            # (cld(M,4), K) UInt32: four consecutive rows per word
    scale::S        # (M,)   Float32, one per output channel
    m::Int          # true row count; `q` is padded up to a multiple of four
end

"""
Packed INT8 weight in Comfy's ConvRot basis.

`q` has the same four-output-row `UInt32` packing as [`QInt8Matrix`](@ref).
The only semantic difference is that the right operand is transformed by the
normalized regular Hadamard matrix, independently in `group_size`-wide blocks,
before multiplication. The representation and kernels are backend-independent.
"""
struct ConvRotQInt8Matrix{Q,S}
    q::Q
    scale::S
    m::Int
    group_size::Int
end

Base.size(A::ConvRotQInt8Matrix) = (A.m, size(A.q, 2))
Base.size(A::ConvRotQInt8Matrix, i::Integer) = i == 1 ? A.m : size(A.q, i)
Base.eltype(::ConvRotQInt8Matrix) = Float16
Base.ndims(::ConvRotQInt8Matrix) = 2

"""Host view of a safetensors INT8 `[M,K]` matrix (stored by Julia as `(K,M)`)."""
struct ConvRotQInt8HostMatrix{Q,S}
    q::Q
    scale::S
    group_size::Int
end

Base.size(A::ConvRotQInt8HostMatrix) = (size(A.q, 2), size(A.q, 1))
Base.size(A::ConvRotQInt8HostMatrix, i::Integer) = size(A)[i]
Base.eltype(::ConvRotQInt8HostMatrix) = Float16
Base.ndims(::ConvRotQInt8HostMatrix) = 2

"""Host view of Comfy's packed asymmetric W4A8+ConvRot matrix."""
struct W4A8ConvRotHostMatrix{Q,R,S,C}
    q::Q                    # checkpoint `[M,K/2]`, Julia `(K/2,M)`
    s_rel::R                # E4M3FN bits, checkpoint `[M,K/group]`
    s_channel::S            # Float32, one per output row
    codebook::C             # Float32[16]
    group_size::Int
    convrot_group_size::Int
end

Base.size(A::W4A8ConvRotHostMatrix) = (size(A.q, 2), 2size(A.q, 1))
Base.size(A::W4A8ConvRotHostMatrix, i::Integer) = size(A)[i]
Base.eltype(::W4A8ConvRotHostMatrix) = Float16
Base.ndims(::W4A8ConvRotHostMatrix) = 2

Base.size(A::QInt8Matrix) = (A.m, size(A.q, 2))
Base.size(A::QInt8Matrix, i::Integer) = i == 1 ? A.m : size(A.q, i)
Base.eltype(::QInt8Matrix) = Float16          # what it dequantises to
Base.ndims(::QInt8Matrix) = 2

# A quantised weight is a pair of resident arrays rather than an AbstractArray
# itself. `residentweights` still passes every weight through `toback`, so carry
# that operation through the wrapper. Each component uses `toback`'s ordinary
# backend-kind check and is therefore unchanged when it already lives there.
toback(backend, A::QInt8Matrix) =
    QInt8Matrix(toback(backend, A.q), toback(backend, A.scale), A.m)

toback(backend, A::ConvRotQInt8Matrix) =
    ConvRotQInt8Matrix(toback(backend, A.q), toback(backend, A.scale), A.m,
                       A.group_size)

# Rows per packed word. Fixed at four by `UInt32`; named so the arithmetic reads.
const Q8ROWS = 4

@kernel cpu=false function checkpoint_q8pack_kernel!(q32, @Const(q),
                                                      K::Int32, M::Int32, MG::Int32)
    i = @index(Global, Linear)
    if i <= MG * K
        @inbounds begin
            l = Int32(i) - Int32(1)
            g = l % MG
            k = l ÷ MG
            word = UInt32(0)
            Base.Cartesian.@nexprs 4 r -> begin
                m = g * Int32(4) + Int32(r - 1)
                if m < M
                    byte = reinterpret(UInt8, q[k + Int32(1) + m * K])
                    word |= UInt32(byte) << (8 * (r - 1))
                end
            end
            q32[i] = word
        end
    end
end

"""Upload and repack a Comfy `int8_tensorwise` ConvRot checkpoint matrix."""
function convrotqint8(backend, q::AbstractMatrix{Int8}, scale;
                      group_size::Integer=256)
    K, M = size(q)
    K % group_size == 0 || throw(DimensionMismatch(
        "ConvRot group size $group_size does not divide K=$K"))
    length(scale) in (1, M) || throw(DimensionMismatch(
        "INT8 scale has $(length(scale)) entries for M=$M"))
    dq = toback(backend, q)
    ds = toback(backend, Float32.(vec(scale)))
    MG = cld(M, Q8ROWS)
    packed = KernelAbstractions.allocate(backend, UInt32, MG, K)
    checkpoint_q8pack_kernel!(backend, 256)(packed, dq, Int32(K), Int32(M), Int32(MG);
                                              ndrange=MG*K)
    ConvRotQInt8Matrix(packed, ds, M, Int(group_size))
end

toback(backend, A::ConvRotQInt8HostMatrix) =
    convrotqint8(backend, A.q, A.scale; group_size=A.group_size)

@inline function f8e4m3fn(x::UInt8)
    sign = (x & 0x80) == 0 ? 1f0 : -1f0
    exponent = Int32((x >> 3) & 0x0f)
    mantissa = Int32(x & 0x07)
    exponent == 0 && return sign * Float32(mantissa) * 0.001953125f0 # 2^-9
    exponent == 15 && mantissa == 7 && return Float32(NaN)
    sign * (1f0 + Float32(mantissa) * 0.125f0) * exp2(Float32(exponent - 7))
end

@kernel cpu=false function w4a8_pack_kernel!(q32, @Const(q), @Const(srel),
                                              @Const(codebook), K::Int32, M::Int32,
                                              MG::Int32, GS::Int32, MGD::Int32,
                                              GOFF::Int32)
    i = @index(Global, Linear)
    if i <= MG * K
        @inbounds begin
            l = Int32(i) - Int32(1)
            rg = l % MG
            k = l ÷ MG
            word = UInt32(0)
            Base.Cartesian.@nexprs 4 r -> begin
                m = rg * Int32(4) + Int32(r - 1)
                if m < M
                    packed = reinterpret(UInt8, q[(k ÷ Int32(2)) + Int32(1) + m * (K ÷ Int32(2))])
                    code = iseven(k) ? (packed & 0x0f) : (packed >> 4)
                    sr = f8e4m3fn(srel[(k ÷ GS) + Int32(1) + m * (K ÷ GS)])
                    v = round(clamp(codebook[Int32(code) + Int32(1)] * sr, -127f0, 127f0))
                    byte = reinterpret(UInt8, unsafe_trunc(Int8, v))
                    word |= UInt32(byte) << (8 * (r - 1))
                end
            end
            # The destination may be a STACK of parts: this one owns the row
            # groups from `GOFF`, and a group is four output rows, so a part
            # whose row count is not a multiple of four cannot be stacked.
            q32[(rg + GOFF) + Int32(1) + k * MGD] = word
        end
    end
end

"""Upload Comfy `asym_w4a8_int8` storage and decode its INT4 grid to packed INT8."""
function w4a8convrot(backend, q::AbstractMatrix{Int8}, s_rel::AbstractMatrix{UInt8},
                     s_channel, codebook; group_size::Integer=16,
                     convrot_group_size::Integer=256)
    KH, M = size(q)
    K = 2KH
    size(s_rel) == (K ÷ group_size, M) || throw(DimensionMismatch(
        "W4A8 relative scales are $(size(s_rel)); expected $((K ÷ group_size, M))"))
    length(s_channel) == M || throw(DimensionMismatch("W4A8 channel scale length mismatch"))
    length(codebook) == 16 || throw(DimensionMismatch("W4A8 codebook must have 16 entries"))
    K % convrot_group_size == 0 || throw(DimensionMismatch(
        "ConvRot group size $convrot_group_size does not divide K=$K"))
    dq, dr = toback(backend, q), toback(backend, s_rel)
    ds = toback(backend, Float32.(vec(s_channel)))
    dc = toback(backend, Float32.(vec(codebook)))
    MG = cld(M, Q8ROWS)
    packed = KernelAbstractions.allocate(backend, UInt32, MG, K)
    w4a8pack!(backend, packed, dq, dr, dc, K, M, group_size, MG, 0)
    ConvRotQInt8Matrix(packed, ds, M, Int(convrot_group_size))
end

"""Decode one W4A8 part into row groups `goff...` of an already-allocated pack."""
function w4a8pack!(backend, packed, q, s_rel, codebook, K::Integer, M::Integer,
                   group_size::Integer, mgdest::Integer, goff::Integer)
    mg = cld(M, Q8ROWS)
    w4a8_pack_kernel!(backend, 256)(packed, q, s_rel, codebook, Int32(K), Int32(M),
                                     Int32(mg), Int32(group_size), Int32(mgdest),
                                     Int32(goff); ndrange=mg*K)
    packed
end

"""
    w4a8convrot(backend, parts) -> ConvRotQInt8Matrix

The stacked form: `fuseqkv`'s three projections, or a gate/up pair, as one
matrix. Each part decodes into its own row range of a single pack, which is the
only way to assemble these — a part's storage is four output rows to a `UInt32`
word after decoding, and the 16-value codebook is per tensor, so there is
nothing to concatenate before the decode.
"""
function w4a8convrot(backend, parts::AbstractVector{<:W4A8ConvRotHostMatrix})
    first_part = first(parts)
    K = size(first_part, 2)
    group_size = first_part.group_size
    convrot_group_size = first_part.convrot_group_size
    all(p -> size(p, 2) == K, parts) || throw(DimensionMismatch(
        "stacked W4A8 matrices disagree on their input width"))
    all(p -> p.group_size == group_size && p.convrot_group_size == convrot_group_size,
        parts) || throw(ArgumentError("stacked W4A8 matrices disagree on their group sizes"))
    all(p -> size(p, 1) % Q8ROWS == 0, parts) || throw(ArgumentError(
        "a stacked W4A8 part must have a multiple of $Q8ROWS output rows"))
    M = sum(p -> size(p, 1), parts)
    MG = cld(M, Q8ROWS)
    packed = KernelAbstractions.allocate(backend, UInt32, MG, K)
    ds = toback(backend, Float32.(reduce(vcat, (vec(p.s_channel) for p in parts))))
    goff = 0
    for p in parts
        w4a8pack!(backend, packed, toback(backend, p.q), toback(backend, p.s_rel),
                  toback(backend, Float32.(vec(p.codebook))), K, size(p, 1),
                  group_size, MG, goff)
        goff += size(p, 1) ÷ Q8ROWS
    end
    ConvRotQInt8Matrix(packed, ds, M, Int(convrot_group_size))
end

toback(backend, A::W4A8ConvRotHostMatrix) =
    w4a8convrot(backend, A.q, A.s_rel, A.s_channel, A.codebook;
                group_size=A.group_size, convrot_group_size=A.convrot_group_size)

"""
    ispackedquantmatrix(W) -> Bool

Whether `W` is a host weight whose packing already presents the `(M, K)`
orientation a GEMM reads — the torch `[M, K]` checkpoint layout, kept as
packed words rather than as an array that can be permuted.

`hoistpermutes` asks this. A dense weight arrives as `(K, M)` and its `t()` view
is materialised into a real `(M, K)` weight; a packed matrix is *stored* that way
already, so the same view is the matrix itself and materialising it would mean
transposing a representation that has no strided form.
"""
ispackedquantmatrix(::Any) = false
ispackedquantmatrix(::ConvRotQInt8HostMatrix) = true
ispackedquantmatrix(::W4A8ConvRotHostMatrix) = true

"""
    stackrows(parts) -> matrix

Concatenate packed checkpoint matrices along their output rows.

`fuseqkv` stacks the three attention projections, or a gate/up pair, into one
GEMM. A dense stack is assembled on the device row range by row range; a packed
one cannot be, because four output rows share a `UInt32` word. It is assembled
in the checkpoint's own layout instead — `q` is stored `(K, M)`, so stacking
output rows is `hcat`, and the per-channel scales stack the same way — and the
result is packed once, by the same path any single matrix takes.
"""
function stackrows(parts::AbstractVector{<:ConvRotQInt8HostMatrix})
    group = first(parts).group_size
    all(p -> p.group_size == group, parts) || throw(ArgumentError(
        "stacked ConvRot matrices disagree on their group size"))
    all(p -> size(p.scale, 1) == 1, parts) || throw(ArgumentError(
        "stacked ConvRot matrices must carry one scale per output channel"))
    ConvRotQInt8HostMatrix(reduce(hcat, (p.q for p in parts)),
                           reduce(hcat, (p.scale for p in parts)), group)
end


"""
Two radix-4 stages of the ConvRot transform, in registers.

A 256-wide group is four stages at strides 1, 4, 16 and 64, and a thread that
holds SIXTEEN elements spaced `S` apart does two of them without talking to any
other thread: `S = 1` covers strides 1 and 4, `S = 16` covers 16 and 64. Two
launches, four traversals of the tensor, no shared memory and no barrier.

The two forms this replaced, measured on Qwen-Image 2.1's largest activation
(4096 x 4118 fp16) on an 8060S:

    a pass per stage, eight traversals          7.87 ms
    one pass, group staged in shared memory    12.13 ms
    two passes, sixteen elements in registers   ? ms

Shared memory looked like the obvious answer and is the slowest of the three:
64-thread workgroups over a 512-byte group spend their time on four barriers,
not on four adds. Unrolling the stages so every stride is a compile-time shift
did not move it either.

Each stage rounds to `eltype(out)`, which is the rounding the pass-per-stage
kernel had, so a group validated against the checkpoint's own decode keeps the
same error. Not the same BITS: against a host implementation of the staged form
this agrees to one fp16 ulp, because the compiler is free to reassociate
`a + b + c - d` and does.
"""
@kernel cpu=false unsafe_indices=true function convrot_pass_kernel!(
        out, @Const(input), ::Val{S}, ::Val{SETS}, ::Val{G}, n::Int64) where {S,SETS,G}
    j = Int64(@index(Global, Linear)) - Int64(1)
    g = j ÷ Int64(SETS)
    w = j - g * Int64(SETS)
    base = g * Int64(G) + (S == 1 ? w * Int64(16) : w) + Int64(1)
    if base <= n
        @inbounds begin
            Base.Cartesian.@nexprs 16 i -> v_i = Float32(input[base + Int64((i - 1) * S)])
            # Stride 1 across the sixteen registers, then stride 4. Both are the
            # same symmetric 4x4, and both round to the output type in between.
            Base.Cartesian.@nexprs 4 q -> begin
                a = v_{4q-3}; b = v_{4q-2}; c = v_{4q-1}; d = v_{4q}
                v_{4q-3} = Float32(eltype(out)(( a + b + c - d) * 0.5f0))
                v_{4q-2} = Float32(eltype(out)(( a + b - c + d) * 0.5f0))
                v_{4q-1} = Float32(eltype(out)(( a - b + c + d) * 0.5f0))
                v_{4q}   = Float32(eltype(out)((-a + b + c + d) * 0.5f0))
            end
            Base.Cartesian.@nexprs 4 q -> begin
                a = v_q; b = v_{q+4}; c = v_{q+8}; d = v_{q+12}
                v_q      = Float32(eltype(out)(( a + b + c - d) * 0.5f0))
                v_{q+4}  = Float32(eltype(out)(( a + b - c + d) * 0.5f0))
                v_{q+8}  = Float32(eltype(out)(( a - b + c + d) * 0.5f0))
                v_{q+12} = Float32(eltype(out)((-a + b + c + d) * 0.5f0))
            end
            Base.Cartesian.@nexprs 16 i -> out[base + Int64((i - 1) * S)] = eltype(out)(v_i)
        end
    end
end

"""
    convrot_passes(input, group_size) -> (strides, sets, ndrange)

The strides a ConvRot group needs, sixteen elements to a thread. A group of 256
is two passes; the general power-of-four case is `log4(G) ÷ 2` of them, and an
odd stage count has no register form here.
"""
function convrot_passes(input, group_size::Integer)
    K = size(input, 1)
    K % group_size == 0 || throw(DimensionMismatch("ConvRot group size does not divide input"))
    stages = round(Int, log(4, group_size))
    4^stages == group_size || throw(ArgumentError("ConvRot group size must be a power of four"))
    iseven(stages) || throw(ArgumentError(
        "ConvRot group $group_size has $stages stages; the register form takes two at a time"))
    strides = Tuple(16^(i - 1) for i in 1:(stages ÷ 2))
    (strides, group_size ÷ 16, length(input) ÷ 16)
end

function convrot(ctx::Ctx, input, group_size::Integer)
    K = size(input, 1)
    K % group_size == 0 || throw(DimensionMismatch("ConvRot group size does not divide input"))
    stages = round(Int, log(4, group_size))
    4^stages == group_size || throw(ArgumentError("ConvRot group size must be a power of four"))
    out = scratch!(ctx, eltype(input), size(input)...)
    strides, sets, ndrange = convrot_passes(input, group_size)
    n = Int64(length(input))
    src = input
    for S in strides
        convrot_pass_kernel!(ctx.backend, 256)(out, src, Val(S), Val(sets),
                                               Val(Int(group_size)), n; ndrange)
        src = out
    end
    out
end

@inline q8byte(w::UInt32, r::Integer) =
    Float32(reinterpret(Int8, UInt8((w >> (8 * r)) & 0x000000ff)))

# ── quantisation ─────────────────────────────────────────────────────────────

@kernel cpu=false function q8scale_kernel!(s, @Const(W), M::Int32, K::Int32)
    m = @index(Global, Linear)
    if m <= M
        @inbounds begin
            mx = 0f0
            mi = Int32(m)
            for k in Int32(0):(K - Int32(1))
                mx = max(mx, abs(Float32(W[mi + k * M])))
            end
            s[m] = mx / 127f0
        end
    end
end

@kernel cpu=false function q8pack_kernel!(q32, @Const(W), @Const(s),
                                          M::Int32, MG::Int32, n::Int64)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            l = Int32(i) - Int32(1)
            g = l % MG                 # 0-based group of four rows
            k = l ÷ MG                 # 0-based column
            w = UInt32(0)
            Base.Cartesian.@nexprs 4 r -> begin
                m = g * Int32(4) + Int32(r - 1) + Int32(1)
                if m <= M
                    sc = s[m]
                    v = sc == 0f0 ? 0f0 : Float32(W[m + k * M]) / sc
                    b = unsafe_trunc(Int8, round(clamp(v, -127f0, 127f0)))
                    w |= UInt32(reinterpret(UInt8, b)) << (8 * (r - 1))
                end
            end
            q32[i] = w
        end
    end
end

# Quantizing straight out of the checkpoint layout — no transpose, every read
# contiguous — was written, made bit-identical, and MEASURED SLOWER. Three
# paired full builds each, alternating, everything hot: 13.65 s with it against
# 11.17 s without (spreads of 0.1 s, disjoint). Fusing it everywhere rather than
# only where the shape suited it was worse again. Removed.
#
# Two lessons in it. The transpose is not what a warm build spends its time on:
# 9.3 s of a 11.2 s build is the 69.6 GB host-to-device copy, and no kernel
# change touches that. And the per-shape microbenchmark that motivated this
# measured `toback(PermutedDimsArray(host))` against
# `quantizeint8(PermutedDimsArray(device))` — only the first included the
# upload, which invented a 2-3x speedup that did not exist. Pair the samples and
# time the whole build.

"""
    quantizeint8(backend, W) -> QInt8Matrix

Pack a device-resident `(M, K)` float weight. `W` is not modified and the caller
drops it.
"""
function quantizeint8(backend, W::AbstractMatrix)
    M, K = size(W)
    MG = cld(M, Q8ROWS)
    s = KernelAbstractions.allocate(backend, Float32, M)
    q32 = KernelAbstractions.allocate(backend, UInt32, MG, K)
    q8scale_kernel!(backend, 256)(s, W, Int32(M), Int32(K); ndrange = M)
    q8pack_kernel!(backend, 256)(q32, W, s, Int32(M), Int32(MG), Int64(MG) * K;
                                 ndrange = MG * K)
    QInt8Matrix(q32, s, M)
end


# ── the GEMV ─────────────────────────────────────────────────────────────────
#
# Split over K as well as rows, for the same reason `Mantle.gemv_splitk_kernel!`
# does: one thread per output row cannot fill the device when M is small, and
# decode's M runs from 1024 to 250624. The scale is applied once, in the
# reduction, so the inner loop is a plain int-to-float multiply-add.
#
# `P` is padded to `MG * 4` rows so the hot loop stores four values with no
# bounds test; the reduction only ever reads the first M of them.

@kernel cpu=false function q8gemv_kernel!(P, @Const(q32), @Const(x), MG::Int32,
                                          MP::Int32, K::Int32, KC::Int32, ntot::Int32)
    lin = @index(Global, Linear)
    if lin <= ntot
        @inbounds begin
            l = Int32(lin) - Int32(1)
            g = l % MG                  # consecutive lanes -> consecutive words
            sp = l ÷ MG                 # 0-based K split
            k1 = min(sp * KC + KC, K)
            Base.Cartesian.@nexprs 4 r -> acc_r = 0f0
            k = sp * KC
            while k < k1
                # One four-byte load of four rows, and one broadcast load of the
                # activation that every lane in the wave shares.
                w = q32[g + Int32(1) + k * MG]
                xv = Float32(x[k + Int32(1)])
                Base.Cartesian.@nexprs 4 r -> begin
                    acc_r = muladd(q8byte(w, r - 1), xv, acc_r)
                end
                k += Int32(1)
            end
            o = g * Int32(4) + Int32(1) + sp * MP
            Base.Cartesian.@nexprs 4 r -> (P[o + Int32(r - 1)] = acc_r)
        end
    end
end

@kernel cpu=false function q8reduce_kernel!(C, @Const(P), @Const(s), @Const(bias),
                                            M::Int32, MP::Int32, S::Int32,
                                            ::Val{HASBIAS}) where {HASBIAS}
    m = @index(Global, Linear)
    if m <= M
        @inbounds begin
            acc = P[m]
            for j in Int32(1):(S - Int32(1))
                acc += P[m + j * MP]
            end
            acc *= s[m]
            HASBIAS && (acc += Float32(bias[m]))
            C[m] = eltype(C)(acc)
        end
    end
end

@kernel cpu=false function q8dequant_kernel!(W, @Const(q32), @Const(s),
                                             M::Int32, MG::Int32, n::Int64)
    i = @index(Global, Linear)
    if i <= n
        @inbounds begin
            l = Int32(i) - Int32(1)
            g = l % MG
            k = l ÷ MG
            w = q32[i]
            Base.Cartesian.@nexprs 4 r -> begin
                m = g * Int32(4) + Int32(r - 1) + Int32(1)
                if m <= M
                    W[m + k * M] = eltype(W)(q8byte(w, r - 1) * s[m])
                end
            end
        end
    end
end

# Threads to aim for. The row axis is `MG`, a quarter of M, because each thread
# already owns four rows.
#
# CLOSEST to the target, not the first split past it. `while S*MG < target` walks
# one doubling too far whenever `MG` does not divide the target neatly, and that
# doubling halves the chunk each thread gets: `26624 x 5120` measures 215 GB/s at
# `S = 8` (53k threads) and 196 at `S = 16` (106k). Measured, GB/s:
#
#     shape (M x K)       S=1   S=2   S=4   S=8  S=16  S=32  S=64
#      8192 x  5120        19    36    58   106   172   182   167
#      1024 x  5120         3     7    17    32    49    69    77
#      5120 x  8192        14    25    41    83   135   130   170
#     26624 x  5120        44    89   174   215   196   207   191
#      5120 x 26624        11    24    40    82   134   206   179
#    250624 x  5120       216   219   202   213   211   206   200
const Q8SPLIT_BIAS = Ref(3)

@inline function q8split(M::Int, K::Int)
    MG = cld(M, Q8ROWS)
    # Biased BELOW the raw thread target. Every doubling also halves the K-chunk
    # each thread reduces and doubles the partials the second pass has to read,
    # and the measurements bracket the target from both sides — `5120 x 26624`
    # wants 0.62x it and `5120 x 8192` 1.25x — so the cheaper side is the right
    # default. `5120 x 26624` is 208 GB/s at 0.62x against 192 at 1.25x.
    target = (Mantle.GEMV_TARGET_THREADS * Q8SPLIT_BIAS[]) ÷ 4
    best, bestd = 1, abs(MG - target)
    S = 1
    while cld(K, 2S) >= Mantle.GEMV_MIN_CHUNK
        S *= 2
        d = abs(S * MG - target)
        d < bestd && ((best, bestd) = (S, d))
    end
    best
end

# ── single-dispatch GEMV ─────────────────────────────────────────────────────
#
# The split-K GEMV above is two dispatches: one writes `S` partial planes, a
# second adds them. That second dispatch is a barrier, and decode issues 257
# matmuls, so it is 257 extra pipeline drains plus a full round trip through the
# partials. Measured on K2 Horizon 32B: the matmuls run at 224.9 GB/s when
# nothing separates them and 214.7 GB/s in the graph, where they cannot overlap.
#
# Here the split lives INSIDE the workgroup. 256 threads are laid out as `RG`
# row-groups by `KS = 256 / RG` k-splits; each thread reduces its own chunk of K
# for four rows, the partials meet in shared memory, and the same workgroup
# writes the finished output. One dispatch, no partial planes.
#
# `RG` is chosen so the launch is about as wide as the split-K version was:
# `cld(MG, RG) * 256` threads. Shared memory is `RG * 4 * KS = 1024` floats
# whatever `RG` is, which is why the layout is written this way round.
const Q8WG = 256

for RG in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    kname = Symbol("q8gemv1_kernel_", RG, "!")
    KS = Q8WG ÷ RG
    @eval @kernel cpu=false function $kname(C, @Const(q32), @Const(x), @Const(sc),
                                            @Const(bias), MG::Int32, M::Int32, K::Int32,
                                            ::Val{HASBIAS}) where {HASBIAS}
        sh = @localmem Float32 (Q8WG * 4,)
        t = @index(Local, Linear) - 1
        wg = @index(Group, Linear) - 1
        r = Int32(t % $RG)                    # row-group within the workgroup
        sp = Int32(t ÷ $RG)                   # which k-split
        g = Int32(wg) * Int32($RG) + r        # global row-group
        KC = cld(K, Int32($KS))
        k1 = min(sp * KC + KC, K)
        @inbounds begin
            Base.Cartesian.@nexprs 4 i -> acc_i = 0f0
            if g < MG
                k = sp * KC
                while k < k1
                    w = q32[g + Int32(1) + k * MG]
                    xv = Float32(x[k + Int32(1)])
                    Base.Cartesian.@nexprs 4 i -> begin
                        acc_i = muladd(q8byte(w, i - 1), xv, acc_i)
                    end
                    k += Int32(1)
                end
            end
            # Partials for this thread's four rows, contiguous per row so the
            # reduction below walks them without a stride.
            base = (r * Int32(4)) * Int32($KS) + sp
            Base.Cartesian.@nexprs 4 i -> (sh[base + Int32(i - 1) * Int32($KS) + Int32(1)] = acc_i)
        end
        @synchronize
        # `RG * 4` outputs per workgroup, `Q8WG` threads: one pass when there are
        # fewer outputs than threads, a strided loop when there are more.
        @inbounds begin
            o = Int32(t)
            while o < Int32($RG) * Int32(4)
                a = 0f0
                b0 = o * Int32($KS)
                for j in Int32(0):(Int32($KS) - Int32(1))
                    a += sh[b0 + j + Int32(1)]
                end
                m = Int32(wg) * Int32($RG) * Int32(4) + o
                if m < M
                    v = a * sc[m + Int32(1)]
                    HASBIAS && (v += Float32(bias[m + Int32(1)]))
                    C[m + Int32(1)] = eltype(C)(v)
                end
                o += Int32(Q8WG)
            end
        end
    end
end

const Q8GEMV1_KERNELS = Dict(1 => q8gemv1_kernel_1!, 2 => q8gemv1_kernel_2!,
                             4 => q8gemv1_kernel_4!, 8 => q8gemv1_kernel_8!,
                             16 => q8gemv1_kernel_16!, 32 => q8gemv1_kernel_32!,
                             64 => q8gemv1_kernel_64!, 128 => q8gemv1_kernel_128!,
                             256 => q8gemv1_kernel_256!)

"""
    Q8SINGLE[] = false

Use the single-dispatch GEMV. **Off by default, because it is slower.**

The reasoning for it was sound and the measurement disagreed: 5.85 tok/s against
6.06 on K2 Horizon 32B. The second dispatch it removes turns out to cost nothing
— running the partial kernel alone, without any reduction, takes 148.5 ms
against 148.9 for both, and 156.1 against 156.0 when serialised. What the split-K
path pays is the RAMP of each GEMV when it cannot overlap the next one, and
moving the reduction into the workgroup does not help that while adding a
barrier every 256 threads must reach.

Kept, and kept off, because it is the natural thing to try next and this is the
measurement that says not to.
"""
const Q8SINGLE = Ref(false)

"""Row-groups per workgroup: aim for the same launch width the split-K rule picks."""
@inline function q8rowgroups(M::Int, K::Int)
    MG = cld(M, Q8ROWS)
    best, bestd = 1, typemax(Int)
    for RG in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        RG <= MG || continue
        cld(K, Q8WG ÷ RG) >= Mantle.GEMV_MIN_CHUNK || continue
        d = abs(cld(MG, RG) * Q8WG - (Mantle.GEMV_TARGET_THREADS * Q8SPLIT_BIAS[]) ÷ 4)
        d < bestd && ((best, bestd) = (RG, d))
    end
    best
end

"""
    q8gemv!(ctx, out, A::QInt8Matrix, x, bias) -> out

`out = A * x` for a single column `x`, dequantising in the inner loop.
"""
function q8gemv!(ctx, out, A::QInt8Matrix, x, bias)
    M, K = size(A)
    MG = size(A.q, 1)
    if Q8SINGLE[]
        RG = q8rowgroups(M, K)
        Q8GEMV1_KERNELS[RG](ctx.backend, Q8WG)(
            out, A.q, x, A.scale, bias === nothing ? A.scale : bias,
            Int32(MG), Int32(M), Int32(K), Val(bias !== nothing);
            ndrange = cld(MG, RG) * Q8WG)
        return out
    end
    MP = MG * Q8ROWS
    S = q8split(M, K)
    KC = cld(K, S)
    P = Mantle.splitscratch(out, MP, 1, S)
    q8gemv_kernel!(ctx.backend, 256)(P, A.q, x, Int32(MG), Int32(MP), Int32(K),
                                     Int32(KC), Int32(MG * S); ndrange = MG * S)
    q8reduce_kernel!(ctx.backend, 256)(
        out, P, A.scale, bias === nothing ? A.scale : bias,
        Int32(M), Int32(MP), Int32(S), Val(bias !== nothing); ndrange = M)
    out
end

"""
    q8dequant(ctx, A::QInt8Matrix) -> fp16 matrix in the workspace

For `N > 1`. A prompt pass does `N` multiply-adds per weight, so it is
compute-bound and can afford one extra pass to hand the cooperative-matrix GEMM
the dense fp16 operand it needs — which is what llama.cpp's `mul_mm` does too,
dequantising a tile into shared memory before the tensor-core loop.
"""
function q8dequant(ctx, A::QInt8Matrix)
    M, K = size(A)
    MG = size(A.q, 1)
    W = scratch!(ctx.ws, ctx.backend, Float16, M, K)
    q8dequant_kernel!(ctx.backend, 256)(W, A.q, A.scale, Int32(M), Int32(MG),
                                        Int64(MG) * K; ndrange = MG * K)
    W
end
