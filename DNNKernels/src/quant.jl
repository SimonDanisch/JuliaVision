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

# Rows per packed word. Fixed at four by `UInt32`; named so the arithmetic reads.
const Q8ROWS = 4

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
