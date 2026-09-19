"""
    PTQ1Matrix

Prism PTQ1_0 weights in their checkpoint representation. Each consecutive
128-value block occupies 28 bytes: 26 bytes of base-3 digits followed by an
FP16 scale. Rows stay packed on the device, so the model never expands its
ternary weights.
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
const PTQ1_ROWS_PER_WG = 4

"""Upload one packed GGUF matrix without dequantising or transposing it."""
function ptq1matrix(backend, file::GGUFFile, tensor::Union{GGUFTensor,AbstractString})
    t = tensor isa GGUFTensor ? tensor : file[tensor]
    t.typeid == GGML_TYPE_PTQ1_0 || throw(ArgumentError("$(t.name) is GGML type $(t.typeid), not PTQ1_0"))
    length(t.dims) == 2 || throw(ArgumentError("$(t.name) is rank $(length(t.dims)), expected a matrix"))
    k, m = t.dims
    k % PTQ1_QK == 0 || throw(ArgumentError("PTQ1 input width $k is not divisible by $PTQ1_QK"))
    host = ggufbytes(file, t)
    data = KernelAbstractions.allocate(backend, UInt8, length(host))
    copyto!(data, host)
    PTQ1Matrix(data, m, k)
end

@inline function ptq1_scale(data, base::Int32)
    bits = UInt16(data[base + Int32(26)]) | (UInt16(data[base + Int32(27)]) << 8)
    Float32(reinterpret(Float16, bits))
end

@inline function ptq1_value(data, base::Int32, e::Int32)
    b = UInt32(0)
    n = Int32(0)
    if e < Int32(80)
        b = UInt32(data[base + (e & Int32(15))])
        n = e >> 4
    elseif e < Int32(120)
        t = e - Int32(80)
        b = UInt32(data[base + Int32(16) + (t & Int32(7))])
        n = t >> 3
    else
        t = e - Int32(120)
        b = UInt32(data[base + Int32(24) + (t & Int32(1))])
        n = t >> 1
    end
    for _ in Int32(1):n
        b = (b * UInt32(3)) & UInt32(0xff)
    end
    Int32((b * UInt32(3)) >> 8) - Int32(1)
end

# Four independent 64-lane reductions share a 256-thread workgroup. A lane
# consumes two trits from every 128-value block. This gives decode enough waves
# even for the 48-element alpha/beta projections, while each packed byte and
# activation is read only by the row that uses it.
@kernel cpu=false function ptq1_mul_kernel!(out, @Const(data), @Const(x), @Const(bias),
                                            M::Int32, K::Int32, N::Int32,
                                            rowgroups::Int32,
                                            ::Val{HASBIAS}) where {HASBIAS}
    partial = @localmem Float32 (PTQ1_WG,)
    t = Int32(@index(Local, Linear) - 1)
    lane = t & Int32(63)
    rowin = t >> 6
    wg = Int32(@index(Group, Linear) - 1)
    rg = wg % rowgroups
    col = wg ÷ rowgroups
    row = rg * Int32(PTQ1_ROWS_PER_WG) + rowin
    blocks = K ÷ Int32(PTQ1_QK)
    acc = 0f0
    if row < M && col < N
        @inbounds for block in Int32(0):(blocks - Int32(1))
            base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
            scale = ptq1_scale(data, base)
            e0 = lane
            e1 = lane + Int32(64)
            xbase = col * K + block * Int32(PTQ1_QK)
            acc = muladd(scale * Float32(ptq1_value(data, base, e0)), Float32(x[xbase + e0 + Int32(1)]), acc)
            acc = muladd(scale * Float32(ptq1_value(data, base, e1)), Float32(x[xbase + e1 + Int32(1)]), acc)
        end
    end
    partial[t + Int32(1)] = acc
    @synchronize
    step = Int32(32)
    while step > Int32(0)
        if lane < step
            partial[t + Int32(1)] += partial[t + step + Int32(1)]
        end
        @synchronize
        step >>= 1
    end
    if lane == Int32(0) && row < M && col < N
        v = partial[t + Int32(1)]
        HASBIAS && (v += Float32(bias[row + Int32(1)]))
        @inbounds out[row + Int32(1) + col * M] = eltype(out)(v)
    end
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
    rows = cld(M, PTQ1_ROWS_PER_WG)
    ptq1_mul_kernel!(ctx.backend, PTQ1_WG)(
        out, A.data, x, bias === nothing ? A.data : bias,
        Int32(M), Int32(K), Int32(N), Int32(rows), Val(bias !== nothing);
        ndrange = rows * N * PTQ1_WG)
    out
end

@kernel cpu=false function ptq1_getrows_kernel!(out, @Const(data), @Const(rows),
                                                M::Int32, K::Int32, N::Int32)
    i = Int32(@index(Global, Linear) - 1)
    if i < K * N
        k = i % K
        n = i ÷ K
        row = Int32(rows[n + Int32(1)])
        if row >= Int32(0) && row < M
            block = k ÷ Int32(PTQ1_QK)
            e = k % Int32(PTQ1_QK)
            blocks = K ÷ Int32(PTQ1_QK)
            base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
            @inbounds out[i + Int32(1)] = eltype(out)(ptq1_scale(data, base) * Float32(ptq1_value(data, base, e)))
        end
    end
end

"""
    ptq1_getrows!(ctx, out, table, rows) -> out

Decode zero-based vocabulary rows into consecutive `K`-element columns.
"""
function ptq1_getrows!(ctx, out, A::PTQ1Matrix, rows)
    N = length(rows)
    length(out) == A.k * N || throw(DimensionMismatch("embedding output must contain $(A.k*N) values"))
    ptq1_getrows_kernel!(ctx.backend, 256)(out, A.data, rows, Int32(A.m), Int32(A.k), Int32(N);
                                                ndrange=A.k * N)
    out
end

@kernel cpu=false function ptq1_dequant_kernel!(out, @Const(data), M::Int32, K::Int32)
    i = Int32(@index(Global, Linear) - 1)
    if i < M * K
        row = i % M
        k = i ÷ M
        block = k ÷ Int32(PTQ1_QK)
        e = k % Int32(PTQ1_QK)
        blocks = K ÷ Int32(PTQ1_QK)
        base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
        @inbounds out[i + Int32(1)] = eltype(out)(ptq1_scale(data, base) * Float32(ptq1_value(data, base, e)))
    end
end

"""Expand a PTQ1 matrix to a device Float32 matrix, for verification only."""
function ptq1_dequant(ctx, A::PTQ1Matrix)
    out = KernelAbstractions.allocate(ctx.backend, Float32, size(A)...)
    ptq1_dequant_kernel!(ctx.backend, 256)(out, A.data, Int32(A.m), Int32(A.k); ndrange=A.m*A.k)
    out
end

# Normalized 1024-wide FWHT. 256 threads process the 512 butterfly pairs in
# two passes per stage. `signs` spans the full input width; the transform itself
# is block diagonal. Forward folded matmuls apply S then H. Embedding lookup is
# the inverse and applies H then S.
@kernel cpu=false function hadamard1024_kernel!(x, @Const(signs), width::Int32,
                                               blocks::Int32,
                                               ::Val{HASSIGNS},
                                               ::Val{INVERSE}) where {HASSIGNS,INVERSE}
    sh = @localmem Float32 (1024,)
    t = Int32(@index(Local, Linear) - 1)
    wg = Int32(@index(Group, Linear) - 1)
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
    @synchronize
    stride = Int32(1)
    while stride < Int32(1024)
        @inbounds for pass in Int32(0):Int32(1)
            p = t + pass * Int32(256)
            group = p ÷ stride
            j = p % stride
            aidx = group * (stride << 1) + j
            bidx = aidx + stride
            a = sh[aidx + Int32(1)]
            b = sh[bidx + Int32(1)]
            sh[aidx + Int32(1)] = a + b
            sh[bidx + Int32(1)] = a - b
        end
        @synchronize
        stride <<= 1
    end
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        v = sh[i + Int32(1)] * 0.03125f0
        if HASSIGNS && INVERSE
            v *= Float32(signs[block * Int32(1024) + i + Int32(1)])
        end
        x[base + i + Int32(1)] = eltype(x)(v)
    end
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
    hadamard1024_kernel!(ctx.backend, 256)(x, signs === nothing ? x : signs,
        Int32(width), Int32(blocks), Val(signs !== nothing), Val(inverse);
        ndrange=blocks * columns * 256)
    x
end
