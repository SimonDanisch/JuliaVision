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
    # n is in 0:4.  Multiplication by 3^n modulo 256 is exactly the recurrence
    # above, without a runtime loop per decoded trit.
    p = n == Int32(0) ? UInt32(1) :
        n == Int32(1) ? UInt32(3) :
        n == Int32(2) ? UInt32(9) :
        n == Int32(3) ? UInt32(27) : UInt32(81)
    b = (b * p) & UInt32(0xff)
    Int32((b * UInt32(3)) >> 8) - Int32(1)
end

# Four independent 64-lane reductions share a 256-thread workgroup. A lane
# consumes two trits from every 128-value block. This gives decode enough waves
# even for the 48-element alpha/beta projections, while each packed byte and
# activation is read only by the row that uses it.
function ptq1_mul_kernel!(out, data, x, bias,
                                            M::Int32, K::Int32, N::Int32,
                                            rowgroups::Int32,
                                            ::Val{HASBIAS},
                                            ::Val{SUBGROUP}) where {HASBIAS,SUBGROUP}
    # A row is always 64 lanes. On wave64 hardware its reduction is one native
    # subgroup instruction. Wave32 needs one partial from each of its two
    # subgroups and a single workgroup barrier before the row leader combines
    # them. This replaces the old six-barrier shared-memory reduction tree on
    # the Radeon decode path while keeping the same kernel portable to NVIDIA.
    partial = KI.localmemory(Float32, Val((PTQ1_ROWS_PER_WG * (64 ÷ SUBGROUP),)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    wg = Int32(KI.get_group_id().x - 1)
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
    reduced = KI.sub_group_reduce_add(acc)
    if SUBGROUP == 64
        if lane == Int32(0) && row < M && col < N
            v = reduced
            HASBIAS && (v += Float32(bias[row + Int32(1)]))
            @inbounds out[row + Int32(1) + col * M] = eltype(out)(v)
        end
    else
        nsub = Int32(64 ÷ SUBGROUP)
        subinrow = lane ÷ Int32(SUBGROUP)
        if sublane == Int32(0)
            partial[rowin * nsub + subinrow + Int32(1)] = reduced
        end
        KI.barrier()
        if lane == Int32(0) && row < M && col < N
            v = partial[rowin * nsub + Int32(1)] +
                partial[rowin * nsub + Int32(2)]
            HASBIAS && (v += Float32(bias[row + Int32(1)]))
            @inbounds out[row + Int32(1) + col * M] = eltype(out)(v)
        end
    end
    return nothing
end

const PTQ1_COLS_PER_WG = 4

# Wide-prefill tile.  The small-N kernels above assign a subgroup to one output
# row, which is the right shape for decode but leaves almost all matrix reuse on
# the table during prompt processing.  This kernel stages a 64x32x32 matrix
# tile: every decoded weight is shared by 32 prompt columns and every activation
# is shared by 64 output rows.
const PTQ1_MM_BM = 64
const PTQ1_MM_BN = 32
const PTQ1_MM_BK = 32

# Prefill tile: a packed ternary weight is decoded once and multiplied by four
# activation columns. Decode keeps the one-column kernel above because four
# accumulators would only add register pressure there.
function ptq1_mul4_kernel!(out, data, x, bias,
                                             M::Int32, K::Int32, N::Int32,
                                             rowgroups::Int32,
                                             ::Val{HASBIAS},
                                             ::Val{SUBGROUP}) where {HASBIAS,SUBGROUP}
    partial = KI.localmemory(Float32, Val((PTQ1_COLS_PER_WG * PTQ1_ROWS_PER_WG *
                                 (64 ÷ SUBGROUP),)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    wg = Int32(KI.get_group_id().x - 1)
    rg = wg % rowgroups
    col0 = (wg ÷ rowgroups) * Int32(PTQ1_COLS_PER_WG)
    row = rg * Int32(PTQ1_ROWS_PER_WG) + rowin
    blocks = K ÷ Int32(PTQ1_QK)
    a0 = 0f0; a1 = 0f0; a2 = 0f0; a3 = 0f0
    if row < M
        @inbounds for block in Int32(0):(blocks - Int32(1))
            base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
            scale = ptq1_scale(data, base)
            e0 = lane
            e1 = lane + Int32(64)
            w0 = scale * Float32(ptq1_value(data, base, e0))
            w1 = scale * Float32(ptq1_value(data, base, e1))
            kbase = block * Int32(PTQ1_QK)
            if col0 < N
                xb = col0 * K + kbase
                a0 = muladd(w0, Float32(x[xb + e0 + Int32(1)]), a0)
                a0 = muladd(w1, Float32(x[xb + e1 + Int32(1)]), a0)
            end
            if col0 + Int32(1) < N
                xb = (col0 + Int32(1)) * K + kbase
                a1 = muladd(w0, Float32(x[xb + e0 + Int32(1)]), a1)
                a1 = muladd(w1, Float32(x[xb + e1 + Int32(1)]), a1)
            end
            if col0 + Int32(2) < N
                xb = (col0 + Int32(2)) * K + kbase
                a2 = muladd(w0, Float32(x[xb + e0 + Int32(1)]), a2)
                a2 = muladd(w1, Float32(x[xb + e1 + Int32(1)]), a2)
            end
            if col0 + Int32(3) < N
                xb = (col0 + Int32(3)) * K + kbase
                a3 = muladd(w0, Float32(x[xb + e0 + Int32(1)]), a3)
                a3 = muladd(w1, Float32(x[xb + e1 + Int32(1)]), a3)
            end
        end
    end
    r0 = KI.sub_group_reduce_add(a0)
    r1 = KI.sub_group_reduce_add(a1)
    r2 = KI.sub_group_reduce_add(a2)
    r3 = KI.sub_group_reduce_add(a3)
    if SUBGROUP == 64
        if lane == Int32(0) && row < M
            HASBIAS && begin
                b = Float32(bias[row + Int32(1)])
                r0 += b; r1 += b; r2 += b; r3 += b
            end
            col0 < N && (@inbounds out[row + Int32(1) + col0 * M] = eltype(out)(r0))
            col0 + Int32(1) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(1)) * M] = eltype(out)(r1))
            col0 + Int32(2) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(2)) * M] = eltype(out)(r2))
            col0 + Int32(3) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(3)) * M] = eltype(out)(r3))
        end
    else
        nsub = Int32(64 ÷ SUBGROUP)
        subinrow = lane ÷ Int32(SUBGROUP)
        if sublane == Int32(0)
            base = rowin * nsub + subinrow + Int32(1)
            partial[base] = r0
            partial[Int32(PTQ1_ROWS_PER_WG) * nsub + base] = r1
            partial[Int32(2 * PTQ1_ROWS_PER_WG) * nsub + base] = r2
            partial[Int32(3 * PTQ1_ROWS_PER_WG) * nsub + base] = r3
        end
        KI.barrier()
        if lane == Int32(0) && row < M
            base = rowin * nsub + Int32(1)
            r0 = partial[base] + partial[base + Int32(1)]
            r1 = partial[Int32(PTQ1_ROWS_PER_WG) * nsub + base] +
                 partial[Int32(PTQ1_ROWS_PER_WG) * nsub + base + Int32(1)]
            r2 = partial[Int32(2 * PTQ1_ROWS_PER_WG) * nsub + base] +
                 partial[Int32(2 * PTQ1_ROWS_PER_WG) * nsub + base + Int32(1)]
            r3 = partial[Int32(3 * PTQ1_ROWS_PER_WG) * nsub + base] +
                 partial[Int32(3 * PTQ1_ROWS_PER_WG) * nsub + base + Int32(1)]
            HASBIAS && begin
                b = Float32(bias[row + Int32(1)])
                r0 += b; r1 += b; r2 += b; r3 += b
            end
            col0 < N && (@inbounds out[row + Int32(1) + col0 * M] = eltype(out)(r0))
            col0 + Int32(1) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(1)) * M] = eltype(out)(r1))
            col0 + Int32(2) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(2)) * M] = eltype(out)(r2))
            col0 + Int32(3) < N && (@inbounds out[row + Int32(1) + (col0 + Int32(3)) * M] = eltype(out)(r3))
        end
    end
    return nothing
end

function ptq1_mul_mm_kernel!(out, data, x,
                                                M::Int32, K::Int32, N::Int32,
                                                mtiles::Int32)
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
                block = k ÷ Int32(PTQ1_QK)
                e = k % Int32(PTQ1_QK)
                base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
                av = ptq1_scale(data, base) * Float32(ptq1_value(data, base, e))
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
    rows = cld(M, PTQ1_ROWS_PER_WG)
    subgroup = ctx.dev.subgroup
    subgroup in (32, 64) || throw(ArgumentError(
        "PTQ1 requires a 32- or 64-lane subgroup, got $subgroup"))
    if N >= 8 && bias === nothing
        mtiles = cld(M, PTQ1_MM_BM)
        ntiles = cld(N, PTQ1_MM_BN)
        KI.Kernel(ctx.backend, ptq1_mul_mm_kernel!)(
            out, A.data, x, Int32(M), Int32(K), Int32(N), Int32(mtiles);
            ndrange = mtiles * ntiles * PTQ1_WG, workgroupsize = PTQ1_WG)
    else
        kernel = N == 1 ? ptq1_mul_kernel! : ptq1_mul4_kernel!
        columns = N == 1 ? N : cld(N, PTQ1_COLS_PER_WG)
        KI.Kernel(ctx.backend, kernel)(
            out, A.data, x, bias === nothing ? A.data : bias,
            Int32(M), Int32(K), Int32(N), Int32(rows), Val(bias !== nothing),
            Val(subgroup);
            ndrange = rows * columns * PTQ1_WG, workgroupsize = PTQ1_WG)
    end
    out
end

function ptq1_getrows_kernel!(out, data, rows,
                                                M::Int32, K::Int32, N::Int32)
    i = Int32(KI.get_global_id().x - 1)
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
    return nothing
end

"""
    ptq1_getrows!(ctx, out, table, rows) -> out

Decode zero-based vocabulary rows into consecutive `K`-element columns.
"""
function ptq1_getrows!(ctx, out, A::PTQ1Matrix, rows)
    N = length(rows)
    length(out) == A.k * N || throw(DimensionMismatch("embedding output must contain $(A.k*N) values"))
    KI.Kernel(ctx.backend, ptq1_getrows_kernel!)(out, A.data, rows, Int32(A.m), Int32(A.k), Int32(N);
                                                ndrange=A.k * N, workgroupsize = 256)
    out
end

function ptq1_dequant_kernel!(out, data, M::Int32, K::Int32)
    i = Int32(KI.get_global_id().x - 1)
    if i < M * K
        row = i % M
        k = i ÷ M
        block = k ÷ Int32(PTQ1_QK)
        e = k % Int32(PTQ1_QK)
        blocks = K ÷ Int32(PTQ1_QK)
        base = (row * blocks + block) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
        @inbounds out[i + Int32(1)] = eltype(out)(ptq1_scale(data, base) * Float32(ptq1_value(data, base, e)))
    end
    return nothing
end

"""Expand a PTQ1 matrix to a device Float32 matrix, for verification only."""
function ptq1_dequant(ctx, A::PTQ1Matrix)
    out = KernelAbstractions.allocate(ctx.backend, Float32, size(A)...)
    KI.Kernel(ctx.backend, ptq1_dequant_kernel!)(out, A.data, Int32(A.m), Int32(A.k); ndrange=A.m*A.k, workgroupsize = 256)
    out
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
            qblock = k ÷ Int32(PTQ1_QK)
            qe = k % Int32(PTQ1_QK)
            base = ((tm + row) * blocks + qblock) * Int32(PTQ1_BLOCK_BYTES) + Int32(1)
            scale = ptq1_scale(data, base)
            Base.Cartesian.@nexprs 8 l -> begin
                sA[row + (e0 + Int32(l - 1)) * lda + Int32(1)] =
                    Float16(scale * Float32(ptq1_value(
                        data, base, qe + Int32(l - 1))))
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
    KI.Kernel(ctx.backend, hadamard1024_kernel!)(x, signs === nothing ? x : signs,
        Int32(width), Int32(blocks), Val(signs !== nothing), Val(inverse);
        ndrange=blocks * columns * 256, workgroupsize = 256)
    x
end
