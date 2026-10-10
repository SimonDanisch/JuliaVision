# A weight element in single precision. The checkpoint's BF16 tensors stay on the
# device as their raw 16-bit words, which is exact (BF16 is the top half of a
# Float32) and half the bytes a decode token reads for them; a raw UInt16 is those
# bits, never an integer value.
@inline _weightf32(w::UInt16) = reinterpret(Float32, UInt32(w) << 16)
@inline _weightf32(w::Real) = Float32(w)

import KernelInterface as KI

function copy_kernel!(out, x, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds out[i] = eltype(out)(x[i]))
    return nothing
end

# Out-of-place signed FWHT for batched projections.  The old path copied the
# complete activation matrix and then transformed it in place.  Loading the
# source directly into shared memory removes that full-device copy.  Recurrent
# layers may simultaneously retain the untransformed fp16 input for their two
# small cooperative projections.
function hadamard1024_out_batch_kernel!(
        out, halfcopy, x, signs, width::Int32, blocks::Int32,
        ::Val{COPYHALF}) where {COPYHALF}
    sh = KI.localmemory(Float32, Val((1024,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    block = wg % blocks
    col = wg ÷ blocks
    base = col * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        v = Float32(x[base + i + Int32(1)])
        COPYHALF && (halfcopy[base + i + Int32(1)] = Float16(v))
        sh[i + Int32(1)] = v * Float32(signs[block * Int32(1024) + i + Int32(1)])
    end
    KI.barrier()
    # A shift and a mask, as in `DNNKernels.hadamard1024_kernel!`: `÷` and `%`
    # by the runtime `stride` were two integer divisions per pair per stage, on a
    # device with no integer divide.
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
    return nothing
end

function gdn_permute_hadamard1024_batch_kernel!(
        out, x, signs, width::Int32, blocks::Int32)
    sh = KI.localmemory(Float32, Val((1024,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    block = wg % blocks
    token = wg ÷ blocks
    base = token * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        inner = block * Int32(1024) + i
        hd = inner % Int32(128)
        rest = inner ÷ Int32(128)
        rep = rest % Int32(3)
        nk = rest ÷ Int32(3)
        src = token * width + hd + nk * Int32(128) + rep * Int32(2048)
        sh[i + Int32(1)] = Float32(x[src + Int32(1)]) *
                           Float32(signs[block * Int32(1024) + i + Int32(1)])
    end
    KI.barrier()
    # A shift and a mask, as in `DNNKernels.hadamard1024_kernel!`: `÷` and `%`
    # by the runtime `stride` were two integer divisions per pair per stage, on a
    # device with no integer divide.
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
    return nothing
end

function dense_ab_split_kernel!(alpha, beta, ab,
                                                  ntokens::Int32)
    i = Int32(KI.get_global_id().x - 1)
    if i < Int32(48) * ntokens
        token = i ÷ Int32(48)
        head = i % Int32(48)
        @inbounds begin
            alpha[i + Int32(1)] = ab[token * Int32(96) + head + Int32(1)]
            beta[i + Int32(1)] = ab[token * Int32(96) + Int32(48) + head + Int32(1)]
        end
    end
    return nothing
end

# One workgroup normalizes one activation column. The weight is shared by all
# columns, while input and output remain column-major just like the PTQ kernels.
function rmsnorm_batch_kernel!(out, x, weight,
                                                 n::Int32, eps::Float32)
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    col = Int32(KI.get_group_id().x - 1)
    base = col * n
    sum = 0f0
    i = t
    while i < n
        @inbounds v = Float32(x[base + i + Int32(1)])
        sum = muladd(v, v, sum)
        i += Int32(256)
    end
    sh[t + Int32(1)] = sum
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        t < step && (sh[t + Int32(1)] += sh[t + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(1)] / Float32(n) + eps))
    i = t
    while i < n
        @inbounds out[base + i + Int32(1)] = eltype(out)(
            Float32(x[base + i + Int32(1)]) * scale * _weightf32(weight[i + Int32(1)]))
        i += Int32(256)
    end
    return nothing
end

# Decode's layer boundary as one pass: the residual add, the RMS norm and the
# signed Hadamard transform the next projections read. Each of the transform's
# 1024-wide workgroups computes the norm over the whole row itself, which is a
# 20 KiB read, rather than waiting a pass for one workgroup to. The residual is
# written to `xout`, never in place, because every workgroup reads all of `xin`.
# Without a branch (`HASBRANCH = false`) it is the norm and transform alone and
# `xout` is not written.
function add_rmsnorm_hadamard_kernel!(xout, norm, had, xin, branch, weight, signs,
                                      n::Int32, eps::Float32,
                                      ::Val{HASBRANCH}, ::Val{SUBGROUP}) where {HASBRANCH,SUBGROUP}
    sh = KI.localmemory(Float32, Val((1024,)), Val(1))
    red = KI.localmemory(Float32, Val((256 ÷ SUBGROUP,)), Val(2))
    t = Int32(KI.get_local_id().x - 1)
    block = Int32(KI.get_group_id().x - 1)
    sum = 0f0
    i = t
    while i < n
        @inbounds v = Float32(xin[i + Int32(1)])
        HASBRANCH && (@inbounds v += Float32(branch[i + Int32(1)]))
        sum = muladd(v, v, sum)
        i += Int32(256)
    end
    lane = t % Int32(SUBGROUP)
    subgroup = t ÷ Int32(SUBGROUP)
    reduced = KI.sub_group_reduce_add(sum)
    lane == Int32(0) && (@inbounds red[subgroup + Int32(1)] = reduced)
    KI.barrier()
    total = 0f0
    for k in Int32(1):Int32(256 ÷ SUBGROUP)
        total += @inbounds red[k]
    end
    scale = inv(sqrt(total / Float32(n) + eps))
    base = block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        e = base + t + j * Int32(256) + Int32(1)
        v = Float32(xin[e])
        if HASBRANCH
            v += Float32(branch[e])
            xout[e] = eltype(xout)(v)
        end
        nv = v * scale * _weightf32(weight[e])
        norm[e] = eltype(norm)(nv)
        sh[t + j * Int32(256) + Int32(1)] = nv * Float32(signs[e])
    end
    KI.barrier()
    stride = Int32(1)
    lg = Int32(0)
    while stride < Int32(1024)
        @inbounds for pass in Int32(0):Int32(1)
            p = t + pass * Int32(256)
            group = p >> lg
            jj = p & (stride - Int32(1))
            aidx = group * (stride << 1) + jj
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
        k = t + j * Int32(256)
        had[base + k + Int32(1)] = eltype(had)(sh[k + Int32(1)] * inv(sqrt(1024f0)))
    end
    return nothing
end

# Residual addition and the normalization which immediately consumes it share
# one read of x.  A workgroup owns one token, keeps the sum in shared memory,
# updates the residual stream, and writes the next normalized input.
function add_rmsnorm_batch_kernel!(out, x, branch,
                                                     weight, n::Int32,
                                                     eps::Float32,
                                                     ::Val{SUBGROUP}) where {SUBGROUP}
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    col = Int32(KI.get_group_id().x - 1)
    base = col * n
    sum = 0f0
    i = t
    while i < n
        @inbounds begin
            v = Float32(x[base + i + Int32(1)]) +
                Float32(branch[base + i + Int32(1)])
            x[base + i + Int32(1)] = eltype(x)(v)
            sum = muladd(v, v, sum)
        end
        i += Int32(256)
    end
    lane = t % Int32(SUBGROUP)
    subgroup = t ÷ Int32(SUBGROUP)
    reduced = DNNKernels.KI.sub_group_reduce_add(sum)
    lane == Int32(0) && (sh[subgroup + Int32(1)] = reduced)
    KI.barrier()
    nsubgroups = Int32(256 ÷ SUBGROUP)
    if subgroup == Int32(0)
        v = lane < nsubgroups ? sh[lane + Int32(1)] : 0f0
        total = DNNKernels.KI.sub_group_reduce_add(v)
        lane == Int32(0) && (sh[Int32(1)] = total)
    end
    KI.barrier()
    scale = inv(sqrt(sh[Int32(1)] / Float32(n) + eps))
    i = t
    while i < n
        @inbounds out[base + i + Int32(1)] = eltype(out)(
            Float32(x[base + i + Int32(1)]) * scale * _weightf32(weight[i + Int32(1)]))
        i += Int32(256)
    end
    return nothing
end

const DENSE_ROWS_PER_WG = 4

# Four output rows per 256-thread workgroup, one 64-lane reduction per row.
# The previous kernel assigned one thread to a whole row: the recurrent 5120x48
# projections therefore launched only 48 threads, each with a 5120-FMA serial
# dependency chain and each lane reading a different, distant row. Here lanes
# walk adjacent K values and the hardware reduction closes the dot product.
# The decode step's two small projections that share an input, alpha and beta
# (48 rows each), in one launch with one workgroup per row. Four rows to a
# workgroup was twelve workgroups a projection, each row a serial 80-step loop over
# 64 lanes: 9.7 us apiece on an M5, at 50 GB/s.
function dense2_gemv_kernel!(out1, out2, w1, w2, x, K::Int32, M1::Int32,
                             ::Val{SUBGROUP}) where {SUBGROUP}
    red = KI.localmemory(Float32, Val((256 ÷ SUBGROUP,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    r = Int32(KI.get_group_id().x - 1)
    second = r >= M1
    row = second ? r - M1 : r
    w = second ? w2 : w1
    acc = 0f0
    k = t
    @inbounds while k < K
        acc = muladd(_weightf32(w[row * K + k + Int32(1)]), Float32(x[k + Int32(1)]), acc)
        k += Int32(256)
    end
    s = KI.sub_group_reduce_add(acc)
    t % Int32(SUBGROUP) == Int32(0) && (@inbounds red[t ÷ Int32(SUBGROUP) + Int32(1)] = s)
    KI.barrier()
    if t == Int32(0)
        v = 0f0
        for i in Int32(1):Int32(256 ÷ SUBGROUP)
            v += @inbounds red[i]
        end
        if second
            @inbounds out2[row + Int32(1)] = eltype(out2)(v)
        else
            @inbounds out1[row + Int32(1)] = eltype(out1)(v)
        end
    end
    return nothing
end

function dense_gemv_batch_kernel!(out, weight, x,
                                                    K::Int32, M::Int32,
                                                    rowgroups::Int32,
                                                    ::Val{SUBGROUP}) where {SUBGROUP}
    partial = KI.localmemory(Float32, Val((DENSE_ROWS_PER_WG * (64 ÷ SUBGROUP),)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    wg = Int32(KI.get_group_id().x - 1)
    rg = wg % rowgroups
    col = wg ÷ rowgroups
    row = rg * Int32(DENSE_ROWS_PER_WG) + rowin
    acc = 0f0
    if row < M
        wbase = row * K
        xbase = col * K
        k = lane
        @inbounds while k < K
            acc = muladd(_weightf32(weight[wbase + k + Int32(1)]),
                         Float32(x[xbase + k + Int32(1)]), acc)
            k += Int32(64)
        end
    end
    reduced = DNNKernels.KI.sub_group_reduce_add(acc)
    if SUBGROUP == 64
        lane == Int32(0) && row < M &&
            (@inbounds out[col * M + row + Int32(1)] = eltype(out)(reduced))
    else
        nsub = Int32(64 ÷ SUBGROUP)
        subinrow = lane ÷ Int32(SUBGROUP)
        sublane == Int32(0) &&
            (partial[rowin * nsub + subinrow + Int32(1)] = reduced)
        KI.barrier()
        lane == Int32(0) && row < M &&
            (@inbounds out[col * M + row + Int32(1)] = eltype(out)(
                partial[rowin * nsub + Int32(1)] + partial[rowin * nsub + Int32(2)]))
    end
    return nothing
end

# A K-major dense weight, as the checkpoint stores it, to an M x K Float32 matrix,
# for a GEMM that reads `A` column-major.
function weight_transpose_kernel!(W, weight, K::Int32, M::Int32)
    i = Int32(KI.get_global_id().x - 1)
    if i < K * M
        m = i % M
        k = i ÷ M
        @inbounds W[i + Int32(1)] = _weightf32(weight[k + m * K + Int32(1)])
    end
    return nothing
end

const DENSE_MM_BM = 32
const DENSE_MM_BN = 32
const DENSE_MM_BK = 32

function dense_mul_mm_kernel!(out, weight, x,
                                                K::Int32, M::Int32, N::Int32,
                                                mtiles::Int32)
    atile = KI.localmemory(Float32, Val((DENSE_MM_BM * DENSE_MM_BK,)), Val(1))
    btile = KI.localmemory(Float32, Val((DENSE_MM_BN * DENSE_MM_BK,)), Val(2))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    mt = wg % mtiles
    nt = wg ÷ mtiles
    m0 = mt * Int32(DENSE_MM_BM)
    n0 = nt * Int32(DENSE_MM_BN)
    lr0 = (t & Int32(15)) * Int32(2)
    lc0 = (t >> 4) * Int32(2)
    a00 = 0f0; a01 = 0f0; a10 = 0f0; a11 = 0f0
    k0 = Int32(0)
    while k0 < K
        ai = t
        while ai < Int32(DENSE_MM_BM * DENSE_MM_BK)
            lr = ai ÷ Int32(DENSE_MM_BK)
            lk = ai % Int32(DENSE_MM_BK)
            row = m0 + lr
            @inbounds atile[ai + Int32(1)] = row < M && k0 + lk < K ?
                _weightf32(weight[row * K + k0 + lk + Int32(1)]) : 0f0
            ai += Int32(256)
        end
        bi = t
        while bi < Int32(DENSE_MM_BN * DENSE_MM_BK)
            lc = bi ÷ Int32(DENSE_MM_BK)
            lk = bi % Int32(DENSE_MM_BK)
            col = n0 + lc
            @inbounds btile[bi + Int32(1)] = col < N && k0 + lk < K ?
                Float32(x[col * K + k0 + lk + Int32(1)]) : 0f0
            bi += Int32(256)
        end
        KI.barrier()
        for lk in Int32(0):Int32(DENSE_MM_BK - 1)
            av0 = atile[(lr0 + Int32(0)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            av1 = atile[(lr0 + Int32(1)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            bv0 = btile[(lc0 + Int32(0)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            bv1 = btile[(lc0 + Int32(1)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            a00 = muladd(av0, bv0, a00); a01 = muladd(av0, bv1, a01)
            a10 = muladd(av1, bv0, a10); a11 = muladd(av1, bv1, a11)
        end
        KI.barrier()
        k0 += Int32(DENSE_MM_BK)
    end
    row0 = m0 + lr0
    col0 = n0 + lc0
    row0 < M && col0 < N && (@inbounds out[row0 + Int32(1) + col0 * M] = eltype(out)(a00))
    row0 < M && col0 + Int32(1) < N &&
        (@inbounds out[row0 + Int32(1) + (col0 + Int32(1)) * M] = eltype(out)(a01))
    row0 + Int32(1) < M && col0 < N &&
        (@inbounds out[row0 + Int32(2) + col0 * M] = eltype(out)(a10))
    row0 + Int32(1) < M && col0 + Int32(1) < N &&
        (@inbounds out[row0 + Int32(2) + (col0 + Int32(1)) * M] = eltype(out)(a11))
    return nothing
end

# SwiGLU's only consumer is the signed Hadamard transform before ffn_down.
# Produce that transformed fp16 operand directly and avoid materialising and
# copying a 17408xN Float32 intermediate.
function swiglu_hadamard1024_batch_kernel!(
        out, gate, up, signs, width::Int32,
        blocks::Int32)
    sh = KI.localmemory(Float32, Val((1024,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    block = wg % blocks
    col = wg ÷ blocks
    base = col * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        g = Float32(gate[base + i + Int32(1)])
        v = (g / (1f0 + exp(-g))) * Float32(up[base + i + Int32(1)])
        sh[i + Int32(1)] = v * Float32(signs[block * Int32(1024) + i + Int32(1)])
    end
    KI.barrier()
    # A shift and a mask, as in `DNNKernels.hadamard1024_kernel!`: `÷` and `%`
    # by the runtime `stride` were two integer divisions per pair per stage, on a
    # device with no integer divide.
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
    return nothing
end

function recurrent_split_batch_kernel!(q, k, v, qkv)
    i = Int32(KI.get_global_id().x - 1)
    token = i ÷ Int32(6144)
    inner = i % Int32(6144)
    qkvbase = token * Int32(10240)
    if inner < Int32(2048)
        @inbounds begin
            q[token * Int32(2048) + inner + Int32(1)] = qkv[qkvbase + inner + Int32(1)]
            k[token * Int32(2048) + inner + Int32(1)] = qkv[qkvbase + Int32(2048) + inner + Int32(1)]
        end
    end
    @inbounds v[token * Int32(6144) + inner + Int32(1)] =
        qkv[qkvbase + Int32(4096) + inner + Int32(1)]
    return nothing
end

function gdn_permute_batch_kernel!(out, x)
    i = Int32(KI.get_global_id().x - 1)
    token = i ÷ Int32(6144)
    inner = i % Int32(6144)
    hd = inner % Int32(128)
    rest = inner ÷ Int32(128)
    rep = rest % Int32(3)
    nk = rest ÷ Int32(3)
    src = token * Int32(6144) + hd + nk * Int32(128) + rep * Int32(2048)
    @inbounds out[i + Int32(1)] = x[src + Int32(1)]
    return nothing
end

function depthwise_conv4_batch_kernel!(out, state, x,
                                                         weight, channels::Int32,
                                                         ntokens::Int32)
    c = Int32(KI.get_global_id().x - 1)
    if c < channels
        si = c * Int32(3) + Int32(1)
        wi = c * Int32(4) + Int32(1)
        @inbounds begin
            x0 = Float32(state[si])
            x1 = Float32(state[si + Int32(1)])
            x2 = Float32(state[si + Int32(2)])
            for token in Int32(0):(ntokens - Int32(1))
                x3 = Float32(x[token * channels + c + Int32(1)])
                y = muladd(_weightf32(weight[wi]), x0,
                    muladd(_weightf32(weight[wi + Int32(1)]), x1,
                    muladd(_weightf32(weight[wi + Int32(2)]), x2,
                           _weightf32(weight[wi + Int32(3)]) * x3)))
                out[token * channels + c + Int32(1)] = eltype(out)(
                    y / (1f0 + exp(-y)))
                x0, x1, x2 = x1, x2, x3
            end
            state[si] = eltype(state)(x0)
            state[si + Int32(1)] = eltype(state)(x1)
            state[si + Int32(2)] = eltype(state)(x2)
        end
    end
    return nothing
end

function gated_delta_net_batch_kernel!(out, state,
        q, k, v, alpha, beta,
        dt_bias, a, z, norm_weight,
        eps::Float32, ntokens::Int32, nheads::Int32, nkeyheads::Int32)
    sh = KI.localmemory(Float32, Val((512,)), Val(1))
    col = Int32(KI.get_local_id().x - 1)
    head = Int32(KI.get_group_id().x - 1)
    keyhead = head % nkeyheads
    sbase = head * Int32(16384) + col * Int32(128)
    for token in Int32(0):(ntokens - Int32(1))
        qoff = token * Int32(2048) + keyhead * Int32(128)
        voff = token * Int32(6144) + head * Int32(128)
        @inbounds begin
            qv = Float32(q[qoff + col + Int32(1)])
            kv = Float32(k[qoff + col + Int32(1)])
            sh[col + Int32(1)] = qv * qv
            sh[Int32(128) + col + Int32(1)] = kv * kv
        end
        KI.barrier()
        step = Int32(64)
        while step > Int32(0)
            if col < step
                @inbounds begin
                    sh[col + Int32(1)] += sh[col + step + Int32(1)]
                    sh[Int32(128) + col + Int32(1)] +=
                        sh[Int32(128) + col + step + Int32(1)]
                end
            end
            KI.barrier()
            step >>= 1
        end
        qscale = inv(sqrt(sh[Int32(1)] + eps))
        kscale = inv(sqrt(sh[Int32(129)] + eps))
        @inbounds begin
            sh[col + Int32(1)] = Float32(q[qoff + col + Int32(1)]) * qscale
            sh[Int32(128) + col + Int32(1)] = Float32(k[qoff + col + Int32(1)]) * kscale
        end
        KI.barrier()

        gatei = token * nheads + head + Int32(1)
        @inbounds begin
            raw_alpha = Float32(alpha[gatei]) + Float32(dt_bias[head + Int32(1)])
            decay = exp((max(raw_alpha, 0f0) + log1p(exp(-abs(raw_alpha)))) *
                        Float32(a[head + Int32(1)]))
            b = 1f0 / (1f0 + exp(-Float32(beta[gatei])))
            sk = 0f0
            for row in Int32(0):Int32(127)
                sk = muladd(decay * Float32(state[sbase + row + Int32(1)]),
                            sh[Int32(128) + row + Int32(1)], sk)
            end
            delta = (Float32(v[voff + col + Int32(1)]) - sk) * b
            y = 0f0
            for row in Int32(0):Int32(127)
                si = sbase + row + Int32(1)
                sv = muladd(decay, Float32(state[si]),
                            sh[Int32(128) + row + Int32(1)] * delta)
                state[si] = eltype(state)(sv)
                y = muladd(sv, sh[row + Int32(1)], y)
            end
            y *= 0.08838834764831845f0
            sh[Int32(256) + col + Int32(1)] = y
            sh[Int32(384) + col + Int32(1)] = y * y
        end
        KI.barrier()
        step = Int32(64)
        while step > Int32(0)
            col < step && (@inbounds sh[Int32(384) + col + Int32(1)] +=
                sh[Int32(384) + col + step + Int32(1)])
            KI.barrier()
            step >>= 1
        end
        @inbounds begin
            invrms = inv(sqrt(sh[Int32(385)] / 128f0 + eps))
            gated = sh[Int32(256) + col + Int32(1)] * invrms *
                    Float32(norm_weight[col + Int32(1)])
            zv = Float32(z[voff + col + Int32(1)])
            out[voff + col + Int32(1)] = eltype(out)(gated * zv / (1f0 + exp(-zv)))
        end
        KI.barrier()
    end
    return nothing
end

function l2normalize_qk_batch_kernel!(q, k, eps::Float32)
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    group = Int32(KI.get_group_id().x - 1)
    base = group * Int32(128)
    @inbounds begin
        qv = Float32(q[base + d + Int32(1)])
        kv = Float32(k[base + d + Int32(1)])
        sh[d + Int32(1)] = qv * qv
        sh[Int32(128) + d + Int32(1)] = kv * kv
    end
    KI.barrier()
    step = Int32(64)
    while step > Int32(0)
        if d < step
            @inbounds begin
                sh[d + Int32(1)] += sh[d + step + Int32(1)]
                sh[Int32(128) + d + Int32(1)] +=
                    sh[Int32(128) + d + step + Int32(1)]
            end
        end
        KI.barrier()
        step >>= 1
    end
    qs = inv(sqrt(sh[Int32(1)] + eps))
    ks = inv(sqrt(sh[Int32(129)] + eps))
    @inbounds begin
        q[base + d + Int32(1)] = eltype(q)(Float32(q[base + d + Int32(1)]) * qs)
        k[base + d + Int32(1)] = eltype(k)(Float32(k[base + d + Int32(1)]) * ks)
    end
    return nothing
end

# One subgroup owns one state column.  Each lane keeps its 2 or 4 rows in
# registers while the complete prompt chunk advances, so the 128x128 recurrent
# state is read and written once per chunk instead of once per token.
# The prefill's DeltaNet recurrence: GDN_COLS state columns of one head per
# subgroup, each lane holding 128 ÷ SUBGROUP rows of every column. One column per
# subgroup was 6144 subgroups, more than the device keeps resident, so the 512-step
# token loop ran in waves: 10.7 ms a layer on an M5 at 512 tokens. Eight columns
# share each token's q, k, decay and beta loads, every subgroup is resident at once,
# and it takes 3.1 ms.
const GDN_COLS = 8

@inline function _gdn_step(s::NTuple{C,NTuple{R,Float32}}, qv::NTuple{R,Float32},
                           kv::NTuple{R,Float32}, v, vb::Int32, decay::Float32,
                           b::Float32) where {C,R}
    sk = ntuple(c -> KI.sub_group_reduce_add(decay * sum(s[c] .* kv)), Val(C))
    delta = ntuple(c -> (@inbounds(Float32(v[vb + Int32(c)])) - sk[c]) * b, Val(C))
    s2 = ntuple(c -> ntuple(r -> muladd(decay, s[c][r], kv[r] * delta[c]), Val(R)), Val(C))
    ys = ntuple(c -> KI.sub_group_reduce_add(sum(s2[c] .* qv)), Val(C))
    return s2, ys
end

@inline _gdn_rows(x, base::Int32, lane::Int32, ::Val{R}, ::Val{SUBGROUP}) where {R,SUBGROUP} =
    ntuple(r -> @inbounds(Float32(x[base + lane + Int32(SUBGROUP * (r - 1)) + Int32(1)])), Val(R))

function gated_delta_state_batch_kernel!(out, state,
        q, k, v, alpha, beta,
        dt_bias, a, ntokens::Int32, ::Val{SUBGROUP}) where {SUBGROUP}
    rows = Val(128 ÷ SUBGROUP)
    t = Int32(KI.get_local_id().x - 1)
    lane = t % Int32(SUBGROUP)
    gsub = Int32(KI.get_group_id().x - 1) * Int32(256 ÷ SUBGROUP) + t ÷ Int32(SUBGROUP)
    head = gsub ÷ Int32(128 ÷ GDN_COLS)
    col0 = (gsub % Int32(128 ÷ GDN_COLS)) * Int32(GDN_COLS)
    keyhead = head % Int32(16)
    sbase = head * Int32(16384) + col0 * Int32(128)
    s = ntuple(c -> _gdn_rows(state, sbase + Int32((c - 1) * 128), lane, rows, Val(SUBGROUP)),
               Val(GDN_COLS))
    dtb = Float32(dt_bias[head + Int32(1)])
    av = Float32(a[head + Int32(1)])
    for token in Int32(0):(ntokens - Int32(1))
        qoff = token * Int32(2048) + keyhead * Int32(128)
        voff = token * Int32(6144) + head * Int32(128) + col0
        gatei = token * Int32(48) + head + Int32(1)
        raw_alpha = Float32(alpha[gatei]) + dtb
        decay = exp((max(raw_alpha, 0f0) + log1p(exp(-abs(raw_alpha)))) * av)
        b = 1f0 / (1f0 + exp(-Float32(beta[gatei])))
        qv = _gdn_rows(q, qoff, lane, rows, Val(SUBGROUP))
        kv = _gdn_rows(k, qoff, lane, rows, Val(SUBGROUP))
        s, ys = _gdn_step(s, qv, kv, v, voff, decay, b)
        # Every lane holds all GDN_COLS sums; the first few write one each.
        lane < Int32(GDN_COLS) &&
            (@inbounds out[voff + lane + Int32(1)] = eltype(out)(ys[lane + 1] * 0.08838834764831845f0))
    end
    @inbounds for c in 1:GDN_COLS, r in 1:(128 ÷ SUBGROUP)
        state[sbase + Int32((c - 1) * 128) + lane + Int32(SUBGROUP * (r - 1)) + Int32(1)] =
            eltype(state)(s[c][r])
    end
    return nothing
end

function gdn_norm_gate_batch_kernel!(out, z,
                                                        norm_weight,
                                                        eps::Float32)
    sh = KI.localmemory(Float32, Val((128,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    group = Int32(KI.get_group_id().x - 1)
    head = group % Int32(48)
    token = group ÷ Int32(48)
    off = token * Int32(6144) + head * Int32(128) + d + Int32(1)
    y = @inbounds Float32(out[off])
    sh[d + Int32(1)] = y * y
    KI.barrier()
    step = Int32(64)
    while step > Int32(0)
        d < step && (@inbounds sh[d + Int32(1)] += sh[d + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    invrms = inv(sqrt(sh[Int32(1)] / 128f0 + eps))
    zv = @inbounds Float32(z[off])
    gated = y * invrms * Float32(norm_weight[d + Int32(1)])
    @inbounds out[off] = eltype(out)(gated * zv / (1f0 + exp(-zv)))
    return nothing
end

# GDN emits value heads in tiled `[hd,nk,rep]` order. Prism folded ssm_out
# against `[hd,rep,nk]`, so this permutation is part of that projection.
# Q projection is `[q(256), gate(256)]` per head, rather than two contiguous
# 6144-vectors. Normalize q per head, rotate its first 64 dimensions, and
# extract the sigmoid gate in one pass.
function prepare_q_kernel!(q, gate, qfull, norm,
                                            position, theta_base::Float32, eps::Float32)
    sh = KI.localmemory(Float32, Val((512,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    h = Int32(KI.get_group_id().x - 1)
    pos = Int32(position[1])
    base = h * Int32(512)
    @inbounds begin
        x = Float32(qfull[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] += sh[Int32(256) + d + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    KI.barrier()
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    KI.barrier()
    o = h * Int32(256) + d + Int32(1)
    @inbounds begin
        q[o] = eltype(q)(sh[d + Int32(1)])
        g = Float32(qfull[base + Int32(256) + d + Int32(1)])
        gate[o] = eltype(gate)(1f0 / (1f0 + exp(-g)))
    end
    return nothing
end

function prepare_q_batch_kernel!(q, gate, qfull, norm,
                                                  position, theta_base::Float32,
                                                  eps::Float32)
    sh = KI.localmemory(Float32, Val((512,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    group = Int32(KI.get_group_id().x - 1)
    token = group ÷ Int32(24)
    h = group % Int32(24)
    pos = Int32(position[1]) + token
    base = token * Int32(12288) + h * Int32(512)
    @inbounds begin
        x = Float32(qfull[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] +=
            sh[Int32(256) + d + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    KI.barrier()
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    KI.barrier()
    o = token * Int32(6144) + h * Int32(256) + d + Int32(1)
    @inbounds begin
        q[o] = eltype(q)(sh[d + Int32(1)])
        g = Float32(qfull[base + Int32(256) + d + Int32(1)])
        gate[o] = eltype(gate)(1f0 / (1f0 + exp(-g)))
    end
    return nothing
end

function prepare_kv_kernel!(kcache, vcache, kscale, vscale,
                                             kproj, vproj,
                                             norm, position,
                                             theta_base::Float32, eps::Float32)
    sh = KI.localmemory(Float32, Val((512,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    h = Int32(KI.get_group_id().x - 1)
    pos = Int32(position[1])
    base = h * Int32(256)
    @inbounds begin
        x = Float32(kproj[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] += sh[Int32(256) + d + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    KI.barrier()
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    KI.barrier()
    cachebase = pos * Int32(1024) + base
    kval = sh[d + Int32(1)]
    vval = Float32(vproj[base + d + Int32(1)])
    sh[Int32(256) + d + Int32(1)] = abs(kval)
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] =
            max(sh[Int32(256) + d + Int32(1)],
                sh[Int32(256) + d + step + Int32(1)]))
        KI.barrier()
        step >>= 1
    end
    ks = max(sh[Int32(257)] / 127f0, floatmin(Float32))
    if d == Int32(0)
        @inbounds kscale[pos * Int32(4) + h + Int32(1)] = ks
    end
    sh[d + Int32(1)] = abs(vval)
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[d + Int32(1)] =
            max(sh[d + Int32(1)], sh[d + step + Int32(1)]))
        KI.barrier()
        step >>= 1
    end
    vs = max(sh[Int32(1)] / 127f0, floatmin(Float32))
    if d == Int32(0)
        @inbounds vscale[pos * Int32(4) + h + Int32(1)] = vs
    end
    @inbounds begin
        # `round` to a FLOAT and `safetrunc` to the int, not `round(Int32, …)`.
        # The typed form carries an `InexactError` branch for NaN and for the
        # out-of-range cases, and constructing that exception ALLOCATES inside
        # the kernel — which is what put `gpu_gc_pool_alloc` in the module and
        # made Lava's `replace_unreachable!` warn that it was lowering the throw
        # path to a `ret undef` of pointer type. `safetrunc`'s docstring in
        # DNNKernels names this exact warning.
        #
        # Same numbers: `round` still rounds to nearest, the clamp still holds
        # the symmetric int8 range, and a NaN now lands on 0 the way torch puts
        # it rather than throwing.
        kcache[cachebase + d + Int32(1)] =
            DNNKernels.safetrunc(Int8, clamp(round(kval / ks), -127f0, 127f0))
        vcache[cachebase + d + Int32(1)] =
            DNNKernels.safetrunc(Int8, clamp(round(vval / vs), -127f0, 127f0))
    end
    return nothing
end

function prepare_kv_batch_kernel!(kcache, vcache, kscale, vscale,
        kproj, vproj, norm, position,
        theta_base::Float32, eps::Float32)
    sh = KI.localmemory(Float32, Val((512,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    group = Int32(KI.get_group_id().x - 1)
    token = group ÷ Int32(4)
    h = group % Int32(4)
    pos = Int32(position[1]) + token
    base = token * Int32(1024) + h * Int32(256)
    @inbounds begin
        x = Float32(kproj[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] +=
            sh[Int32(256) + d + step + Int32(1)])
        KI.barrier()
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    KI.barrier()
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    KI.barrier()
    cachebase = pos * Int32(1024) + h * Int32(256)
    kval = sh[d + Int32(1)]
    vval = Float32(vproj[base + d + Int32(1)])
    sh[Int32(256) + d + Int32(1)] = abs(kval)
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] =
            max(sh[Int32(256) + d + Int32(1)],
                sh[Int32(256) + d + step + Int32(1)]))
        KI.barrier()
        step >>= 1
    end
    ks = max(sh[Int32(257)] / 127f0, floatmin(Float32))
    d == Int32(0) && (@inbounds kscale[pos * Int32(4) + h + Int32(1)] = ks)
    sh[d + Int32(1)] = abs(vval)
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[d + Int32(1)] =
            max(sh[d + Int32(1)], sh[d + step + Int32(1)]))
        KI.barrier()
        step >>= 1
    end
    vs = max(sh[Int32(1)] / 127f0, floatmin(Float32))
    d == Int32(0) && (@inbounds vscale[pos * Int32(4) + h + Int32(1)] = vs)
    @inbounds begin
        # `round` to a FLOAT and `safetrunc` to the int, not `round(Int32, …)`.
        # The typed form carries an `InexactError` branch for NaN and for the
        # out-of-range cases, and constructing that exception ALLOCATES inside
        # the kernel — which is what put `gpu_gc_pool_alloc` in the module and
        # made Lava's `replace_unreachable!` warn that it was lowering the throw
        # path to a `ret undef` of pointer type. `safetrunc`'s docstring in
        # DNNKernels names this exact warning.
        #
        # Same numbers: `round` still rounds to nearest, the clamp still holds
        # the symmetric int8 range, and a NaN now lands on 0 the way torch puts
        # it rather than throwing.
        kcache[cachebase + d + Int32(1)] =
            DNNKernels.safetrunc(Int8, clamp(round(kval / ks), -127f0, 127f0))
        vcache[cachebase + d + Int32(1)] =
            DNNKernels.safetrunc(Int8, clamp(round(vval / vs), -127f0, 127f0))
    end
    return nothing
end

# Decode attention uses online softmax, so it needs no score buffer even for an
# 8K context. One workgroup owns a query head and each lane owns one value
# dimension. Scores are reduced over the 256 lanes, then all lanes update their
# own weighted-value accumulator.
# Decode attention as two passes, the first split over the context.
#
# The kernel this replaced gave each of the 24 query heads one workgroup and
# walked every cached position in order, with a tree reduction and three
# barriers per position. At 2048 positions that was 2.2 ms a layer and 35 of a
# 104 ms token on an M5, for 4 MB of int8 K/V a layer that streams in ~35 us.
#
# Here a workgroup owns one KV head and ATTN_CHUNK positions, and all six query
# heads that share the KV head, so each cached row is read once rather than six
# times. It writes each head's running max, sum and weighted V for its chunk;
# `attention_combine_kernel!` merges the chunks. The grid is sized for the
# cache's capacity and a chunk wholly past `position` returns at once, so the
# recorded plan serves every position up to it.
const ATTN_CHUNK = 64

# Max and sum over a 32-lane segment, landing on the segment's first lane. A
# segment boundary is a multiple of 32 in any subgroup this runs on, and lane
# `l`'s value only ever reaches lanes below it, so the leaders are exact.
@inline function _segmax32(v::Float32)
    v = max(v, KI.shfl_down(v, Int32(16))); v = max(v, KI.shfl_down(v, Int32(8)))
    v = max(v, KI.shfl_down(v, Int32(4))); v = max(v, KI.shfl_down(v, Int32(2)))
    max(v, KI.shfl_down(v, Int32(1)))
end
@inline function _segsum32(v::Float32)
    v += KI.shfl_down(v, Int32(16)); v += KI.shfl_down(v, Int32(8))
    v += KI.shfl_down(v, Int32(4)); v += KI.shfl_down(v, Int32(2))
    v + KI.shfl_down(v, Int32(1))
end

# Helpers rather than closures in the kernel: a closure over a variable the loop
# reassigns is boxed, and a boxed value is a dynamic call on the device.
@inline _qrow(q, base::Int32, ::Val{DL}) where {DL} =
    ntuple(e -> @inbounds(Float32(q[base + Int32(e)])), Val(DL))
@inline _scores6(qr::NTuple{6}, kv) = ntuple(j -> KI.sub_group_reduce_add(sum(qr[j] .* kv)), Val(6))
@inline _accum6(acc::NTuple{6,Float32}, w, pl::Int32, v::Float32) = ntuple(Val(6)) do j
    muladd(@inbounds(w[Int32(j - 1) * Int32(ATTN_CHUNK) + pl + Int32(1)]), v, acc[j])
end

# Loads first, then the arithmetic. A loop that loads and uses one position at a
# time stalls each work-item on every load in turn, and with a few hundred
# workgroups that latency was the kernels' cost: the split decode pass took 412
# us at 8192 positions on an M5, against ~130 for its K/V to stream. Issuing a
# subgroup's K rows, and eight positions of V, before any is used took it to 229.
@inline _kwords(k32, kb::Int32, ::Val{NW}) where {NW} =
    ntuple(u -> @inbounds(k32[kb + Int32(u)]), Val(NW))
@inline _kbytes(w::NTuple{NW,UInt32}) where {NW} = ntuple(Val(4 * NW)) do e  # signed bytes, one shift pair each
    word = w[((e - 1) >> 2) + 1]
    Float32(reinterpret(Int32, word << (24 - 8 * ((e - 1) & 3))) >> 24)
end
@inline function _pv8(acc::NTuple{6,Float32}, w, vcache, vscale, p0::Int32, pl0::Int32,
                      np::Int32, kvh::Int32, t::Int32)
    Base.Cartesian.@nexprs 8 u -> begin
        plu_u = pl0 + Int32(u - 1)
        pu_u = p0 + min(plu_u, np - Int32(1))
        v_u = @inbounds Float32(vcache[pu_u * Int32(1024) + kvh * Int32(256) + t + Int32(1)]) *
              vscale[pu_u * Int32(4) + kvh + Int32(1)]
    end
    Base.Cartesian.@nexprs 8 u -> (plu_u < np && (acc = _accum6(acc, w, plu_u, v_u)))
    acc
end

# The scores of one chunk: subgroup `sg` takes positions sg, sg + nsg, ..., each
# lane DL dims of all six heads, and every K row it needs is loaded up front.
@inline function _scorechunk!(w, qr, k32, kscale, p0::Int32, np::Int32, kvh::Int32,
                              lane::Int32, sg::Int32, ::Val{SUBGROUP}) where {SUBGROUP}
    nsg = Int32(256 ÷ SUBGROUP)
    d0 = lane * Int32(256 ÷ SUBGROUP)
    nw = Val(256 ÷ SUBGROUP ÷ 4)
    Base.Cartesian.@nexprs 16 i -> if i <= ATTN_CHUNK ÷ (256 ÷ SUBGROUP)
        pl_i = sg + nsg * Int32(i - 1)
        p_i = p0 + min(pl_i, np - Int32(1))
        kw_i = _kwords(k32, (p_i * Int32(1024) + kvh * Int32(256) + d0) >> 2, nw)
        ks_i = @inbounds kscale[p_i * Int32(4) + kvh + Int32(1)] * 0.0625f0
    end
    Base.Cartesian.@nexprs 16 i -> if i <= ATTN_CHUNK ÷ (256 ÷ SUBGROUP)
        sc_i = _scores6(qr, _kbytes(kw_i))
        if lane == Int32(0)
            Base.Cartesian.@nexprs 6 j -> (@inbounds w[Int32((j - 1) * ATTN_CHUNK) + pl_i + Int32(1)] =
                pl_i < np ? sc_i[j] * ks_i : -Inf32)
        end
    end
    return nothing
end

function attention_partial_kernel!(pmax, psum, pacc, q, kcache, vcache,
                                   kscale, vscale, position, nchunks::Int32,
                                   ::Val{SUBGROUP}) where {SUBGROUP}
    # scores, then the softmax weights in place: six heads x ATTN_CHUNK
    w = KI.localmemory(Float32, Val((6 * ATTN_CHUNK,)), Val(1))
    stat = KI.localmemory(Float32, Val((12,)), Val(2))
    t = Int32(KI.get_local_id().x - 1)
    lane = t % Int32(SUBGROUP)
    sg = t ÷ Int32(SUBGROUP)
    wg = Int32(KI.get_group_id().x - 1)
    kvh = wg & Int32(3)
    c = wg >> 2
    pos = Int32(position[1])
    p0 = c * Int32(ATTN_CHUNK)
    # Uniform across the workgroup, so returning before the barriers is safe.
    p0 > pos && return nothing
    np = min(Int32(ATTN_CHUNK), pos - p0 + Int32(1))
    # Scores: one subgroup per position, each lane DL contiguous dims of all six
    # heads' queries, held in registers for the whole chunk.
    dl = Val(256 ÷ SUBGROUP)
    d0 = lane * Int32(256 ÷ SUBGROUP)
    qr = ntuple(j -> _qrow(q, (kvh * Int32(6) + Int32(j - 1)) * Int32(256) + d0, dl), Val(6))
    k32 = reinterpret(UInt32, kcache)
    _scorechunk!(w, qr, k32, kscale, p0, np, kvh, lane, sg, Val(SUBGROUP))
    KI.barrier()
    # Softmax over the chunk: 32 work-items per head, two positions each.
    if t < Int32(6 * 32)
        j = t >> 5
        r = t & Int32(31)
        a = @inbounds w[j * Int32(ATTN_CHUNK) + r + Int32(1)]
        b = @inbounds w[j * Int32(ATTN_CHUNK) + r + Int32(33)]
        m = _segmax32(max(a, b))
        r == Int32(0) && (@inbounds stat[j + Int32(1)] = m)
    end
    KI.barrier()
    if t < Int32(6 * 32)
        j = t >> 5
        r = t & Int32(31)
        m = @inbounds stat[j + Int32(1)]
        ea = exp(@inbounds(w[j * Int32(ATTN_CHUNK) + r + Int32(1)]) - m)
        eb = exp(@inbounds(w[j * Int32(ATTN_CHUNK) + r + Int32(33)]) - m)
        @inbounds w[j * Int32(ATTN_CHUNK) + r + Int32(1)] = ea
        @inbounds w[j * Int32(ATTN_CHUNK) + r + Int32(33)] = eb
        l = _segsum32(ea + eb)
        r == Int32(0) && (@inbounds stat[Int32(6) + j + Int32(1)] = l)
    end
    KI.barrier()
    # Weighted V: one work-item per dimension, six heads at once.
    acc = (0f0, 0f0, 0f0, 0f0, 0f0, 0f0)
    pl = Int32(0)
    while pl < np
        acc = _pv8(acc, w, vcache, vscale, p0, pl, np, kvh, t)
        pl += Int32(8)
    end
    Base.Cartesian.@nexprs 6 j -> begin
        slot = (kvh * Int32(6) + Int32(j - 1)) * nchunks + c
        @inbounds pacc[slot * Int32(256) + t + Int32(1)] = acc[j]
        if t == Int32(0)
            @inbounds pmax[slot + Int32(1)] = stat[j]
            @inbounds psum[slot + Int32(1)] = stat[6 + j]
        end
    end
    return nothing
end

function attention_combine_kernel!(out, pmax, psum, pacc, gate, position,
                                   nchunks::Int32)
    d = Int32(KI.get_local_id().x - 1)
    h = Int32(KI.get_group_id().x - 1)
    used = (Int32(position[1]) + Int32(ATTN_CHUNK)) ÷ Int32(ATTN_CHUNK)
    base = h * nchunks
    m = -Inf32
    for c in Int32(0):(used - Int32(1))
        m = max(m, @inbounds pmax[base + c + Int32(1)])
    end
    l = 0f0
    o = 0f0
    for c in Int32(0):(used - Int32(1))
        f = exp(@inbounds(pmax[base + c + Int32(1)]) - m)
        l = muladd(f, @inbounds(psum[base + c + Int32(1)]), l)
        o = muladd(f, @inbounds(pacc[(base + c) * Int32(256) + d + Int32(1)]), o)
    end
    i = h * Int32(256) + d + Int32(1)
    @inbounds out[i] = eltype(out)((o / l) * Float32(gate[i]))
    return nothing
end


# Prefill attention: one workgroup per ATTN_TOKENS consecutive tokens and one KV
# head, walking the context in 64 positions with an online softmax for the
# 6 x ATTN_TOKENS queries that share it. A prompt has enough tokens to fill the
# device this way, so there is no split over the context.
#
# Each work-item scores one (position, token) pair for all six heads, a whole
# 256-wide dot product each against queries staged in shared memory, so no
# subgroup reduction is needed and each K row is read once for every token in the
# group; each V value is then read once for all 24 queries. One token per
# workgroup, with a six-way subgroup reduction per position, took 416 ms a layer
# for 2048 tokens at positions 6144-8191 on an M5; this takes 252, which made
# attention the second-largest pass of a long prompt instead of a match for the
# GEMM. The kernel before that walked one position at a time with two barriers
# each: 21 s for a 2048-token prompt.
const ATTN_TOKENS = 4

@inline _segmax8(v::Float32) =
    (v = max(v, KI.shfl_down(v, Int32(4))); v = max(v, KI.shfl_down(v, Int32(2))); max(v, KI.shfl_down(v, Int32(1))))
@inline _segsum8(v::Float32) =
    (v += KI.shfl_down(v, Int32(4)); v += KI.shfl_down(v, Int32(2)); v + KI.shfl_down(v, Int32(1)))
@inline _acc24(acc::NTuple{24,Float32}, w, pl::Int32, v::Float32) = ntuple(Val(24)) do qi
    muladd(@inbounds(w[Int32((qi - 1) * 64) + pl + Int32(1)]), v, acc[qi])
end
@inline _resc24(acc::NTuple{24,Float32}, stat) = ntuple(qi -> acc[qi] * @inbounds(stat[72 + qi]), Val(24))
@inline _bytes4(word::UInt32) =
    (Float32(reinterpret(Int32, word << 24) >> 24), Float32(reinterpret(Int32, word << 16) >> 24),
     Float32(reinterpret(Int32, word << 8) >> 24), Float32(reinterpret(Int32, word) >> 24))
@inline _fma6x4(a::NTuple{6,Float32}, q4, i0::Int32, kf::NTuple{4,Float32}) = ntuple(Val(6)) do j
    qv = @inbounds q4[i0 + Int32((j - 1) * 64)]
    muladd(qv[1].value, kf[1], muladd(qv[2].value, kf[2],
        muladd(qv[3].value, kf[3], muladd(qv[4].value, kf[4], a[j]))))
end
# Six heads' dot products of one K row against queries starting at `qb` in `qs`.
@inline function _dots6(qs, qb::Int32, k4, kb4::Int32)
    a = (0f0, 0f0, 0f0, 0f0, 0f0, 0f0)
    q4 = reinterpret(NTuple{4,VecElement{Float32}}, qs)
    for wv in Int32(0):Int32(15)
        kw = @inbounds k4[kb4 + wv + Int32(1)]
        i0 = (qb >> 2) + wv * Int32(4) + Int32(1)
        a = _fma6x4(a, q4, i0, _bytes4(kw[1].value))
        a = _fma6x4(a, q4, i0 + Int32(1), _bytes4(kw[2].value))
        a = _fma6x4(a, q4, i0 + Int32(2), _bytes4(kw[3].value))
        a = _fma6x4(a, q4, i0 + Int32(3), _bytes4(kw[4].value))
    end
    a
end

function attention_prefill_kernel!(out, q, gate, kcache, vcache, kscale, vscale,
                                   position, ntokens::Int32)
    qs = KI.localmemory(Float32, Val((ATTN_TOKENS * 6 * 256,)), Val(1))
    w = KI.localmemory(Float32, Val((ATTN_TOKENS * 6 * 64,)), Val(2))
    # per query: running max | running sum | this chunk's max | rescale factor
    stat = KI.localmemory(Float32, Val((96,)), Val(3))
    t = Int32(KI.get_local_id().x - 1)
    wg = Int32(KI.get_group_id().x - 1)
    kvh = wg & Int32(3)
    t0 = (wg >> 2) * Int32(ATTN_TOKENS)
    nt = min(Int32(ATTN_TOKENS), ntokens - t0)
    base = Int32(position[1]) + t0          # the first token's position
    i = t
    while i < Int32(ATTN_TOKENS * 6 * 256)
        u = i ÷ Int32(1536)
        r = i - u * Int32(1536)
        @inbounds qs[i + Int32(1)] = u < nt ?
            Float32(q[(t0 + u) * Int32(6144) + kvh * Int32(1536) + r + Int32(1)]) : 0f0
        i += Int32(256)
    end
    if t < Int32(6 * ATTN_TOKENS)
        @inbounds stat[t + Int32(1)] = -Inf32
        @inbounds stat[Int32(24) + t + Int32(1)] = 0f0
    end
    KI.barrier()
    k4 = reinterpret(NTuple{4,VecElement{UInt32}}, kcache)
    maxpos = base + nt - Int32(1)
    acc = ntuple(_ -> 0f0, Val(24))
    pl = t & Int32(63)
    u = t >> 6
    for p0 in Int32(0):Int32(64):maxpos
        np = min(Int32(64), maxpos - p0 + Int32(1))
        p = p0 + pl
        # Token u sees positions up to its own: causal within the group too.
        if u < nt && p <= base + u
            sc = _dots6(qs, u * Int32(1536), k4, (p * Int32(1024) + kvh * Int32(256)) >> 4)
            ks = @inbounds kscale[p * Int32(4) + kvh + Int32(1)] * 0.0625f0
            Base.Cartesian.@nexprs 6 j -> (@inbounds w[(u * Int32(6) + Int32(j - 1)) * Int32(64) + pl + Int32(1)] = sc[j] * ks)
        else
            Base.Cartesian.@nexprs 6 j -> (@inbounds w[(u * Int32(6) + Int32(j - 1)) * Int32(64) + pl + Int32(1)] = -Inf32)
        end
        KI.barrier()
        # Online softmax, eight work-items to a query. The chunk's max lands on
        # each segment's first lane, so it goes through shared memory first.
        if t < Int32(8 * 6 * ATTN_TOKENS)
            qi = t >> 3
            wb = qi * Int32(64) + (t & Int32(7)) * Int32(8)
            cm = -Inf32
            for e in Int32(0):Int32(7)
                cm = max(cm, @inbounds w[wb + e + Int32(1)])
            end
            cm = _segmax8(cm)
            (t & Int32(7)) == Int32(0) && (@inbounds stat[Int32(48) + qi + Int32(1)] = max(stat[qi + Int32(1)], cm))
        end
        KI.barrier()
        if t < Int32(8 * 6 * ATTN_TOKENS)
            qi = t >> 3
            wb = qi * Int32(64) + (t & Int32(7)) * Int32(8)
            m = @inbounds stat[Int32(48) + qi + Int32(1)]
            l = 0f0
            for e in Int32(0):Int32(7)
                # A query past the prompt's end has seen nothing, and stays at -Inf.
                ex = m == -Inf32 ? 0f0 : exp(@inbounds(w[wb + e + Int32(1)]) - m)
                @inbounds w[wb + e + Int32(1)] = ex
                l += ex
            end
            l = _segsum8(l)
            if (t & Int32(7)) == Int32(0)
                f = m == -Inf32 ? 0f0 : exp(@inbounds(stat[qi + Int32(1)]) - m)
                @inbounds stat[Int32(72) + qi + Int32(1)] = f
                @inbounds stat[Int32(24) + qi + Int32(1)] = muladd(@inbounds(stat[Int32(24) + qi + Int32(1)]), f, l)
                @inbounds stat[qi + Int32(1)] = m
            end
        end
        KI.barrier()
        acc = _resc24(acc, stat)
        for e in Int32(0):(np - Int32(1))
            pp = p0 + e
            v = @inbounds Float32(vcache[pp * Int32(1024) + kvh * Int32(256) + t + Int32(1)]) *
                vscale[pp * Int32(4) + kvh + Int32(1)]
            acc = _acc24(acc, w, e, v)
        end
        KI.barrier()
    end
    Base.Cartesian.@nexprs 24 qi -> begin
        uq = Int32((qi - 1) ÷ 6)
        if uq < nt
            o = (t0 + uq) * Int32(6144) + (kvh * Int32(6) + Int32((qi - 1) % 6)) * Int32(256) + t + Int32(1)
            @inbounds out[o] = eltype(out)((acc[qi] / stat[Int32(24 + qi)]) * Float32(gate[o]))
        end
    end
    return nothing
end

# Batched attention keeps the 256 value dimensions mapped one-per-thread, but
# reduces each QK dot with subgroup instructions.  The original batch kernel
# used eight full-workgroup barriers for every cached token; this needs two.
function select_last_kernel!(out, x, width::Int32, ntokens::Int32)
    i = Int32(KI.get_global_id().x - 1)
    i < width && (@inbounds out[i + Int32(1)] =
        x[(ntokens - Int32(1)) * width + i + Int32(1)])
    return nothing
end


# The greedy pick on the device, written beside the logits it reads. Taking it on
# the host meant downloading all 248320 logits and scanning them between every
# two steps: 2.4 ms of a 64 ms token on an M5, with the device idle throughout.
# Julia's `argmax` order: the first maximum under `isless`, so NaN outranks
# every number, as it does on the host.
@inline _better(va, ia, vb, ib) = isless(va, vb) || (isequal(va, vb) && ib < ia)

function argmax_kernel!(best, x, n::Int32)
    vals = KI.localmemory(Float32, Val((256,)), Val(1))
    idxs = KI.localmemory(Int32, Val((256,)), Val(2))
    t = Int32(KI.get_local_id().x - 1)
    bv = -Inf32
    bi = typemax(Int32)
    i = t
    while i < n
        v = @inbounds Float32(x[i + Int32(1)])
        if _better(bv, bi, v, i)
            bv = v
            bi = i
        end
        i += Int32(256)
    end
    @inbounds vals[t + Int32(1)] = bv
    @inbounds idxs[t + Int32(1)] = bi
    KI.barrier()
    step = Int32(128)
    while step > Int32(0)
        if t < step
            vb = @inbounds vals[t + step + Int32(1)]
            ib = @inbounds idxs[t + step + Int32(1)]
            if _better(@inbounds(vals[t + Int32(1)]), @inbounds(idxs[t + Int32(1)]), vb, ib)
                @inbounds vals[t + Int32(1)] = vb
                @inbounds idxs[t + Int32(1)] = ib
            end
        end
        KI.barrier()
        step >>= 1
    end
    t == Int32(0) && (@inbounds best[1] = idxs[1])
    return nothing
end
