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
        KI.barrier()
        stride <<= 1
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
        KI.barrier()
        stride <<= 1
    end
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
    return nothing
end

function add_kernel!(out, a, b, n::Int32)
    i = Int32(KI.get_global_id().x)
    i <= n && (@inbounds out[i] = eltype(out)(Float32(a[i]) + Float32(b[i])))
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

function rmsnorm_kernel!(out, x, weight, n::Int32, eps::Float32)
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    sum = 0f0
    i = t
    while i < n
        @inbounds v = Float32(x[i + Int32(1)])
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
        @inbounds out[i + Int32(1)] = eltype(out)(Float32(x[i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
        i += Int32(256)
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
            Float32(x[base + i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
        i += Int32(256)
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
            Float32(x[base + i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
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
function dense_gemv_kernel!(out, weight, x,
                                              K::Int32, M::Int32,
                                              ::Val{SUBGROUP}) where {SUBGROUP}
    partial = KI.localmemory(Float32, Val((DENSE_ROWS_PER_WG * (64 ÷ SUBGROUP),)), Val(1))
    t = Int32(KI.get_local_id().x - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    row = Int32(KI.get_group_id().x - 1) * Int32(DENSE_ROWS_PER_WG) + rowin
    acc = 0f0
    if row < M
        base = row * K
        k = lane
        @inbounds while k < K
            acc = muladd(Float32(weight[base + k + Int32(1)]),
                         Float32(x[k + Int32(1)]), acc)
            k += Int32(64)
        end
    end
    reduced = DNNKernels.KI.sub_group_reduce_add(acc)
    if SUBGROUP == 64
        if lane == Int32(0) && row < M
            @inbounds out[row + Int32(1)] = eltype(out)(reduced)
        end
    else
        nsub = Int32(64 ÷ SUBGROUP)
        subinrow = lane ÷ Int32(SUBGROUP)
        if sublane == Int32(0)
            partial[rowin * nsub + subinrow + Int32(1)] = reduced
        end
        KI.barrier()
        if lane == Int32(0) && row < M
            @inbounds out[row + Int32(1)] = eltype(out)(
                partial[rowin * nsub + Int32(1)] +
                partial[rowin * nsub + Int32(2)])
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
            acc = muladd(Float32(weight[wbase + k + Int32(1)]),
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
                Float32(weight[row * K + k0 + lk + Int32(1)]) : 0f0
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

function swiglu_kernel!(out, gate, up, n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n
        @inbounds begin
            g = Float32(gate[i])
            out[i] = eltype(out)((g / (1f0 + exp(-g))) * Float32(up[i]))
        end
    end
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
        KI.barrier()
        stride <<= 1
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

function recurrent_split_kernel!(q, k, v, qkv)
    i = Int32(KI.get_global_id().x - 1)
    if i < Int32(6144)
        if i < Int32(2048)
            @inbounds begin
                q[i + Int32(1)] = qkv[i + Int32(1)]
                k[i + Int32(1)] = qkv[Int32(2048) + i + Int32(1)]
            end
        end
        @inbounds v[i + Int32(1)] = qkv[Int32(4096) + i + Int32(1)]
    end
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
                y = muladd(Float32(weight[wi]), x0,
                    muladd(Float32(weight[wi + Int32(1)]), x1,
                    muladd(Float32(weight[wi + Int32(2)]), x2,
                           Float32(weight[wi + Int32(3)]) * x3)))
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
function gated_delta_state_batch_kernel!(out, state,
        q, k, v, alpha, beta,
        dt_bias, a, ntokens::Int32,
        groupsperhead::Int32, ::Val{SUBGROUP}) where {SUBGROUP}
    t = Int32(KI.get_local_id().x - 1)
    lane = t % Int32(SUBGROUP)
    sub = t ÷ Int32(SUBGROUP)
    wg = Int32(KI.get_group_id().x - 1)
    head = wg ÷ groupsperhead
    colgroup = wg % groupsperhead
    nsubgroups = Int32(256 ÷ SUBGROUP)
    col = colgroup * nsubgroups + sub
    keyhead = head % Int32(16)
    sbase = head * Int32(16384) + col * Int32(128)

    s0 = @inbounds Float32(state[sbase + lane + Int32(1)])
    s1 = @inbounds Float32(state[sbase + lane + Int32(SUBGROUP) + Int32(1)])
    s2 = 0f0; s3 = 0f0
    if SUBGROUP == 32
        s2 = @inbounds Float32(state[sbase + lane + Int32(65)])
        s3 = @inbounds Float32(state[sbase + lane + Int32(97)])
    end

    for token in Int32(0):(ntokens - Int32(1))
        qoff = token * Int32(2048) + keyhead * Int32(128)
        voff = token * Int32(6144) + head * Int32(128)
        gatei = token * Int32(48) + head + Int32(1)
        raw_alpha = Float32(alpha[gatei]) + Float32(dt_bias[head + Int32(1)])
        decay = exp((max(raw_alpha, 0f0) + log1p(exp(-abs(raw_alpha)))) *
                    Float32(a[head + Int32(1)]))
        b = 1f0 / (1f0 + exp(-Float32(beta[gatei])))
        q0 = @inbounds Float32(q[qoff + lane + Int32(1)])
        q1 = @inbounds Float32(q[qoff + lane + Int32(SUBGROUP) + Int32(1)])
        k0 = @inbounds Float32(k[qoff + lane + Int32(1)])
        k1 = @inbounds Float32(k[qoff + lane + Int32(SUBGROUP) + Int32(1)])
        partial = decay * (s0 * k0 + s1 * k1)
        q2 = 0f0; q3 = 0f0; k2 = 0f0; k3 = 0f0
        if SUBGROUP == 32
            q2 = @inbounds Float32(q[qoff + lane + Int32(65)])
            q3 = @inbounds Float32(q[qoff + lane + Int32(97)])
            k2 = @inbounds Float32(k[qoff + lane + Int32(65)])
            k3 = @inbounds Float32(k[qoff + lane + Int32(97)])
            partial += decay * (s2 * k2 + s3 * k3)
        end
        sk = DNNKernels.KI.sub_group_reduce_add(partial)
        delta = (@inbounds Float32(v[voff + col + Int32(1)]) - sk) * b
        s0 = muladd(decay, s0, k0 * delta)
        s1 = muladd(decay, s1, k1 * delta)
        yp = s0 * q0 + s1 * q1
        if SUBGROUP == 32
            s2 = muladd(decay, s2, k2 * delta)
            s3 = muladd(decay, s3, k3 * delta)
            yp += s2 * q2 + s3 * q3
        end
        y = DNNKernels.KI.sub_group_reduce_add(yp)
        lane == Int32(0) &&
            (@inbounds out[voff + col + Int32(1)] = eltype(out)(y * 0.08838834764831845f0))
    end

    @inbounds begin
        state[sbase + lane + Int32(1)] = eltype(state)(s0)
        state[sbase + lane + Int32(SUBGROUP) + Int32(1)] = eltype(state)(s1)
        if SUBGROUP == 32
            state[sbase + lane + Int32(65)] = eltype(state)(s2)
            state[sbase + lane + Int32(97)] = eltype(state)(s3)
        end
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
function gdn_permute_kernel!(out, x)
    i = Int32(KI.get_global_id().x - 1)
    if i < Int32(6144)
        hd = i % Int32(128)
        rest = i ÷ Int32(128)
        rep = rest % Int32(3)
        nk = rest ÷ Int32(3)
        src = hd + nk * Int32(128) + rep * Int32(2048)
        @inbounds out[i + Int32(1)] = x[src + Int32(1)]
    end
    return nothing
end

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
        kcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, kval / ks), -127, 127))
        vcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, vval / vs), -127, 127))
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
        kcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, kval / ks), -127, 127))
        vcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, vval / vs), -127, 127))
    end
    return nothing
end

# Decode attention uses online softmax, so it needs no score buffer even for an
# 8K context. One workgroup owns a query head and each lane owns one value
# dimension. Scores are reduced over the 256 lanes, then all lanes update their
# own weighted-value accumulator.
function decode_attention_kernel!(out, q, gate,
                                                    kcache, vcache,
                                                    kscale, vscale,
                                                    position)
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    h = Int32(KI.get_group_id().x - 1)
    pos = Int32(position[1])
    kvh = h ÷ Int32(6)
    qv = Float32(q[h * Int32(256) + d + Int32(1)])
    high = -floatmax(Float32)
    denom = 0f0
    acc = 0f0
    for p in Int32(0):pos
        cachei = p * Int32(1024) + kvh * Int32(256) + d + Int32(1)
        scalei = p * Int32(4) + kvh + Int32(1)
        @inbounds sh[d + Int32(1)] = qv * Float32(kcache[cachei]) * kscale[scalei]
        KI.barrier()
        step = Int32(128)
        while step > Int32(0)
            d < step && (sh[d + Int32(1)] += sh[d + step + Int32(1)])
            KI.barrier()
            step >>= 1
        end
        score = sh[Int32(1)] * 0.0625f0
        newhigh = max(high, score)
        oldscale = exp(high - newhigh)
        newscale = exp(score - newhigh)
        denom = denom * oldscale + newscale
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei]) * vscale[scalei]
        high = newhigh
        KI.barrier()
    end
    o = h * Int32(256) + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
    return nothing
end

function decode_attention_batch_kernel!(out, q, gate,
        kcache, vcache, kscale, vscale,
        position)
    sh = KI.localmemory(Float32, Val((256,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    group = Int32(KI.get_group_id().x - 1)
    token = group ÷ Int32(24)
    h = group % Int32(24)
    pos = Int32(position[1]) + token
    kvh = h ÷ Int32(6)
    querybase = token * Int32(6144) + h * Int32(256)
    qv = Float32(q[querybase + d + Int32(1)])
    high = -floatmax(Float32)
    denom = 0f0
    acc = 0f0
    for p in Int32(0):pos
        cachei = p * Int32(1024) + kvh * Int32(256) + d + Int32(1)
        scalei = p * Int32(4) + kvh + Int32(1)
        @inbounds sh[d + Int32(1)] = qv * Float32(kcache[cachei]) * kscale[scalei]
        KI.barrier()
        step = Int32(128)
        while step > Int32(0)
            d < step && (sh[d + Int32(1)] += sh[d + step + Int32(1)])
            KI.barrier()
            step >>= 1
        end
        score = sh[Int32(1)] * 0.0625f0
        newhigh = max(high, score)
        oldscale = exp(high - newhigh)
        newscale = exp(score - newhigh)
        denom = denom * oldscale + newscale
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei]) * vscale[scalei]
        high = newhigh
        KI.barrier()
    end
    o = querybase + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
    return nothing
end

# Batched attention keeps the 256 value dimensions mapped one-per-thread, but
# reduces each QK dot with subgroup instructions.  The original batch kernel
# used eight full-workgroup barriers for every cached token; this needs two.
function decode_attention_batch_fast_kernel!(out, q,
        gate, kcache, vcache, kscale,
        vscale, position, ::Val{SUBGROUP}) where {SUBGROUP}
    sh = KI.localmemory(Float32, Val((9,)), Val(1))
    d = Int32(KI.get_local_id().x - 1)
    lane = d % Int32(SUBGROUP)
    sub = d ÷ Int32(SUBGROUP)
    nsub = Int32(256 ÷ SUBGROUP)
    group = Int32(KI.get_group_id().x - 1)
    token = group ÷ Int32(24)
    h = group % Int32(24)
    pos = Int32(position[1]) + token
    kvh = h ÷ Int32(6)
    querybase = token * Int32(6144) + h * Int32(256)
    qv = @inbounds Float32(q[querybase + d + Int32(1)])
    high = -floatmax(Float32)
    denom = 0f0
    acc = 0f0
    for p in Int32(0):pos
        cachei = p * Int32(1024) + kvh * Int32(256) + d + Int32(1)
        scalei = p * Int32(4) + kvh + Int32(1)
        partial = @inbounds qv * Float32(kcache[cachei]) * kscale[scalei]
        reduced = DNNKernels.KI.sub_group_reduce_add(partial)
        lane == Int32(0) && (sh[sub + Int32(1)] = reduced)
        KI.barrier()
        if d == Int32(0)
            score = 0f0
            for i in Int32(1):nsub
                score += sh[i]
            end
            sh[Int32(9)] = score * 0.0625f0
        end
        KI.barrier()
        score = sh[Int32(9)]
        newhigh = max(high, score)
        oldscale = exp(high - newhigh)
        newscale = exp(score - newhigh)
        denom = denom * oldscale + newscale
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei]) * vscale[scalei]
        high = newhigh
    end
    o = querybase + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
    return nothing
end

function select_last_kernel!(out, x, width::Int32, ntokens::Int32)
    i = Int32(KI.get_global_id().x - 1)
    i < width && (@inbounds out[i + Int32(1)] =
        x[(ntokens - Int32(1)) * width + i + Int32(1)])
    return nothing
end

