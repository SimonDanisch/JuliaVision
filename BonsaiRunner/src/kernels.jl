const KA = KernelAbstractions

@kernel cpu=false function copy_kernel!(out, @Const(x), n::Int32)
    i = Int32(@index(Global, Linear))
    i <= n && (@inbounds out[i] = eltype(out)(x[i]))
end

# Out-of-place signed FWHT for batched projections.  The old path copied the
# complete activation matrix and then transformed it in place.  Loading the
# source directly into shared memory removes that full-device copy.  Recurrent
# layers may simultaneously retain the untransformed fp16 input for their two
# small cooperative projections.
@kernel cpu=false function hadamard1024_out_batch_kernel!(
        out, halfcopy, @Const(x), @Const(signs), width::Int32, blocks::Int32,
        ::Val{COPYHALF}) where {COPYHALF}
    sh = @localmem Float32 (1024,)
    t = Int32(@index(Local, Linear) - 1)
    wg = Int32(@index(Group, Linear) - 1)
    block = wg % blocks
    col = wg ÷ blocks
    base = col * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        v = Float32(x[base + i + Int32(1)])
        COPYHALF && (halfcopy[base + i + Int32(1)] = Float16(v))
        sh[i + Int32(1)] = v * Float32(signs[block * Int32(1024) + i + Int32(1)])
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
end

@kernel cpu=false function gdn_permute_hadamard1024_batch_kernel!(
        out, @Const(x), @Const(signs), width::Int32, blocks::Int32)
    sh = @localmem Float32 (1024,)
    t = Int32(@index(Local, Linear) - 1)
    wg = Int32(@index(Group, Linear) - 1)
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
end

@kernel cpu=false function add_kernel!(out, @Const(a), @Const(b), n::Int32)
    i = Int32(@index(Global, Linear))
    i <= n && (@inbounds out[i] = eltype(out)(Float32(a[i]) + Float32(b[i])))
end

@kernel cpu=false function dense_ab_split_kernel!(alpha, beta, @Const(ab),
                                                  ntokens::Int32)
    i = Int32(@index(Global, Linear) - 1)
    if i < Int32(48) * ntokens
        token = i ÷ Int32(48)
        head = i % Int32(48)
        @inbounds begin
            alpha[i + Int32(1)] = ab[token * Int32(96) + head + Int32(1)]
            beta[i + Int32(1)] = ab[token * Int32(96) + Int32(48) + head + Int32(1)]
        end
    end
end

@kernel cpu=false function rmsnorm_kernel!(out, @Const(x), @Const(weight), n::Int32, eps::Float32)
    sh = @localmem Float32 (256,)
    t = Int32(@index(Local, Linear) - 1)
    sum = 0f0
    i = t
    while i < n
        @inbounds v = Float32(x[i + Int32(1)])
        sum = muladd(v, v, sum)
        i += Int32(256)
    end
    sh[t + Int32(1)] = sum
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        t < step && (sh[t + Int32(1)] += sh[t + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(1)] / Float32(n) + eps))
    i = t
    while i < n
        @inbounds out[i + Int32(1)] = eltype(out)(Float32(x[i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
        i += Int32(256)
    end
end

# One workgroup normalizes one activation column. The weight is shared by all
# columns, while input and output remain column-major just like the PTQ kernels.
@kernel cpu=false function rmsnorm_batch_kernel!(out, @Const(x), @Const(weight),
                                                 n::Int32, eps::Float32)
    sh = @localmem Float32 (256,)
    t = Int32(@index(Local, Linear) - 1)
    col = Int32(@index(Group, Linear) - 1)
    base = col * n
    sum = 0f0
    i = t
    while i < n
        @inbounds v = Float32(x[base + i + Int32(1)])
        sum = muladd(v, v, sum)
        i += Int32(256)
    end
    sh[t + Int32(1)] = sum
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        t < step && (sh[t + Int32(1)] += sh[t + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(1)] / Float32(n) + eps))
    i = t
    while i < n
        @inbounds out[base + i + Int32(1)] = eltype(out)(
            Float32(x[base + i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
        i += Int32(256)
    end
end

# Residual addition and the normalization which immediately consumes it share
# one read of x.  A workgroup owns one token, keeps the sum in shared memory,
# updates the residual stream, and writes the next normalized input.
@kernel cpu=false function add_rmsnorm_batch_kernel!(out, x, @Const(branch),
                                                     @Const(weight), n::Int32,
                                                     eps::Float32,
                                                     ::Val{SUBGROUP}) where {SUBGROUP}
    sh = @localmem Float32 (256,)
    t = Int32(@index(Local, Linear) - 1)
    col = Int32(@index(Group, Linear) - 1)
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
    @synchronize
    nsubgroups = Int32(256 ÷ SUBGROUP)
    if subgroup == Int32(0)
        v = lane < nsubgroups ? sh[lane + Int32(1)] : 0f0
        total = DNNKernels.KI.sub_group_reduce_add(v)
        lane == Int32(0) && (sh[Int32(1)] = total)
    end
    @synchronize
    scale = inv(sqrt(sh[Int32(1)] / Float32(n) + eps))
    i = t
    while i < n
        @inbounds out[base + i + Int32(1)] = eltype(out)(
            Float32(x[base + i + Int32(1)]) * scale * Float32(weight[i + Int32(1)]))
        i += Int32(256)
    end
end

const DENSE_ROWS_PER_WG = 4

# Four output rows per 256-thread workgroup, one 64-lane reduction per row.
# The previous kernel assigned one thread to a whole row: the recurrent 5120x48
# projections therefore launched only 48 threads, each with a 5120-FMA serial
# dependency chain and each lane reading a different, distant row. Here lanes
# walk adjacent K values and the hardware reduction closes the dot product.
@kernel cpu=false function dense_gemv_kernel!(out, @Const(weight), @Const(x),
                                              K::Int32, M::Int32,
                                              ::Val{SUBGROUP}) where {SUBGROUP}
    partial = @localmem Float32 (DENSE_ROWS_PER_WG * (64 ÷ SUBGROUP),)
    t = Int32(@index(Local, Linear) - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    row = Int32(@index(Group, Linear) - 1) * Int32(DENSE_ROWS_PER_WG) + rowin
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
        @synchronize
        if lane == Int32(0) && row < M
            @inbounds out[row + Int32(1)] = eltype(out)(
                partial[rowin * nsub + Int32(1)] +
                partial[rowin * nsub + Int32(2)])
        end
    end
end

@kernel cpu=false function dense_gemv_batch_kernel!(out, @Const(weight), @Const(x),
                                                    K::Int32, M::Int32,
                                                    rowgroups::Int32,
                                                    ::Val{SUBGROUP}) where {SUBGROUP}
    partial = @localmem Float32 (DENSE_ROWS_PER_WG * (64 ÷ SUBGROUP),)
    t = Int32(@index(Local, Linear) - 1)
    lane = t & Int32(63)
    sublane = t % Int32(SUBGROUP)
    rowin = t >> 6
    wg = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        lane == Int32(0) && row < M &&
            (@inbounds out[col * M + row + Int32(1)] = eltype(out)(
                partial[rowin * nsub + Int32(1)] + partial[rowin * nsub + Int32(2)]))
    end
end

const DENSE_MM_BM = 32
const DENSE_MM_BN = 32
const DENSE_MM_BK = 32

@kernel cpu=false function dense_mul_mm_kernel!(out, @Const(weight), @Const(x),
                                                K::Int32, M::Int32, N::Int32,
                                                mtiles::Int32)
    atile = @localmem Float32 (DENSE_MM_BM * DENSE_MM_BK,)
    btile = @localmem Float32 (DENSE_MM_BN * DENSE_MM_BK,)
    t = Int32(@index(Local, Linear) - 1)
    wg = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        for lk in Int32(0):Int32(DENSE_MM_BK - 1)
            av0 = atile[(lr0 + Int32(0)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            av1 = atile[(lr0 + Int32(1)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            bv0 = btile[(lc0 + Int32(0)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            bv1 = btile[(lc0 + Int32(1)) * Int32(DENSE_MM_BK) + lk + Int32(1)]
            a00 = muladd(av0, bv0, a00); a01 = muladd(av0, bv1, a01)
            a10 = muladd(av1, bv0, a10); a11 = muladd(av1, bv1, a11)
        end
        @synchronize
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
end

@kernel cpu=false function swiglu_kernel!(out, @Const(gate), @Const(up), n::Int32)
    i = Int32(@index(Global, Linear))
    if i <= n
        @inbounds begin
            g = Float32(gate[i])
            out[i] = eltype(out)((g / (1f0 + exp(-g))) * Float32(up[i]))
        end
    end
end

# SwiGLU's only consumer is the signed Hadamard transform before ffn_down.
# Produce that transformed fp16 operand directly and avoid materialising and
# copying a 17408xN Float32 intermediate.
@kernel cpu=false function swiglu_hadamard1024_batch_kernel!(
        out, @Const(gate), @Const(up), @Const(signs), width::Int32,
        blocks::Int32)
    sh = @localmem Float32 (1024,)
    t = Int32(@index(Local, Linear) - 1)
    wg = Int32(@index(Group, Linear) - 1)
    block = wg % blocks
    col = wg ÷ blocks
    base = col * width + block * Int32(1024)
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        g = Float32(gate[base + i + Int32(1)])
        v = (g / (1f0 + exp(-g))) * Float32(up[base + i + Int32(1)])
        sh[i + Int32(1)] = v * Float32(signs[block * Int32(1024) + i + Int32(1)])
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
    scale = inv(sqrt(Float32(1024)))
    @inbounds for j in Int32(0):Int32(3)
        i = t + j * Int32(256)
        out[base + i + Int32(1)] = eltype(out)(sh[i + Int32(1)] * scale)
    end
end

@kernel cpu=false function recurrent_split_batch_kernel!(q, k, v, @Const(qkv))
    i = Int32(@index(Global, Linear) - 1)
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
end

@kernel cpu=false function recurrent_split_kernel!(q, k, v, @Const(qkv))
    i = Int32(@index(Global, Linear) - 1)
    if i < Int32(6144)
        if i < Int32(2048)
            @inbounds begin
                q[i + Int32(1)] = qkv[i + Int32(1)]
                k[i + Int32(1)] = qkv[Int32(2048) + i + Int32(1)]
            end
        end
        @inbounds v[i + Int32(1)] = qkv[Int32(4096) + i + Int32(1)]
    end
end

@kernel cpu=false function gdn_permute_batch_kernel!(out, @Const(x))
    i = Int32(@index(Global, Linear) - 1)
    token = i ÷ Int32(6144)
    inner = i % Int32(6144)
    hd = inner % Int32(128)
    rest = inner ÷ Int32(128)
    rep = rest % Int32(3)
    nk = rest ÷ Int32(3)
    src = token * Int32(6144) + hd + nk * Int32(128) + rep * Int32(2048)
    @inbounds out[i + Int32(1)] = x[src + Int32(1)]
end

@kernel cpu=false function depthwise_conv4_batch_kernel!(out, state, @Const(x),
                                                         @Const(weight), channels::Int32,
                                                         ntokens::Int32)
    c = Int32(@index(Global, Linear) - 1)
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
end

@kernel cpu=false function gated_delta_net_batch_kernel!(out, state,
        @Const(q), @Const(k), @Const(v), @Const(alpha), @Const(beta),
        @Const(dt_bias), @Const(a), @Const(z), @Const(norm_weight),
        eps::Float32, ntokens::Int32, nheads::Int32, nkeyheads::Int32)
    sh = @localmem Float32 (512,)
    col = Int32(@index(Local, Linear) - 1)
    head = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        step = Int32(64)
        while step > Int32(0)
            if col < step
                @inbounds begin
                    sh[col + Int32(1)] += sh[col + step + Int32(1)]
                    sh[Int32(128) + col + Int32(1)] +=
                        sh[Int32(128) + col + step + Int32(1)]
                end
            end
            @synchronize
            step >>= 1
        end
        qscale = inv(sqrt(sh[Int32(1)] + eps))
        kscale = inv(sqrt(sh[Int32(129)] + eps))
        @inbounds begin
            sh[col + Int32(1)] = Float32(q[qoff + col + Int32(1)]) * qscale
            sh[Int32(128) + col + Int32(1)] = Float32(k[qoff + col + Int32(1)]) * kscale
        end
        @synchronize

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
        @synchronize
        step = Int32(64)
        while step > Int32(0)
            col < step && (@inbounds sh[Int32(384) + col + Int32(1)] +=
                sh[Int32(384) + col + step + Int32(1)])
            @synchronize
            step >>= 1
        end
        @inbounds begin
            invrms = inv(sqrt(sh[Int32(385)] / 128f0 + eps))
            gated = sh[Int32(256) + col + Int32(1)] * invrms *
                    Float32(norm_weight[col + Int32(1)])
            zv = Float32(z[voff + col + Int32(1)])
            out[voff + col + Int32(1)] = eltype(out)(gated * zv / (1f0 + exp(-zv)))
        end
        @synchronize
    end
end

@kernel cpu=false function l2normalize_qk_batch_kernel!(q, k, eps::Float32)
    sh = @localmem Float32 (256,)
    d = Int32(@index(Local, Linear) - 1)
    group = Int32(@index(Group, Linear) - 1)
    base = group * Int32(128)
    @inbounds begin
        qv = Float32(q[base + d + Int32(1)])
        kv = Float32(k[base + d + Int32(1)])
        sh[d + Int32(1)] = qv * qv
        sh[Int32(128) + d + Int32(1)] = kv * kv
    end
    @synchronize
    step = Int32(64)
    while step > Int32(0)
        if d < step
            @inbounds begin
                sh[d + Int32(1)] += sh[d + step + Int32(1)]
                sh[Int32(128) + d + Int32(1)] +=
                    sh[Int32(128) + d + step + Int32(1)]
            end
        end
        @synchronize
        step >>= 1
    end
    qs = inv(sqrt(sh[Int32(1)] + eps))
    ks = inv(sqrt(sh[Int32(129)] + eps))
    @inbounds begin
        q[base + d + Int32(1)] = eltype(q)(Float32(q[base + d + Int32(1)]) * qs)
        k[base + d + Int32(1)] = eltype(k)(Float32(k[base + d + Int32(1)]) * ks)
    end
end

# One subgroup owns one state column.  Each lane keeps its 2 or 4 rows in
# registers while the complete prompt chunk advances, so the 128x128 recurrent
# state is read and written once per chunk instead of once per token.
@kernel cpu=false function gated_delta_state_batch_kernel!(out, state,
        @Const(q), @Const(k), @Const(v), @Const(alpha), @Const(beta),
        @Const(dt_bias), @Const(a), ntokens::Int32,
        groupsperhead::Int32, ::Val{SUBGROUP}) where {SUBGROUP}
    t = Int32(@index(Local, Linear) - 1)
    lane = t % Int32(SUBGROUP)
    sub = t ÷ Int32(SUBGROUP)
    wg = Int32(@index(Group, Linear) - 1)
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
end

@kernel cpu=false function gdn_norm_gate_batch_kernel!(out, @Const(z),
                                                        @Const(norm_weight),
                                                        eps::Float32)
    sh = @localmem Float32 (128,)
    d = Int32(@index(Local, Linear) - 1)
    group = Int32(@index(Group, Linear) - 1)
    head = group % Int32(48)
    token = group ÷ Int32(48)
    off = token * Int32(6144) + head * Int32(128) + d + Int32(1)
    y = @inbounds Float32(out[off])
    sh[d + Int32(1)] = y * y
    @synchronize
    step = Int32(64)
    while step > Int32(0)
        d < step && (@inbounds sh[d + Int32(1)] += sh[d + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    invrms = inv(sqrt(sh[Int32(1)] / 128f0 + eps))
    zv = @inbounds Float32(z[off])
    gated = y * invrms * Float32(norm_weight[d + Int32(1)])
    @inbounds out[off] = eltype(out)(gated * zv / (1f0 + exp(-zv)))
end

# GDN emits value heads in tiled `[hd,nk,rep]` order. Prism folded ssm_out
# against `[hd,rep,nk]`, so this permutation is part of that projection.
@kernel cpu=false function gdn_permute_kernel!(out, @Const(x))
    i = Int32(@index(Global, Linear) - 1)
    if i < Int32(6144)
        hd = i % Int32(128)
        rest = i ÷ Int32(128)
        rep = rest % Int32(3)
        nk = rest ÷ Int32(3)
        src = hd + nk * Int32(128) + rep * Int32(2048)
        @inbounds out[i + Int32(1)] = x[src + Int32(1)]
    end
end

# Q projection is `[q(256), gate(256)]` per head, rather than two contiguous
# 6144-vectors. Normalize q per head, rotate its first 64 dimensions, and
# extract the sigmoid gate in one pass.
@kernel cpu=false function prepare_q_kernel!(q, gate, @Const(qfull), @Const(norm),
                                            @Const(position), theta_base::Float32, eps::Float32)
    sh = @localmem Float32 (512,)
    d = Int32(@index(Local, Linear) - 1)
    h = Int32(@index(Group, Linear) - 1)
    pos = Int32(position[1])
    base = h * Int32(512)
    @inbounds begin
        x = Float32(qfull[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] += sh[Int32(256) + d + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    @synchronize
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    @synchronize
    o = h * Int32(256) + d + Int32(1)
    @inbounds begin
        q[o] = eltype(q)(sh[d + Int32(1)])
        g = Float32(qfull[base + Int32(256) + d + Int32(1)])
        gate[o] = eltype(gate)(1f0 / (1f0 + exp(-g)))
    end
end

@kernel cpu=false function prepare_q_batch_kernel!(q, gate, @Const(qfull), @Const(norm),
                                                  @Const(position), theta_base::Float32,
                                                  eps::Float32)
    sh = @localmem Float32 (512,)
    d = Int32(@index(Local, Linear) - 1)
    group = Int32(@index(Group, Linear) - 1)
    token = group ÷ Int32(24)
    h = group % Int32(24)
    pos = Int32(position[1]) + token
    base = token * Int32(12288) + h * Int32(512)
    @inbounds begin
        x = Float32(qfull[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] +=
            sh[Int32(256) + d + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    @synchronize
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    @synchronize
    o = token * Int32(6144) + h * Int32(256) + d + Int32(1)
    @inbounds begin
        q[o] = eltype(q)(sh[d + Int32(1)])
        g = Float32(qfull[base + Int32(256) + d + Int32(1)])
        gate[o] = eltype(gate)(1f0 / (1f0 + exp(-g)))
    end
end

@kernel cpu=false function prepare_kv_kernel!(kcache, vcache, kscale, vscale,
                                             @Const(kproj), @Const(vproj),
                                             @Const(norm), @Const(position),
                                             theta_base::Float32, eps::Float32)
    sh = @localmem Float32 (512,)
    d = Int32(@index(Local, Linear) - 1)
    h = Int32(@index(Group, Linear) - 1)
    pos = Int32(position[1])
    base = h * Int32(256)
    @inbounds begin
        x = Float32(kproj[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] += sh[Int32(256) + d + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    @synchronize
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    @synchronize
    cachebase = pos * Int32(1024) + base
    kval = sh[d + Int32(1)]
    vval = Float32(vproj[base + d + Int32(1)])
    sh[Int32(256) + d + Int32(1)] = abs(kval)
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] =
            max(sh[Int32(256) + d + Int32(1)],
                sh[Int32(256) + d + step + Int32(1)]))
        @synchronize
        step >>= 1
    end
    ks = max(sh[Int32(257)] / 127f0, floatmin(Float32))
    if d == Int32(0)
        @inbounds kscale[pos * Int32(4) + h + Int32(1)] = ks
    end
    sh[d + Int32(1)] = abs(vval)
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[d + Int32(1)] =
            max(sh[d + Int32(1)], sh[d + step + Int32(1)]))
        @synchronize
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
end

@kernel cpu=false function prepare_kv_batch_kernel!(kcache, vcache, kscale, vscale,
        @Const(kproj), @Const(vproj), @Const(norm), @Const(position),
        theta_base::Float32, eps::Float32)
    sh = @localmem Float32 (512,)
    d = Int32(@index(Local, Linear) - 1)
    group = Int32(@index(Group, Linear) - 1)
    token = group ÷ Int32(4)
    h = group % Int32(4)
    pos = Int32(position[1]) + token
    base = token * Int32(1024) + h * Int32(256)
    @inbounds begin
        x = Float32(kproj[base + d + Int32(1)])
        sh[d + Int32(1)] = x
        sh[Int32(256) + d + Int32(1)] = x*x
    end
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] +=
            sh[Int32(256) + d + step + Int32(1)])
        @synchronize
        step >>= 1
    end
    scale = inv(sqrt(sh[Int32(257)] / 256f0 + eps))
    @inbounds sh[d + Int32(1)] *= scale * Float32(norm[d + Int32(1)])
    @synchronize
    if d < Int32(32)
        angle = Float32(pos) * exp(-log(theta_base) * Float32(2d) / 64f0)
        c, s = cos(angle), sin(angle)
        a = sh[d + Int32(1)]
        b = sh[d + Int32(32) + Int32(1)]
        sh[d + Int32(1)] = a*c - b*s
        sh[d + Int32(32) + Int32(1)] = a*s + b*c
    end
    @synchronize
    cachebase = pos * Int32(1024) + h * Int32(256)
    kval = sh[d + Int32(1)]
    vval = Float32(vproj[base + d + Int32(1)])
    sh[Int32(256) + d + Int32(1)] = abs(kval)
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[Int32(256) + d + Int32(1)] =
            max(sh[Int32(256) + d + Int32(1)],
                sh[Int32(256) + d + step + Int32(1)]))
        @synchronize
        step >>= 1
    end
    ks = max(sh[Int32(257)] / 127f0, floatmin(Float32))
    d == Int32(0) && (@inbounds kscale[pos * Int32(4) + h + Int32(1)] = ks)
    sh[d + Int32(1)] = abs(vval)
    @synchronize
    step = Int32(128)
    while step > Int32(0)
        d < step && (sh[d + Int32(1)] =
            max(sh[d + Int32(1)], sh[d + step + Int32(1)]))
        @synchronize
        step >>= 1
    end
    vs = max(sh[Int32(1)] / 127f0, floatmin(Float32))
    d == Int32(0) && (@inbounds vscale[pos * Int32(4) + h + Int32(1)] = vs)
    @inbounds begin
        kcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, kval / ks), -127, 127))
        vcache[cachebase + d + Int32(1)] = Int8(clamp(round(Int32, vval / vs), -127, 127))
    end
end

# Decode attention uses online softmax, so it needs no score buffer even for an
# 8K context. One workgroup owns a query head and each lane owns one value
# dimension. Scores are reduced over the 256 lanes, then all lanes update their
# own weighted-value accumulator.
@kernel cpu=false function decode_attention_kernel!(out, @Const(q), @Const(gate),
                                                    @Const(kcache), @Const(vcache),
                                                    @Const(kscale), @Const(vscale),
                                                    @Const(position))
    sh = @localmem Float32 (256,)
    d = Int32(@index(Local, Linear) - 1)
    h = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        step = Int32(128)
        while step > Int32(0)
            d < step && (sh[d + Int32(1)] += sh[d + step + Int32(1)])
            @synchronize
            step >>= 1
        end
        score = sh[Int32(1)] * 0.0625f0
        newhigh = max(high, score)
        oldscale = exp(high - newhigh)
        newscale = exp(score - newhigh)
        denom = denom * oldscale + newscale
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei]) * vscale[scalei]
        high = newhigh
        @synchronize
    end
    o = h * Int32(256) + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
end

@kernel cpu=false function decode_attention_batch_kernel!(out, @Const(q), @Const(gate),
        @Const(kcache), @Const(vcache), @Const(kscale), @Const(vscale),
        @Const(position))
    sh = @localmem Float32 (256,)
    d = Int32(@index(Local, Linear) - 1)
    group = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        step = Int32(128)
        while step > Int32(0)
            d < step && (sh[d + Int32(1)] += sh[d + step + Int32(1)])
            @synchronize
            step >>= 1
        end
        score = sh[Int32(1)] * 0.0625f0
        newhigh = max(high, score)
        oldscale = exp(high - newhigh)
        newscale = exp(score - newhigh)
        denom = denom * oldscale + newscale
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei]) * vscale[scalei]
        high = newhigh
        @synchronize
    end
    o = querybase + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
end

# Batched attention keeps the 256 value dimensions mapped one-per-thread, but
# reduces each QK dot with subgroup instructions.  The original batch kernel
# used eight full-workgroup barriers for every cached token; this needs two.
@kernel cpu=false function decode_attention_batch_fast_kernel!(out, @Const(q),
        @Const(gate), @Const(kcache), @Const(vcache), @Const(kscale),
        @Const(vscale), @Const(position), ::Val{SUBGROUP}) where {SUBGROUP}
    sh = @localmem Float32 (9,)
    d = Int32(@index(Local, Linear) - 1)
    lane = d % Int32(SUBGROUP)
    sub = d ÷ Int32(SUBGROUP)
    nsub = Int32(256 ÷ SUBGROUP)
    group = Int32(@index(Group, Linear) - 1)
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
        @synchronize
        if d == Int32(0)
            score = 0f0
            for i in Int32(1):nsub
                score += sh[i]
            end
            sh[Int32(9)] = score * 0.0625f0
        end
        @synchronize
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
end

@kernel cpu=false function select_last_kernel!(out, @Const(x), width::Int32, ntokens::Int32)
    i = Int32(@index(Global, Linear) - 1)
    i < width && (@inbounds out[i + Int32(1)] =
        x[(ntokens - Int32(1)) * width + i + Int32(1)])
end

function _copy!(backend, out, x)
    length(out) == length(x) || throw(DimensionMismatch())
    copy_kernel!(backend, 256)(out, x, Int32(length(x)); ndrange=length(x)); out
end

function _rmsnorm!(backend, out, x, weight, eps)
    rmsnorm_kernel!(backend, 256)(out, x, weight, Int32(length(x)), Float32(eps); ndrange=256); out
end

function _densegemv!(backend, out, weight, x)
    K = length(x); M = length(out)
    length(weight) == K*M || throw(DimensionMismatch("dense weight is not $K×$M"))
    subgroup = Mantle.caps(backend).subgroup
    subgroup in (32, 64) || throw(ArgumentError(
        "dense GEMV requires a 32- or 64-lane subgroup, got $subgroup"))
    groups = cld(M, DENSE_ROWS_PER_WG)
    dense_gemv_kernel!(backend, 256)(out, weight, x, Int32(K), Int32(M),
        Val(subgroup); ndrange=groups * 256)
    out
end
