const KA = KernelAbstractions

@kernel cpu=false function copy_kernel!(out, @Const(x), n::Int32)
    i = Int32(@index(Global, Linear))
    i <= n && (@inbounds out[i] = eltype(out)(x[i]))
end

@kernel cpu=false function add_kernel!(out, @Const(a), @Const(b), n::Int32)
    i = Int32(@index(Global, Linear))
    i <= n && (@inbounds out[i] = eltype(out)(Float32(a[i]) + Float32(b[i])))
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

@kernel cpu=false function swiglu_kernel!(out, @Const(gate), @Const(up), n::Int32)
    i = Int32(@index(Global, Linear))
    if i <= n
        @inbounds begin
            g = Float32(gate[i])
            out[i] = eltype(out)((g / (1f0 + exp(-g))) * Float32(up[i]))
        end
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
