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

@kernel cpu=false function swiglu_kernel!(out, @Const(gate), @Const(up), n::Int32)
    i = Int32(@index(Global, Linear))
    if i <= n
        @inbounds begin
            g = Float32(gate[i])
            out[i] = eltype(out)((g / (1f0 + exp(-g))) * Float32(up[i]))
        end
    end
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

@kernel cpu=false function prepare_kv_kernel!(kcache, vcache, @Const(kproj), @Const(vproj),
                                             @Const(norm), @Const(position), theta_base::Float32,
                                             eps::Float32)
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
    @inbounds begin
        kcache[cachebase + d + Int32(1)] = eltype(kcache)(sh[d + Int32(1)])
        vcache[cachebase + d + Int32(1)] = eltype(vcache)(vproj[base + d + Int32(1)])
    end
end

# Decode attention uses online softmax, so it needs no score buffer even for an
# 8K context. One workgroup owns a query head and each lane owns one value
# dimension. Scores are reduced over the 256 lanes, then all lanes update their
# own weighted-value accumulator.
@kernel cpu=false function decode_attention_kernel!(out, @Const(q), @Const(gate),
                                                    @Const(kcache), @Const(vcache),
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
        @inbounds sh[d + Int32(1)] = qv * Float32(kcache[cachei])
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
        @inbounds acc = acc * oldscale + newscale * Float32(vcache[cachei])
        high = newhigh
        @synchronize
    end
    o = h * Int32(256) + d + Int32(1)
    @inbounds out[o] = eltype(out)((acc / denom) * Float32(gate[o]))
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
