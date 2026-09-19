@inline _sigmoid_f32(x::Float32) = 1f0 / (1f0 + exp(-x))
@inline _softplus_f32(x::Float32) = max(x, 0f0) + log1p(exp(-abs(x)))
@inline _silu_f32(x::Float32) = x * _sigmoid_f32(x)

"""
    depthwise_conv4!(ctx, out, state, x, weight) -> out

One-token causal depthwise convolution for Qwen3.5's recurrent blocks. `state`
is a mutable `3×channels` history and `weight` is `4×channels`, both with
the tap dimension contiguous. The returned sample has SiLU applied, matching
the Gated DeltaNet graph.
"""
@kernel cpu=false function depthwise_conv4_kernel!(out, state, @Const(x), @Const(weight),
                                                   channels::Int32)
    c = Int32(@index(Global, Linear) - 1)
    if c < channels
        si = c * Int32(3) + Int32(1)
        wi = c * Int32(4) + Int32(1)
        @inbounds begin
            x0 = Float32(state[si])
            x1 = Float32(state[si + Int32(1)])
            x2 = Float32(state[si + Int32(2)])
            x3 = Float32(x[c + Int32(1)])
            y = muladd(Float32(weight[wi]), x0,
                muladd(Float32(weight[wi + Int32(1)]), x1,
                muladd(Float32(weight[wi + Int32(2)]), x2,
                       Float32(weight[wi + Int32(3)]) * x3)))
            state[si] = eltype(state)(x1)
            state[si + Int32(1)] = eltype(state)(x2)
            state[si + Int32(2)] = eltype(state)(x3)
            out[c + Int32(1)] = eltype(out)(_silu_f32(y))
        end
    end
end

function depthwise_conv4!(ctx, out, state, x, weight)
    channels = length(x)
    length(out) == channels || throw(DimensionMismatch("convolution output has the wrong length"))
    length(state) == 3channels || throw(DimensionMismatch("convolution state must be 3×channels"))
    length(weight) == 4channels || throw(DimensionMismatch("convolution weight must be 4×channels"))
    depthwise_conv4_kernel!(ctx.backend, 256)(out, state, x, weight, Int32(channels); ndrange=channels)
    out
end

# One workgroup owns one value head. The 128 lanes are its output columns; a
# lane walks the 128 rows of that column, so state never needs an atomic update.
# q/k and the final RMS reduction use shared memory. Raw alpha/beta gates are
# accepted directly, avoiding four tiny launch-and-materialize steps per layer.
@kernel cpu=false function gated_delta_net_kernel!(out, state,
        @Const(q), @Const(k), @Const(v), @Const(alpha), @Const(beta),
        @Const(dt_bias), @Const(a), @Const(z), @Const(norm_weight),
        eps::Float32, nheads::Int32, nkeyheads::Int32)
    sh = @localmem Float32 (512,)
    col = Int32(@index(Local, Linear) - 1)
    head = Int32(@index(Group, Linear) - 1)
    keyhead = head % nkeyheads
    qoff = keyhead * Int32(128)
    voff = head * Int32(128)

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
                sh[Int32(128) + col + Int32(1)] += sh[Int32(128) + col + step + Int32(1)]
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

    raw_alpha = Float32(alpha[head + Int32(1)]) + Float32(dt_bias[head + Int32(1)])
    decay = exp(_softplus_f32(raw_alpha) * Float32(a[head + Int32(1)]))
    b = _sigmoid_f32(Float32(beta[head + Int32(1)]))
    sbase = head * Int32(16384) + col * Int32(128)
    sk = 0f0
    @inbounds for row in Int32(0):Int32(127)
        sk = muladd(decay * Float32(state[sbase + row + Int32(1)]),
                    sh[Int32(128) + row + Int32(1)], sk)
    end
    delta = (Float32(v[voff + col + Int32(1)]) - sk) * b
    y = 0f0
    @inbounds for row in Int32(0):Int32(127)
        si = sbase + row + Int32(1)
        sv = muladd(decay, Float32(state[si]), sh[Int32(128) + row + Int32(1)] * delta)
        state[si] = eltype(state)(sv)
        y = muladd(sv, sh[row + Int32(1)], y)
    end
    y *= 0.08838834764831845f0 # inv(sqrt(128))
    sh[Int32(256) + col + Int32(1)] = y
    sh[Int32(384) + col + Int32(1)] = y * y
    @synchronize

    step = Int32(64)
    while step > Int32(0)
        if col < step
            @inbounds sh[Int32(384) + col + Int32(1)] += sh[Int32(384) + col + step + Int32(1)]
        end
        @synchronize
        step >>= 1
    end
    invrms = inv(sqrt(sh[Int32(385)] / 128f0 + eps))
    @inbounds begin
        gated = sh[Int32(256) + col + Int32(1)] * invrms * Float32(norm_weight[col + Int32(1)])
        gated *= _silu_f32(Float32(z[voff + col + Int32(1)]))
        out[voff + col + Int32(1)] = eltype(out)(gated)
    end
end

"""
    gated_delta_net!(ctx, out, state, q, k, v, alpha, beta,
                     dt_bias, a, z, norm_weight; eps=1f-6) -> out

Qwen3.5 one-token Gated DeltaNet update, including q/k L2 normalization,
raw-gate activation, recurrent state update, per-head RMSNorm, and the SiLU z
gate. Shapes are fixed by the architecture: q/k `128×16`, v/z/out `128×48`,
and state `128×128×48`.
"""
function gated_delta_net!(ctx, out, state, q, k, v, alpha, beta,
                          dt_bias, a, z, norm_weight; eps::Real=1f-6)
    length(q) == 128*16 || throw(DimensionMismatch("q must be 128×16"))
    length(k) == 128*16 || throw(DimensionMismatch("k must be 128×16"))
    length(v) == 128*48 || throw(DimensionMismatch("v must be 128×48"))
    length(z) == 128*48 || throw(DimensionMismatch("z must be 128×48"))
    length(out) == 128*48 || throw(DimensionMismatch("out must be 128×48"))
    length(state) == 128*128*48 || throw(DimensionMismatch("state must be 128×128×48"))
    all(length(x) == 48 for x in (alpha, beta, dt_bias, a)) ||
        throw(DimensionMismatch("alpha, beta, dt_bias and a must have 48 values"))
    length(norm_weight) == 128 || throw(DimensionMismatch("norm weight must have 128 values"))
    gated_delta_net_kernel!(ctx.backend, 128)(out, state, q, k, v, alpha, beta,
        dt_bias, a, z, norm_weight, Float32(eps), Int32(48), Int32(16);
        ndrange=48*128)
    out
end
