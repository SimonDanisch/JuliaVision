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
function depthwise_conv4_kernel!(out, state, x, weight,
                                                   channels::Int32)
    c = Int32(KI.get_global_id().x - 1)
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
    return nothing
end

function depthwise_conv4!(ctx, out, state, x, weight)
    channels = length(x)
    length(out) == channels || throw(DimensionMismatch("convolution output has the wrong length"))
    length(state) == 3channels || throw(DimensionMismatch("convolution state must be 3×channels"))
    length(weight) == 4channels || throw(DimensionMismatch("convolution weight must be 4×channels"))
    harnesslaunch!(ctx.backend, depthwise_conv4_kernel!, out, state, x, weight, Int32(channels); ndrange=channels, workgroupsize = 256)
    out
end

# One workgroup owns one value head: GDN_LANES work-items per state column, each
# holding 128 ÷ GDN_LANES of its rows as float4 runs `4(l + GDN_LANES j) .. +3`, so a
# sub-group's load reads whole runs of every column it covers. A work-item per
# column walking its 128 rows alone put the columns 512 bytes apart: a 64-wide wave
# touched 64 cache lines per load, and 48 workgroups were 96 waves on a 40-CU
# Radeon 8060S, 231 us a layer where the state moves in 26. An M5's caches hid it
# (47 us, at its bandwidth).
#
# A column's lanes sit in one sub-group, which is what the lane sums rely on; the
# column is derived from the sub-group ids because KI leaves which work-items share
# a sub-group unspecified. Raw alpha/beta gates are accepted directly.
const GDN_LANES = 4

# The sum over a column's L lanes, in every one of them. Floating-point addition
# commutes, so each lane's butterfly arrives at the same bits.
@inline _gdn_lanesum(x::Float32, lane::Int32, ::Val{1}) = x
@inline _gdn_lanesum(x::Float32, lane::Int32, ::Val{L}) where {L} =
    _gdn_lanesum(x + KI.shfl(x, lane ⊻ Int32(L ÷ 2)), lane, Val(L ÷ 2))

@inline _gdn_dot4(a, b) = (a[1].value * b[1].value + a[2].value * b[2].value) +
                          (a[3].value * b[3].value + a[4].value * b[4].value)
@inline _gdn_sum2(a) = _gdn_dot4(a, a)

# Run `j` of this lane's rows from `x4`, a float4 view, starting at float4 `base`.
@inline _gdn_runs(x4, base::Int32, ::Val{R}, ::Val{L}) where {R,L} =
    ntuple(j -> @inbounds(x4[base + Int32(L * (j - 1)) + Int32(1)]), Val(R))

@inline _gdn_scale4(a, s::Float32) =
    (VecElement(a[1].value * s), VecElement(a[2].value * s),
     VecElement(a[3].value * s), VecElement(a[4].value * s))

@inline _gdn_update4(s4, decay::Float32, k4, delta::Float32) =
    (VecElement(muladd(decay, s4[1].value, k4[1].value * delta)),
     VecElement(muladd(decay, s4[2].value, k4[2].value * delta)),
     VecElement(muladd(decay, s4[3].value, k4[3].value * delta)),
     VecElement(muladd(decay, s4[4].value, k4[4].value * delta)))

function gated_delta_net_kernel!(out, state,
        q, k, v, alpha, beta,
        dt_bias, a, z, norm_weight,
        eps::Float32, nheads::Int32, nkeyheads::Int32, ::Val{L}) where {L}
    V4 = NTuple{4,VecElement{Float32}}
    runs = Val(32 ÷ L)
    sh = KI.localmemory(Float32, Val((128,)), Val(1))
    sg = Int32(KI.get_sub_group_size())
    lane = Int32(KI.get_sub_group_local_id() - 1)
    slot = Int32(KI.get_sub_group_id() - 1) * sg + lane
    col = slot ÷ Int32(L)
    l = slot % Int32(L)
    head = Int32(KI.get_group_id().x - 1)
    keyhead = head % nkeyheads

    q4 = _gdn_runs(reinterpret(V4, q), keyhead * Int32(32) + l, runs, Val(L))
    k4 = _gdn_runs(reinterpret(V4, k), keyhead * Int32(32) + l, runs, Val(L))
    qscale = inv(sqrt(_gdn_lanesum(sum(map(_gdn_sum2, q4)), lane, Val(L)) + eps))
    kscale = inv(sqrt(_gdn_lanesum(sum(map(_gdn_sum2, k4)), lane, Val(L)) + eps))
    qn = map(x -> _gdn_scale4(x, qscale), q4)
    kn = map(x -> _gdn_scale4(x, kscale), k4)

    @inbounds begin
        raw_alpha = Float32(alpha[head + Int32(1)]) + Float32(dt_bias[head + Int32(1)])
        decay = exp(_softplus_f32(raw_alpha) * Float32(a[head + Int32(1)]))
        b = _sigmoid_f32(Float32(beta[head + Int32(1)]))
    end
    s4 = reinterpret(V4, state)
    sbase = head * Int32(4096) + col * Int32(32) + l
    s = _gdn_runs(s4, sbase, runs, Val(L))
    sk = decay * _gdn_lanesum(sum(map(_gdn_dot4, s, kn)), lane, Val(L))
    voff = head * Int32(128)
    delta = (@inbounds(Float32(v[voff + col + Int32(1)])) - sk) * b
    s2 = map((x, kv) -> _gdn_update4(x, decay, kv, delta), s, kn)
    @inbounds for j in 1:(32 ÷ L)
        s4[sbase + Int32(L * (j - 1)) + Int32(1)] = s2[j]
    end
    y = _gdn_lanesum(sum(map(_gdn_dot4, s2, qn)), lane, Val(L)) *
        0.08838834764831845f0 # inv(sqrt(128))
    l == Int32(0) && (@inbounds sh[col + Int32(1)] = y)
    KI.barrier()

    # Every column's lanes read the same 128 ÷ L of the head's outputs, so each
    # arrives at the same sum of squares without a second barrier.
    ss = 0f0
    @inbounds for i in Int32(0):Int32(128 ÷ L - 1)
        yi = sh[l + Int32(L) * i + Int32(1)]
        ss = muladd(yi, yi, ss)
    end
    invrms = inv(sqrt(_gdn_lanesum(ss, lane, Val(L)) / 128f0 + eps))
    if l == Int32(0)
        @inbounds begin
            gated = y * invrms * Float32(norm_weight[col + Int32(1)])
            gated *= _silu_f32(Float32(z[voff + col + Int32(1)]))
            out[voff + col + Int32(1)] = eltype(out)(gated)
        end
    end
    return nothing
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
    # The kernel reads these as float4 runs.
    all(x -> eltype(x) === Float32, (state, q, k)) ||
        throw(ArgumentError("state, q and k must be Float32"))
    harnesslaunch!(ctx.backend, gated_delta_net_kernel!, out, state, q, k, v, alpha, beta,
        dt_bias, a, z, norm_weight, Float32(eps), Int32(48), Int32(16), Val(GDN_LANES);
        ndrange=48*128*GDN_LANES, workgroupsize = 128*GDN_LANES)
    out
end
