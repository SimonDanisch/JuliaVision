# Codes to audio: `EnhancedCodec.decode`, the length regulator, the CFM's Euler
# solve with classifier-free guidance (`flow_matching.py`, `solve_euler`) and
# BigVGAN, as one graph.

"""
    linspace32(a, b, n) -> Vector{Float32}

`torch.linspace(a, b, n)` on a CUDA device: float32, the first half counted from
`a` and the second back from `b` (ATen's `RangeFactories.cu`).
"""
function linspace32(a::Float32, b::Float32, n::Int)
    step = (b - a) / Float32(n - 1)
    half = n ÷ 2
    return Float32[i < half ? a + step * Float32(i) : b - step * Float32(n - 1 - i) for i in 0:(n - 1)]
end

"""
    eulerschedule(steps) -> (t, dt)

`solve_euler`'s times: `dt` is the span's own difference at each step and `t`
accumulates it in float32, from 0.
"""
function eulerschedule(steps::Int)
    span = linspace32(0f0, 1f0, steps + 1)
    t, dt = Float32[], Float32[]
    tk = span[1]
    for k in 1:steps
        d = span[k + 1] - span[k]
        push!(t, tk); push!(dt, d)
        tk = tk + d
    end
    return t, dt
end

"""`int(S_infer.shape[1] * 1.72 * duration_factor)`: mel frames for `n` codes."""
melframes(n::Int, duration::Real) = trunc(Int, 2n * 1.72 * Float64(duration))

# ── kernels ──────────────────────────────────────────────────────────────────

"""
The CFM's fixed inputs at batch 2, conditioned then null: `prompt` `(T, 80, 2)`
is the reference mel over its frames and zero after, `mu` `(512, T, 2)` the
prompt features then the codes' features, `style` `(192, 2)`.
"""
function stackprompt!(prompt, mel, nprompt::Int32, T::Int32)
    g = Int32(KI.get_global_id().x)
    g <= T * Int32(80) || return nothing
    r, c = (g - Int32(1)) % T + Int32(1), (g - Int32(1)) ÷ T + Int32(1)
    @inbounds prompt[r, c, 1] = r <= nprompt ? mel[r, c, 1] : 0f0
    @inbounds prompt[r, c, 2] = 0f0
    return nothing
end

function stackmu!(mu, prompt, cond, nprompt::Int32, T::Int32)
    g = Int32(KI.get_global_id().x)
    n, p = Int32(512) * T, Int32(512) * nprompt
    g <= n || return nothing
    # Both inputs contain frames. Clamp both reads and select their values:
    # selecting an input pointer instead produced zero-filled output on RTX
    # 4000 Ada with the current Lava compiler (see the GPU packing regression).
    @inbounds a = prompt[min(g, p)]
    @inbounds b = cond[max(g - p, Int32(1))]
    @inbounds mu[g] = ifelse(g <= p, a, b)
    @inbounds mu[g + n] = 0f0
    return nothing
end

function stackstyle!(style2, style)
    g = Int32(KI.get_global_id().x)
    g <= Int32(192) || return nothing
    @inbounds style2[g, 1] = style[g]
    @inbounds style2[g, 2] = 0f0
    return nothing
end

"""The prompt frames of the noise zeroed, as `solve_euler` does before the loop."""
function zeroprompt!(x, nprompt::Int32, T::Int32)
    g = Int32(KI.get_global_id().x)
    g <= T * Int32(80) || return nothing
    (g - Int32(1)) % T + Int32(1) <= nprompt && (@inbounds x[g] = 0f0)
    return nothing
end

"""This step's estimator inputs: the state twice and the time twice."""
function stackstate!(x2, t2, x, tsched, index, n::Int32)
    g = Int32(KI.get_global_id().x)
    g <= n || return nothing
    @inbounds x2[g] = x[g]
    @inbounds x2[g + n] = x[g]
    if g == Int32(1)
        @inbounds t2[1] = tsched[index[1]]
        @inbounds t2[2] = tsched[index[1]]
    end
    return nothing
end

"""
`x += dt ((1 + r) v_cond - r v_null)`, then the prompt frames zeroed: one
`solve_euler` step.
"""
function cfmstep!(x, v, dts, index, a::Float32, r::Float32, nprompt::Int32, T::Int32)
    g = Int32(KI.get_global_id().x)
    n = T * Int32(80)
    g <= n || return nothing
    @inbounds if (g - Int32(1)) % T + Int32(1) <= nprompt
        x[g] = 0f0
    else
        d = a * v[g] - r * v[g + n]
        x[g] = x[g] + dts[index[1]] * d
    end
    return nothing
end

"""The generated frames, `x[nprompt+1:T, :]`, as BigVGAN's `(y, 80, 1)` input."""
function generatedframes!(out, x, nprompt::Int32, T::Int32, y::Int32)
    g = Int32(KI.get_global_id().x)
    g <= y * Int32(80) || return nothing
    r, c = (g - Int32(1)) % y + Int32(1), (g - Int32(1)) ÷ y + Int32(1)
    @inbounds out[r, c, 1] = x[nprompt + r, c, 1]
    return nothing
end

"""`torch.clamp(32767 * wav, -32767, 32767)`."""
function pcmscale!(out, wav, n::Int32)
    g = Int32(KI.get_global_id().x)
    g <= n || return nothing
    @inbounds out[g] = clamp(32767f0 * wav[g], -32767f0, 32767f0)
    return nothing
end

# ── the graph ────────────────────────────────────────────────────────────────

"""
    synthesize(tts, v, codes; duration, steps, cfg, noise, seed) -> Vector{Float32}

One segment's audio from its codes, at 22.05 kHz and in 16-bit scale
(`clamp(32767 wav)`), as one graph: codec, regulator to `melframes(n, duration)`
frames, the Euler solve over the reference prompt and the new frames with
classifier-free guidance at `cfg`, BigVGAN on the new frames.

`noise` is the solve's start, `(frames + new, 80)`; without it one is drawn
from `seed` (upstream draws `torch.randn` on its device, which no other
device reproduces).
"""
function synthesize(tts, v::Voice, codes::AbstractVector{<:Integer}; kw...)
    return synthesizereplay(tts, v, codes; kw...)
end


function synthesizereplay(tts, v::Voice, codes; duration::Real=1.0,
                          steps::Int=25, cfg::Real=0.7, noise=nothing, seed::Integer=1234)
    n = length(codes)
    y = melframes(n, duration)
    T = frames(v) + y
    z = noise === nothing ? randn(Random.Xoshiro(seed), Float32, T, 80, 1) :
                            reshape(Float32.(noise), T, 80, 1)
    # Shape, solve settings and the leased voice output resources define the
    # graph. Tokens, style and solver noise are stores within each replay.
    key = (:synthesis, n, y, frames(v), steps, Float32(cfg), v.prompt, v.mel)
    mp = get!(tts.networks.bigvgan.scratch, key) do
        g, out, owned = synthgraph(tts, v, codes; duration, steps, cfg, noise=z, seed)
        Mantle.Readback(g, out)
        plan = Mantle.record!(Mantle.Plan(g; budget=Inf))
        inputs = (; codes=owned.codes, style=owned.style, noise=owned.noise)
        DNNKernels.RecordedPlan(plan, inputs, (out,), Any[values(owned)...], true)
    end
    mp.inputs.codes[:] = Int64.(codes)
    mp.inputs.style[:] = v.style
    mp.inputs.noise[:] = vec(z)
    future = Mantle.Readback(mp.plan.graph, only(mp.outputs))
    Mantle.run!(mp.plan)
    return vec(fetch(future))
end

"""
    synthgraph(tts, v, codes; duration, steps, cfg, noise, seed) -> (graph, out, owned)

[`synthesize`](@ref)'s graph, declared and not run: `out` is the PCM buffer it
writes, `owned` the buffers to free after the run (`out` among them).
"""
function synthgraph(tts, v::Voice, codes::AbstractVector{<:Integer}; duration::Real = 1.0,
                    steps::Int = 25, cfg::Real = 0.7, noise = nothing, seed::Integer = 1234)
    g = Mantle.Graph(tts.device)
    mel, owned = synthesismel!(g, tts, v, codes; duration, steps, cfg, noise, seed)
    wav = emit!(g, tts.networks.bigvgan, (; t = melframes(length(codes), duration)), mel)
    out = Mantle.Buffer(tts.device, Float32, length(wav))
    Mantle.dispatch!(g, pcmscale!, (out, wav, Int32(length(wav))), length(wav); name = "pcm")
    return g, out, merge(owned, (; out))
end

"""Declare codec, regulator and flow solve; optionally bind the generated mel."""
function synthesismel!(g, tts, v::Voice, codes::AbstractVector{<:Integer}; duration::Real = 1.0,
                       steps::Int = 25, cfg::Real = 0.7, noise = nothing,
                       seed::Integer = 1234, out = nothing)
    dev, N = tts.device, tts.networks
    n = length(codes)
    y = melframes(n, duration)
    Tm = frames(v)
    T = Tm + y
    z = noise === nothing ? randn(Random.Xoshiro(seed), Float32, T, 80, 1) : reshape(Float32.(noise), T, 80, 1)
    size(z) == (T, 80, 1) || throw(DimensionMismatch("noise is $(size(z)), expected ($T, 80)"))
    tsched, dts = eulerschedule(steps)
    codesb = Mantle.Buffer(dev, reshape(Int64.(codes), n, 1))
    # Float32 whatever the codec's precision: the regulator reads float32, and a
    # bound output is converted on store.
    S = Mantle.Transient.Buffer(g, Float32, (1024, 2n, 1))
    emit!(g, N.codec, (; n), codesb; out = S)
    index = Mantle.Buffer(dev, nearestindex(2n, y))
    cond = emit!(g, N.regulator, (; t = 2n, y), S, index)
    prompt2 = Mantle.Transient.Buffer(g, Float32, (T, 80, 2))
    mu2 = Mantle.Transient.Buffer(g, Float32, (512, T, 2))
    style2 = Mantle.Transient.Buffer(g, Float32, (192, 2))
    styleb = Mantle.Buffer(dev, v.style)
    Mantle.dispatch!(g, stackprompt!, (prompt2, v.mel, Int32(Tm), Int32(T)), 80T; name = "cfm/prompt")
    Mantle.dispatch!(g, stackmu!, (mu2, v.prompt, cond, Int32(Tm), Int32(T)), 512T; name = "cfm/mu")
    Mantle.dispatch!(g, stackstyle!, (style2, styleb), 192; name = "cfm/style")
    x = Mantle.Buffer(dev, z)
    Mantle.dispatch!(g, zeroprompt!, (x, Int32(Tm), Int32(T)), 80T; name = "cfm/zero prompt")
    tb, dtb = Mantle.Buffer(dev, tsched), Mantle.Buffer(dev, dts)
    a, r = Float32(1 + cfg), Float32(cfg)
    Mantle.repeat!(g, steps) do i
        x2 = Mantle.Transient.Buffer(g, Float32, (T, 80, 2))
        t2 = Mantle.Transient.Buffer(g, Float32, 2)
        Mantle.dispatch!(g, stackstate!, (x2, t2, x, tb, i, Int32(80T)), 80T; name = "cfm/state")
        vel = emit!(g, N.dit, (; t = T), x2, prompt2, t2, style2, mu2)
        Mantle.dispatch!(g, cfmstep!, (x, vel, dtb, i, a, r, Int32(Tm), Int32(T)), 80T; name = "cfm/step")
    end
    mel = out === nothing ? Mantle.Transient.Buffer(g, Float32, (y, 80, 1)) : out
    Mantle.dispatch!(g, generatedframes!, (mel, x, Int32(Tm), Int32(T), Int32(y)), 80y; name = "cfm/frames")
    return mel, (; codes=codesb, index, style=styleb, noise=x, time=tb, delta=dtb)
end

"""Samples of silence between segments: `int(22050 * 200 / 1000)`."""
silence(ms::Real; rate::Int = SAMPLERATE) = trunc(Int, rate * ms / 1000)

"""
    fadeout!(x; ms = 20) -> x

`fade_out_tail`: the last `ms` milliseconds times a raised cosine from 1 to 0
(`0.5 (1 + cos(linspace(0, π, n)))`, float32).
"""
function fadeout!(x::AbstractVector{Float32}; ms::Real = 20.0, rate::Int = SAMPLERATE)
    n = min(trunc(Int, rate * ms / 1000), length(x))
    n <= 1 && return x
    ramp = 0.5f0 .* (1f0 .+ cos.(linspace32(0f0, Float32(π), n)))
    x[(end - n + 1):end] .*= ramp
    return x
end
