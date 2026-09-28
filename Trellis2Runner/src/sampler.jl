"""
    FlowSampler(; steps, strength, rescale, interval, rescale_t, sigma_min)

TRELLIS.2's `FlowEulerGuidanceIntervalSampler` settings for one stage, as
`pipeline.json` gives them. Not diffusers' scheduler, so not
`DNNKernels.flowschedule`: the time grid stays Float64 (Python floats) and only
meets Float32 where a Python scalar multiplies a tensor.

- `strength`: classifier-free guidance, `strength * pos + (1 - strength) * neg`.
  1 runs the conditioned pass alone.
- `interval`: guidance only while `interval[1] <= t <= interval[2]`; outside it
  the step is the conditioned pass alone.
- `rescale`: CFG rescale, mixing the guided x0 with one rescaled to the
  conditioned x0's standard deviation. 0 is off.
- `rescale_t`: the time warp `r t / (1 + (r - 1) t)`.
"""
struct FlowSampler
    steps::Int
    strength::Float64
    rescale::Float64
    interval::Tuple{Float64,Float64}
    rescale_t::Float64
    sigma_min::Float64
end

FlowSampler(; steps::Integer = 12, strength::Real, rescale::Real = 0.0,
            interval::Tuple{Real,Real} = (0.0, 1.0), rescale_t::Real,
            sigma_min::Real = 1e-5) =
    FlowSampler(steps, strength, rescale, Float64.(interval), rescale_t, sigma_min)

"""
    tschedule(s::FlowSampler) -> Vector{Float64}

`steps + 1` times from 1 to 0: `np.linspace(1, 0, steps + 1)`, then
`r * t / (1 + (r - 1) * t)`, all in Float64 as numpy computes it.
"""
function tschedule(s::FlowSampler)
    r = s.rescale_t
    return [r * t / (1 + (r - 1) * t) for t in linspace(1.0, 0.0, s.steps + 1)]
end

"""Whether step `k` (at time `t`) is guided: `GuidanceIntervalSamplerMixin`."""
guided(s::FlowSampler, t::Float64) = s.strength != 1 && s.interval[1] <= t <= s.interval[2]

"""
    guidanceruns(s) -> Vector{Tuple{UnitRange{Int},Bool}}

The steps as maximal runs that are all guided or all not. The time falls
monotonically, so an interval is one run: TRELLIS.2's sparse structure is ten
guided steps then two plain ones, the texture model twelve plain ones.
"""
function guidanceruns(s::FlowSampler)
    ts = tschedule(s)
    runs = Tuple{UnitRange{Int},Bool}[]
    k = 1
    while k <= s.steps
        gk = guided(s, ts[k])
        j = k
        while j < s.steps && guided(s, ts[j + 1]) == gk
            j += 1
        end
        push!(runs, (k:j, gk))
        k = j + 1
    end
    return runs
end

"""
    sampleflow!(g, model, name, sampler, x, cond, negcond, extra; dims, group) -> x

Declare a stage's whole sampler into `g`: every step's model calls, the guidance
and the Euler update `x -= (t - t_prev) v`. Each run of guided or plain steps is
a `repeat!` loop, so the model is declared once per kind of step (twice for a
guided one, conditioned and not) and not once per step; what differs between
steps is read on the device, by the loop index, out of the stage's schedule.

`x` is the state, a `Buffer` the graph updates in place: the noise going in, the
sample coming out. `cond`, `negcond` and `extra` (what a stage fixes for all its
steps: the rotary table and, for the texture models, the shape latent) are the
graph's resources the model reads.

The model sees `t * 1000` as one Float32, `torch.tensor([1000 * t])`: the
product in Float64, one rounding, done on the host into the schedule.
"""
function sampleflow!(g::Mantle.Graph, model::Model, name::AbstractString, s::FlowSampler,
                     x, cond, negcond, extra::Tuple = (); dims::NamedTuple = (;),
                     group::Int = 256)
    dev = model.device
    aten = model.graphs[name]
    ts = tschedule(s)
    n = length(x)
    a = Float32(1 - s.sigma_min)
    w, wneg = Float32(s.strength), Float32(1 - s.strength)
    r, rkeep = Float32(s.rescale), Float32(1 - s.rescale)
    velocity(xin, tt, c) = only(values(emitgraph!(g, model, name; dims,
        inputs = Dict(zip(aten.inputs, (xin, tt, c, extra...))))))
    for (steps, isguided) in guidanceruns(s)
        # This run's schedule, indexed by the loop's iteration.
        t1000 = Mantle.Buffer(dev, Float32[Float32(1000 * ts[k]) for k in steps])
        dt = Mantle.Buffer(dev, Float32[Float32(ts[k] - ts[k + 1]) for k in steps])
        b = Mantle.Buffer(dev, Float32[Float32(s.sigma_min + (1 - s.sigma_min) * ts[k])
                                       for k in steps])
        Mantle.repeat!(g, length(steps)) do i
            tt = Mantle.Transient.Buffer(g, Float32, 1)
            Mantle.dispatch!(g, steptime!, (tt, t1000, i), 1; name = "$name/t")
            pos = velocity(x, tt, cond)
            if !isguided
                Mantle.dispatch!(g, eulerstep!, (x, pos, dt, i), n; name = "$name/step")
                return
            end
            neg = velocity(x, tt, negcond)
            if s.rescale > 0
                ratio = Mantle.Transient.Buffer(g, Float32, 1)
                Mantle.dispatch!(g, cfgratio!, (ratio, x, pos, neg, a, b, w, wneg, i, Val(group)),
                                 group; group, name = "$name/cfg ratio")
                Mantle.dispatch!(g, cfgstep!,
                                 (x, pos, neg, ratio, a, b, w, wneg, r, rkeep, dt, i), n;
                                 name = "$name/cfg step")
            else
                Mantle.dispatch!(g, cfgstep!, (x, pos, neg, w, wneg, dt, i), n;
                                 name = "$name/cfg step")
            end
        end
    end
    return x
end

# ── the kernels of one step ──────────────────────────────────────────────────

"""The model's time input for this step: `1000 t` out of the run's schedule."""
function steptime!(tt, t1000, index)
    @inbounds tt[1] = t1000[index[1]]
    return nothing
end

"""A plain step: `x -= dt v`."""
function eulerstep!(x, v, dt, index)
    i = KI.get_global_id().x
    i <= length(x) || return nothing
    @inbounds x[i] -= dt[index[1]] * v[i]
    return nothing
end

"""
The CFG-rescale ratio `std(x0_pos) / std(x0_cfg)`, unbiased over every element,
as torch computes it (upstream reduces over all but the batch axis, and the
batch is 1). One workgroup: each item sums a strided share in Float64, and a
tree over workgroup memory adds the shares.
"""
function cfgratio!(ratio, x, pos, neg, a::Float32, b, w::Float32, wneg::Float32,
                   index, ::Val{G}) where {G}
    l = KI.get_local_id().x
    bk = @inbounds b[index[1]]
    n = length(x)
    s1 = s2 = s3 = s4 = 0.0
    i = l
    while i <= n
        @inbounds xi, p = x[i], pos[i]
        pred = w * p + wneg * neg[i]
        # `_pred_to_xstart` for both predictions.
        u = Float64(a * xi - bk * p)
        v = Float64(a * xi - bk * pred)
        s1 += u; s2 += u * u
        s3 += v; s4 += v * v
        i += G
    end
    sums = KI.localmemory(Float64, Val((4, G)), Val(1))
    @inbounds sums[1, l], sums[2, l], sums[3, l], sums[4, l] = s1, s2, s3, s4
    KI.barrier()
    half = G ÷ 2
    while half >= 1
        if l <= half
            @inbounds for c in 1:4
                sums[c, l] += sums[c, l + half]
            end
        end
        KI.barrier()
        half ÷= 2
    end
    if l == 1
        m = Float64(n)
        @inbounds varpos = (sums[2, 1] - sums[1, 1]^2 / m) / (m - 1)
        @inbounds varcfg = (sums[4, 1] - sums[3, 1]^2 / m) / (m - 1)
        @inbounds ratio[1] = Float32(sqrt(varpos) / sqrt(varcfg))
    end
    return nothing
end

"""A guided step without rescale: `x -= dt (w pos + (1 - w) neg)`."""
function cfgstep!(x, pos, neg, w::Float32, wneg::Float32, dt, index)
    i = KI.get_global_id().x
    i <= length(x) || return nothing
    @inbounds x[i] -= dt[index[1]] * (w * pos[i] + wneg * neg[i])
    return nothing
end

"""
A guided step with CFG rescale: the guided x0 mixed with itself scaled by
`ratio` (see [`cfgratio!`](@ref)), back to a velocity through
`_xstart_to_pred`, and `x -= dt pred`.
"""
function cfgstep!(x, pos, neg, ratio, a::Float32, b, w::Float32, wneg::Float32,
                  r::Float32, rkeep::Float32, dt, index)
    i = KI.get_global_id().x
    i <= length(x) || return nothing
    k = @inbounds index[1]
    @inbounds xi = x[i]
    @inbounds pred = w * pos[i] + wneg * neg[i]
    bk = @inbounds b[k]
    x0cfg = a * xi - bk * pred
    x0 = r * (x0cfg * @inbounds(ratio[1])) + rkeep * x0cfg
    @inbounds x[i] = xi - dt[k] * ((a * xi - x0) / bk)
    return nothing
end
