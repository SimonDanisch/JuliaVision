"""
Wan 2.2 text-to-video: the loop around the three graphs.

The models themselves are ordinary DNNKernels graphs — `umt5_encoder`, `wan_dit`,
`wanvae_decoder`. What is *not* a graph is the sampler: a flow-matching ODE
solved by stepping a schedule, with two model evaluations per step. That lives
here, in plain Julia, because it is scheduler arithmetic over a handful of
scalars and putting it in a graph would fix the step count at export time.

    sigmas = flowsigmas(50; shift = 5.0)
    latents = generate(pipe, tokens; steps = 50, guidance = 5.0)

Structure of one step, matching `wan/text2video.py`:

    v_cond   = dit(x, t, context)
    v_uncond = dit(x, t, context_null)
    v        = v_uncond + g * (v_cond - v_uncond)      # classifier-free guidance
    x       += (sigma_next - sigma) * v                # Euler on the flow ODE

Euler rather than Wan's UniPC/DPM-Solver: it is the first-order member of the
same family, needs no solver state, and is exact to compare against. A
higher-order solver reaches the same place in fewer steps and is a drop-in
replacement for `step` below.
"""

"""
    flowsigmas(steps; shift = 5.0) -> Vector{Float64}

Wan's `get_sampling_sigmas`: a uniform 1→0 ramp warped by `shift`, which biases
the schedule toward high noise. The endpoint 0 is dropped, so `sigmas[1] == 1`
and there are exactly `steps` entries.
"""
function flowsigmas(steps::Integer; shift::Real = 5.0)
    s = [1.0 - k / steps for k in 0:(steps - 1)]        # linspace(1,0,steps+1)[1:steps]
    return [shift * σ / (1 + (shift - 1) * σ) for σ in s]
end

"""
    flowtimesteps(sigmas; train_steps = 1000) -> Vector{Float64}

The timestep each sigma corresponds to, which is what the model is conditioned
on. Flow matching parameterises time as the noise level itself, scaled by the
training horizon.
"""
flowtimesteps(sigmas; train_steps::Integer = 1000) = sigmas .* train_steps

"""
    eulerstep(x, v, sigma, sigma_next) -> x'

One Euler step of the flow ODE. `v` is the model's velocity prediction; the
sample moves along it by the change in noise level. `sigma_next` is 0 on the
last step, which lands exactly on the data manifold.
"""
eulerstep(x, v, sigma::Real, sigma_next::Real) = x .+ eltype(x)(sigma_next - sigma) .* v

"""
    cfg(v_cond, v_uncond, scale) -> v

Classifier-free guidance: extrapolate away from the unconditional prediction.
`scale == 1` is the conditional model alone; larger follows the prompt harder at
the cost of diversity.
"""
cfg(v_cond, v_uncond, scale::Real) =
    v_uncond .+ eltype(v_cond)(scale) .* (v_cond .- v_uncond)

"""
Everything one generation needs: the three graphs, their weights, and the
backend they run on.

Held together rather than passed separately because the sampler calls the
transformer twice per step and rebuilding its weight table each time would
dominate the loop.
"""
struct WanPipeline
    encoder::Graph
    encoderweights::Dict{String,Any}
    dit::Graph
    ditweights::Dict{String,Any}
    vae::Graph
    vaeweights::Dict{String,Any}
    backend::Any
    # graph name -> (slab, plan, workspace, lazy set). Built on first call and
    # reused: the plan depends only on the graph, and the transformer runs
    # `2 * steps` times off the same one.
    scratch::Dict{String,Any}
end

function WanPipeline(enc, encw, dit, ditw, vae, vaew, backend)
    enc, encw = prepare(enc, encw)
    dit, ditw = prepare(dit, ditw)
    vae, vaew = prepare(vae, vaew)
    WanPipeline(enc, todevice(enc, encw, backend), dit, todevice(dit, ditw, backend),
                vae, todevice(vae, vaew, backend), backend, Dict{String,Any}())
end

"""
    prepare(g, host) -> (g, host)

Host-side graph preparation, before anything is uploaded. The same pass the
MatAnyone driver runs (`Model`), for the same reason and now on the model that
made the cost obvious.

`hoistpermutes` materialises each transposed *weight* operand into a real weight
buffer laid out the way the GEMM wants it. In the transformer that is not a
tidy-up, it is nearly the whole forward: `Lava.mul!` takes a different kernel for
a `Transpose` operand and it runs at 55 GFLOP/s against 563 on contiguous memory,
and all 26 `addmm` weights arrive permuted, so 97% of the DiT's 1980 ms was one
op taking the slow path. Permuting a constant belongs at load time; done here it
is paid once per pipeline instead of `2 * steps` times per generation, and it is
bit-exact — `permutedims` reorders elements, it does not compute with them.

The 30 permutes this leaves alone are views of *activations*, which change every
call and so cannot be hoisted anywhere.
"""
function prepare(g::Graph, host::AbstractDict)
    w = Dict{String,Any}(host)
    g, n = hoistpermutes(g, w)
    @debug "DNNKernels: $(g.name) hoisted $n permuted weights"
    return (g, w)
end

"""
    realinputs(g) -> Vector{String}

The graph's genuine inputs, i.e. everything `torch.export` did not lift from a
module constant. Lifted ones are prefixed `c_` and are satisfied from the weight
table, so they must not be counted when addressing inputs positionally.
"""
realinputs(g::Graph) = [i for i in g.inputs if !startswith(i, "c_")]

"""
    ascomplex(v, b) -> v

safetensors has no complex dtype, so an exporter saves a complex constant as its
real view: `view_as_real` appends a trailing axis of 2, which the layout reversal
turns into a *leading* one. Restore the complex value when the buffer the graph
declares is complex — `c_m_freqs` is the rotary table, and feeding it as a real
array survives 40 further ops before surfacing as a `cat` bounds error.
"""
function ascomplex(v::AbstractArray, b::Buffer)
    (b.dtype <: Complex && !(eltype(v) <: Complex)) || return v
    size(v, 1) == 2 || error("complex buffer $(b.id): expected a leading pair axis, got $(size(v))")
    return materialize(reinterpret(reshape, Complex{eltype(v)}, v))
end
ascomplex(v, ::Buffer) = v
ascomplex(v, ::Nothing) = v

"""
    todevice(g, host, backend) -> Dict

`g`'s weight table on `backend`. Uploading once, when the pipeline is built, is
the whole point of holding the three graphs together: the transformer runs
`2 * steps` times and re-uploading 1.7 GB per call would cost more than the
arithmetic.

Constants that `torch.export` lifted to inputs live in the same file under their
`c_`-stripped name, so they are resolved here too — and a complex one is restored
*before* the upload, since the pair axis has to collapse on the host.

Only what the graph *reads* is uploaded, which is narrower than what it declares
in two ways. The exporters save one file per *model*, so the VAE's file carries
its encoder too and the decoder graph never reads a byte of it — 1.4 GB of VRAM
that is not merely wasted but is the difference between the decode fitting on
this card and not. And `prepare` leaves every hoisted weight's original behind,
still declared and now read by nothing; uploading those too would have doubled
the transformer's 1.6 GB and turned a speedup into an OOM.
"""
function todevice(g::Graph, host::AbstractDict, backend)
    live = liveroots(g)
    decl = Dict{String,Buffer}()
    for (id, b) in g.buffers
        # Liveness gates the weights only. A lifted constant is a declared
        # *input*, and `rungraph` insists on satisfying every one of them even
        # when no op reads it — `c__grid_sizes` is such an input, and it is three
        # integers, not the megabytes this filter is here for.
        b.kind === :weight && id in live && (decl[b.key] = b)
        b.kind === :external && startswith(id, "c_") && (decl[id[3:end]] = b)
    end
    return Dict{String,Any}(k => toback(backend, ascomplex(host[k], b))
                            for (k, b) in decl if haskey(host, k))
end

"""
    liveroots(g) -> Set{String}

The ids of the buffers the graph's ops actually read, with views resolved to the
buffer they are a view of. A weight reached only through a dead view is not in
here, which is the point.
"""
function liveroots(g::Graph)
    roots = Set{String}()
    for id in Iterators.flatten((Iterators.flatten(op.ins for op in g.ops), g.outputs))
        b = get(g.buffers, id, nothing)
        while b !== nothing && b.kind === :view && !isempty(b.of)
            b = get(g.buffers, b.of, nothing)
        end
        b === nothing || push!(roots, b.id)
    end
    return roots
end

"""
    scratchfor(pipe, name, g) -> (slab, plan, workspace, lazy)

The planned slab for one graph, built once. Without it every intermediate stays
alive for the whole graph: the VAE decoder is 1326 ops at 256×256×9, so its
transients sum to far more than this machine has — planning them into one slab
is what makes the decode run at all, not an optimisation.
"""
function scratchfor(pipe::WanPipeline, name::AbstractString, g::Graph)
    get!(pipe.scratch, name) do
        plan = planslab(g, (;))
        slab = KernelAbstractions.allocate(pipe.backend, UInt8, max(plan.bytes, 1))
        @debug "DNNKernels: $name slab $(round(plan.bytes / 2^20, digits = 1)) MB"
        (slab, plan, Workspace(pipe.backend), fusableset(g))
    end
end

"""
    rungraph(pipe, name, g, weights, inputs) -> output

Run one graph to its single output on the pipeline's backend and slab. `inputs`
maps external buffer ids to values; anything not supplied is looked up in
`weights` under the `c_`-stripped name, which is how `torch.export` names the
constants it lifts. Caller-supplied inputs are moved to the backend here, so the
sampler can keep working in plain host arrays between steps if it wants to.
"""
function rungraph(pipe::WanPipeline, name::AbstractString, g::Graph,
                  weights::AbstractDict, inputs::AbstractDict)
    ins = Dict{String,Any}(k => toback(pipe.backend, v) for (k, v) in inputs)
    for id in g.inputs
        haskey(ins, id) && continue
        startswith(id, "c_") && haskey(weights, id[3:end]) ||
            error("graph $(g.name): input $id was not supplied")
        ins[id] = weights[id[3:end]]
    end
    slab, plan, ws, lazy = scratchfor(pipe, name, g)
    vals = execute!(g, ins, weights; dims = (;), backend = pipe.backend, slab, plan, ws, lazy)
    return value(Ctx(vals, g, (;), pipe.backend), g.outputs[1])
end

"""
    fitshape(v, want) -> v

Reshape a value to the shape the next graph declares for it, when the two differ
only in singleton axes. `StaticWan.forward` ends in `unsqueeze(0)`, so the
transformer emits a batch axis its own latent input does not have, and the VAE
wants that axis back — carrying the mismatch through `eulerstep` instead would
broadcast the latent one rank larger on every step, silently, for 50 of them.
"""
function fitshape(v::AbstractArray, want::Dims)
    size(v) == want && return v
    length(v) == prod(want) ||
        error("cannot fit $(size(v)) into $want: $(length(v)) vs $(prod(want)) elements")
    return reshape(v, want)
end

"""
    inshape(g, id) -> Dims

The Julia shape of a graph input: the declared torch shape reversed, which is the
layout every array in this runtime uses.
"""
inshape(g::Graph, id::AbstractString) = Dims(reverse(Int.(g.buffers[id].shape)))

"""
    generate(pipe, tokens, tokens_null; steps, guidance, latent, shift) -> frames

Text-to-video. `tokens`/`tokens_null` are the prompt and the empty prompt already
tokenised; `latent` is the starting noise, whose shape fixes the clip's size (the
exported transformer pins it).

Returns decoded frames, so the caller never sees a latent.
"""
function generate(pipe::WanPipeline, tokens, tokens_null, latent; kw...)
    enc(tk) = rungraph(pipe, "encoder", pipe.encoder, pipe.encoderweights,
                       Dict{String,Any}(realinputs(pipe.encoder)[1] => tk))
    return generate(pipe, latent; context = enc(tokens), contextnull = enc(tokens_null), kw...)
end

"""
    generate(pipe, latent; context, contextnull, ...) -> frames

The sampler proper, with the text already encoded. Split out because the encoder
runs once while the transformer runs `2 * steps` times, and because it lets the
loop be exercised without a text encoder attached.
"""
function generate(pipe::WanPipeline, latent; context, contextnull,
                  steps::Integer = 50, guidance::Real = 5.0, shift::Real = 5.0,
                  progress = nothing)
    sigmas = flowsigmas(steps; shift = shift)
    ts = flowtimesteps(sigmas)
    # `g.inputs` leads with the constants export lifted (`c_m_freqs`,
    # `c__grid_sizes`), so positional indexing picks the wrong ones — take the
    # real inputs by name.
    din = realinputs(pipe.dit)
    length(din) == 3 ||
        error("wan_dit expects (latent, timestep, context); got $(din)")
    xshape = inshape(pipe.dit, din[1])
    # The sampler's own arithmetic — guidance and the Euler step — runs on the
    # backend too. Leaving the latent on the host makes `cfg` broadcast a host
    # array against a device one, which fails at kernel compilation rather than
    # falling back, and would otherwise copy the state across the bus twice a step.
    x = toback(pipe.backend, fitshape(latent, xshape))
    ctxpos = toback(pipe.backend, context)
    ctxneg = toback(pipe.backend, contextnull)
    dit(xk, t, c) = fitshape(
        rungraph(pipe, "dit", pipe.dit, pipe.ditweights,
                 Dict{String,Any}(din[1] => xk, din[2] => t, din[3] => c)), xshape)
    for k in 1:steps
        t = fill(eltype(x)(ts[k]), 1)
        v = cfg(dit(x, t, ctxpos), dit(x, t, ctxneg), guidance)
        σnext = k == steps ? 0.0 : sigmas[k + 1]
        x = eulerstep(x, v, sigmas[k], σnext)
        progress === nothing || progress(k, steps)
    end

    z = realinputs(pipe.vae)[1]
    return rungraph(pipe, "vae", pipe.vae, pipe.vaeweights,
                    Dict{String,Any}(z => fitshape(x, inshape(pipe.vae, z))))
end
