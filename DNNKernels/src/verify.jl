"""
Layer-by-layer verification against the PyTorch reference activations.

lava-dnn.md, Verification order: "Layer-by-layer, not end-to-end. The first
mismatching layer is the bug."

A fixed tolerance cannot express that. Float32 reassociation makes the absolute
error grow as √n through a reduction, and a squashing nonlinearity then rescales
whatever came in - `segment` reaches a conv with relative error 3e-6 on a
magnitude-1100 tensor, and the sigmoid after it reports 2e-5 "relative" purely
because its own range is 1. Neither is a bug, and no single threshold separates
them from one.

So the criterion is **amplification**: an op is suspect when its output error is
much larger than the error already present on its inputs. A genuine bug creates
error out of nothing (the aliasing bug that corrupted `full` produced 1.0 from
inputs that were exact), while drift only ever carries error forward.

**That last clause is false at an ill-conditioned op, for every implementation.**
Whisper's fp16 encoder amplifies 1576x at `add_42` — block 20's feed-forward,
at eight of its 1500 frames — and PyTorch's own fp16 amplifies **415x** at the
same node against its own fp32, on the same `max|Δ|` this criterion uses. (In
*rel rms* the same event is 12x, because the eight frames are a thousandth of
the tensor: which statistic is quoted changes the number by 35x and not the
conclusion.) Flagging the op says nothing about whose arithmetic
it is. Where a *second* reference precision exists, the control is to ask whether
the reference amplifies there too; `tools/verify_whisper.jl` does exactly that
before counting a flagged node as a failure. Without that control, this criterion
will report a correct half-precision port as broken at whichever op the model is
worst conditioned at.
"""

struct LayerDiff
    index::Int
    id::String
    aten::String
    maxabs::Float64
    relative::Float64
    inflow::Float64      # largest error already on the inputs
    shape::Tuple
end

"""
    maxerr(a, b)

Largest absolute difference, with `-Inf` treated as equal to `-Inf`.

`scalar_tensor(-inf)` feeds the attention mask, and `(-Inf) - (-Inf)` is NaN.
Since every comparison against NaN is false, a plain `maximum(abs.(a .- b))`
makes such a buffer pass *whatever* it contains - so a genuinely NaN result
would have been reported as correct. Anything that differs and is not finite is
an outright mismatch.
"""
function maxerr(a, b)
    m = 0.0
    for (x, y) in zip(a, b)
        isequal(x, y) && continue
        d = abs(Float64(x) - Float64(y))
        isfinite(d) || return Inf
        d > m && (m = d)
    end
    m
end

"""
    tohost(x)

A device result as a host `Array`, and anything already on the host unchanged.

The comparison below walks both tensors element by element, which on a device
array is one transfer per element — where it is permitted at all. `verifygraph`
was CPU-only for that reason, and SAM 2's encoder had to be checked by a
separate end-to-end script (`tools/bench_sam2.jl`) because 1493 ops at 1024x1024
are not something a CPU backend finishes. Downloading each result once makes the
layer-by-layer criterion available on whichever backend actually runs the model,
which for anything the size of Whisper's encoder is the only one there is.
"""
tohost(x) = x
tohost(x::Array) = x
tohost(x::AbstractArray) = collect(x)

"""
    weightkeys(g) -> Set{String}

The `weights.safetensors` names a graph's ops actually read, following views to
their parent: a transposed weight reaches a `mm` as a `:view` buffer, so
scanning `op.ins` for `:weight` alone finds none of the projections in a
transformer.
"""
function weightkeys(g::Graph)
    ks = Set{String}()
    function walk(id)
        b = get(g.buffers, id, nothing)
        b === nothing && return
        b.kind === :weight && !isempty(b.key) && push!(ks, b.key)
        b.kind === :view && !isempty(b.of) && walk(b.of)
        return
    end
    for op in g.ops, i in op.ins
        walk(i)
    end
    ks
end

"""
    verifygraph(graphpath, refs, weights; dims, backend, atol, amplify)

`refs` is the dict from `readsafetensors`, keyed `"<graph>/in<i>"` and
`"<graph>/node/<node>"`. Inputs are taken from the refs so each graph is checked
in isolation - an upstream graph's error cannot mask a downstream one.

Flags the first op whose error exceeds `atol` *and* is more than `amplify` times
the error on its inputs.

Takes a loaded `Graph` as well as a path, so a *prefix* of a graph can be
checked on its own. Whisper's encoder is 32 identical blocks; running two of
them covers every op type it uses for a sixteenth of the arithmetic, which is
the difference between a CPU-backend check that finishes and one that does not.
"""
function verifygraph(g::Graph, refs::AbstractDict, weights::AbstractDict;
                     dims, backend=KernelAbstractions.CPU(),
                     atol=1e-4, rtol=1e-3, rtol16=3e-2, amplify=4.0, verbose=true)
    inputs = Dict{String,Any}()
    for (i, name) in enumerate(g.inputs)
        k = "$(g.name)/in$(i-1)"
        haskey(refs, k) || error("no reference input $k")
        inputs[name] = toback(backend, refs[k])
    end

    # `readsafetensors` returns host arrays and `execute!` does not upload —
    # `Model` does that on the way in, and this entry point has no `Model`. Only
    # the weights this graph names, so verifying a two-block prefix of a 2.4 GB
    # encoder does not upload the other thirty blocks. A no-op when they are
    # already resident, and when the backend is the CPU.
    used = weightkeys(g)
    weights = Dict{String,Any}(k => (k in used ? toback(backend, v) : v)
                               for (k, v) in weights)

    # With a workspace, so this checks the path that actually runs. Op bodies
    # branch on `ctx.ws === nothing` — `native_layer_norm` centres into scratch
    # when it has one and allocates when it does not — and verifying only the
    # allocating branch leaves the shipped one unverified.
    ws = Workspace(backend)
    values = execute!(g, inputs, weights; dims, backend, ws)

    # A flipped predicate changes the graph's behaviour discontinuously, so
    # everything after it diverges for a reason that is not a bug. Pin the tie
    # points to the reference and re-run, so the layers downstream are still
    # checked strictly rather than drowned by the consequence.
    pinned, flips = Dict{String,Any}(), Dict{String,Int}()
    for op in g.ops
        k = "$(g.name)/node/$(op.out)"
        got = get(values, op.out, nothing)
        (got === nothing || got isa Tuple || !haskey(refs, k)) && continue
        eltype(got) === Bool || continue
        size(got) == size(refs[k]) || continue
        n = count(tohost(got) .!= refs[k])
        n > 0 && (pinned[op.out] = toback(backend, refs[k]); flips[op.out] = n)
    end
    isempty(pinned) || (values = execute!(g, inputs, weights; dims, backend, ws, overrides=pinned))

    err = Dict{String,Float64}()          # per-buffer error carried so far
    # fp16 precision is transitive: once a value has passed through an fp16
    # buffer it only carries fp16 accuracy, whatever dtype it is stored in
    # afterwards. aggregate() (tensor_utils.py) divides p by 1-p with p clamped
    # to 1-1e-7, which amplifies that inherited error by orders of magnitude in
    # a buffer torch keeps in fp32.
    half = Dict{String,Bool}()
    for id in g.inputs
        err[id] = 0.0
        half[id] = eltype(inputs[id]) === Float16
    end
    diffs = LayerDiff[]
    ties = Tuple{String,Int,Int}[]
    worst, worstid, checked = 0.0, "", 0

    # a view inherits its parent's error; a weight has none
    function inerr(id)
        haskey(err, id) && return err[id]
        b = get(g.buffers, id, nothing)
        b === nothing && return 0.0
        b.kind === :view && !isempty(b.of) && return inerr(b.of)
        0.0
    end

    function inhalf(id)
        haskey(half, id) && return half[id]
        b = get(g.buffers, id, nothing)
        b === nothing && return false
        b.dtype === Float16 && return true
        b.kind === :view && !isempty(b.of) && return inhalf(b.of)
        false
    end

    for (i, op) in enumerate(g.ops)
        k = "$(g.name)/node/$(op.out)"
        got = get(values, op.out, nothing)
        got === nothing && continue
        got isa Tuple && (got = got[1]; k = "$(g.name)/node/$(op.out).0")
        haskey(refs, k) || continue
        got = tohost(got)
        want = refs[k]
        flow = maximum(Float64[inerr(x) for x in op.ins]; init=0.0)

        if size(got) != size(want)
            push!(diffs, LayerDiff(i, op.id, op.aten, Inf, Inf, flow, size(got)))
            verbose && println("  [$i/$(length(g.ops))] SHAPE $(op.id) ($(op.aten)): " *
                               "$(size(got)) vs reference $(size(want))")
            break
        end

        # A boolean output is a discontinuous function of its inputs, so an
        # fp32-epsilon difference upstream flips it outright rather than
        # perturbing it. object_transformer.py:193 compares logits against
        # their own max, which is an exact tie whenever the winning channel is
        # the one being tested - any reimplementation, or a different cuDNN
        # version, can break that tie the other way. Report it and keep going;
        # a systematically wrong predicate shows up as a large fraction.
        if eltype(got) === Bool || eltype(want) === Bool
            if haskey(flips, op.out)
                n = flips[op.out]
                push!(ties, (op.id, n, length(got)))
                verbose && println("  [$i/$(length(g.ops))] TIE $(op.id) ($(op.aten)): " *
                                   "$n/$(length(got)) elements differ " *
                                   "($(round(100n / length(got), sigdigits=2))%), " *
                                   "pinned to reference to isolate downstream layers")
            end
            err[op.out] = 0.0
            half[op.out] = any(inhalf, op.ins)
            continue
        end

        d = maxerr(got, want)
        err[op.out] = d
        checked += 1
        scale = maximum(abs.(Float64.(want)); init=0.0)
        rel = scale > 0 ? d / scale : d
        d > worst && ((worst, worstid) = (d, op.id))

        # Two conditions, because either alone gives false positives.
        #
        # Relative gate: the error must exceed the dtype's own precision,
        # otherwise it is just rounding. Per-dtype, and taken over the inputs as
        # well as the output - fp16 has eps 9.8e-4, and an fp16 value feeding an
        # op whose result is fp32 still only carries fp16 precision.
        #
        # Growth: the error must have actually appeared *here*. A reduction over
        # n terms legitimately scales its inputs' absolute error by ~Σ|w|, and an
        # elementwise op simply carries it forward, so an op that reports exactly
        # what it was handed has not done anything wrong. A genuine bug creates
        # error from inputs that were fine.
        tainted = eltype(want) === Float16 || any(inhalf, op.ins)
        half[op.out] = tainted
        gate = tainted ? rtol16 : rtol
        if d > atol && rel > gate && d > amplify * max(flow, eps(Float32))
            push!(diffs, LayerDiff(i, op.id, op.aten, d, rel, flow, size(got)))
            verbose && println("  [$i/$(length(g.ops))] FIRST MISMATCH $(op.id) ($(op.aten)): " *
                               "max|Δ| = $(round(d, sigdigits=4)) (rel $(round(rel, sigdigits=3))) " *
                               "from inputs carrying $(round(flow, sigdigits=3)), shape $(size(got))")
            break
        end
    end

    if isempty(diffs) && verbose
        println("  $(g.name): $checked/$(length(g.ops)) ops within tolerance " *
                "(worst accumulated max|Δ| = $(round(worst, sigdigits=4)) at $worstid)" *
                (isempty(ties) ? "" : ", $(length(ties)) tie-sensitive predicate(s)"))
    end
    (isempty(diffs), diffs, ties)
end

"""
    coverage(graphpath) -> (implemented, missing)

Which ATen ops in a graph have a `runop!` method, without executing it.
"""
function coverage(g::Graph)
    impl, miss = String[], String[]
    for op in unique(o.aten for o in g.ops)
        # the catch-all `::Val{T} where T` matches everything, so hasmethod is
        # useless here; the dispatched-on parameter is concrete only for a real
        # implementation
        sig = Base.unwrap_unionall(which(runop!, Tuple{Ctx,Op,Val{Symbol(op)}}).sig)
        isconcretetype(sig.parameters[4]) ? push!(impl, op) : push!(miss, op)
    end
    (sort(impl), sort(miss))
end

# Path forms, kept because `tools/` scripts hold paths to trees they generated
# themselves. Library and test callers take the `Graph` methods above.
coverage(graphpath::AbstractString) = coverage(loadgraph(graphpath))

verifygraph(graphpath::AbstractString, refs::AbstractDict, weights::AbstractDict; kw...) =
    verifygraph(loadgraph(graphpath), refs, weights; kw...)
