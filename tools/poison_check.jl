# A read-before-write detector for a declared graph.
#
# Nothing in a graph may read a buffer before its producer writes it, and the
# reason that is easy to get wrong is that freshly pooled memory is ZERO: a read
# of an unwritten buffer returns a plausible number, so the graph is correct right
# up until the memory holds something else.
#
# MatAnyone's matte came back entirely NaN on Metal for exactly that reason. An
# `add.Tensor` feeding a `native_layer_norm` was deferred into the norm's fused
# kernel, which stores the sum where the NORM sits in the order -- fifty passes
# after two other readers of it. Zeros hid it on every backend; the Metal placer
# had handed those bytes to another transient, and there it was fatal. It took
# most of a day, because the graph was correct with `keepall` and with
# `alias = false` and so looked like a placement or a barrier bug, and it is
# neither: serialising all 446 passes did not help, no write left its buffer, and
# the placer's intervals never overlapped. THIS found it in one step.
#
# The test: run each graph twice with identical inputs, once as the pool hands the
# memory over and once with every TRANSIENT pre-filled with NaN. The outputs must
# agree bit for bit.
#
#     julia --project=. tools/poison_check.jl
#
# Two rules learned the hard way:
#
#   * Poison TRANSIENTS only. Filling a resident weight makes every cast of it NaN
#     and says nothing at all -- the first run of this reported two "bugs" that
#     were both a weight it had destroyed.
#   * `fill!(s, NaN)` passes a `Float64` and will not compile for an fp32 array on
#     a device with no `double`. `eltype(s)(NaN)`.
#
# Two FALSE POSITIVES, both confirmed by running unpoisoned twice, which is the
# check to make before believing a hit:
#
#   * a stochastic op. Kokoro's `SineGen` draws `rand`/`randn_like`, so two runs
#     differ by construction. `noise = ZeroNoise()` removes that term.
#   * an atomic scatter-add. `index_put` with `accumulate = true` accumulates in
#     whatever order the atomics land, so Kokoro's vocoder differs by one fp16 ULP
#     between any two runs, poisoned or not.
#
# Result as of 2026-09-22, on Metal: every graph of SAM 2.1, MatAnyone,
# DepthAnything, RIFE, Whisper, BasicVSR++ and Kokoro is clean. NeuralLUT is not
# covered -- it keeps no `weights` field to re-emit from.

using Mantle, KernelAbstractions, DNNKernels
const DK = DNNKernels
const M = Mantle

"""A plausible host array for buffer `b` at `dims`; non-zero, so a dropped read shows."""
function fakeinput(b, dims)
    sz = DK.evalshape(b.shape, dims)
    T = b.dtype
    if T <: AbstractFloat
        n = max(prod(sz), 1)
        # `range` refuses length 1 with distinct endpoints, and a 0-d or 1-element
        # tensor is common (a scalar timestep).
        v = n == 1 ? [0.25f0] : collect(range(-0.5f0, 0.5f0; length = n))
        return T.(reshape(v, sz))
    elseif T === Bool
        return rand(Bool, sz)
    elseif T <: Integer
        return T.(reshape(mod.(1:max(prod(sz), 1), 3), sz))
    end
    zeros(T, sz)
end

"""`gr`'s outputs, with every transient pre-filled with NaN when `poison`."""
function runpoison(dev, gr, weights, dims, inputs; poison::Bool)
    mg, ec = DK.emitgraph(dev, gr, DK.residentweights(dev, gr, weights), dims;
                          keepall = true, noise = DK.ZeroNoise())
    if poison
        for (id, r) in ec.res
            b = get(gr.buffers, id, nothing)
            b === nothing && continue
            b.kind === :transient || continue
            s = M.storage(r)
            s isa AbstractArray && eltype(s) <: AbstractFloat && fill!(s, eltype(s)(NaN))
        end
    end
    for (id, x) in inputs
        haskey(ec.res, id) || continue
        dst = M.storage(ec.res[id])
        copyto!(dst, reshape(convert(Array{eltype(dst)}, x), size(dst)))
    end
    p = M.Plan(mg); M.record!(p); M.run!(p); M.waitidle(dev)
    outs = [DK.tohost(M.storage(ec.res[id])) for id in gr.outputs]
    M.free!(p); DK.freeowned!(ec)
    outs
end

bitsame(x, y) = length(x) == length(y) &&
    all(((u, v),) -> (isnan(u) && isnan(v)) || u === v, zip(vec(x), vec(y)))

"""Which outputs of `gr` change when its transients start as NaN instead of zero."""
function poisoncheck(dev, gr, weights, dims)
    ins = Dict{String,Any}(id => fakeinput(gr.buffers[id], dims) for id in gr.inputs)
    a = runpoison(dev, gr, weights, dims, ins; poison = false)
    b = runpoison(dev, gr, weights, dims, ins; poison = true)
    bad = String[]
    for (k, (x, y)) in enumerate(zip(a, b))
        x isa AbstractArray && eltype(x) <: AbstractFloat || continue
        bitsame(x, y) || push!(bad, gr.outputs[k])
    end
    # Before believing a hit: does it differ run to run anyway? See the notes above.
    if !isempty(bad)
        c = runpoison(dev, gr, weights, dims, ins; poison = false)
        flaky = [gr.outputs[k] for (k, (x, y)) in enumerate(zip(a, c))
                 if x isa AbstractArray && eltype(x) <: AbstractFloat && !bitsame(x, y)]
        bad = setdiff(bad, flaky)
        isempty(flaky) || @info "nondeterministic even unpoisoned — not a read-before-write" flaky
    end
    bad
end

"""Every graph a `Model` or a single-graph runner holds."""
graphsof(o) = hasproperty(o, :graphs) ? o.graphs :
              hasproperty(o, :model) && hasproperty(o.model, :graphs) ? o.model.graphs :
              hasproperty(o, :graph) ? Dict("graph" => o.graph) :
              error("no graphs on $(typeof(o))")

"""Run the check over every graph of `model` and print one line each."""
function checkmodel(model, dims; label = "")
    dev = hasproperty(model, :device) ? model.device : M.todevice(model.backend)
    ws = hasproperty(model, :weights) ? model.weights : model.model.weights
    for (name, gr) in sort(collect(graphsof(model)); by = first)
        bad = try
            poisoncheck(dev, gr, ws, dims)
        catch e
            println("  ", rpad(label * name, 30), "skipped: ",
                    first(replace(sprint(showerror, e), '\n' => " | "), 90))
            continue
        end
        println("  ", rpad(label * name, 30),
                isempty(bad) ? "ok" : "READ-BEFORE-WRITE in $(bad)")
        flush(stdout)
    end
end
