# Where a declared model's frame actually goes, one op at a time.
#
# Mantle's profiler does not populate per-pass GPU time on every backend, and the
# obvious substitutes both mislead: a low DISPATCH overhead says the GPU is busy and
# nothing about whether it is busy at peak, and timing graph PREFIXES is not
# monotone because `dropdead` gives each truncation a different set of live ops.
#
# So each op is emitted on its own, with its first input promoted to a graph input,
# and replayed. That over-counts slightly (an op alone pays a submission the frame
# amortises) and it does not see fusion between neighbours, but it finds the ops that
# are orders of magnitude off, which is what a first pass is for.
#
# It found RIFE's: seven transposed convolutions at 0.03 TFLOP/s, 438 ms of a 581 ms
# frame, while the forward convolutions beside them ran on Apple's library at 2.5-9.7.
using Mantle, KernelAbstractions, DNNKernels
const DK = DNNKernels
const M = Mantle

"""
`op` alone as a graph, with every operand it does not own promoted to a graph input.

EVERY one, not just the first: an op whose second input is another op's result
cannot stand alone otherwise, and that is most of them — on Qwen-Image 2.1's
denoiser, promoting only the first left 119 of 140 ops untimed, which is exactly
the ops a census exists to find. A `:weight` keeps its kind so the real tensor is
uploaded; a VIEW keeps its kind too, because its parent is what has to be promoted
and the view resolves off it.
"""
function soloop(g::DK.Graph, op::DK.Op)
    bufs = Dict{String,DK.Buffer}()
    inputs = String[]
    function promote!(id)
        haskey(bufs, id) && return
        b = g.buffers[id]
        if b.kind === :view && !isempty(b.of)
            promote!(b.of)                      # the parent carries the bytes
            bufs[id] = b
        elseif b.kind === :weight
            bufs[id] = b
        else
            bufs[id] = DK.Buffer(b.id, :input, b.shape, b.dtype, b.key, b.live, b.of,
                                 b.viewop, b.attrs)
            push!(inputs, id)
        end
    end
    foreach(promote!, op.ins)
    bufs[op.out] = g.buffers[op.out]
    DK.Graph(g.name, g.symbols, inputs, [op.out], bufs, [collect(keys(bufs))...], [op],
             Vector{String}[])
end

"""Milliseconds for `op` alone, and how many of its dispatches were library calls."""
function timeop(dev, g, weights, dims, op; R = 4)
    gr = soloop(g, op)
    mg, ec = DK.emitgraph(dev, gr, DK.residentweights(dev, gr, weights), dims)
    nlib = sum(count(d -> d.ndrange === nothing, p.dispatches) for p in mg.passes)
    p = M.Plan(mg); M.record!(p)
    # After `record!`: a graph input has no storage until the plan is materialised.
    for id in gr.inputs
        xs = M.storage(ec.res[id])
        xs isa AbstractArray && eltype(xs) <: AbstractFloat || continue
        copyto!(xs, convert(Array{eltype(xs)},
                reshape(Float32.((1:length(xs)) .% 37) ./ 37f0 .- 0.5f0, size(xs))))
    end
    M.run!(p); M.waitidle(dev)
    t = minimum([@elapsed (M.run!(p); M.waitidle(dev)) for _ in 1:R])
    M.free!(p); DK.freeowned!(ec)
    (t * 1000, nlib, length(mg.passes))
end

"""Every op of `g`, slowest first. Ops that cannot stand alone are reported as such."""
function census(dev, g, weights, dims; top = 15, showfailures = 5)
    rows = Any[]
    failures = Any[]
    for op in g.ops
        isempty(op.ins) && continue
        try
            ms, nlib, np = timeop(dev, g, weights, dims, op)
            push!(rows, (op.id, op.aten, ms, nlib, np))
        catch e
            # KEPT, not swallowed: "could not stand alone" is a count, and a
            # count does not say whether the op needs an operand this cannot
            # promote or whether its emit is broken. Those want opposite
            # responses and the message is what tells them apart.
            push!(rows, (op.id, op.aten, NaN, -1, 0))
            push!(failures, (op.id, op.aten, sprint(showerror, e)))
        end
    end
    if !isempty(failures)
        println(length(failures), " op(s) could not stand alone; first ",
                min(showfailures, length(failures)), ":")
        for (id, aten, msg) in first(failures, showfailures)
            println("  ", rpad(id, 22), rpad(aten, 30), first(replace(msg, '\n' => ' '), 110))
        end
    end
    ok = [r for r in rows if !isnan(r[3])]
    tot = sum(r[3] for r in ok; init = 0.0)
    sort!(ok; by = r -> r[3], rev = true)
    println("summed ", round(tot, digits=1), " ms over ", length(ok), " ops (",
            count(r -> isnan(r[3]), rows), " could not stand alone)")
    println(rpad("op", 22), rpad("aten", 30), lpad("ms", 8), lpad("%", 7), lpad("lib", 5))
    for r in first(ok, top)
        println(rpad(r[1], 22), rpad(r[2], 30), lpad(round(r[3], digits=2), 8),
                lpad(round(100r[3]/tot, digits=1), 7), lpad(r[4], 5))
    end
    bykind = Dict{String,Float64}()
    for r in ok; bykind[r[2]] = get(bykind, r[2], 0.0) + r[3]; end
    println("\nby aten:")
    for (k, v) in first(sort(collect(bykind); by = last, rev = true), 10)
        println("  ", rpad(k, 32), lpad(round(v, digits=1), 8), " ms  ",
                lpad(round(100v/tot, digits=1), 5), "%")
    end
    ok
end

# ── the same question asked of a RECORDED plan, one command at a time ────────
#
# `census` above re-emits each op alone, which cannot see what a command costs
# where it actually sits. `commandcensus` replays single commands out of a
# recording instead. Three things make it wrong if they are left out, and all
# three produced confident numbers before they were:
#
#   * WARM UP. The first replay of a command loads its pipeline variant — 49 ms
#     against 0.37 warm, on an M5 that also reports "exceeded compiled variants
#     footprint limit", so variants are being evicted and reloaded. Measured
#     cold, this tree's Qwen-Image 2.1 denoiser came to 3.425 s of kernel time;
#     warm it is 1.416, and the warm figure is the one that matches a standalone
#     benchmark of the same kernel (3.01 ms against 2.90).
#   * WAIT. `closesubmit!` commits and returns a fence; it does not block. Time
#     it alone and a 9 ms segment reads 0.0.
#   * MAP the commands. An ICB command index is not a pass index: `planshape`
#     skips `:update` passes and inserts a writer per predicate CHANGE. Kokoro
#     has 1051 passes and 983 commands, and indexing one by the other named two
#     convolutions as a cast and a split.
#
# The floor is still real — a single command costs a submission the frame
# amortises — so a command whose work is well under ~0.1 ms is reported as
# roughly that floor. Sum by KERNEL and compare ratios, not absolutes.

"""`(command, pass)` pairs for one recorded piece, by `planshape`'s own rule."""
function commandowners(pl, chunk, head::Bool)
    owner = Int[]
    head && push!(owner, 0)                 # the range reset owns no pass
    prev = nothing
    for i in chunk
        pp = pl.passes[i]
        pp.pass.kind === :update && continue
        pred = pp.pass.predicate
        (pred === nothing || M.samepredicate(pred, prev)) || push!(owner, i)
        prev = pred
        for d in pp.dispatches
            d isa M.Call || push!(owner, i)
        end
    end
    owner
end

"""Seconds for one warm replay of commands `lo:hi` of `part`."""
function replaycost(dev, part, ids, slot, lo, hi; R = 5)
    seg = M.MetalSegment(lo, hi, slot)
    sub = M.opensubmit!(dev, ids)          # warm: the variant load is not the work
    M.executesegment!(dev, sub, part, seg)
    M.closesubmit!(dev, sub); M.waitidle(dev)
    sub = M.opensubmit!(dev, ids)
    for _ in 1:R
        M.executesegment!(dev, sub, part, seg)
    end
    M.closesubmit!(dev, sub)
    return (@elapsed M.waitidle(dev)) / R   # closesubmit! does NOT wait
end

"""
    commandcensus(dev, plan; R = 5, top = 12) -> Dict

Every ICB command of a recorded plan, warm, summed by kernel.

Refuses rather than guesses when the command map and the recording disagree:
that mismatch is what silently renames every row.
"""
function commandcensus(dev, pl; R = 5, top = 12)
    rec = pl.recording
    rec === nothing && throw(ArgumentError("commandcensus: this plan was not recorded."))
    parts = rec isa M.RecordingParts ? rec.parts : [rec]
    maxp = rec isa M.RecordingParts ? pl.record_maxpasses : length(pl.passes)
    agg = Dict{String,Tuple{Int,Float64}}()
    for (pi, part) in enumerate(parts)
        chunk = ((pi - 1) * maxp + 1):min(pi * maxp, length(pl.passes))
        _, nwriters, _ = M.planshape(pl, chunk)
        owner = commandowners(pl, chunk, pi == 1 && nwriters > 0)
        length(owner) == part.ncommands || throw(ErrorException(
            "commandcensus: piece $pi maps $(length(owner)) commands and the " *
            "recording holds $(part.ncommands). Every name below would be the " *
            "wrong one, so this refuses rather than reporting them."))
        ids = M.ensureresident!(dev, part)
        slot = fill(-1, part.ncommands)
        for s in part.segments, c in s.first:s.last
            slot[c] = s.slot
        end
        for c in 1:part.ncommands
            i = owner[c]
            i == 0 && continue
            k = string(pl.passes[i].pass.dispatches[1].kernel)
            n, tot = get(agg, k, (0, 0.0))
            agg[k] = (n + 1, tot + replaycost(dev, part, ids, slot[c], c, c; R))
        end
    end
    total = sum(v[2] for v in values(agg); init = 0.0)
    println("kernel commands: ", round(total, digits = 3), " s over ",
            sum(v[1] for v in values(agg); init = 0), " commands")
    for (k, (n, s)) in first(sort(collect(agg); by = x -> x[2][2], rev = true), top)
        println("  ", rpad(first(k, 32), 34), "n=", rpad(n, 6),
                lpad(round(s, digits = 3), 7), " s  ",
                lpad(round(1000s / n, digits = 3), 8), " ms each")
    end
    agg
end
