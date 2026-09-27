"""
Per-pass device time of a recorded DNNKernels plan, on any backend.

    include("tools/passtimes.jl")
    t = passtimes(recordedplan)              # one row per compute pass
    byop(t)                                  # summed by the graph op the pass came from
    compareops(byop(vulkan), byop(rocm))     # the same ops side by side

Each compute pass is re-dispatched ALONE on the plan's own storage, in plan order,
and timed with the device idle on both sides. That is the measurement that works
the same on every backend: Vulkan has a timestamp profiler and ROCm has none, and
comparing two different instruments is how a gap gets invented.

What it costs and why it is still the right number:

  * Each pass pays a launch and a synchronise that the recorded plan does not, so
    the rows sum to a little more than a replay. On the Qwen-Image VAE the sum was
    within 5% of the wall time of a replay.
  * The declared `Dispatch` is re-dispatched, with its own `group`. Taking the
    group from the compiled dispatch instead, where it can be `nothing`, launched
    a convolution in the wrong shape and timed a kernel that computed garbage.
  * Plan order matters because transients alias: a pass run out of order reads
    whatever a later pass left in its input.

Only compute passes are timed. A copy or update pass is counted in `skipped`.
"""

using Mantle, Printf

"""The graph op a pass was emitted for: its name up to the first `.`."""
opof(name::AbstractString) = first(split(name, '.'))

"""
    passtimes(plan; reps = 3) -> NamedTuple

`plan` is a DNNKernels `RecordedPlan` or a Mantle `Plan`, recorded and already run
once so its storage holds real values. Returns `(rows, skipped, total_ms)`, rows
being `(index, name, op, kernels, ms)` with `ms` the fastest of `reps` runs.
"""
passtimes(rp; reps::Integer = 3) = passtimes(rp.plan; reps)

function passtimes(pl::Mantle.Plan; reps::Integer = 3)
    dev = pl.graph.dev
    rows = NamedTuple{(:index, :name, :op, :kernels, :ms),
                      Tuple{Int,String,String,String,Float64}}[]
    skipped = 0
    for (i, pp) in enumerate(pl.passes)
        p = pp.pass
        if p.kind !== :compute || isempty(p.dispatches)
            skipped += 1
            continue
        end
        g = Mantle.Graph(dev)
        for d in p.dispatches
            args = map(Mantle.storage, d.args)
            if d.ndrange === nothing
                Mantle.dispatch!(g, d.kernel, args; name = p.name)
            else
                Mantle.dispatch!(g, d.kernel, args, d.ndrange; group = d.group, name = p.name)
            end
        end
        one = Mantle.record!(Mantle.Plan(g))
        Mantle.run!(one); Mantle.waitidle(dev)            # compile and warm
        ms = Inf
        for _ in 1:reps
            Mantle.waitidle(dev)
            t = time_ns()
            Mantle.run!(one); Mantle.waitidle(dev)
            ms = min(ms, (time_ns() - t) / 1e6)
        end
        Mantle.free!(one)
        kernels = join(unique(string(nameof(d.kernel)) for d in p.dispatches), ",")
        push!(rows, (; index = i, name = p.name, op = opof(p.name), kernels, ms))
    end
    return (; rows, skipped, total_ms = sum(r.ms for r in rows; init = 0.0))
end

"""Rows summed by graph op, largest first: `(op, kernels, passes, ms)`."""
function byop(t)
    d = Dict{String,Tuple{String,Int,Float64}}()
    for r in t.rows
        k, n, ms = get(d, r.op, (r.kernels, 0, 0.0))
        d[r.op] = (k, n + 1, ms + r.ms)
    end
    sort([(; op, kernels, passes, ms) for (op, (kernels, passes, ms)) in d]; by = x -> -x.ms)
end

"""
    compareops(a, b; top = 30, io = stdout)

Two `byop` tables joined on the op, sorted by how much more `a` spends than `b`.
The op is the join key because the two backends may emit different passes for
one op; what has to match is the op the graph asked for.
"""
function compareops(a, b; top::Integer = 30, io::IO = stdout, labels = ("a", "b"))
    da = Dict(r.op => r for r in a)
    db = Dict(r.op => r for r in b)
    ops = collect(union(keys(da), keys(db)))
    gap(op) = (haskey(da, op) ? da[op].ms : 0.0) - (haskey(db, op) ? db[op].ms : 0.0)
    sort!(ops; by = op -> -gap(op))
    @printf(io, "%-34s %10s %10s %9s   %s | %s\n", "op", labels[1], labels[2], "gap",
            "kernels " * labels[1], labels[2])
    for op in first(ops, top)
        ra, rb = get(da, op, nothing), get(db, op, nothing)
        @printf(io, "%-34s %8.2f ms %8.2f ms %7.2f ms   %s | %s\n", op,
                ra === nothing ? 0.0 : ra.ms, rb === nothing ? 0.0 : rb.ms, gap(op),
                ra === nothing ? "-" : ra.kernels, rb === nothing ? "-" : rb.kernels)
    end
    @printf(io, "%-34s %8.2f ms %8.2f ms %7.2f ms\n", "total",
            sum(r.ms for r in a; init = 0.0), sum(r.ms for r in b; init = 0.0),
            sum(r.ms for r in a; init = 0.0) - sum(r.ms for r in b; init = 0.0))
    return nothing
end
