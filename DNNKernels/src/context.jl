"""
What a running graph carries with it: the execution context and the per-run
diagnostics.

Both live here rather than next to `execute!` because the kernel entry points
take a `Ctx` — `sdpa(ctx, q, k, v, …)`, `convolution!(ctx, out, x, w, …)` — and a
method signature naming a type needs that type to exist when the method is
defined. `graph.jl` is the only thing above this file.
"""

"""
    Diagnostics(; optimes, opdouble, opdoublefilter, planmisses, launches)

Per-run instrumentation. Every field is off by default and free when off: the
cost of an inactive diagnostic is one `=== nothing` on a field of a struct the
caller already holds.

This was five module-level `Ref`s — `OPTIMES`, `OPDOUBLE`, `OPDOUBLEFILTER`,
`PLAN_MISSES`, `LAUNCH_PROBE`. The defence for a global was that they must be
reachable from a call stack that does not thread them, which described the old
signatures rather than constraining anything: `Ctx` already reached all 62
`runop!` methods, and the kernel entry points now take it in place of their
`(backend, ws)` pair — one argument where there were two.

What that buys, beyond the count: **two differently instrumented runs can exist
in one process**, which is the whole point of a measurement object, and the tests
no longer save and restore module state that a failure would leave flipped.

    d = Diagnostics(optimes = Dict{String,Tuple{Int,Float64}}())
    execute!(g, inputs, weights; dims, backend, diag = d)
    sort(collect(d.optimes); by = x -> -x[2][2])

A `Model` carries one from construction (`m.diag`), so instrumenting a whole
`step!` is `m.diag.optimes = Dict{String,Tuple{Int,Float64}}()` and no signature
changes.
"""
Base.@kwdef mutable struct Diagnostics
    # ── `aten name => (count, milliseconds)`, by synchronising around every op.
    #
    # Deliberately a *serialising* measurement — it is the only kind that
    # attributes device time to a source-level op — so the totals run longer than
    # the step does. On this model the difference is small: removing every
    # barrier from a step only buys ~3 ms of 46, i.e. the dispatches barely
    # overlap anyway.
    optimes::Union{Nothing,Dict{String,Tuple{Int,Float64}}} = nothing

    # ── Name of an ATen op to run *twice* per step.
    #
    # The step's wall time then grows by that op's real cost, which is the only
    # way to attribute device time here without the measurement swamping what it
    # measures: syncing around each op costs ~0.4 ms a time and buries a 640-op
    # step under 240 ms of its own barriers, and per-dispatch timestamps
    # serialise and inflate just as badly. Running an idempotent op a second time
    # perturbs nothing — the result is identical — and the difference is measured
    # on an otherwise untouched step.
    #
    # `"*"` matches every aten, which is only useful together with
    # `opdoublefilter`: it costs a *set* of ops spanning several atens in one
    # measurement instead of one run per aten, so the answer carries one run's
    # error rather than the sum of eight.
    opdouble::String = ""

    # ── Narrow `opdouble` to a subset of an op family: `(ctx, op) -> Bool`.
    #
    # Attribution has to happen *in situ*. Standalone convolution microbenchmarks
    # on this setup are unusable — identical code timed one shape at 16 us and
    # 116 us in consecutive runs — while doubling an op inside a captured,
    # replayed step has been stable all along, because the measurement is a whole
    # step and the perturbation is the only thing that changes. This makes that
    # technique reach a single shape instead of a whole family, which is what
    # per-shape convolution cost needs.
    opdoublefilter::Union{Nothing,Function} = nothing

    # ── Every id `dest` could not place, as `id => (count, bytes)`.
    #
    # The counterpart to Lava's allocation trace, which sees only what reaches
    # the pool and is therefore blind to a `Recycler` hit — memory that is just
    # as resident. This answers the question that one cannot: *which op* is still
    # allocating outside the plan, and how much. Getting SAM 2's encoder from
    # 1 649 MB of unplanned allocation per call to 106 MB was four rounds of
    # reading this and `Lava.dump_alloc_trace()` together.
    #
    # A miss is not automatically a bug: a tuple output the planner skips, or a
    # handler asking for a dtype the reservation was not sized for, both land
    # here legitimately. It is a list of candidates, not of faults.
    planmisses::Union{Nothing,Dict{String,Tuple{Int,Int}}} = nothing

    # ── Every `launch!` as `(ndrange, workgroup) => (count, groups)`.
    #
    # For finding launches that do not fill the device. A grid of 64 workgroups
    # on a 48-SM card leaves most of it idle however good the kernel is, and that
    # is invisible in a per-op timing table — it shows up only as one op being
    # inexplicably slow. `Lava.with_dispatch_timing` says *which dispatch*; this
    # says *which launch site and what shape*.
    launches::Union{Nothing,Dict{Tuple{Dims,Dims},Tuple{Int,Dims}}} = nothing
end

"""Ids `dest` could not place, largest first."""
function planmisses(d::Diagnostics)
    m = d.planmisses
    m === nothing && return NamedTuple[]
    [(; id, count = v[1], bytes = v[2]) for (id, v) in
     sort(collect(m); by = x -> -x[2][2])]
end

# Parameterised on the backend AND the slab type — see the note on `Model` in
# driver.jl. The slab matters as much as the backend: `slabview` has two methods,
# and the `::Vector{UInt8}` one returns a HOST array. With `slab::Any` both are
# live, so `dest` hands back something untyped, `get_backend(out)` in `launch!`
# cannot be resolved, and inference explores `Kernel{CPU}` — which is how 3 948
# CPU kernel specialisations (`cpu_ndmap!`, `cpu_conv2d_igemm!`, …) ended up in
# SAM 2's package image despite nothing here ever running on the CPU. Their
# `__run` takes every argument as `Any`, so their call edges span whole method
# tables and any newly loaded package throws them away.
struct Ctx{B,S}
    values::Dict{String,Any}
    graph::Graph
    dims::NamedTuple
    backend::B
    slab::S                       # UInt8 scratch slab, or nothing
    plan::Any                     # Slab (plan.jl), or nothing
    outid::Base.RefValue{String}  # output id of the op currently running
    ws::Any                       # Workspace for kernel-internal scratch, or nothing
    lazy::Any                     # Set of ids that may stay unmaterialised (fuse.jl)
    rec::Any                      # Recycler for unplanned allocations, or nothing
    diag::Diagnostics             # per-run instrumentation, all off by default
end

Ctx(values, graph, dims, backend) =
    Ctx(values, graph, dims, backend, nothing, nothing, Ref(""), nothing, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid) =
    Ctx(values, graph, dims, backend, slab, plan, outid, nothing, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid, ws) =
    Ctx(values, graph, dims, backend, slab, plan, outid, ws, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy) =
    Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy, rec) =
    Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy, rec, Diagnostics())

"""
    Ctx(backend; ws = nothing, rec = nothing, diag = Diagnostics())

A context with no graph behind it.

For the two callers that are not running a graph and still need the backend, the
workspace and the diagnostics in one object: the driver's memory read, which sits
*between* graphs, and any direct call to a kernel entry point — which is what a
test does, and the reason `sdpa(ctx, …)` is as reachable from a test as
`sdpa(…; backend, ws)` was.

The empty `Graph` costs one small allocation and is never read: nothing on this
path asks for `ctx.graph`, because nothing on it has an op id.
"""
Ctx(backend; ws = nothing, rec = nothing, diag = Diagnostics()) =
    Ctx(Dict{String,Any}(), Graph("", String[], String[], String[],
                                  Dict{String,Buffer}(), String[], Op[]),
        NamedTuple(), backend, nothing, nothing, Ref(""), ws, nothing, rec, diag)
