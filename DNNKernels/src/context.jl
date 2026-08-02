"""
What a running graph carries with it: the execution context and the per-run
diagnostics.

Both live here rather than next to `execute!` because the kernel entry points
take a `Ctx` — `sdpa(ctx, q, k, v, …)`, `convolution!(ctx, out, x, w, …)` — and a
method signature naming a type needs that type to exist when the method is
defined. `graph.jl` is the only thing above this file.
"""

"""
    Device(backend) -> Device

What the kernels need to know about the device they are about to run on,
answered **once per context**, at construction, from a live device.

This is `kernel-library-review.md` finding 9. Five numbers were hardcoded at
their call sites — 48 shader cores, 32 lanes to a subgroup, a 48 KB shared
budget, a 256-thread launch group, a 48-workgroup occupancy floor — each written
while there was one card to be wrong about. Across an Ada desktop, an RDNA3 iGPU
whose compute subgroup may be 64, and lavapipe, they are wrong in fact and not
merely in principle: `NW * 32` computes the wrong workgroup size on a wave64
device, and every tiling decision keyed on it inherits the error.

None of these may be captured at module scope (`GUARDRAILS.md` §8): they describe
a device, so a second device must get its own answers.

## Where the device comes from

`Lava.vk_context(backend)`, which resolves a pinned backend to the context it was
built with and an unpinned one to whichever is current.

That accessor did not exist when this struct was written. `LavaBackend(ctx)` kept
`ctx.default_bq` and **discarded `ctx`**, and a `BatchQueue` holds a
`Vulkan.Device` rather than the `VkContext` that owns it — so there was no path
at all from a backend to its device, and both project briefs asserted there was
("the carrier already exists"). Adding the field was one line; finding that it
was missing took running on a second vendor.

So a `Device` built from a pinned backend now describes **that** device, and two
of them can be alive at once. What is still single-device is one layer down:
Lava's four module-scope caches hold device-owned handles keyed without the
device (`GUARDRAILS.md` §8). Nothing here depends on those, but the two-device
acceptance test does.
"""
struct Device
    coopmat::Bool          # cooperative-matrix GEMM usable here
    tile::Int              # its tile extent
    subgroup::Int          # lanes per subgroup — 32 on NVIDIA, 32 *or* 64 on RDNA3
    coopmatsubgroup::Int   # …and the width a cooperative-matrix kernel gets
    sharedbudget::Int      # bytes of `@localmem` one workgroup may claim
    workgrouplimit::Int    # threads per workgroup
    cores::Int             # shader cores / SMs; 0 when the device will not say
    launchgroup::Int       # threads `launch!` asks for by default
end

"""
The `VkContext` a backend runs on. See the note in [`Device`](@ref) — this is the
single place that becomes per-backend once Lava's `LavaBackend` keeps its context.
"""
vkcontext(::Any) = nothing
vkcontext(b::Lava.LavaBackend) = Lava.vk_context(b)

"""
Device facts for a backend that is not Lava's — the CPU verification path.

Deliberately not "the RTX 4000 Ada's numbers minus the GPU bits": a CPU run must
not take a tensor-core path, and the shared budget is the portable Vulkan floor
so that anything computed from it stays valid rather than merely plausible.
"""
Device(::Any) = Device(false, 16, 1, 1, 48 * 1024, 1024, 0, 256)

function Device(backend::Lava.LavaBackend)
    ctx = vkcontext(backend)
    Device(Lava.coopmat_gemm_available(ctx),
           Lava.GEMM_TILE,
           Lava.device_subgroup_size(ctx),
           # NOT the device default. Lava PINS any module declaring
           # `CooperativeMatrixKHR` to `COOPMAT_SUBGROUP` via
           # `VK_EXT_subgroup_size_control` (`pipeline.jl`, the
           # `PipelineShaderStageRequiredSubgroupSizeCreateInfo` branch), because
           # its coopmat kernels index subgroups as `tid ÷ 32` and a cooperative
           # matrix is subgroup-scoped.
           #
           # So a coopmat kernel's workgroup must be sized in units of THIS, not
           # of `subgroup`. Getting that wrong is invisible on a wave32 card and
           # silently wrong on RDNA 3.5, where the device default is 64 and the
           # pipeline still runs 32 — the launch would ask for `NW * 64` threads
           # while the kernel indexes `NW * 2` subgroups. Which is worse than the
           # literal `32` it replaces, so it is spelled out here once.
           Lava.COOPMAT_SUBGROUP,
           Lava.max_shared_memory(ctx),
           Lava.WORKGROUP_LIMIT[],
           Lava.shader_core_count(ctx),
           # 256 measured best for `launch!`; see `launchgroup`. Kept as a number
           # on the device rather than a global because it is a workgroup size,
           # and the limit it must respect is the device's.
           min(256, Lava.WORKGROUP_LIMIT[]))
end

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
struct Ctx{B,S,P,W,L,R}
    values::Dict{String,Any}
    graph::Graph
    dims::NamedTuple
    backend::B
    dev::Device                   # what this backend's device can do — see `Device`

    # ── Let an attention whose extents do not divide the tile take the fused
    # cooperative-matrix path anyway, padded and masked.
    #
    # A property of *this graph run*, which is why it is here and not on `Device`
    # (it is not a device fact) and not in a plan (the plan is per call). SAM 2's
    # decoder wants it and its encoder does not: the decoder's attentions are 23
    # tokens and want exactly this, while the encoder has six `Lq = 16` calls that
    # would go along at 50% waste and cost **+2.12 ms of encode** for nothing.
    # Measured interleaved in one process on the autocast export, which is the
    # only form that means anything for a 4 ms call on a shared card: decode
    # 8.24 -> 4.22 ms with the clamp, encode 118.63 -> 120.75.
    #
    # This was a `Ref` that `sam2.jl` set around the decode and restored in a
    # `finally`, defended on the grounds that "`flashcm_tiling` reads it six
    # frames down, inside `runop!`". That is the same defence the five diagnostics
    # `Ref`s made, and it has the same answer: `Ctx` already reaches every
    # `runop!`. Being a field also removes the hazard the `finally` existed for —
    # there is no longer any state that a decoder error could leave switched on
    # for the next encode.
    clampattn::Bool
    slab::S                       # UInt8 scratch slab, or nothing
    plan::P                       # Slab (plan.jl), or nothing
    outid::Base.RefValue{String}  # output id of the op currently running
    ws::W                         # Workspace for kernel-internal scratch, or nothing
    lazy::L                       # Set of ids that may stay unmaterialised (fuse.jl)
    rec::R                        # Recycler for unplanned allocations, or nothing
    diag::Diagnostics             # per-run instrumentation, all off by default
end

# `dev` is derived from `backend` and never passed in. It costs one query per
# context — once per graph execution — and each of those reads a field Lava
# filled at device creation, so it does not go near the driver on the hot path.
#
# ONE constructor, all keywords. There were five telescoping positional forms
# (review finding 6), which is a shape that only ever grows: `slab, plan, outid,
# ws, lazy, rec, diag` is seven arguments in a fixed order with `nothing` as most
# of them, and adding `clampattn` in the middle of that would have been a bug
# nobody could see. A keyword is also self-documenting at the call site.
#
# The four remaining `Any` fields became type PARAMETERS rather than named types
# because `Slab`, `Workspace` and `Recycler` are all defined in files included
# after this one. Parameters cost nothing here: there are two live combinations,
# the graph path and the bare `Ctx(backend)`, not a combinatorial spread. And the
# typing is load-bearing rather than tidiness — see the note above on `slab`,
# where `::Any` put 3 948 CPU kernel specialisations into SAM 2's package image.
function Ctx(values, graph, dims, backend;
             slab = nothing, plan = nothing, outid = Ref(""), ws = nothing,
             lazy = nothing, rec = nothing, diag::Diagnostics = Diagnostics(),
             clampattn::Bool = false)
    Ctx(values, graph, dims, backend, Device(backend), clampattn,
        slab, plan, outid, ws, lazy, rec, diag)
end

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
Ctx(backend; kw...) =
    Ctx(Dict{String,Any}(), Graph("", String[], String[], String[],
                                  Dict{String,Buffer}(), String[], Op[]),
        NamedTuple(), backend; kw...)
