"""
Running an ATen graph as a Mantle plan.

Mantle derives lifetimes and barriers from what each pass *declares* it touches.
An ATen graph declares nothing of the kind: `execute!` calls a handler, the
handler asks `dest` for storage and launches whatever it likes. So the bridge is
a declaration, and the declaration is **discovered by running the graph once**.

It could not be derived. Two of SAM 2's reads are invisible to any walk over the
graph JSON — an element of a multi-output op, and the read a `contiguous` copy
makes of the buffer it copies from — and both came back as a wrong tensor rather
than as an error, because a read nobody declares is a barrier nobody emits and an
interval that ends too early. `storageids!` carries the two paragraphs.

Discovery stays the mechanism. What changes here is that its *result* is cached:
a graph's declarations depend on the graph, its shapes and the version of the op
handlers, none of which vary between two processes running the same model, so a
cold process reads them instead of executing the graph to rediscover them.

## Why this is `src` and not `ext`

It was an extension on a weak dependency, on the argument that DNNKernels runs
perfectly well on its own `planslab` and Mantle is one more answer to `place`.
That argument is retired: **Mantle is the API this package builds GPU execution
with**, not an alternative to it. Placement, lifetimes, barriers, the recording
and its replay are all Mantle's, and a package whose execution model is optional
has two execution models to keep correct.

It also puts the backend in the right place. Nothing in this file names Lava —
the old `using Lava` here was already dead — because everything it needs is
Mantle's own surface: `Device`, `Buffer`, `Transient.Buffer`, `custom!`, `use`,
`Plan`, `run!`, `bake!`. Which backend supplies the mechanism underneath is
Mantle's extension to decide, so a Metal or WebGPU backend is a change there and
none here.

`planslab` stays, and stays useful: it is what discovery runs against, and
`checkslab` is an independent check on the placement Mantle produces. What it is
no longer is a second way to execute.
"""
const KA = KernelAbstractions

# ── the two plans `dest` can be given ─────────────────────────────────────────

"""
What `dest` hands out once Mantle has placed the graph.

One entry per buffer id: the array the placer gave that resource, which for a
transient is a window into the arena and for a persistent buffer is the buffer
itself. Filled in after `Plan`, because a transient has no storage before then.
"""
struct MantlePlan
    views::Dict{String,Any}
    bytes::Dict{String,Int}
end

MantlePlan() = MantlePlan(Dict{String,Any}(), Dict{String,Int}())

function place(p::MantlePlan, slab, id::AbstractString, ::Type{T}, dims) where {T}
    v = get(p.views, id, nothing)
    v === nothing && return nothing
    # The same rule the slab planner applies: the reservation is what discovery
    # asked for, and a larger request would overrun into whatever Mantle aliased
    # next door.
    prod(dims) * sizeof(T) <= p.bytes[id] || return nothing
    slabview(T, v, dims, 0)
end

"""
The plan a discovery run carries.

It serves `dest` out of DNNKernels' ordinary slab — so the run is the one that
would have happened anyway, with correct values for the next op to be discovered
against — and records every id it is asked for on the way. `writes` is emptied by
the caller before each op, so what is left in it afterwards is that op's writes.

`order` is the ids in the sequence they were first requested. Nothing needs that
sequence to run, but the cache does: rebuilding the resources without executing
the graph has to create them in an order some later process can reproduce, and
"whatever the dictionary iterated" is not one.
"""
struct Discovery
    mgraph::Any                     # the Mantle graph the transients belong to
    dev::Any
    esc::Set{String}
    res::Dict{String,Any}           # buffer id -> the Mantle resource holding it
    bytes::Dict{String,Int}
    order::Vector{String}
    writes::Vector{String}
    slab::Slab                      # DNNKernels' own plan, to serve the run
    store::Any
end

function place(d::Discovery, slab, id::AbstractString, ::Type{T}, dims) where {T}
    n = alignup(prod(dims) * sizeof(T))
    have = get(d.bytes, id, 0)
    if have == 0
        d.res[id] = resourcefor(d.mgraph, d.dev, id, d.esc, n)
        d.bytes[id] = n
        push!(d.order, id)
    elseif n > have
        error("buffer $id was asked for $have bytes and then $n: the op sequence " *
              "is not static, so one discovery run cannot describe it")
    end
    push!(d.writes, id)
    place(d.slab, d.store, id, T, dims)
end

"""
One buffer's Mantle resource: transient unless it escapes.

An escaping buffer is read after the plan has run — by the next graph, or by the
caller reading an output — so it cannot be a transient whose bytes Mantle is free
to hand to something else at its last use.

Its own function because the cached path has to make exactly the same decision
without a run to make it during, and two spellings of one rule is how they drift.
"""
resourcefor(mgraph, dev, id::AbstractString, esc::Set{String}, n::Integer) =
    id in esc ? M.Buffer(dev, UInt8, n) : M.Transient.Buffer(mgraph, UInt8, n)

# ── what a read touches ───────────────────────────────────────────────────────

"""
    storageids!(out, graph, res, byout, id) -> out

The ids whose storage a read of `id` actually lands on.

A read does not always land where it is spelled. A value the fuser left lazy has
no storage at all — `emit` returned the `Broadcasted` and never called `dest` —
so reading it reads its operands, recursively. A view `makeview` resolves lazily
(a reshape, a slice, a `PermutedDimsArray`) reads its parent's storage.

Anything with an entry in `res` answers for itself and stops the walk, which is
what makes a view that *was* materialised come out as itself rather than as the
buffer it was copied from.

The one id that is not spelled anywhere in the graph is an element of a
multi-output op: `native_layer_norm` names a tuple with no single shape, so the
planner reserves `"native_layer_norm.0"` and the graph reaches it through a
`getitem` view of the tuple. Walking past that to the tuple, and then to the op
that produced it, declares a read of the op's *inputs* — which is a live read of
memory that has already been handed to something else, and the reason SAM 2's
first attempt came out as NaN.
"""
function storageids!(out, g::Graph, res, byout, id::AbstractString, depth::Int = 0)
    depth > 64 && return out
    haskey(res, id) && (push!(out, id); return out)
    b = get(g.buffers, id, nothing)
    b === nothing && return out
    if !isempty(b.of)
        if occursin("getitem", b.viewop)
            k = string(b.of, '.', Int(b.attrs["arg1"]))
            haskey(res, k) && (push!(out, k); return out)
        end
        return storageids!(out, g, res, byout, b.of, depth + 1)
    end
    o = get(byout, id, nothing)
    o === nothing && return out         # a weight, a graph input, a host constant
    for i in o.ins
        storageids!(out, g, res, byout, i, depth + 1)
    end
    out
end

# ── discovery ─────────────────────────────────────────────────────────────────

"""What one op turned out to touch."""
struct OpUse
    reads::Vector{String}
    writes::Vector{String}
    scratch::Bool
end

Base.:(==)(a::OpUse, b::OpUse) =
    a.reads == b.reads && a.writes == b.writes && a.scratch == b.scratch

"""
Everything a discovery run produces that building a plan needs.

Deliberately not the resources themselves: those belong to one Mantle graph on
one device, and the point of caching is to build them again somewhere else. What
survives is the *declaration* — per op what it touched, per buffer how many bytes
it asked for, and the order the buffers were first asked for in.
"""
struct Declarations
    uses::Vector{OpUse}
    order::Vector{String}
    bytes::Dict{String,Int}
    wsbytes::Int
end

Base.:(==)(a::Declarations, b::Declarations) =
    a.uses == b.uses && a.order == b.order && a.bytes == b.bytes && a.wsbytes == b.wsbytes

"""
    discover(graph, ctx, disc, byout) -> Vector{OpUse}

Run the graph once, recording per op what it read and what it wrote.

A separate phase from building the passes, rather than one interleaved with them,
because the workspace resource is one declaration whose *size* is only known once
the whole run has grown it. Interleaving would mean either declaring it before it
exists or reaching back into passes already built.
"""
function discover(graph::Graph, ctx::Ctx, disc::Discovery, byout)
    map(graph.ops) do op
        empty!(disc.writes)
        ctx.outid[] = op.out
        reset!(ctx.ws)
        ctx.values[op.out] = coerce(timeop!(ctx, op), graph.buffers[op.out])
        # Reads after the run, not before it. An op that reads a view of a
        # permuted parent makes the copy itself, on the way in — so at the moment
        # it started, the buffer it is about to read had no storage to name, and
        # the walk would have gone past it to the parent. 51 of SAM 2's encoder
        # buffers are that copy.
        reads = Set{String}()
        for inp in op.ins
            storageids!(reads, graph, disc.res, byout, inp)
        end
        # Making that copy is itself a read, of the buffer being copied *from*,
        # and the walk above cannot see it: it stops at the copy, which is the
        # right answer for every later op and the wrong one for this one. Left
        # out, the parent's interval ends at the op that produced it, the placer
        # hands its bytes on, and `permutedims!` copies whatever landed there.
        for id in disc.writes
            b = get(graph.buffers, id, nothing)
            (b === nothing || isempty(b.of)) && continue
            storageids!(reads, graph, disc.res, byout, b.of)
        end
        OpUse(sort!(collect(reads)), sort!(unique(disc.writes)), ctx.ws.used > 0)
    end
end

"""
    rediscover(dev, graph, inputs, weights, dims; clampattn) -> Declarations

One discovery run, from scratch. The expensive path, and the only one that is
ever *correct by construction* — the cache is checked against this.

**The Mantle graph it discovers into is its own, and throwaway.** `place` on a
`Discovery` creates a resource per buffer, and for a non-escaping one that is a
`Transient.Buffer` registered in whatever graph it was handed. Given the graph
the plan is being built into, discovery leaves a transient there for every buffer
— and `build` then creates the real ones and declares its passes against *those*,
so the discovery set is used by no pass at all. `Liveness` refuses that, by
design: "a transient nothing uses" is a mistake in the graph.

It only ever bit the fresh-discovery path, which is to say a cold cache, which is
to say the first run on any machine — and never the cached path, which is what
every subsequent run takes. That is the worst possible distribution for a bug and
the reason this function now owns the graph instead of accepting one.
"""
function rediscover(dev, graph::Graph, inputs, weights, dims;
                    clampattn::Bool = false, produced::AbstractDict = Dict{String,Int}())
    backend = M.backend(dev)
    byout = Dict(o.out => o for o in graph.ops)
    dslab = planslab(graph, dims)
    dstore = KA.allocate(backend, UInt8, max(dslab.bytes, 1))
    # Its own graph: nothing here is ever run, and the resources exist only to
    # record sizes and to answer `storageids!`.
    disc = Discovery(M.Graph(dev), dev, escaping(graph), Dict{String,Any}(), Dict{String,Int}(),
                     String[], String[], dslab, dstore)
    # Inputs another graph in the same plan writes. Seeding them into `res` is
    # the whole mechanism: `storageids!` stops at anything `res` knows (line ~137)
    # and otherwise walks past a graph input as "no storage", which is right when
    # the host wrote it before any pass and wrong when a pass writes it. Without
    # this the reads are never declared, so no barrier orders the handover
    # against the graph that consumes it.
    for id in sort!(collect(keys(produced)))
        disc.res[id] = M.Transient.Buffer(disc.mgraph, UInt8, max(produced[id], 1))
        disc.bytes[id] = produced[id]
        push!(disc.order, id)
    end
    ctxd = Ctx(Dict{String,Any}(), graph, dims, backend;
               slab = dstore, plan = disc, ws = Workspace(backend),
               lazy = fusableset(graph), clampattn)
    seed!(ctxd, graph, inputs, weights)
    uses = discover(graph, ctxd, disc, byout)
    Declarations(uses, copy(disc.order), copy(disc.bytes),
                 ctxd.ws.buf === nothing ? 0 : length(ctxd.ws.buf))
end

# ── the cache ─────────────────────────────────────────────────────────────────

"""
    DISCOVERY_VERSION

Generation of the cached declarations. **Bump after changing what an op handler
allocates** — a new `dest` call, a view that is now materialised where it used to
stay lazy, a fused expression that is not fused any more.

Separate from `KERNELS_VERSION`, and the distinction is the whole point of having
two. `KERNELS_VERSION` guards *SPIR-V*: edit a kernel body and the frozen module
is stale. This guards *declarations*: a kernel body can change completely without
moving a single read or write, and a handler can change which buffers it touches
without emitting one new instruction. Neither version subsumes the other, and one
constant covering both would be bumped for every kernel edit and throw away a
cache that was still correct.
"""
const DISCOVERY_VERSION = "1"

"""
    graphkey(graph, dims) -> String

What the cached declarations are keyed by: the graph's structure, the shapes it
was discovered at, and the version above.

Structure rather than the file it was loaded from. Two processes running the same
model must hit, and a graph edited in place must miss — a path and an mtime give
the opposite of both. Everything discovery's answer can depend on goes in: the op
sequence and their inputs, and per buffer the fields `place` and `storageids!`
read.
"""
function graphkey(graph::Graph, dims, produced = String[])
    h = hash(DISCOVERY_VERSION)
    h = hash(graph.name, h)
    for (k, v) in sort!(collect(pairs(dims)); by = first)
        h = hash(k, hash(v, h))
    end
    for o in graph.ops
        h = hash(o.aten, hash(o.out, h))
        for i in o.ins
            h = hash(i, h)
        end
    end
    # Sorted, because `buffers` is a Dict and its iteration order is not stable
    # across processes — the one way this key could differ for identical graphs.
    for id in sort!(collect(keys(graph.buffers)))
        b = graph.buffers[id]
        h = hash(id, hash(b.kind, hash(b.dtype, hash(b.of, hash(b.viewop, h)))))
        for s in b.shape
            h = hash(s, h)
        end
    end
    for o in graph.outputs
        h = hash(o, h)
    end
    # Which inputs are produced *inside* the plan, because that changes the
    # answer: an input the host writes is invisible to `storageids!` and declares
    # no read, while one a pass writes has storage Mantle must order against. The
    # same decoder therefore has two sets of declarations — standalone and
    # chained — and they must not share a cache entry.
    for id in sort!(collect(produced))
        h = hash(id, h)
    end
    string(graph.name, "_", string(h, base = 16))
end

cachedir() = joinpath(first(Base.DEPOT_PATH), "scratchspaces", "dnnkernels_discovery")
cachepath(key::AbstractString) = joinpath(cachedir(), key * ".json")

"""
JSON rather than `Serialization`, for a cache that outlives the session that
wrote it. `Serialization` is not stable across Julia versions and fails loudly
only sometimes; this is four fields of strings and integers, and being able to
read one in a text editor is worth more here than the bytes it costs.
"""
function save(path::AbstractString, d::Declarations)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.write(io, (uses = [(reads = u.reads, writes = u.writes, scratch = u.scratch)
                                 for u in d.uses],
                         order = d.order,
                         bytes = d.bytes,
                         wsbytes = d.wsbytes))
    end
    path
end

function load(path::AbstractString)
    o = JSON3.read(read(path, String))
    Declarations([OpUse(String.(u.reads), String.(u.writes), u.scratch) for u in o.uses],
                 String.(o.order),
                 Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(o.bytes)),
                 Int(o.wsbytes))
end

"""
    declarations(dev, graph, inputs, weights, dims; …) -> Declarations

The graph's declarations, from the cache when it has them and from a run when it
does not.

`refresh = true` runs discovery anyway and overwrites the entry, which is how the
cache is checked: the two `Declarations` must be `==`, and a difference is either
a stale entry or a handler whose behaviour depends on something the key does not
cover. Both are worth failing over.

A corrupt or unreadable entry is a *miss*, not an error. The cache is an
optimisation; the run behind it is always available, and a cache that can wedge a
process is worse than no cache.
"""
function declarations(dev, graph::Graph, inputs, weights, dims;
                      clampattn::Bool = false, cache::Bool = true, refresh::Bool = false,
                      produced::AbstractDict = Dict{String,Int}())
    cache || return rediscover(dev, graph, inputs, weights, dims; clampattn, produced)
    path = cachepath(graphkey(graph, dims, collect(keys(produced))))
    if !refresh && isfile(path)
        try
            return load(path)
        catch err
            @warn "DNNKernels: unreadable discovery cache, rediscovering" path err
        end
    end
    d = rediscover(dev, graph, inputs, weights, dims; clampattn, produced)
    try
        save(path, d)
    catch err
        @warn "DNNKernels: could not write discovery cache" path err
    end
    d
end

# ── building the graph ────────────────────────────────────────────────────────

"""What a built graph needs to be run and read."""
struct GraphPlan
    graph::Graph
    mgraph::Any
    plan::Any
    ctx::Ctx
    res::Dict{String,Any}
    esc::Set{String}
    uses::Vector{OpUse}
    viewids::Vector{String}
    scratch::Any                    # the workspace resource, or nothing
end

"""Seed the value table the way `execute!` does: weights, inputs, host constants."""
function seed!(ctx::Ctx, graph::Graph, inputs, weights)
    for id in graph.order
        b = graph.buffers[id]
        if b.kind === :weight
            haskey(weights, b.key) || error("missing weight $(b.key)")
            ctx.values[id] = weights[b.key]
        elseif b.kind === :external && haskey(inputs, id)
            ctx.values[id] = inputs[id]
        elseif b.kind === :host
            ctx.values[id] = evalexpr(String(b.attrs["expr"]), ctx.dims)
        end
    end
    ctx
end

"""
    build(dev, graph, inputs, weights; dims, …) -> GraphPlan

One Mantle pass per op, declared from the graph's declarations and compiled.

The declarations come from the cache when it has them, so a cold process builds
this plan without executing the graph at all. When they come from a run instead,
that run happens in its own context, out of DNNKernels' own slab: it shares
nothing with the recording context but the graph and the weights, because it
leaves materialised views cached in its value table and those point at slab
storage the plan does not own.
"""
function build(dev, graph::Graph, inputs::AbstractDict, weights::AbstractDict;
               dims, clampattn::Bool = false, alias::Bool = true,
               coalesce::Bool = true, policy = M.Overlap(),
               cache::Bool = true, refresh::Bool = false)
    mg = M.Graph(dev)
    part = addpasses!(mg, dev, graph, inputs, weights; dims, clampattn, cache, refresh)
    finish!(mg, [part]; alias, coalesce, policy)
    only(part.plans)
end

"""
    addpasses!(mg, dev, graph, inputs, weights; dims, produced, …) -> part

Declare one Mantle pass per op of `graph` into `mg`, without compiling anything.

Split out of [`build`](@ref) so several graphs can be laid into **one** Mantle
graph and share a single `Plan`. That is what makes a chain one queue submission
instead of one per graph, and it is also what lets the placer see both graphs'
transients at once rather than as two tenants of an arena.

`produced` maps the ids of this graph's *inputs* that another pass in the same
plan writes, to the resource holding each. They are declared rather than assumed:
see the note in `rediscover`.
"""
function addpasses!(mg, dev, graph::Graph, inputs::AbstractDict, weights::AbstractDict;
                    dims, clampattn::Bool = false, cache::Bool = true, refresh::Bool = false,
                    produced::AbstractDict = Dict{String,Any}())
    backend = M.backend(dev)
    esc = escaping(graph)

    pbytes = Dict{String,Int}(id => M.capacity(r) for (id, r) in produced)
    decl = declarations(dev, graph, inputs, weights, dims;
                        clampattn, cache, refresh, produced = pbytes)

    # The resources, in the order discovery first asked for them. On the cached
    # path this is the only thing that creates them, which is why the order is
    # part of the record rather than left to a Dict's iteration.
    res = Dict{String,Any}()
    for (id, r) in produced
        res[id] = r                      # supplied by whoever writes it
    end
    for id in decl.order
        haskey(res, id) && continue
        res[id] = resourcefor(mg, dev, id, esc, decl.bytes[id])
    end

    mplan = MantlePlan()
    ctxr = Ctx(Dict{String,Any}(), graph, dims, backend;
               plan = mplan, ws = Workspace(backend), lazy = fusableset(graph), clampattn,
               diag = Diagnostics(; planmisses = Dict{String,Tuple{Int,Int}}()))
    seed!(ctxr, graph, inputs, weights)

    # One resource for the kernel workspace, pre-sized to the high-water mark
    # discovery reached so the record run's `scratch!` never grows it. A grow
    # would swap the buffer the declared barriers name for a different one.
    scratch = decl.wsbytes == 0 ? nothing : M.Buffer(dev, UInt8, decl.wsbytes)
    scratch === nothing || (ctxr.ws.buf = M.storage(scratch))

    for (op, u) in zip(graph.ops, decl.uses)
        M.custom!(mg, op.aten) do p
            # One `use` per resource, not one per role: two of them on the same
            # resource in one pass is two entries in the usage sequence, and two
            # transitions derived where the pass has one state.
            for id in union(u.reads, u.writes)
                M.use(p, res[id]; read = id in u.reads, write = id in u.writes)
            end
            u.scratch && M.use(p, scratch; read = true, write = true)
            return () -> runrecorded!(ctxr, graph, op)
        end
    end

    # Not compiled here: `finish!` does that once, for every graph laid into `mg`.
    (; graph, mgraph = mg, ctx = ctxr, res, esc, decl, mplan, scratch, plans = Any[])
end

"""
    finish!(mg, parts; …) -> plan

Compile the one Mantle graph every part was declared into, and give each part its
`GraphPlan`.

Every part shares the returned `Plan`, which is the point: one compile, one
placement over all of their transients, and one queue submission when it is baked.
"""
function finish!(mg, parts; alias::Bool = true, coalesce::Bool = true, policy = M.Overlap())
    plan = M.Plan(mg; alias, coalesce, policy)
    pos = IdDict(p => i for (i, p) in enumerate(mg.passes))
    issorted([pos[pp.pass] for pp in plan.passes]) || error(
        "the scheduler reordered the passes. A custom body is host code reading a " *
        "value table its predecessors filled, so it cannot run out of declaration " *
        "order; use policy = Overlap(), where declaration order dominates the score.")

    for part in parts
        # After `Plan`, not before: a transient has no storage until the placer
        # gives it some, and these views are what `dest` hands out.
        for (id, r) in part.res
            part.mplan.views[id] = M.storage(r)
            part.mplan.bytes[id] = part.decl.bytes[id]
        end
        push!(part.plans,
              GraphPlan(part.graph, mg, plan, part.ctx, part.res, part.esc, part.decl.uses,
                        [id for (id, b) in part.graph.buffers if b.kind === :view],
                        part.scratch))
    end
    plan
end

"""
    Handover(from, fromid, to, toid)

One value passed from one graph in a chain to the next: output `fromid` of part
`from` becomes input `toid` of part `to`.

The conversion is real work — SAM 2's encoder emits `Float16` where its decoder
declares `Float32` — so this is a pass like any other, declared and ordered rather
than issued between two submissions by the host.
"""
struct Handover
    from::Int
    fromid::String
    to::Int
    toid::String
end

"""Several graphs and the values passed between them, as one plan."""
struct ChainPlan
    plan::Any
    parts::Vector{GraphPlan}
    dsts::Dict{Tuple{Int,String},Any}       # (part, input id) -> the typed view
end

"""
    buildchain(dev, specs, handovers, weights; dims, …) -> ChainPlan

Several ATen graphs laid into **one** Mantle graph, with the values passed
between them declared as passes.

`specs` is a vector of `(; graph, inputs, clampattn)`, in execution order.

Why one graph rather than one plan each: a plan is one queue submission when it
is baked, so a chain of two is two, plus however many the host issues for the
conversions between them. Declaring the conversions and compiling the lot
together makes the whole chain a single submission — and lets the placer see
every graph's transients at once instead of sharing an arena between tenants,
which is a strictly better position to place from.

The handover destinations are created *before* the consuming graph is declared,
because its discovery has to see them as storage. An input the host writes is
invisible to `storageids!` and declares no read; one a pass writes must declare
one, or nothing orders the consumer against the handover.
"""
function buildchain(dev, specs, handovers, weights::AbstractDict;
                    dims, alias::Bool = true, coalesce::Bool = true, policy = M.Overlap(),
                    cache::Bool = true, refresh::Bool = false)
    mg = M.Graph(dev)

    # Destinations first: sized from the consumer's own declaration of the input,
    # so the conversion lands in exactly the shape and dtype that graph expects.
    dsts = Dict{Tuple{Int,String},Any}()
    for h in handovers
        b = specs[h.to].graph.buffers[h.toid]
        sz = evalshape(b.shape, dims)
        n = alignup(prod(sz) * sizeof(b.dtype))
        r = M.Buffer(dev, UInt8, n)
        dsts[(h.to, h.toid)] = (res = r, view = slabview(b.dtype, M.storage(r), sz, 0))
    end

    parts = Any[]
    for (i, s) in enumerate(specs)
        mine = [h for h in handovers if h.to == i]
        produced = Dict{String,Any}(h.toid => dsts[(i, h.toid)].res for h in mine)
        inputs = merge(Dict{String,Any}(s.inputs),
                       Dict{String,Any}(h.toid => dsts[(i, h.toid)].view for h in mine))
        push!(parts, addpasses!(mg, dev, s.graph, inputs, weights;
                                dims, s.clampattn, cache, refresh, produced))

        # Then everything this part hands on, so the passes go in execution order.
        for h in handovers
            h.from == i || continue
            src = parts[i].res[h.fromid]
            dst = dsts[(h.to, h.toid)]
            ctx = parts[i].ctx
            M.custom!(mg, "handover $(h.fromid)") do p
                M.use(p, src; read = true)
                M.use(p, dst.res; write = true)
                # Resolved at record time, not now: the producer's value table is
                # filled by its own passes during the same recording.
                return () -> (dst.view .= value(ctx, h.fromid); nothing)
            end
        end
    end

    plan = finish!(mg, parts; alias, coalesce, policy)
    ChainPlan(plan, [only(p.plans) for p in parts], dsts)
end

"""Run the whole chain — one `run!`, one submission when baked — and return the
last graph's outputs."""
function runchain!(c::ChainPlan; barriers::Symbol = :derived)
    foreach(invalidateviews!, c.parts)
    M.run!(c.plan; barriers)
    gp = last(c.parts)
    Tuple(value(gp.ctx, o) for o in gp.graph.outputs)
end

"""Record the whole chain once and replay it thereafter. See [`bakeplan!`](@ref)
for why the invalidation has to happen before the capture and not after."""
function bakechain!(c::ChainPlan)
    foreach(invalidateviews!, c.parts)
    M.bake!(c.plan)
    c
end

"""One op inside its pass: the three lines `execute!` runs, against Mantle's storage."""
function runrecorded!(ctx::Ctx, graph::Graph, op::Op)
    ctx.outid[] = op.out
    reset!(ctx.ws)
    ctx.values[op.out] = coerce(timeop!(ctx, op), graph.buffers[op.out])
    nothing
end

"""
    runplan(gp) -> outputs

Run the plan and resolve the graph's declared outputs.

The cached views go first. `makeview` memoises what it resolves, which is right
within one run and wrong across two: a view that is merely lazy still points at
the parent's live bytes and would be fine, but one `contiguous` had to *copy*
holds the previous run's numbers and nothing would refresh it.
"""
function runplan(gp::GraphPlan; barriers::Symbol = :derived)
    invalidateviews!(gp)
    M.run!(gp.plan; barriers)
    Tuple(value(gp.ctx, o) for o in gp.graph.outputs)
end

"""
Drop the memoised views, which is required before *any* run of this graph.

`makeview` memoises what it resolves. That is right within one run and wrong
across two: a view that is merely lazy still points at the parent's live bytes,
but one `contiguous` had to **copy** holds the previous run's numbers and nothing
refreshes it.
"""
function invalidateviews!(gp::GraphPlan)
    for id in gp.viewids
        delete!(gp.ctx.values, id)
    end
    gp
end

"""
    bakeplan!(gp) -> gp

Record this graph once and replay it on every later [`runplan`](@ref).

Not `Mantle.bake!` directly, and the difference is not a wrapper. Recording a
`custom!` pass **runs its body**, so baking a graph performs a full execution of
it — which means it has to do everything a run does *first*, and the thing a run
does first is [`invalidateviews!`](@ref).

Skipping it bakes the previous run's numbers into the recording, where no later
invalidation can reach them: the capture is already wrong, and the replay then
reproduces it faithfully forever. Measured, on SAM 2's encoder: 791 of 791
buffers wrong from the second op, with the three computed outputs off by 1.2,
2.1 and 15.3 — while the three constant outputs matched, which is what made it
look like a partial failure rather than a total one.
"""
function bakeplan!(gp::GraphPlan)
    invalidateviews!(gp)
    M.bake!(gp.plan)
    gp
end

# ── checks on what the plan came out as ───────────────────────────────────────

"""Ids `dest` could not place from Mantle's plan — expected to be empty."""
misses(gp::GraphPlan) = planmisses(gp.ctx.diag)

"""
    unread(gp) -> Vector{String}

Placed transients no pass declares a read of.

A dead store is possible in principle; in a graph that has been through
`dropdead` it is usually a symptom. The reader exists and was attributed to some
*other* id, so this buffer's interval ends at its write, the placer hands its
bytes to the next thing, and the read lands on them.

Cheaper to ask than to diff 640 intermediates, which cannot be done at all once
the plan has run: the intermediates live in aliased memory and hold whatever took
it over. It is what named the 51 copies `contiguous` forces on SAM 2's encoder,
whose only reader is the op that makes them — declared as writes and not as
reads, because the walk ran before the op did.
"""
unread(gp::GraphPlan) =
    let r = Set{String}()
        for u in gp.uses, id in u.reads
            push!(r, id)
        end
        sort!([id for id in keys(gp.res) if !(id in r) && !(id in gp.esc)])
    end

"""
    shortlived(gp) -> Vector{(id, mantle, dnn)}

Placed buffers whose Mantle interval ends before DNNKernels' own planner says
they are dead.

The two derive lifetimes differently — `planslab` walks the graph and the fusion
chains, Mantle counts which pass touched what — so they are an independent check
on each other, and the direction that matters is one-sided: an interval that ends
too *early* is memory handed to something else while it is still being read.

Not every entry is a fault, in either of two ways. `lifetimes` gives a
multi-output op one interval for the whole tuple, so each element inherits the
longest element's, while Mantle gives each its own. And a `contiguous` copy is
given its *root's* lifetime by `planslab`, deliberately, so that it can never be
placed on top of the buffer it is a copy of; declaring the read that copy makes
says the same thing exactly, and leaves the copy free for the rest of the root's
life. On SAM 2's decoder that is the one entry left after filtering the tuples.

What it is for is the third case. It is how the `contiguous` copy's read of the
buffer it copies *from* was found — not declared at all, so the parent's interval
ended at the op that produced it and `permutedims!` copied whatever the placer
had put there since.
"""
function shortlived(gp::GraphPlan)
    ext = Base.get_extension(Mantle, :MantleLavaExt)
    lt = lifetimes(gp.graph, fusableset(gp.graph))
    out = Tuple{String,Tuple{Int,Int},Tuple{Int,Int}}[]
    for (id, r) in gp.res
        r isa ext.TransientBuffer || continue
        i = findlast('.', id)
        k = i !== nothing && haskey(lt, id[1:(i - 1)]) ? id[1:(i - 1)] :
            haskey(lt, id) ? id : rootbuffer(gp.graph, id)
        haskey(lt, k) || continue
        r.last < lt[k][2] && push!(out, (id, (r.first, r.last), lt[k]))
    end
    sort!(out, by = x -> x[2][2] - x[3][2])
end

