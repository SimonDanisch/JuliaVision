"""
Graph execution.

`runop!` dispatches on `Val{op}` so the op set extends by adding a method, not
by editing a switch. Anything elementwise or reducing goes through broadcasting
and `AcceleratedKernels`; only the index-shaped work reaches a `launch!`.

Views are Julia views, and they stay lazy. In the reversed layout a torch
`view`/`reshape` on contiguous data is a Julia `reshape`, a `permute` is a
`PermutedDimsArray` and a `slice` is a `view` - none of them allocating, which
is the "parent + offset + strides" the plan asks for without a bespoke
descriptor type.

Reading a slice out into a fresh array goes through `materialize`, never
`getindex`; see its docstring.
"""

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
end

Ctx(values, graph, dims, backend) =
    Ctx(values, graph, dims, backend, nothing, nothing, Ref(""), nothing, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid) =
    Ctx(values, graph, dims, backend, slab, plan, outid, nothing, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid, ws) =
    Ctx(values, graph, dims, backend, slab, plan, outid, ws, nothing, nothing)
Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy) =
    Ctx(values, graph, dims, backend, slab, plan, outid, ws, lazy, nothing)

"""
    Recycler

Hands the same device buffers back to the same allocation sites on every step.

The planner covers the transients, but what it skips — tuple-shaped outputs,
values that escape into the next graph, an op body's own scratch — is allocated
fresh per step, and Lava's pool is a bump allocator, so those addresses march
upward and never repeat. Measured on MatAnyone2: twelve consecutive steps gave
twelve distinct address tuples for `State`'s four arrays.

That is what blocks `Lava.capture`/`replay!`. A replayed command buffer carries
the device addresses it was recorded with, so replaying a step is only correct if
every buffer it touches is at the address it had at capture. The recycler makes
that true by construction rather than by hoping the pool's free lists hand back
what they took: a call site asks for the *n*-th allocation of a graph call, and
gets the same array object every step as long as the element type and shape
match. They do, because the op sequence of a graph is fixed.

`bank` alternates per step. A value produced in step `k` is read in step `k + 1`
(`State.sensory`, `lastmask`, `lastpixfeat`, `lastmskvalue` all live exactly one
step), so a single set of buffers would have step `k + 1` overwriting what it is
still reading. Two banks give the write somewhere else to land; nothing here
lives longer than one step, so two is enough.
"""
mutable struct Recycler
    banks::NTuple{2,Vector{Any}}
    bank::Int          # 1 or 2
    n::Int             # allocation ordinal within the current graph call
    misses::Int        # allocations the recycler could not serve, for diagnosis
end

Recycler() = Recycler((Any[], Any[]), 1, 0, 0)

"""Start a graph call: ordinals restart, the bank stays as `flip!` left it."""
startcall!(r::Recycler) = (r.n = 0; r)
startcall!(::Nothing) = nothing

"""Move to the other bank. Called once per step, not once per graph."""
flip!(r::Recycler) = (r.bank = 3 - r.bank; r)
flip!(::Nothing) = nothing

"""
    arraytype(backend, T, Val(N)) -> Type

The array type `KernelAbstractions.allocate(backend, T, dims...)` returns.

Exists so `recycle!` can hand back a CONCRETE type. Its banks are `Vector{Any}`,
so the cache-hit path infers as `AbstractArray{T}`; unioned with the allocate
path that widens `rawalloc`, then `dest`, and then `launch!`'s default
`backend = get_backend(out)` infers as `Union{Any, CPU, …, LavaBackend}`. Once
`CPU` is in that union, inference explores `Kernel{CPU}` and `KA.__run` — whose
arguments are all `Any` — and those land in the package image. Their call edges
span whole method tables, so loading any package throws them away along with
everything inferred through them. That was 3 948 CPU kernel specialisations in
SAM 2's image for a model that never runs on the CPU.

The fallback stays abstract, so a backend without a method still works.
"""
arraytype(backend, ::Type{T}, ::Val{N}) where {T,N} = AbstractArray{T,N}
arraytype(::Lava.LavaBackend, ::Type{T}, ::Val{N}) where {T,N} = Lava.LavaArray{T,N}

function recycle!(r::Recycler, backend, ::Type{T}, dims::Dims{N}) where {T,N}
    A = arraytype(backend, T, Val(N))
    v = r.banks[r.bank]
    i = (r.n += 1)
    if i <= length(v)
        b = v[i]
        b isa A && size(b) === dims && return b::A
    end
    r.misses += 1
    b = KernelAbstractions.allocate(backend, T, dims...)::A
    i <= length(v) ? (v[i] = b) : push!(v, b)
    return b
end

"""
    emit(ctx, bc) -> array | Broadcasted

Result of an elementwise op: the `Broadcasted` itself when the graph says this
value is consumed exactly once by another elementwise op (see `fuse.jl`), and a
materialised array otherwise.
"""
@inline function emit(ctx::Ctx, bc)
    inst = Base.Broadcast.instantiate(bc)
    l = ctx.lazy
    (l !== nothing && ctx.outid[] in l) && return inst
    d = dest(ctx, ctx.graph.buffers[ctx.outid[]].dtype, size(inst)...)
    d .= inst
    d
end

"""
    dest(ctx, T, dims...) -> array

The destination for the op currently running. When the graph has a static plan
this is a view into the slab at that buffer's planned offset — no allocation.
Falls back to a fresh allocation for buffers the planner skipped (tuple outputs)
and for handlers that have not been converted to write into their destination
yet, so the two can coexist while the conversion proceeds.
"""
@inline function dest(ctx::Ctx, id::AbstractString, ::Type{T}, dims::Integer...) where {T}
    p, sl = ctx.plan, ctx.slab
    if p !== nothing && sl !== nothing
        off = get(p.offsets, id, nothing)
        # The slot was reserved from the *declared* dtype and shape. A caller
        # asking for a different element type (ops that allocate in `eltype(x)`)
        # could otherwise overrun into the neighbouring buffer, so fall back to a
        # real allocation unless it demonstrably fits.
        if off !== nothing && prod(dims) * sizeof(T) <= get(p.sizes, id, 0)
            # `derive` hands back a real device array sharing the slab's buffer
            # rather than a `reshape(reinterpret(view(...)))` stack. Worth ~7% on
            # Lava (20.6 -> 22.1 steps/s), presumably from the simpler index
            # arithmetic and fewer wrapper layers reaching each kernel.
            #
            # Not, as first assumed, because it switches broadcasts from the
            # Cartesian kernel to the linear one — it does not, and it should not:
            # GPUArrays only uses the linear path when every operand is linearly
            # indexable *and* the shapes match exactly, which is untrue for most
            # ops here. `broadcast_kernel_cartesian` dominating the dispatch count
            # is correct behaviour, not a pathology.
            #
            # Offsets are 256-byte aligned so the element offset is exact.
            return slabview(T, sl, dims, off)
        end
    end
    m = PLAN_MISSES[]
    if m !== nothing
        c, b = get(m, id, (0, 0))
        m[id] = (c + 1, b + prod(dims) * sizeof(T))
    end
    rawalloc(ctx, T, dims)
end

"""
    PLAN_MISSES[] :: Union{Nothing,Dict{String,Tuple{Int,Int}}}

Set to a dict to record every id `dest` could not place, as
`id => (count, bytes)`. Off by default and free when off.

The counterpart to Lava's allocation trace, which sees only what reaches the
pool and is therefore blind to a `Recycler` hit — memory that is just as
resident. This answers the question that one cannot: *which op* is still
allocating outside the plan, and how much. Getting SAM 2's encoder from 1 649 MB
of unplanned allocation per call to 106 MB was four rounds of reading this and
`Lava.dump_alloc_trace()` together.

A miss is not automatically a bug: a tuple output the planner skips, or a
handler asking for a dtype the reservation was not sized for, both land here
legitimately. It is a list of candidates, not of faults.
"""
const PLAN_MISSES = Ref{Any}(nothing)

"""Ids `dest` could not place, largest first."""
function planmisses()
    m = PLAN_MISSES[]
    m === nothing && return NamedTuple[]
    [(; id, count = v[1], bytes = v[2]) for (id, v) in
     sort(collect(m); by = x -> -x[2][2])]
end

"""
    rawalloc(ctx, T, dims) -> array

Storage the planner did not place — the one place every unplanned allocation in a
graph goes through. Served by the `Recycler` when one is attached, so the address
repeats every step; a plain allocation otherwise. Named apart from `alloc`, which
is the ops' entry point and goes through `dest` and the plan first.
"""
@inline function rawalloc(ctx::Ctx, ::Type{T}, dims::Dims) where {T}
    r = ctx.rec
    r === nothing && return KernelAbstractions.allocate(ctx.backend, T, dims...)
    return recycle!(r, ctx.backend, T, dims)
end

"""
    slabview(T, slab, dims, byteoffset) -> array

A `T`-typed array of shape `dims` sharing the slab's storage.

Two implementations because `GPUArrays.derive` only exists for device arrays,
and the CPU backend's slab is an ordinary `Vector{UInt8}`. Without the host
method every `Model` built on the CPU backend threw `MethodError: no method
matching derive(::Type{Float32}, ::Vector{UInt8}, ...)` on its first planned op
— which is why nothing was exercising `foldbatchnorm`/`foldrelu`/`hoistcasts`
end-to-end: `runtests.jl` calls `verifygraph` on the graph JSON directly and
never constructs a `Model` at all.
"""
slabview(::Type{T}, sl, dims, off::Int) where {T} =
    GPUArrays.derive(T, sl, dims, off ÷ sizeof(T))

function slabview(::Type{T}, sl::Vector{UInt8}, dims, off::Int) where {T}
    n = prod(dims) * sizeof(T)
    reshape(reinterpret(T, view(sl, (off + 1):(off + n))), dims)
end

"""Same, for the op currently running."""
@inline dest(ctx::Ctx, ::Type{T}, dims::Integer...) where {T} =
    dest(ctx, ctx.outid[], T, dims...)

"""
    opdest(ctx, xs...) -> array

Destination for the current op, sized by the broadcast of `xs` and typed by the
graph's *declared* dtype for this buffer. Using the declared dtype means the
conversion `coerce` would otherwise do happens inside the broadcast instead of
as a second pass.
"""
@inline function opdest(ctx::Ctx, xs...)
    T = ctx.graph.buffers[ctx.outid[]].dtype
    dest(ctx, T, Broadcast.broadcast_shape(map(size, xs)...)...)
end

"""
    jdim(d, n) -> Int

Torch dimension index (0-based, possibly negative) to Julia dimension for an
`n`-dimensional array whose shape is reversed.
"""
jdim(d::Integer, n::Integer) = d >= 0 ? n - Int(d) : -Int(d)

attr(op::Op, k, default=nothing) = get(op.attrs, k, default)
# `Vector{Int}` is what `plainattr` already produced at load time, so return it
# untouched rather than allocating a copy per call.
ints(x::Vector{Int}) = x
ints(x) = x isa AbstractVector ? Int.(x) : Int(x)

"""
    value(ctx, id)

Resolve a buffer, materialising views lazily and recursively.
"""
function value(ctx::Ctx, id::AbstractString)
    haskey(ctx.values, id) && return ctx.values[id]
    b = get(ctx.graph.buffers, id, nothing)
    b === nothing && error("unknown buffer $id")
    b.kind === :view || error("buffer $id of kind $(b.kind) was never produced")
    v = makeview(ctx, b)
    ctx.values[id] = v
    v
end

"""
    contiguous(a)

Materialise a permuted view. Torch's `reshape`/`view` on a non-contiguous tensor
copies; Julia's `reshape` instead stacks a second lazy wrapper, and the resulting
`ReshapedArray{PermutedDimsArray{...}}` is not recognised as a GPU array by
either backend — `Adapt`'s wrapper union is one level deep, so broadcast falls
back to `DefaultArrayStyle` and scalar-indexes from the host. Collapsing it here,
once, with a real device-side `permutedims` keeps the nesting from ever forming.
"""
contiguous(a) = a
contiguous(a::PermutedDimsArray{T,N,perm}) where {T,N,perm} =
    permutedims(parent(a), perm)

"""
    contiguous(ctx, id, a)

`contiguous` into the slot the planner reserved for view `id`.

These copies are the largest thing left outside the plan — 51 of them, 290 MB,
on every SAM 2 encode — and they were invisible to `planslab` because a view is
not an op output. `materialisedview` names them, so `dest` has a slot to give;
where it has not (`plan === nothing`, or a shape the reservation does not cover)
this falls back to allocating exactly as before.
"""
contiguous(ctx::Ctx, id::AbstractString, a) = contiguous(a)
function contiguous(ctx::Ctx, id::AbstractString, a::PermutedDimsArray{T,N,perm}) where {T,N,perm}
    d = dest(ctx, id, T, size(a)...)
    permutedims!(d, parent(a), perm)
    d
end

"""View ops that only reinterpret the shape, leaving the element order alone."""
const SHAPEONLY = ("view.default", "_unsafe_view.default", "unsqueeze.default",
                   "squeeze.dims", "squeeze.dim")

"""
    lazyreshape(bc, dims) -> Broadcasted | nothing

`reshape` of an unevaluated elementwise expression, without evaluating it: the
same expression over reshaped operands.

Shape-only views are where fusion used to stop. 82 of the 344 elementwise ops
here are read through a `view`/`unsqueeze` and nothing else, so leaving them
lazy was pointless — `makeview` would hand the `Broadcasted` to `reshape`, which
has no method for it. Reshaping the operands instead is exact whenever every
operand already has the result's shape, because then the expression is a plain
elementwise map and reshaping cannot disturb which elements meet.

`nothing` when some operand is *smaller* than the result (a scalar-like axis
being broadcast): there the element at linear index `i` of the result does not
come from index `i` of that operand, and reshaping the two independently would
silently pair up the wrong elements. The caller materialises instead.
"""
reshapable(sz, x) = true
reshapable(sz, x::AbstractArray) = size(x) == sz
reshapable(sz, x::Base.Broadcast.Broadcasted) = all(a -> reshapable(sz, a), x.args)

relazy(dims, x) = x
relazy(dims, x::AbstractArray) = reshape(x, dims)
relazy(dims, bc::Base.Broadcast.Broadcasted) =
    Base.Broadcast.broadcasted(bc.f, map(a -> relazy(dims, a), bc.args)...)

function lazyreshape(bc::Base.Broadcast.Broadcasted, dims)
    sz = size(bc)
    prod(dims) == prod(sz) || return nothing
    reshapable(sz, bc) || return nothing
    Base.Broadcast.instantiate(relazy(dims, bc))
end

function makeview(ctx::Ctx, b::Buffer)
    parent = value(ctx, b.of)
    op = b.viewop
    a = b.attrs
    if occursin("getitem", op)
        return parent[Int(a["arg1"]) + 1]
    elseif op in SHAPEONLY
        shp = evalshape(b.shape, ctx.dims)
        if parent isa Base.Broadcast.Broadcasted
            r = lazyreshape(parent, shp)
            r === nothing || return r
            parent = Base.materialize(parent)
        end
        return reshape(contiguous(ctx, b.id, parent), shp)
    elseif op == "permute.default"
        perm = ints(a["arg1"])
        n = length(perm)
        # torch permutes dims of the un-reversed shape; reverse both the order
        # of the permutation and the indices it names
        jperm = ntuple(i -> n - perm[n - i + 1], n)
        return PermutedDimsArray(parent, jperm)
    elseif op in ("t.default", "transpose.int")
        n = ndims(parent)
        d1 = op == "t.default" ? 0 : Int(a["arg1"])
        d2 = op == "t.default" ? 1 : Int(a["arg2"])
        perm = collect(1:n)
        perm[jdim(d1, n)], perm[jdim(d2, n)] = perm[jdim(d2, n)], perm[jdim(d1, n)]
        return PermutedDimsArray(parent, Tuple(perm))
    elseif op == "slice.Tensor"
        # The end index can be a symbolic size, which does not survive as an
        # attr, so take the extent from the view's own declared shape - that is
        # authoritative and already resolved. Using the parent's extent instead
        # silently returns the whole (possibly padded) axis.
        n = ndims(parent)
        d = jdim(Int(get(a, "arg1", 0)), n)
        lo = Int(get(a, "arg2", 0))
        # torch clamps a negative start into range rather than wrapping past the
        # front: `x[-2:]` on a length-1 axis is the whole axis, not index -1.
        # Without the clamp this indexes `0:0` and throws a BoundsError several
        # ops later, where the slice is no longer visible.
        lo = lo < 0 ? clamp(size(parent, d) + lo, 0, size(parent, d)) :
                      min(lo, size(parent, d))
        len = evalshape(b.shape, ctx.dims)[d]
        idx = ntuple(i -> i == d ? ((lo + 1):(lo + len)) : Colon(), n)
        return view(parent, idx...)
    elseif op == "select.int"
        n = ndims(parent)
        d = jdim(Int(a["arg1"]), n)
        i = Int(a["arg2"])
        i = i < 0 ? size(parent, d) + i : i
        idx = ntuple(k -> k == d ? (i + 1) : Colon(), n)
        return view(parent, idx...)
    elseif op == "expand.default"
        # replicate the extent-1 axes up to the declared shape
        target = evalshape(b.shape, ctx.dims)
        c = contiguous(parent)
        p = ndims(c) < length(target) ?
            reshape(c, size(c)..., ntuple(_ -> 1, length(target) - ndims(c))...) : c
        all(i -> size(p, i) == target[i], 1:length(target)) && return p
        return repeat(p, ntuple(i -> size(p, i) == target[i] ? 1 : target[i], length(target))...)
    elseif op in ("alias.default", "detach.default")
        return parent
    elseif op == "split_with_sizes.default"
        n = ndims(parent)
        sizes = ints(a["arg1"])
        d = jdim(Int(get(a, "arg2", 0)), n)
        offs = 0
        out = Any[]
        for s in sizes
            idx = ntuple(i -> i == d ? ((offs + 1):(offs + s)) : Colon(), n)
            push!(out, view(parent, idx...))
            offs += s
        end
        return Tuple(out)
    end
    error("unhandled view op $op for $(b.id)")
end

"""
    alloc(ctx, id, dims...) -> array

Output storage on the execution backend, with the dtype the graph declares.
Nothing allocates a host `Array` directly - every buffer an op writes has to
live where the kernel runs.
"""
# Both forms go through the static plan: `alloc` is what ops already call, so
# routing it here converts every one of them at once rather than op by op.
alloc(ctx::Ctx, id::AbstractString, dims::Integer...) =
    dest(ctx, id, ctx.graph.buffers[id].dtype, dims...)

alloc(ctx::Ctx, ::Type{T}, dims::Integer...) where {T} =
    dest(ctx, ctx.outid[], T, dims...)

"""
    tupledest(ctx, i, T, dims...) -> array

Storage for element `i` (0-based) of the multi-output op currently running.

`planslab` places those elements under `"<buffer>.<i>"` because the tuple itself
has no single shape to reserve. A handler that allocates its result any other way
— `KernelAbstractions.allocate`, or an ordinary broadcast that calls `similar` —
gets a fresh device buffer on every call, which is what drove SAM 2's encoder to
free and re-reserve ~9 GB per frame through Lava's OOM-reclaim path.
"""
@inline tupledest(ctx::Ctx, i::Integer, ::Type{T}, dims::Integer...) where {T} =
    dest(ctx, string(ctx.outid[], '.', i), T, dims...)

"""
    tupledtype(ctx, i, default) -> Type

The dtype the graph declares for element `i` (0-based) of the running op's tuple
result, or `default` when it declares none.

Asking matters because `dest` only hands out a planned slot when the request
*fits* it, and the slot was reserved from the declared dtype. Under autocast
these outputs are fp16 while the handlers naturally allocate in their
accumulator type, fp32 — twice the bytes, so every one of them missed its slot
and fell back to a fresh allocation. That is 48 attentions and 96 layer norms per
encode landing outside the static plan, which is what put a gigabyte in the
recycler and Lava's pool into its reclaim path. Writing in the declared type is
also what the reference does: PyTorch's autocast returns fp16 from both.
"""
@inline function tupledtype(ctx::Ctx, i::Integer, default)
    b = get(ctx.graph.buffers, ctx.outid[], nothing)
    b === nothing && return default
    dts = get(b.attrs, "dtypes", nothing)
    (dts === nothing || length(dts) < i + 1 || dts[i + 1] === nothing) && return default
    dts[i + 1]
end

sync(ctx::Ctx) = KernelAbstractions.synchronize(ctx.backend)

"""
    allocate!(ctx, id) -> Array

Output storage for an op. Individually allocated for now; `arena.jl` replaces
this with offsets into one block once the intervals are packed.
"""
function allocate!(ctx::Ctx, id::AbstractString)
    b = ctx.graph.buffers[id]
    rawalloc(ctx, b.dtype, Dims(evalshape(b.shape, ctx.dims)))
end

"""
    coerce(value, buffer)

Force an op result to the dtype the exported graph declares for it.

Under autocast - which is what MatAnyone2 ships - the dtype of every buffer is
part of the graph: conv and matmul are fp16, reductions and `cat` stay fp32, and
the boundaries are explicit `_to_copy` nodes. Julia's own promotion rules do not
know that policy (mixing an fp32 norm weight with an fp16 activation promotes to
fp32 and would silently keep the whole tail in fp32), so the declared dtype is
applied after every op. That makes the reference's precision decisions
authoritative rather than something we re-derive.
"""
coerce(v, b::Buffer) = v
function coerce(v::AbstractArray, b::Buffer)
    (isempty(b.shape) && ndims(v) > 0) && return v
    eltype(v) === b.dtype ? v : b.dtype.(v)
end
# A value left lazy by `emit` carries no storage yet, so there is nothing to
# convert; `fusableset` only defers across ops that already agree on dtype, so
# the declared precision is preserved by construction.
coerce(v::Base.Broadcast.Broadcasted, ::Buffer) = v

"""
    execute!(graph, inputs, weights; dims, backend) -> Dict

Run every op in order. Returns the full value table so the verification pass can
diff any intermediate, not just the outputs.
"""
function execute!(graph::Graph, inputs::AbstractDict, weights::AbstractDict;
                  dims, backend=KernelAbstractions.CPU(),
                  overrides::AbstractDict=Dict{String,Any}(),
                  slab=nothing, plan=nothing, ws=nothing, lazy=nothing, rec=nothing)
    ctx = Ctx(Dict{String,Any}(), graph, dims, backend, slab, plan, Ref(""), ws, lazy, rec)
    for id in graph.order
        b = graph.buffers[id]
        if b.kind === :weight
            haskey(weights, b.key) || error("missing weight $(b.key)")
            ctx.values[id] = weights[b.key]
        elseif b.kind === :external && haskey(inputs, id)
            ctx.values[id] = inputs[id]
        elseif b.kind === :host
            ctx.values[id] = evalexpr(String(b.attrs["expr"]), dims)
        end
    end
    for (i, op) in enumerate(graph.ops)
        # `overrides` pins an op's result, which the verifier uses to isolate a
        # layer downstream of a tie-broken predicate
        if haskey(overrides, op.out)
            ctx.values[op.out] = overrides[op.out]
            continue
        end
        try
            ctx.outid[] = op.out          # tells `dest` which slab slot to hand out
            reset!(ctx.ws)                # kernel scratch does not outlive its op
            ctx.values[op.out] = coerce(timeop!(ctx, op), graph.buffers[op.out])
        catch e
            e isa MethodError && e.f === runop! &&
                error("op $(i)/$(length(graph.ops)) `$(op.aten)` (id $(op.id)) has no method")
            rethrow()
        end
    end
    # No sync here. `synchronize` is a full flush-and-wait, and nothing between
    # graphs needs it: a step chains eight graphs whose results stay on the
    # device, the backend keeps its own dispatches ordered, and any host read
    # (`Array`, a scalar transfer) synchronises implicitly. Syncing per graph
    # cost eight blocking round-trips per step, which showed up as ~12% of wall
    # time in `cuStreamSynchronize` plus the task machinery around it.
    #
    # Callers that need the results on the host synchronise themselves; `matte`
    # does so by copying the alpha back.
    ctx.values
end

runop!(ctx::Ctx, op::Op, ::Val{T}) where {T} =
    error("unimplemented ATen op `$(op.aten)` (id $(op.id))")

"""
    OPTIMES :: Union{Nothing,Dict{String,Tuple{Int,Float64}}}

Set to a dict to make `execute!` synchronise around every op and accumulate
`aten name => (count, milliseconds)`. Off by default and free when off.

This is deliberately a *serialising* measurement — it is the only kind that
attributes device time to a source-level op — so the totals run longer than the
step does. On this model the difference is small: removing every barrier from a
step only buys ~3 ms of 46, i.e. the dispatches barely overlap anyway.
"""
const OPTIMES = Ref{Any}(nothing)

"""
    OPDOUBLEFILTER :: Ref{Any}

Narrow `OPDOUBLE` to a subset of an op family: `(ctx, op) -> Bool`, or `nothing`
for all of them.

Attribution has to happen *in situ*. Standalone convolution microbenchmarks on
this setup are unusable — identical code timed one shape at 16 us and 116 us in
consecutive runs — while doubling an op inside a captured, replayed step has been
stable all along, because the measurement is a whole step and the perturbation is
the only thing that changes. This makes that technique reach a single shape
instead of a whole family, which is what per-shape convolution cost needs.
"""
const OPDOUBLEFILTER = Ref{Any}(nothing)

@inline function opdoublewanted(ctx::Ctx, op::Op)
    f = OPDOUBLEFILTER[]
    f === nothing && return true
    return f(ctx, op)::Bool
end

@inline function timeop!(ctx::Ctx, op::Op)
    t = OPTIMES[]
    if t !== nothing
        KernelAbstractions.synchronize(ctx.backend)
        t0 = time_ns()
        r = runop!(ctx, op, op.tag)
        KernelAbstractions.synchronize(ctx.backend)
        n, ms = get(t, op.aten, (0, 0.0))
        t[op.aten] = (n + 1, ms + (time_ns() - t0) / 1e6)
        return r
    end
    # `isempty` first: the string compare is not free 640 times a step.
    d = OPDOUBLE[]
    if !isempty(d) && (d == "*" || op.aten == d) && opdoublewanted(ctx, op)
        runop!(ctx, op, op.tag)
    end
    runop!(ctx, op, op.tag)
end

"""
    OPDOUBLE :: String

Name of an ATen op to run *twice* per step. The step's wall time then grows by
that op's real cost, which is the only way to attribute device time here without
the measurement swamping what it measures: syncing around each op costs ~0.4 ms
a time and buries a 640-op step under 240 ms of its own barriers, and per-
dispatch timestamps serialise and inflate just as badly. Running an idempotent
op a second time perturbs nothing — the result is identical — and the difference
is measured on an otherwise untouched step.

`"*"` matches every aten, which is only useful together with `OPDOUBLEFILTER`:
it costs a *set* of ops spanning several atens in one measurement instead of one
run per aten, so the answer carries one run's error rather than the sum of eight.
"""
const OPDOUBLE = Ref{String}("")
