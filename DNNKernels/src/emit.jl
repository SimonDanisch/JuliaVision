"""
Lowering an ATen graph into a Mantle graph.

The exported graph is at torch's granularity: one op per `aten::` call, tensors
with shapes, views declared as buffers naming their parent. A Mantle graph is at
KERNEL granularity: one pass per dependent stage, a dispatch per launch, and
every byte a resource the placer owns. This file is the lowering between them,
and it is why this package has no allocator of its own.

Nothing here executes. `emitgraph` walks the ops declaring what each one will do,
`Plan` then runs all seven of Mantle's phases — `Dag`, `Schedule`, `Liveness`,
`Place`, `Aliasing`, `Barriers`, `Pipelines` — and `run!` walks or replays the
result. An op's body says `dispatch!(p, kernel, args, ndrange)` rather than
launching a kernel itself, which is the whole of what the 88 `emitop!` methods
are.

Three things follow from declaring rather than allocating:

  * the intermediates are laid out by `Liveness`/`Place`/`Aliasing`, which see
    the whole graph before a single byte is touched.
  * op-internal scratch appears in no ATen graph (the export is at torch
    granularity and a transposed copy of `q` is our kernel's business, not the
    model's), so lowering is where it becomes a declaration: a transient like
    any other, aliased against the rest of the graph and not only within its op.
  * a declared view is a `Mantle.ResourceView`, the descriptor of parent, offset
    and strides, and not a Julia wrapper something has to walk back down to
    recover them from.
"""

"""
What an op declares into, and what it declares against.

`res` maps an ATen buffer id to its Mantle resource — a `Transient.Buffer` for an
intermediate, a `Buffer` for anything that outlives the plan, a plain value for a
host scalar. A multi-output op's results are keyed `"\$(id)#\$(i)"`, since the
graph gives the op one output id and the tuple elements need one resource each.
"""
struct EmitCtx{G,D}
    aten::Graph
    g::G                            # the Mantle graph being built
    dev::D
    dims::NamedTuple
    res::Dict{String,Any}
    esc::Set{String}
    # Which op is emitting, so `dest` can answer without being passed the op.
    outid::Base.RefValue{String}
    # Every `Mantle.Buffer` `make` allocated here, so `freeowned!` can give them
    # back. A transient belongs to the plan and `Mantle.free!(plan)` returns it;
    # an OWNED buffer belongs to nobody, and `Mantle.free!(::Buffer)` is
    # "explicit, and still never called for you: skipping it is a leak the pool
    # can report" — one escaping buffer per emit. Recorded rather than recovered
    # by scanning `res` for the type: `res` also holds the caller's resident
    # weights and host scalars, and
    # the emit knows what it allocated.
    owned::Vector{Any}
end

"""
    emitgraph(dev, aten, weights, dims; keepall = false, skip = ())

Declare `aten` into a fresh Mantle graph. Runs nothing.

`keepall` gives EVERY buffer storage of its own instead of a transient the placer
may alias, which is what layer-by-layer verification needs: an intermediate's
value after the whole plan has run is not its value at its own op unless nothing
else was allowed to reuse those bytes. It costs the sum of every buffer rather
than the placer's peak (5.68 GiB against ~1 for SAM 2's encoder at 1024x1024),
so it is for `verifygraph` and not for running a model.

`skip` names ops NOT to emit, for the same caller: `verifygraph` pins a buffer
whose predicate flipped to the reference value and re-runs, so the layers
downstream are still checked strictly instead of drowning in the consequence. A
skipped op's output buffer still exists — `declare!` made it — and the caller
writes it before the run.
"""
function emitgraph(dev, aten::Graph, weights::AbstractDict, dims::NamedTuple;
                   keepall::Bool = false, skip = ())
    g = M.Graph(dev)
    live = consumedids(aten; all = keepall)
    esc = keepall ? live : escaping(aten)
    emitctx = EmitCtx(aten, g, dev, dims, Dict{String,Any}(), esc, Ref(""), Any[])
    shapes = resultshapes(aten)
    # A REFUSAL must not leak what it had already allocated. `declare!` gives
    # every escaping buffer storage before the first op is emitted, so an op
    # without an `emitop!` -- the refusal this path exists to give -- throws with
    # the whole graph's owned buffers already allocated and no context in the
    # caller's hands to free them from. Rethrown unchanged: the error is the
    # answer, and only the cleanup is added.
    try
        for id in aten.order
            declare!(emitctx, aten.buffers[id], weights, live, shapes)
        end
        for op in aten.ops
            op.out in skip && continue
            emitctx.outid[] = op.out
            emitop!(emitctx, op, op.tag)
        end
    catch
        freeowned!(emitctx)
        rethrow()
    end
    # An OUTPUT that is a view is resolved by nobody else.
    #
    # A view is materialised on demand, by the op that reads it -- and the
    # caller's read is not an op. SAM 2's decoder returns `slice_2` and
    # `slice_3`, two windows onto the mask stack that nothing inside the graph
    # touches, so `planfor` looked them up in `res` and got a `KeyError`.
    # Resolved here, a shape-only output view costs a descriptor and a
    # materialised one gets its fill pass like any other.
    try
        for id in aten.outputs
            operand(emitctx, id)
        end
    catch
        freeowned!(emitctx)
        rethrow()
    end
    return g, emitctx
end

"""
    resultshapes(aten) -> Dict{String,Any}

The shape of each element of a multi-output result, taken from the `getitem`
view that reads it.

A `getitem` is the identity on the bytes it names, so a view and the element it
names are the same tensor and their two shapes are one fact written twice. They
disagree in one direction only, and both Kokoro exports do it in 119 buffers
between them, against 1317 that agree across every other model. The VIEW is
symbolic and the op's `shapes` metadata holds that symbol evaluated at the trace.
`[1, "t", 512]` against `[1, 30, 512]`, `[1, 128, "120*f + 1"]` against
`[1, 128, 11761]`. It is not one op's quirk either: in those two graphs it is
every `native_layer_norm`, every `_native_batch_norm_legit`, every sdpa and
every `lstm.input`.

The view is the shape that survives because it is the more general statement and
because it is what every consumer indexes with: `viewfor` answers a `getitem`
with the PARENT's resource, so a resource that does not match the view is one
nothing can read correctly. Declared from the metadata instead, kokorotext plans
30 columns for whatever `t` it is called with, and silently: the extents are
concrete, so nothing downstream has a symbol left to disagree with.

Two views of one element that disagree are refused rather than ordered. So is a
view whose RANK differs from the metadata's, which is a disagreement about what
the tensor IS rather than about how long one axis is.
"""
function resultshapes(aten::Graph)
    out = Dict{String,Any}()
    for (_, b) in aten.buffers
        (b.kind === :view && occursin("getitem", b.viewop)) || continue
        key = "$(b.of)#$(Int(b.attrs["arg1"]))"
        prev = get(out, key, nothing)
        prev === nothing || collect(prev) == collect(b.shape) || error(
            "DNNKernels: two `getitem` views of `$key` give it different " *
            "shapes, $(prev) and $(b.shape). They name the same bytes, so one " *
            "is wrong and nothing here can tell which.")
        out[key] = b.shape
    end
    return out
end

"""The shape to declare element `key` with: its reader's, checked against the
export's own metadata for rank."""
function resultshape(shapes::Dict{String,Any}, key::AbstractString, meta)
    s = get(shapes, key, nothing)
    s === nothing && return meta
    length(s) == length(meta) || error(
        "DNNKernels: `$key` is declared $(collect(meta)) by the op that " *
        "produces it and read as $(collect(s)) by its `getitem`. Those are " *
        "different ranks, so they are not the same tensor described twice.")
    return s
end

"""
One buffer's resource.

The kinds are the export's, and the decision each one makes is Mantle's:

  * `:weight` and `:host` are values that already exist — the safetensor on the
    device, and a scalar evaluated from `dims`.
  * `:external` is written by the caller and `:view` is resolved on demand
    (`viewfor`), through the parent it names.
  * everything else is an intermediate, and an intermediate is a TRANSIENT unless
    it escapes. An escaping buffer is read after the plan has run — by the next
    graph, or by the caller reading an output — so it cannot be memory the placer
    may hand to something else at its last use.
"""
function declare!(emitctx::EmitCtx, b::Buffer, weights::AbstractDict,
                  live::Set{String}, shapes::Dict{String,Any})
    b.kind === :view && return                      # `viewfor`, on demand
    if b.kind === :weight
        haskey(weights, b.key) || error("missing weight $(b.key)")
        emitctx.res[b.id] = weights[b.key]
        return
    end
    if b.kind === :host
        emitctx.res[b.id] = evalexpr(String(b.attrs["expr"]), emitctx.dims)
        return
    end
    # A multi-output op declares `shapes`/`dtypes` instead of `shape`/`dtype`;
    # each element gets its own resource, and an element the runtime never reads
    # (sdpa's philox seed and offset) has `nothing` for a dtype and gets none.
    if haskey(b.attrs, "shapes")
        for (i, (shape, T)) in enumerate(zip(b.attrs["shapes"], b.attrs["dtypes"]))
            (shape === nothing || T === nothing) && continue
            key = "$(b.id)#$(i - 1)"
            key in live || continue
            emitctx.res[key] = make(emitctx, b.id, T,
                                    evalshape(resultshape(shapes, key, shape),
                                              emitctx.dims))
        end
        return
    end
    b.id in live || return
    emitctx.res[b.id] = make(emitctx, b.id, b.dtype, evalshape(b.shape, emitctx.dims))
end

"""
    consumedids(aten; all = false) -> Set{String}

Every buffer, and every element of a multi-output buffer, that something in the
graph consumes: an op's input, a view's parent, or a declared output.

`all = true` answers every id instead, which is `emitgraph`'s `keepall`: the same
set in the same shape (the `"\$(id)#\$(i)"` keys included), so one function
decides what a buffer is called in both modes.

A declaration needs this and an interpreted run did not. torch's schema returns
four values from flash attention and two from `max_pool2d_with_indices`, and a
graph reads one of each — so the export declares buffers nothing touches, and
running lazily meant an unread one simply never got allocated. Declared, a
transient is placed from its USE, so one with none has nothing to place and
Mantle refuses it by name (measured: 102 of SAM 2's encoder's 1301).

`dropdead` prunes dead OPS and cannot see this: the op is live, one of its
results is not.

An `:external` buffer is always consumed — it is an input, and a graph that
ignores one still has to be handed it.
"""
function consumedids(g::Graph; all::Bool = false)
    live = Set{String}(g.outputs)
    for o in g.ops, i in o.ins
        push!(live, i)
    end
    for (_, b) in g.buffers
        all && push!(live, b.id)
        b.kind === :external && push!(live, b.id)
        if all && haskey(b.attrs, "shapes")
            for i in eachindex(b.attrs["shapes"])
                push!(live, "$(b.id)#$(i - 1)")
            end
        end
        b.kind === :view || continue
        push!(live, b.of)
        # A `getitem` names ONE element of a tuple, which is the only way an
        # element is read; the bare id would be the whole tuple and nothing takes
        # that.
        occursin("getitem", b.viewop) &&
            push!(live, "$(b.of)#$(Int(b.attrs["arg1"]))")
    end
    # An op whose RESULT IS ONE OF ITS INPUTS declares no storage of its own —
    # the emit registers the input under the output's id. Declaring a buffer as
    # well would leave it unread, which `Liveness` refuses, and copying into it
    # would leave the caller's bytes unwritten.
    for o in g.ops
        isaliasing(o) && delete!(live, o.out)
    end
    return live
end

"""
    isaliasing(op) -> Bool

Whether this op's result IS one of its inputs rather than a new value.

Two shapes, both from `foldcacheupdate`, which rewrites a KV cache update once
it has proved the cache slice is a view of a graph input nothing reads first:

  * an `index_put` marked `inplace`, which writes THROUGH its `self`;
  * an `alias.default` OP, which is what the `cat` it folded becomes — the
    stack is the identity on the tensor the puts wrote.

`alias.default` is also a VIEW op (`SHAPEONLY_VIEWS`), and that is the same
statement from the buffer side rather than a second rule: a view of a buffer and
an op that returns its input both name bytes someone else owns.
"""
isaliasing(op::Op) =
    op.aten == "alias.default" ||
    (op.aten == "index_put.default" &&
     get(op.attrs, "inplace", false) === true)

"""
    residentweights(dev, aten, weights) -> Dict{String,Any}

The weights this graph names, on `dev`.

`emitgraph` puts a weight straight into `res` as a kernel operand, so a host
array reaches the compiler as a non-bitstype argument: "Argument 4 to your kernel
function is of type `Tuple{Matrix{Float32}, …}`, which is not a bitstype". Every
caller has to upload, so the rule is here rather than repeated: `hoistconstants`
did not and `verifygraph` did, in its own words.

Only what THIS graph names, which is the selectivity `verifygraph` needs —
checking a two-block prefix of a 2.4 GB encoder must not upload the other thirty
blocks. Every `:weight` buffer rather than only the ones an op reads: a
buffer whose only reader is a VIEW still needs its resource, and `declare!` walks
`aten.order`, so it asks for all of them.

`toback` is the identity on something already resident, so this is a no-op on a
`Model`'s own dict.
"""
function residentweights(dev, aten::Graph, weights::AbstractDict)
    out = Dict{String,Any}()
    be = M.backend(dev)
    for (_, b) in aten.buffers
        (b.kind === :weight && !isempty(b.key)) || continue
        haskey(out, b.key) && continue
        haskey(weights, b.key) || continue     # `declare!` names the missing one
        out[b.key] = toback(be, weights[b.key])
    end
    return out
end

"""A resource of this shape: transient unless the id escapes or is written from
outside.

An EMPTY buffer is never a transient. A transient is memory the placer manages
and there is nothing here to manage — nothing can read an element of it and no
pass can be launched over it, so `Liveness` would refuse it as unused. torch
produces empty tensors legitimately (SAM 2's decoder concatenates a `(1, 0,
256)` on the branch with no boxes) and the emits skip the pass, so the
declaration still has to answer something: a zero-byte buffer of its own.
"""
function make(emitctx::EmitCtx, id::AbstractString, ::Type{T}, dims::Dims) where {T}
    b = emitctx.aten.buffers[id]
    isowned = prod(dims) == 0 || b.kind === :external ||
              id in emitctx.esc || id in emitctx.aten.outputs
    isowned || return M.Transient.Buffer(emitctx.g, T, dims)
    buf = M.Buffer(emitctx.dev, T, dims)
    push!(emitctx.owned, buf)
    return buf
end

"""
    freeowned!(emitctx) -> Int

Give every buffer this emit allocated back to the pool, and return how many.

`Mantle.free!(::Buffer)` is explicit by design and nothing finalises a `Buffer`,
so a caller that emits a graph and drops it leaks one region per escaping
buffer. Under `keepall` that is EVERY buffer in the graph, which is what
`declaredvalues` asks for: measured on a churn of 256 MB per round, a pool that
is freed grows 258 MB over twelve rounds and one that is not grows 6.2 GB over
twenty-four, every byte of it unreachable and unreclaimable. Repeated over a
parity sweep it reached 114 GB of system memory and the kernel killed the
desktop.

Idempotent by emptying the list, because a caller that frees and is then freed
again by a `finally` must not retire a region twice: `retire!` appends to the
pool's pending list and a second append releases bytes that already belong to
somebody else.

The transients are NOT here; they belong to the plan and `Mantle.free!(plan)`
returns them.
"""
function freeowned!(emitctx::EmitCtx)
    n = length(emitctx.owned)
    for b in emitctx.owned
        M.free!(b)
    end
    empty!(emitctx.owned)
    return n
end

"""
    viewfor(ctx, id) -> ResourceView

A view buffer as a descriptor over the resource that owns its bytes.

Cached in `res`, because a `ResourceView` has to be ONE object per buffer: `use`
interns by identity, and two descriptors for the same window would be two
resources with no hazard between them.

Only the shape-only views are handled by construction — `view`, `_unsafe_view`,
`unsqueeze`, `squeeze` — which reinterpret extents over the parent's bytes
starting at its first element. Anything else (a `slice`, a `permute`, a `select`)
changes where the elements are, so it needs its own storage and a pass to fill
it; `materialisedview` in `plan.jl` is the predicate that says which, and those
are declared as ordinary transients by the op that produces them. Throwing here
rather than guessing is deliberate: a wrong offset is silent.
"""
function viewfor(emitctx::EmitCtx, id::AbstractString)
    haskey(emitctx.res, id) && return emitctx.res[id]
    b = emitctx.aten.buffers[id]
    b.kind === :view || error("buffer $id is not a view")
    # A `getitem` is not a window onto anything: it names one element of a
    # multi-output result under the `"$(id)#$(i)"` key. So this is a lookup,
    # where `makeview` had to index the tuple the op had already returned.
    #
    # The parent is resolved FIRST when the key is not there yet, because a
    # multi-output result comes from two places: `declare!` makes the keys for an
    # OP's results from the export's `shapes`, and `splitview!` makes them for a
    # `split_with_sizes`, which is a VIEW and so is resolved on demand. A bare
    # lookup found MatAnyone's `readout_query` split absent — nothing had asked
    # for the split itself, only for its pieces.
    if occursin("getitem", b.viewop)
        key = "$(b.of)#$(Int(b.attrs["arg1"]))"
        haskey(emitctx.res, key) || viewfor(emitctx, b.of)
        haskey(emitctx.res, key) || error(
            "DNNKernels: `$(b.viewop)` ($(id)) names element " *
            "$(Int(b.attrs["arg1"])) of `$(b.of)`, which declared no resource " *
            "for it. A multi-output op declares one per entry of its `shapes`, " *
            "and an entry the export left as `nothing` gets none.")
        return (emitctx.res[id] = emitctx.res[key])
    end
    b.viewop == "split_with_sizes.default" &&
        return splitview!(emitctx, id, b, operand(emitctx, b.of))
    # The whole chain as one descriptor, asked BEFORE the parent is resolved,
    # because resolving it is what materialises every level in between.
    #
    # Two answers come out of it and they are the same question. A chain whose
    # composed strides are the CONTIGUOUS ones names a run of its root's memory,
    # so it is a `ResourceView` and costs nothing — that covers a shape-only
    # view, a trailing-axis `select.int` and anything else that happens to land
    # dense. This is what makes an in-place `index_put` possible: the KV cache
    # reaches it as `select.int(self_k, 0, i)`, torch's dim 0 is Julia's LAST, so
    # the window is a run and a write through it lands in the caller's cache.
    #
    # Anything else MOVES its elements and needs storage of its own, filled in
    # ONE pass off that root.
    s = stridedoperand(emitctx, id)
    if s !== nothing
        if s.strides == colstrides(s.dims)
            v = M.viewof(s.parent, s.dims; offset = s.offset)
            emitctx.res[id] = v
            return v
        end
        return materialise(emitctx, id, b, s)
    end
    parent = operand(emitctx, b.of)
    # A SHAPE-ONLY view of a parent whose own chain could not be described: the
    # parent is dense storage by the time it is resolved, so this is a
    # descriptor over it.
    if b.viewop in SHAPEONLY_VIEWS
        v = M.viewof(parent, evalshape(b.shape, emitctx.dims))
        emitctx.res[id] = v
        return v
    end
    # Anything else MOVES its elements, so it needs storage of its own and one
    # pass that fills it. `contiguous` did the permute case with `permutedims!`
    # at run time and left the rest as lazy Julia wrappers; a wrapper is not
    # something a kernel can be handed, and a resource is what the placer can
    # alias.
    #
    # Through `make`, so a view that ESCAPES is owned and not a transient, on the
    # same rule every other buffer is. `Transient.Buffer` unconditionally is what
    # was here, and SAM 2's decoder returns two materialised views (`slice_2` and
    # `slice_3`) that nothing inside the graph reads: two transients with no use
    # between them, which the placer correctly overlapped, so the mask's first
    # three elements came back holding the IoU scores. Silent -- the shapes and
    # dtypes were right and 196,605 of 196,608 elements were too.
    od = evalshape(b.shape, emitctx.dims)
    ast, off = viewstrides(emitctx, b, parent, od)
    out = make(emitctx, id, eltype(parent), od)
    M.dispatch!(emitctx.g, stridedcopy!, (out, od, parent, ast, off), prod(od);
                name = "$(id).$(first(split(b.viewop, '.')))")
    emitctx.res[id] = out
    return out
end

"""
    materialise(ctx, id) -> resource

View `id` with storage of its own, in ONE pass wherever the chain allows it.

`viewstrides` reads a view against its PARENT, so materialising a chain that way
materialises every level of it: `permute(permute(x))` is two copies of the same
elements. `s` is the chain composed into one descriptor by
[`stridedoperand`](@ref), and this is one pass off its root. 161 of SAM 2's
encoder copies were intermediate levels of a chain whose top could not be
described.
"""
function materialise(emitctx::EmitCtx, id::AbstractString, b, s::StridedOperand)
    out = make(emitctx, id, eltype(s.parent), s.dims)
    M.dispatch!(emitctx.g, stridedcopy!,
                (out, s.dims, s.parent, s.strides, s.offset), prod(s.dims);
                name = "$(id).$(first(split(b.viewop, '.')))")
    emitctx.res[id] = out
    return out
end

"""
    splitview!(emitctx, id, b, parent) -> Tuple

`split_with_sizes`, which is a VIEW in the export and several resources here: one
per piece, registered under `"\$(id)#\$(i)"` so the `getitem` above is a lookup
like any other multi-output result.

The pieces are consecutive boxes along one axis, so each is a `stridedcopy!` from
the parent at that piece's offset — the same pass a `slice` gets, which is what a
piece IS. Except along the LAST axis, where consecutive boxes are consecutive
bytes and a piece is a `viewof` with an offset and no pass at all; that is the
case MatAnyone's `readout_query` has.

Registering all the pieces at once, rather than one per `getitem`, is what makes
the offsets add up: each piece's offset is the sum of the extents before it, and
a lazy per-piece path would have to recompute that from the piece's own index.
"""
function splitview!(emitctx::EmitCtx, id::AbstractString, b::Buffer, parent)
    n = ndims(parent)
    # `intlist` and not `ints`: a split size can be SYMBOLIC. MatAnyone's
    # `readout_query` splits by a length the graph carries as a symbol, and
    # `Int("n")` is a `MethodError` about `Int64` several frames from the split.
    sizes = intlist(emitctx.dims, emitctx.res, b.attrs["arg1"])
    d = jdim(Int(get(b.attrs, "arg2", 0)), n)
    ast = colstrides(ntuple(k -> size(parent, k), n))
    sum(sizes) == size(parent, d) || error(
        "DNNKernels: `split_with_sizes` ($(id)) splits axis $d of $(size(parent)) " *
        "into $(sizes), which sums to $(sum(sizes)).")
    pieces = Any[]
    off = 0
    for (i, len) in enumerate(sizes)
        od = ntuple(k -> k == d ? len : size(parent, k), n)
        key = "$(id)#$(i - 1)"
        piece = if d == n
            # Contiguous: the pieces partition the trailing axis, so this is a
            # window over the parent's bytes and costs a descriptor.
            M.viewof(parent, od; offset = off * ast[d])
        else
            dst = make(emitctx, id, eltype(parent), od)
            M.dispatch!(emitctx.g, stridedcopy!,
                        (dst, od, parent, ast, off * ast[d]), prod(od);
                        name = "$(id).split$(i)")
            dst
        end
        emitctx.res[key] = piece
        push!(pieces, piece)
        off += len
    end
    out = Tuple(pieces)
    emitctx.res[id] = out
    return out
end

"""
    fromend(i, extent, id, op) -> i

A torch index, normalised: NEGATIVE counts from the end, as everywhere in
Python.

`select.int(0, -1)` is "the last element" and it is what MatAnyone's three
graphs use to read the last entry of an `arange`. Taken literally the offset is
`-1 * stride`, which puts `stridedcopy!` one element BEFORE the parent — it read
whatever was next to it in the pool, so the value was garbage that changed
between runs and between graphs (-4.29e37 in `readout_query`, 1.49e-5 in
`encode_mask_shallow`) and nothing failed. The interpreted path never had this:
Julia's `view` is handed `end`-relative indices already resolved by `makeview`.

Refuses rather than clamping an index still outside after normalising. A view
that starts outside its parent cannot be what the graph meant, and reading
adjacent memory is the failure this exists to stop.
"""
function fromend(i::Int, extent::Int, id, op)
    j = i < 0 ? i + extent : i
    0 <= j <= extent || error(
        "DNNKernels: view $(id) is a `$op` indexing $(i) into an axis of " *
        "extent $(extent), which is $(j) counted from the start. A view that " *
        "begins outside its parent would read whatever is next to it.")
    return j
end

"""
    viewstrides(emitctx, b, parent, od) -> (strides, offset)
    viewstrides(emitctx, b, ps, pst, od) -> (strides, offset)

How view `b` reads its parent: one parent stride per axis of the view's own
shape, and where the view starts.

Torch indexes the UN-REVERSED shape, so its axis `d` is Julia axis `n - d`
(`jdim`) and its indices are 0-based. Every conversion below is that one fact
applied to a different attribute, which is why they are together rather than one
per op.

`pst` is the parent's own strides. A materialising view reads a dense parent and
passes `colstrides(size(parent))`; [`stridedoperand`](@ref) composes a chain of
views by passing the strides it has already accumulated, which is the same
arithmetic over a parent that is itself strided.
"""
viewstrides(emitctx::EmitCtx, b, parent, od::Dims) =
    viewstrides(emitctx, b, size(parent), colstrides(size(parent)), od)

function viewstrides(emitctx::EmitCtx, b, ps::Dims, pst::Dims, od::Dims)
    n = length(ps)
    op = b.viewop
    if op == "permute.default"
        perm = ints(b.attrs["arg1"])
        length(perm) == n || error(
            "DNNKernels: view $(b.id) permutes $(length(perm)) axes of a " *
            "$(n)-d parent.")
        # Julia output axis `jo` is torch axis `n - jo`, whose parent axis is
        # `perm[n - jo + 1]`, which is Julia parent axis `n - perm[...]`.
        return (ntuple(jo -> pst[n - perm[n - jo + 1]], n), 0)
    elseif op == "expand.default"
        # A repeated axis has stride 0, which is exactly what a broadcast is.
        # Read off the SHAPES, so this one form needs the parent dense.
        pst == colstrides(ps) || error(
            "DNNKernels: view $(b.id) expands a parent that is itself strided, " *
            "which `bcstrides` cannot describe. Materialise the parent first.")
        return (bcstrides(od, ps), 0)
    elseif op == "slice.Tensor"
        jd = jdim(Int(b.attrs["arg1"]), n)
        start = fromend(Int(get(b.attrs, "arg2", 0)), ps[jd], b.id, op)
        step = Int(get(b.attrs, "arg4", 1))
        return (ntuple(k -> k == jd ? pst[k] * step : pst[k], n), start * pst[jd])
    elseif op == "select.int"
        jd = jdim(Int(b.attrs["arg1"]), n)
        i = fromend(Int(b.attrs["arg2"]), ps[jd], b.id, op)
        # The axis is DROPPED, so the view has rank n-1 and the axis's
        # contribution is a constant offset.
        keep = Tuple(k for k in 1:n if k != jd)
        return (ntuple(j -> pst[keep[j]], n - 1), i * pst[jd])
    end
    error("DNNKernels: view $(b.id) is a `$op`, which has no declared form. " *
          "A view that moves its elements needs its parent strides and offset " *
          "in `viewstrides`; a view that only reinterprets the shape belongs in " *
          "`SHAPEONLY_VIEWS`.")
end

"""
    unitaxisstrides(ps, pst, od) -> strides or nothing

`od`'s strides when it is `ps` with extent-1 axes dropped or inserted, which is
what `squeeze` and `unsqueeze` are.

An axis of extent 1 contributes nothing to an address, so those two move no
elements whatever the parent's strides are. `nothing` for any other reshape: a
genuine one is the same elements in the same order only over a dense parent.

Matched on the SHAPES rather than on the attribute, so `squeeze.dim`,
`squeeze.dims` and the argument-free spelling are one case.
"""
function unitaxisstrides(ps::Dims, pst::Dims, od::Dims)
    st = Int[]
    i = 1
    for j in eachindex(od)
        while i <= length(ps) && ps[i] != od[j] && ps[i] == 1
            i += 1
        end
        if i <= length(ps) && ps[i] == od[j]
            push!(st, pst[i])
            i += 1
        elseif od[j] == 1
            # An axis the output INSERTS: never indexed past 0, so the stride is
            # free and the next axis's keeps the descriptor readable.
            push!(st, i <= length(pst) ? pst[i] : 1)
        else
            return nothing
        end
    end
    while i <= length(ps)
        ps[i] == 1 || return nothing
        i += 1
    end
    return Tuple(st)
end

"""
The views whose mapping is an offset and a per-axis stride, so a kernel that
takes strides can read them in place. Anything else moves elements and is
materialised — see [`viewfor`](@ref).
"""
const STRIDEDVIEWS = ("permute.default", "slice.Tensor", "select.int")

"""
    stridedoperand(ctx, id) -> StridedOperand or nothing

`id` as a strided read of a resource, or `nothing` when some level of its view
chain cannot be described that way.

Composed level by level, each one through [`viewstrides`](@ref) over the strides
the level below already accumulated, so `permute(slice(qkv))` is one descriptor
and no pass. A SHAPE-ONLY level re-derives its strides from the new shape, which
is only the same elements in the same order when what it reshapes is dense.

`nothing` for a view already materialised (`res` has it, so that resource is the
answer) and for a parent this cannot describe — materialising the PARENT to
stride over it could copy more than the level being saved, so that decision
stays with `viewfor`.
"""
function stridedoperand(emitctx::EmitCtx, id::AbstractString)
    haskey(emitctx.res, id) && return nothing
    b = get(emitctx.aten.buffers, id, nothing)
    (b === nothing || b.kind !== :view) && return nothing
    (b.viewop in STRIDEDVIEWS || b.viewop in SHAPEONLY_VIEWS) || return nothing
    pb = get(emitctx.aten.buffers, b.of, nothing)
    pb === nothing && return nothing
    od = evalshape(b.shape, emitctx.dims)
    pd = pb.kind === :view ? stridedoperand(emitctx, b.of) : nothing
    if pd === nothing
        # A parent that is a view this cannot describe, and is not already
        # materialised, is where the chain stops. A `getitem` is not that: it
        # LOOKS UP one element of a multi-output result, so it answers with a
        # resource that is already there and dense, and is a root like an op's
        # own output. 99 of SAM 2's encoder copies are a permute over one.
        (pb.kind === :view && !haskey(emitctx.res, b.of) &&
         !occursin("getitem", pb.viewop)) && return nothing
        parent = operand(emitctx, b.of)
        isresource(parent) || return nothing
        root, ps, pst, poff = parent, size(parent), colstrides(size(parent)), 0
    else
        root, ps, pst, poff = pd.parent, pd.dims, pd.strides, pd.offset
    end
    if b.viewop in SHAPEONLY_VIEWS
        st = unitaxisstrides(ps, pst, od)
        st === nothing || return StridedOperand(root, od, st, poff)
        pst == colstrides(ps) || return nothing
        return StridedOperand(root, od, colstrides(od), poff)
    end
    st, off = viewstrides(emitctx, b, ps, pst, od)
    return StridedOperand(root, od, st, poff + off)
end

"""
    operand(ctx, id)

What an op reads for input `id`: a resource, a view of one, or a host value.
"""
function operand(emitctx::EmitCtx, id::AbstractString)
    haskey(emitctx.res, id) && return emitctx.res[id]
    b = get(emitctx.aten.buffers, id, nothing)
    b === nothing && error("unknown buffer $id")
    b.kind === :view && return viewfor(emitctx, id)
    error("buffer $id of kind $(b.kind) was never declared")
end

"""
One operand of an op by POSITION, which is not the same as by index into `ins`.

torch folds a scalar operand into the schema, so it arrives in `attrs` under its
positional slot and `ins` holds only the tensors, consumed in order. Either side
can be the scalar: `1 - sigmoid(x)` exports as
`sub.Tensor(ins = [sigmoid], arg0 = 1)`.

Indexing `op.ins` directly works only while every op reaching it has tensors on
both sides, and is a `BoundsError` the first time one does not. Same rule as
`runop!`'s `operand(ctx, op, pos)`, and `ARGKEY` is shared with it.
"""
function operand(emitctx::EmitCtx, op::Op, pos::Int)
    key = argkey(pos)
    haskey(op.attrs, key) && return numattr(emitctx.dims, op.attrs[key])
    idx = pos - count(p -> haskey(op.attrs, argkey(p)), 1:(pos - 1))
    idx <= length(op.ins) || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) has no operand at position $pos")
    return operand(emitctx, op.ins[idx])
end

"""    dest(ctx) -> resource

Where the op being emitted writes. `dest(ctx, i)` for element `i` of a
multi-output op, zero-based as the export numbers them.

Refuses by NAME when the buffer has no resource, because a bare `KeyError` on an
op id says nothing about why. Only one thing causes it: `declare!` gives storage
to the buffers something CONSUMES (`consumedids`), so an op writing one nothing
reads has no destination — and such an op is dead. `dropdead` removes those, and
the driver runs it before planning; a graph reaching here with one has not been
through that pass.
"""
function dest(emitctx::EmitCtx, i::Union{Nothing,Integer} = nothing)
    key = i === nothing ? emitctx.outid[] : "$(emitctx.outid[])#$(i)"
    haskey(emitctx.res, key) || error(
        "DNNKernels: op `$(emitctx.outid[])` writes `$key`, which nothing in " *
        "the graph consumes, so it has no destination. That op is dead and " *
        "`dropdead` is the pass that removes it.")
    return emitctx.res[key]
end

"""Every declared element of a multi-output op's result, in order."""
dests(emitctx::EmitCtx, n::Integer) = ntuple(i -> dest(emitctx, i - 1), n)

"""
    maybedest(emitctx, i) -> resource or nothing

Element `i` of a multi-output op's result where the export declared one.

torch returns four values from flash attention and two from
`max_pool2d_with_indices`, and a graph that reads only the first declares only
the first — `declare!` gives a resource to each buffer the graph HAS. The
interpreted path fabricated the others with `similar(out, 0)`, which is a real
allocation standing in for something nothing reads; `nothing` says the same
thing and costs nothing.
"""
maybedest(emitctx::EmitCtx, i::Integer) =
    get(emitctx.res, "$(emitctx.outid[])#$(i)", nothing)

"""
    destor(emitctx, i, T, dims) -> resource

Element `i` of a multi-output op's result, or a TRANSIENT of that shape where the
export declared none.

A kernel that writes a result needs somewhere to put it whether or not the graph
reads it: `layernorm_kernel!` always writes the mean and the reciprocal standard
deviation, and `bnstats!` always writes both of its. `maybedest` is for the
results an emit can simply not produce; this is for the ones a kernel produces
anyway, and then they are scratch the placer aliases like any other transient.

Written out rather than as `something(maybedest(...), scratch(...))`, because
`something` evaluates BOTH arguments: the scratch was declared even when the
export had a destination, and a transient nothing then passes to a pass is one
`Liveness` refuses by name. It took `verifygraph`'s `keepall` to fire -- there,
every result has a destination -- but the leak was not conditional on that: any
graph that reads a layer norm's mean would have hit it.
"""
function destor(emitctx::EmitCtx, i::Integer, ::Type{T}, dims::Dims) where {T}
    d = maybedest(emitctx, i)
    d === nothing || return d
    return scratch(emitctx, T, dims...)
end

"""
    scratch(ctx, T, dims...) -> TransientBuffer

An op's own working buffer, declared into the graph.

Scratch appears in no ATen graph: the export is at torch granularity, and a
split-K accumulator or a transposed copy of `q` is a property of our kernels.
Declared here, its liveness is whatever its uses say, so the placer aliases it
against the whole graph like any other transient rather than only within the op
that asked for it.
"""
scratch(emitctx::EmitCtx, ::Type{T}, dims::Integer...) where {T} =
    M.Transient.Buffer(emitctx.g, T, map(Int, dims))

"""The fallback, so an unported op says which one it is rather than failing four
frames down in `dispatch!`."""
emitop!(emitctx::EmitCtx, op::Op, ::Val{A}) where {A} = error(
    "DNNKernels.emitop!: no emit method for `$(op.aten)` (op $(op.id)). The op " *
    "declares its dispatches instead of launching them; see `emit.jl` for the " *
    "patterns and `runop!` for the host-side reference.")

# ── elementwise, as a dispatch ────────────────────────────────────────────────
#
# `runop!` wrote these as `emit(ctx, Base.broadcasted(f, xs...))`, which returned
# the `Broadcasted` itself when `fuse.jl` said the value was consumed exactly once
# — so the fused kernel was generated at the CONSUMER's launch, pairwise, by
# GPUArrays' broadcast machinery. That is why SAM 2's encoder showed one
# `fused.elementwise` op and 1014 broadcast dispatches.
#
# The fusion DECISION was always static (`fuseops` at model build). What was at
# run time was the codegen. Declared, there is nothing to defer: the group is
# known, so it emits one dispatch.
#
# The kernel is `ew!` from `kernels/elementwise.jl`: macro-free, ONE method over
# a tuple of operands, taking the output shape and each operand's effective
# strides as plain arguments. So broadcast is not GPUArrays' either: it is
# `bcindex` over strides computed at emit time from shapes the
# graph already states, which is what lets NeuralLUT's `(1, 1, 1, 1, 3)` factor
# meet its `(33, 33, 33, 3, 3)` LUT with no materialised copy and no second
# dispatch.

"""
    elementwise!(ctx, op, f, ins...) -> resource

One pass, one dispatch: `dest(ctx) .= f.(ins...)`.

`f` is an argument rather than a type parameter of a wrapper, so a closure over
scalars, leaky ReLU's slope or an epsilon, needs no operand of its own.

Nothing here says what it touches. `ew!` stores through its first argument and
reads through the operand tuple, and that is read off the kernel body by
`Mantle.argument_usage` — so the destination is written, the operands are only
read, and two ops that read the same weight do not serialise against each other.
"""
function elementwise!(emitctx::EmitCtx, op::Op, f, ins...)
    out = dest(emitctx)
    od = size(out)
    length(out) == 0 && return out      # see `mapbody!`
    ops, sts = operandtuples(od, ins)
    M.dispatch!(emitctx.g, ew!, (out, od, ops, sts, f), length(out); name = op.id)
    return out
end

"""
The operands and their effective strides, as the two tuples [`ew!`](@ref) walks
in step.

One kernel over any number of operands, rather than one per arity. What that
needs from core is a `::Tuple` method on `resolve` and `storage`, so a nested
tuple of resources reaches the kernel resolved, and `devicepointeroffsets` not
counting a tuple as a level of nesting, so a `resize!` under a recorded plan
finds the operands' addresses (`Mantle.nestinglevels`).
"""
operandtuples(od::Dims, ::Tuple{}) = ((), ())
function operandtuples(od::Dims, ins::Tuple)
    x = first(ins)
    isresource(x) || error(
        "DNNKernels: elementwise operand of type $(typeof(x)) has no bytes to " *
        "index. A host scalar belongs in the function, as `Base.Fix2(f, x)`, " *
        "rather than in the operand list; `binary!` is where that is decided.")
    ops, sts = operandtuples(od, Base.tail(ins))
    if x isa StridedOperand
        w = stridedwindow(x, od)
        w === nothing || return ((w[1], ops...), (w[2], sts...))
        error("DNNKernels: a strided operand of $(x.dims) cannot be read at an " *
              "output shape of $(od). `binary!` asks `strideview` for one only " *
              "where the ranks agree, so this is a caller that did not.")
    end
    return ((x, ops...), (bcstrides(od, size(x)), sts...))
end

"""
    stridedwindow(s, od) -> (resource, strides) or nothing

`s` as something [`ew!`](@ref) can index, and the strides to read it with.

`ew!` addresses each operand as `operand[bcindex(lin, od, strides)]` — a linear
index built from the OUTPUT coordinates — so an operand needs no shape of its
own, only a base and one stride per output axis. The base is a ONE-DIMENSIONAL
window over the root, which is what folds the descriptor's offset in without the
kernel taking an offset at all.

An axis of extent 1 against a wider output gets stride 0, the same as
`bcstrides` gives a dense operand. `nothing` where the ranks disagree or the
window would reach past the root, which is where the caller materialises
instead.
"""
function stridedwindow(s::StridedOperand, od::Dims)
    length(od) == length(s.dims) || return nothing
    st = ntuple(length(od)) do k
        s.dims[k] == od[k] ? s.strides[k] : s.dims[k] == 1 ? 0 : -1
    end
    any(<(0), st) && return nothing
    span = 1 + sum((od[k] - 1) * st[k] for k in eachindex(od); init = 0)
    s.offset + span <= length(s.parent) || return nothing
    return (M.viewof(s.parent, (span,); offset = s.offset), st)
end

"""
Whether an operand is MEMORY, as opposed to a host scalar.

Three things qualify and the third is easy to miss: a `Buffer` or a
`Transient.Buffer`, a `ResourceView` of one, and a plain device array, which is
what a WEIGHT is. `Model` uploads weights with `KA.allocate`, so they are not
Mantle resources; `use` interns them all the same, which is Mantle fencing bytes
it did not place.

Asking only about `Resource`/`ResourceView` said no to every weight, and the
consequence was not an error: `binary!` read that as "host scalar", bound a
3x3x33x33x33 LUT into a closure with `Fix2`, and the kernel failed to compile
with `unsupported call to jl_alloc_genericmemory_unchecked` — a broadcast inside
the kernel, four frames from anything naming a weight.
"""
isresource(x) = x isa M.Resource || x isa M.ResourceView || x isa AbstractArray
# A [`StridedOperand`](@ref) is memory too: a root, an offset into it and a
# stride per axis. `operandtuples` turns it into the window `ew!` indexes.
isresource(::StridedOperand) = true

"""
    binary!(ctx, op, f) -> resource

A two-operand op, with a scalar side bound into `f` if it has one.

`mul.Tensor` can be given a tensor and a number, and a kernel argument has to
have bytes to index, so the number goes into the function rather than into the
operand list. `Fix1`/`Fix2` put it in the closure's TYPE, so it costs no
argument and no memory.
"""
function binary!(emitctx::EmitCtx, op::Op, f0)
    # Strided where the operand is a view `ew!` can index in place: a residual
    # add reads a permute of its producer, and materialising that is a pass over
    # every element to read every element once.
    sa = strideview(emitctx, op, 1)
    sb = strideview(emitctx, op, 2)
    a = sa === nothing ? operand(emitctx, op, 1) : sa
    b = sb === nothing ? operand(emitctx, op, 2) : sb
    # The FOLDED ACTIVATION, composed into the same closure so it stays one pass.
    #
    # `foldrelu` folds a relu into `convolution.default` **or `add.Tensor`**, and
    # only the convolution and `addmm` emits read `act`. So every relu folded
    # into an add was dropped: the result was finite, the right shape and dtype,
    # and differed from the reference only where the sum was negative. Found by
    # `tools/two_route_parity.jl` on MatAnyone's `add_94` -- interpreted min 0.0
    # against declared min -2.68 -- and `foldrelu`'s own note says which adds
    # those are: "a residual block ends `add(conv, skip) -> relu`, so the relus
    # on the largest feature maps follow an *add*", 36 of them.
    #
    # Composed and not a second dispatch, because that is what folding bought.
    g = actfn(Symbol(get(op.attrs, "act", "none")))
    f = g === identity ? f0 : (x, y) -> g(f0(x, y))
    isresource(a) && isresource(b) && return elementwise!(emitctx, op, f, a, b)
    isresource(a) && return elementwise!(emitctx, op, Base.Fix2(f, b), a)
    isresource(b) && return elementwise!(emitctx, op, Base.Fix1(f, a), b)
    error("DNNKernels: `$(op.aten)` (op $(op.id)) has a host scalar on both " *
          "sides, so it is a constant and `constfold` should have removed it.")
end

# ── the ops ──────────────────────────────────────────────────────────────────

emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("mul.Tensor")}) = binary!(emitctx, op, *)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("div.Tensor")}) = binary!(emitctx, op, /)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("add.Tensor")}) = binary!(emitctx, op, +)

"""
`aten::clone`, which is a copy.

ONE pass, from wherever the elements are: a clone of a permuted view is a
`stridedcopy!` off the root, where materialising the view first and copying that
is two passes over the same bytes. SAM 2's encoder has 90 of these pairs.

Otherwise `identity` over one operand, which is the same dispatch as any other
elementwise op.
"""
function emitclone!(emitctx::EmitCtx, op::Op)
    s = strideview(emitctx, op, 1)
    out = dest(emitctx)
    (s === nothing || size(s) != size(out)) &&
        return elementwise!(emitctx, op, identity, operand(emitctx, op, 1))
    od = size(out)
    M.dispatch!(emitctx.g, stridedcopy!, (out, od, s.parent, s.strides, s.offset),
                prod(od); name = op.id)
    return out
end

emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("clone.default")}) =
    emitclone!(emitctx, op)

function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("leaky_relu.default")})
    x = operand(emitctx, op, 1)
    s = eltype(dest(emitctx))(something(get(op.attrs, "arg1", nothing), 0.01))
    # The slope is captured, so it travels in the closure and not as an operand.
    elementwise!(emitctx, op, v -> v >= zero(v) ? v : s * v, x)
end

# ── one function of one operand ──────────────────────────────────────────────
#
"""
The one operand of a unary op, strided where [`ew!`](@ref) can index it in place.
"""
function unaryoperand(emitctx::EmitCtx, op::Op)
    s = strideview(emitctx, op, 1)
    return s === nothing ? operand(emitctx, op, 1) : s
end

# From `UNARY_FUSED`, the same table `runop!` generated its methods from and
# `fusedfunc` reads to build a `FusedOp`. Read a third time here rather than
# listed again: two lists for one fact took about an hour to diverge the first
# time, when a fusion emitting `x -> inv(sqrt(x))` was not the `rsqrt.default`
# anybody had tested.
for (name, f) in UNARY_FUSED
    @eval emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol($name)}) =
        elementwise!(emitctx, op, $f, unaryoperand(emitctx, op))
end

"""
`aten::sub.Tensor(a, b, alpha)`, which is `a - alpha * b`.

`alpha` defaults to 1, and when it is 1 the multiply is not emitted at all --
the closure is `-` itself, so the kernel is the same one `add.Tensor` compiles.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("sub.Tensor")})
    k = alpha(op)
    return k == 1 ? binary!(emitctx, op, -) :
                    binary!(emitctx, op, (x, y) -> x - k * y)
end

emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("eq.Scalar")}) = binary!(emitctx, op, ==)
# `ge`/`le` in both spellings and `bitwise_and`, which is `logical_and` on a
# `Bool` tensor and the same kernel either way. `binary!` picks the arity: a host
# scalar on one side becomes a `Fix1`/`Fix2` in the closure and not an operand.
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("ge.Tensor")}) = binary!(emitctx, op, >=)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("ge.Scalar")}) = binary!(emitctx, op, >=)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("le.Tensor")}) = binary!(emitctx, op, <=)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("le.Scalar")}) = binary!(emitctx, op, <=)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("bitwise_and.Tensor")}) =
    binary!(emitctx, op, &)
# `gt`/`lt` against a scalar. `binary!` reads the operand positionally, so the
# scalar arrives from `attrs` and travels in the closure, not as an operand.
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("gt.Scalar")}) = binary!(emitctx, op, >)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("lt.Scalar")}) = binary!(emitctx, op, <)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("gt.Tensor")}) = binary!(emitctx, op, >)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("lt.Tensor")}) = binary!(emitctx, op, <)
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("logical_and.default")}) =
    binary!(emitctx, op, &)

"""
`aten::_to_copy`, a dtype conversion as one elementwise pass.

Which conversion is `castfn`'s. Both routes ask that one function, so neither
can round, truncate or saturate differently from the other.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("_to_copy.default")})
    a = operand(emitctx, op, 1)
    return elementwise!(emitctx, op, castfn(eltype(dest(emitctx)), eltype(a)), a)
end

"""
`aten::clamp` with either bound optional, and either bound possibly symbolic --
the iSTFT clamps an index against the frame count, so a bound becomes a function
of the sequence length once the graph is length-generic.

The bounds travel in the CLOSURE and not as operands: they are host scalars, and
a kernel argument has to have bytes to index. `clampbounds` keeps them in the
operand's own type when that type is an integer, because a `Float32` bound
promotes the result and storing that into an integer destination is a `convert`
with an `InexactError` path -- a throw inside a kernel, whose exception
allocation is a hostcall on AMDGPU and dead code the compiler still emits
everywhere else.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("clamp.default")})
    x = operand(emitctx, op, 1)
    l, h = clampbounds(eltype(x), emitctx.dims, get(op.attrs, "arg1", nothing),
                       get(op.attrs, "arg2", nothing))
    return elementwise!(emitctx, op, v -> clamp(v, l, h), x)
end

"""
`aten::where.self(cond, a, b)`, as one three-operand pass.

A zero VALUE is captured and not the type: a closure capturing `T` has a
`Type{Float32}` field, which is not isbits, and a kernel cannot take a
non-bitstype argument. Either branch may also be a host scalar, which `binary!`
handles for two operands and this does for three.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("where.self")})
    c = operand(emitctx, op, 1)
    a = operand(emitctx, op, 2)
    b = operand(emitctx, op, 3)
    z = zero(eltype(dest(emitctx)))
    isresource(a) && isresource(b) &&
        return elementwise!(emitctx, op, (p, x, y) -> ifelse(p, oftype(z, x), oftype(z, y)),
                            c, a, b)
    isresource(a) && return elementwise!(emitctx, op,
        (p, x) -> ifelse(p, oftype(z, x), oftype(z, b)), c, a)
    isresource(b) && return elementwise!(emitctx, op,
        (p, y) -> ifelse(p, oftype(z, a), oftype(z, y)), c, b)
    return elementwise!(emitctx, op, p -> ifelse(p, oftype(z, a), oftype(z, b)), c)
end

"""
`full`, `full_like` and `empty.memory_format`: a constant into the declared
buffer.

`empty` leaves the contents undefined in torch, so zeroing is a legal
implementation and the only one under which a graph that wrongly reads the
result fails the same way twice instead of intermittently. SAM 2's decoder uses
it for the `(1, 0, 256)` tensor concatenated onto the sparse embeddings when
there are no boxes, where there is nothing to fill either way.

The VALUE can be symbolic, not just the shape: a graph that materialises its own
sequence length writes `full((1,), t)`.
"""
function emitfill!(emitctx::EmitCtx, op::Op, v; name = op.id)
    out = dest(emitctx)
    # See `mapbody!`: an empty result needs no pass, and `empty.memory_format`
    # is where they come from.
    length(out) == 0 && return out
    M.dispatch!(emitctx.g, M.fill_kernel!, (out, convert(eltype(out), v)), length(out);
                name)
    return out
end

emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("full.default")}) =
    emitfill!(emitctx, op, numattr(emitctx.dims, op.attrs["arg1"]))
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("full_like.default")}) =
    emitfill!(emitctx, op, numattr(emitctx.dims, op.attrs["arg1"]))
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("empty.memory_format")}) =
    emitfill!(emitctx, op, 0)
# A 0-d tensor holding one number, which is the same fill over one element.
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("scalar_tensor.default")}) =
    emitfill!(emitctx, op, numattr(emitctx.dims, op.attrs["arg0"]))

"""
`aten::pow.Tensor_Scalar`, with the small integer exponents written out.

`x^2` as a multiply rather than a call to `pow` is not a micro-optimisation on a
GPU: `pow` is a library call with a branchy implementation, and the exponent is
a host scalar so the specialisation is free. Anything else goes through `^`.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("pow.Tensor_Scalar")})
    a = operand(emitctx, op, 1)
    e = operand(emitctx, op, 2)
    if e isa Real && isinteger(e)
        n = Int(e)
        n == 1 && return elementwise!(emitctx, op, identity, a)
        n == 2 && return elementwise!(emitctx, op, x -> x * x, a)
        n == 3 && return elementwise!(emitctx, op, x -> x * x * x, a)
        n == -1 && return elementwise!(emitctx, op, inv, a)
        return elementwise!(emitctx, op, x -> intpow(x, n), a)
    end
    return elementwise!(emitctx, op, Base.Fix2(^, e), a)
end

"""
`aten::gelu`, in whichever formulation the export asked for.

`approximate = "tanh"` selects the cheap one and torch's default is exact. Read
through `atenarg` because `approximate` is a keyword in almost every PyTorch
source that writes it, and picking the wrong formulation is a silent accuracy
change rather than an error. Both evaluate in `accum(T)` and round once, which is
what PyTorch does for a half tensor.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("gelu.default")})
    f = String(atenarg(op, 1, "approximate", "none")) == "tanh" ? gelutanh : geluexact
    return elementwise!(emitctx, op, f, operand(emitctx, op, 1))
end

"""`aten::copy_`'s functional form: the SOURCE is what lands in the
destination, and the first argument is only there to give the shape."""
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("copy.default")}) =
    elementwise!(emitctx, op, identity, operand(emitctx, op, 2))

"""
`fused.elementwise` — the group [`fuseops`](@ref) collapses a chain of
elementwise ops into.

Its `FusedOp` is a plain callable, so it needs no kernel of its own: it is the
`f` of one `ew!` dispatch over the group's operands. Passing it as an argument is
the function barrier that keeps the per-element call static -- it comes out of a
`Dict{String,Any}`, so a body that read it inline would dispatch dynamically once
per element.
"""
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("fused.elementwise")}) =
    elementwise!(emitctx, op, op.attrs["fused"], map(i -> operand(emitctx, i), op.ins)...)

# ── repeat, and a reduction ──────────────────────────────────────────────────

"""
`aten::repeat`, as one gather into the planned buffer.

The source is padded with trailing singleton axes to the output's rank, which is
the declaration-time form of the `reshape` loop `runop!` ran: torch prepends
singleton dims when the repeat spec is longer than the rank, and a prepend in
torch's order is an append in the reversed one.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("repeat.default")})
    a = operand(emitctx, op.ins[1])
    out = dest(emitctx)
    od = size(out)
    id = ntuple(k -> k <= ndims(a) ? size(a, k) : 1, length(od))
    M.dispatch!(emitctx.g, tilecopy!, (out, od, a, id), prod(od); name = op.id)
    return out
end

"""
    folddims(emitctx, op, dims, combine, init; pre = identity, post = identity) -> out

One reduction over `dims`, as one dispatch over the output.

`od` carries the reduction: the input's shape with a `1` on each reduced axis.
The output resource may have those axes dropped (`keepdim = false`), which
changes its shape and not its bytes, so the kernel indexes both linearly.

`combine` and `init` are what separate `sum` from `prod` from `any`. They are
arguments of `folddims!` rather than four kernels, because the index arithmetic
is the whole of what those ops have in common and all of what they do.

**`init` decides the accumulator's type** — see `folddims!`. For a float
reduction that is `accum(eltype(a))` and not `eltype(a)`, and `foldincasts`
depends on it.

`pre` and `post` are the op's OWN map steps, the one on each side of the fold: a
p-norm is `sum(abs2)` followed by a `sqrt`, and `mean` is a sum followed by a
division. Both happen inside the reduction kernel, so neither costs a pass.
`pre` composes with whatever `foldpremap` folded in, and in that order,
because the folded map is a step of the value the graph fed this op. No
graph reaches that composition today, because `PREMAPPABLE` lists the three
plain reductions and not the norm; the rule is here rather than there because
which ops a pass folds into is not something this function may assume.
"""
function folddims(emitctx::EmitCtx, op::Op, dims, combine, init;
                  pre = identity, post = identity)
    a = operand(emitctx, op.ins[1])
    out = dest(emitctx)
    id = size(a)
    od = ntuple(k -> k in dims ? 1 : id[k], length(id))
    prod(od) == length(out) ||
        error("DNNKernels: `$(op.aten)` (op $(op.id)) reduces $(id) over " *
              "$(dims) to $(prod(od)) elements, and its output buffer holds " *
              "$(length(out)).")
    # `foldpremap` folds a map step into the reduction; `identity` is the
    # unfolded case, so there is one kernel rather than two.
    folded = something(premap(op), identity)
    f = folded === identity ? pre : pre === identity ? folded : pre ∘ folded
    M.dispatch!(emitctx.g, folddims!, (out, od, a, id, f, combine, init, post),
                prod(od); name = op.id)
    return out
end

"""The axes an `arg1`-style reduction names, in Julia's order."""
reduceddims(op::Op, n::Int) =
    Tuple(jdim(d, n) for d in ints(op.attrs["arg1"]))

"""`aten::sum.dim_IntList`."""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("sum.dim_IntList")})
    a = operand(emitctx, op.ins[1])
    A = accum(eltype(a))
    return folddims(emitctx, op, reduceddims(op, ndims(a)), +, zero(A))
end

"""
`aten::prod.dim_int`, which is `sum` with the other operator and the other
identity.

The accumulator is `accum(eltype(a))` for the same reason a sum's is: a product
over a long axis leaves `Float16`'s range far sooner than a sum does, and it does
so silently.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("prod.dim_int")})
    a = operand(emitctx, op.ins[1])
    A = accum(eltype(a))
    return folddims(emitctx, op, (jdim(Int(op.attrs["arg1"]), ndims(a)),), *, one(A))
end

"""
`aten::any.dim`: `|` over `Bool`, so the accumulator is `false` and NOT
`accum(eltype(a))`.

The operand may be any type — torch's `any` is "nonzero somewhere" — so the map
step is the test and the fold is the or. Written as `!iszero` rather than
`x -> x != 0`, which needs a zero of the operand's type in the closure and so a
non-isbits `Type` field.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("any.dim")})
    a = operand(emitctx, op.ins[1])
    return folddims(emitctx, op, (jdim(Int(op.attrs["arg1"]), ndims(a)),), |, false)
end

"""`aten::all.dim`, `any.dim`'s mirror: `&` from `true`."""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("all.dim")})
    a = operand(emitctx, op.ins[1])
    return folddims(emitctx, op, (jdim(Int(op.attrs["arg1"]), ndims(a)),), &, true)
end

"""
`aten::linalg_vector_norm(ord, dims, keepdim)`: the fold with a map on each side
of it.

Order 2 is `sqrt(sum(abs2))` and the general one is `sum(abs(x)^p)^(1/p)`, so
both are `folddims` with a `pre` and a `post` and neither is a reduction of its
own. `ord == 2` is spelled separately because `sqrt` is not `^(1/2)` on a GPU:
it is one instruction against a `log`/`exp` pair, and every graph here asks for
2.

A non-finite or non-positive order is REFUSED rather than run. `p = Inf` is the
maximum absolute value and `p = 0` counts the nonzeros; both are different
reductions, and `acc^(1/Inf)` is `acc^0`, which is 1 for every input. The
interpreted path computes exactly that and returns ones.

The dims are `nothing` for torch's "over everything", which is every axis.
"""
function emitop!(emitctx::EmitCtx, op::Op,
                 ::Val{Symbol("linalg_vector_norm.default")})
    a = operand(emitctx, op.ins[1])
    A = accum(eltype(a))
    ord = Float64(something(get(op.attrs, "arg1", nothing), 2))
    isfinite(ord) && ord > 0 || error(
        "DNNKernels: `linalg_vector_norm` (op $(op.id)) asks for order $(ord). " *
        "Only a finite positive order is a sum of powers; `Inf` is a maximum " *
        "and `0` is a count, and each is its own reduction.")
    spec = get(op.attrs, "arg2", nothing)
    dims = spec === nothing ? ntuple(identity, ndims(a)) :
           Tuple(jdim(d, ndims(a)) for d in ints(spec))
    ord == 2 && return folddims(emitctx, op, dims, +, zero(A);
                                pre = abs2, post = sqrt)
    p, invp = A(ord), A(1 / ord)
    return folddims(emitctx, op, dims, +, zero(A);
                    pre = x -> abs(x)^p, post = acc -> acc^invp)
end

"""
`aten::arange.start_step(start, end, step)`, as one dispatch.

The three scalars may be **fractional** — DINOv3's rotary embedding asks for
`arange(0.5, 32, 1)`, the patch centres — so they are read with `numattr` and
not `intattr`.

The length is aten's own `ceil((end - start) / step)` and not a Julia range's.
Those disagree whenever the step is not 1: `arange(0, 5, 2)` is `[0, 2, 4]` to
torch, three elements, where `0:2:5` is also three but `0:2:4` and `0:2:5` are
not the same range to reason about. `test_arange.jl` pins both, and the length is
checked against the declared buffer rather than allocated to fit — a declared
graph has already placed those bytes.

`end` may be a host value the graph computed rather than an attribute, which is
why `op.ins` is consulted first.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("arange.start_step")})
    # NOT the positional accessor: that assumes every position is either an
    # attribute or an `ins` entry, and `start`/`step` here may be neither --
    # torch defaults them. Same three reads `runop!` made.
    start = numattr(emitctx.dims, something(get(op.attrs, "arg0", nothing), 0))
    stop  = length(op.ins) >= 1 ? operand(emitctx, op.ins[1]) :
                                  numattr(emitctx.dims, op.attrs["arg1"])
    step  = numattr(emitctx.dims, something(get(op.attrs, "arg2", nothing), 1))
    out = dest(emitctx)
    n = max(0, ceil(Int, (Float64(stop) - Float64(start)) / Float64(step)))
    n == length(out) || error(
        "DNNKernels: `arange.start_step` (op $(op.id)) is $start:$step:$stop, " *
        "which is $n elements, and its output buffer holds $(length(out)).")
    T = eltype(out)
    M.dispatch!(emitctx.g, arange!, (out, n, T(start), T(step)), n; name = op.id)
    return out
end

"""
`aten::index.Tensor` — torch's advanced indexing, as one gather.

Torch indexes the UN-REVERSED shape, so entry `k` of `arg1` addresses Julia
dimension `ndims - k + 1`, and the values are 0-based. They stay 0-based: a
0-based value is the source coordinate the kernel adds to an offset, so no
`.+ 1` pass over the index is needed.

Two shapes, and telling them apart is the part that is easy to get silently
wrong, because Julia spells the other thing the same way. `x[i, j]` with two
vectors selects `length(i) * length(j)` elements; torch broadcasts `i` against
`j` and selects `length(i)` of them, one per position. `indexseparable` is when
the two agree — each index tensor varying along its own broadcast axis — which is
SAM 2's position-embedding interpolation, sixteen times.

Both are a gather and both keep their index on the device, so neither needs a
host round trip (`collect(vec(x))[vec(lin)]`), which could not be recorded.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("index.Tensor")})
    x = operand(emitctx, op, 1)
    spec = op.attrs["arg1"]
    n = ndims(x)
    dims, idxs = Int[], Any[]
    for (k, e) in enumerate(spec)
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error(
            "DNNKernels: `index.Tensor` (op $(op.id)) indexes dim $k of a " *
            "$(n)-d input.")
        push!(dims, jd)
        push!(idxs, operand(emitctx, String(e)[2:end]))   # attrs store "\$name"
    end
    out = dest(emitctx)
    isempty(dims) && return elementwise!(emitctx, op, identity, x)
    # Ascending Julia dimension. `spec` is in torch order, so reversing each
    # entry's axis walks the Julia dims backwards; `indexgather!` pairs the j-th
    # indexed dimension with the j-th index tensor, and unsorted that is
    # transposed.
    p = sortperm(dims)
    dims, idxs = dims[p], Tuple(idxs[p])
    od = size(out)
    if length(dims) == 1 || indexseparable(dims, idxs)
        size(x) isa NTuple{length(od),Int} || error(
            "DNNKernels: `index.Tensor` (op $(op.id)) is separable, so its " *
            "output has the input's rank $(ndims(x)) and holds $(length(od)).")
        M.dispatch!(emitctx.g, indexgather!, (out, od, x, size(x), idxs, Tuple(dims)),
                    prod(od); name = op.id)
    else
        length(dims) == n || error(
            "DNNKernels: `index.Tensor` (op $(op.id)) pairs $(length(dims)) " *
            "index tensors over a $(n)-d input. Mixing paired and sliced axes " *
            "is where torch also has to decide WHERE the gathered axis goes, " *
            "and the separable half of that is handled above.")
        sts = Tuple(bcstrides(od, size(i)) for i in idxs)
        M.dispatch!(emitctx.g, indexpaired!, (out, od, x, size(x), idxs, sts),
                    prod(od); name = op.id)
    end
    return out
end

"""
    mapbody!(emitctx, op, body, out, args...) -> out

Declare `body(I, args...)` at every index of `out`, as one dispatch.

The kernel is `ndmap_flat!`, which is the one `launch!` picks for a
linearly-indexable destination, and the body is the same function the
interpreted path passed it. So the index arithmetic and the arithmetic are
literally that code -- an op declared through this differs from the op that ran
in when it is submitted and in nothing else. `Mantle.FastDiv32` is why the flat
variant is the fast one: the coordinate decomposition is a magic-number multiply
rather than N-1 real divisions.
"""
function mapbody!(emitctx::EmitCtx, op::Op, body, out, args...; name = op.id)
    n = length(out)
    # An EMPTY result needs no pass, and `Mantle.dispatch!` refuses one rather
    # than carrying a dispatch that does nothing. torch produces empty tensors
    # legitimately -- SAM 2's decoder concatenates a `(1, 0, 256)` on the branch
    # with no boxes -- so the skip belongs here, where the shape is known, and
    # the refusal belongs there, where a zero ndrange would otherwise become a
    # `DivideError` inside `KernelAbstractions.partition`.
    n == 0 && return out
    M.dispatch!(emitctx.g, ndmap_flat!,
                (body, out, map(M.FastDiv32, size(out)), n, args...), n; name)
    return out
end

"""
`aten::gather(dim, index)`: `out[c] = a[c with c[dim] = index[c]]`.

The index has the result's shape and names the coordinate on ONE axis. That is
a different gather from `index.Tensor`'s, where an index array is a coordinate
list for a whole axis and several of them form an outer product. So it is
`gatherdim!` and not `indexgather!`, and the distinction is in those two
docstrings.

Torch requires the index to be no larger than the source on every axis it does
not name; checked, because reading past the source is the failure the check
exists to stop and the two extents come from different buffers.

`runop!` restricted this to the case where every axis but `dim` is a singleton,
which is Kokoro's, and ran it on the HOST. The general form is one dispatch.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("gather.default")})
    # `ins`, not positions: torch's schema is `gather(self, dim, index)`, so the
    # index is the second TENSOR and the third argument, and `dim` is the scalar
    # in `attrs["arg1"]` below.
    a = operand(emitctx, op.ins[1])
    idx = operand(emitctx, op.ins[2])
    out = dest(emitctx)
    n = ndims(a)
    ndims(idx) == n || error(
        "DNNKernels: `gather` (op $(op.id)) indexes a $(n)-d source with a " *
        "$(ndims(idx))-d index. torch gives the index the source's rank.")
    d = jdim(Int(op.attrs["arg1"]), n)
    1 <= d <= n || error(
        "DNNKernels: `gather` (op $(op.id)) names torch dim " *
        "$(Int(op.attrs["arg1"])) of a $(n)-d source.")
    od = size(idx)
    all(k -> k == d || od[k] <= size(a, k), 1:n) || error(
        "DNNKernels: `gather` (op $(op.id)) has an index of $(od) over a " *
        "source of $(size(a)), which is larger than the source on an axis it " *
        "does not gather. Every such axis passes its coordinate straight " *
        "through, so the read would leave the source.")
    prod(od) == length(out) || error(
        "DNNKernels: `gather` (op $(op.id)) gathers $(prod(od)) elements and " *
        "its output buffer holds $(length(out)).")
    M.dispatch!(emitctx.g, gatherdim!, (out, od, a, size(a), idx, d), prod(od);
                name = op.id)
    return out
end

"""`aten::upsample_nearest2d`, as one gather at the output's resolution."""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("upsample_nearest2d.vec")})
    x = operand(emitctx, op, 1)
    out = dest(emitctx)
    return mapbody!(emitctx, op, upsample_nearest, out, x,
                    Float32(size(x, 1) / size(out, 1)),
                    Float32(size(x, 2) / size(out, 2)))
end

"""
`aten::upsample_bilinear2d`, the same gather one interpolation wider.

`align_corners` is read and not assumed. torch exposes both conventions and the
exported graphs use both — BasicVSR++'s SPyNet upsamples flow with `true` and its
pyramid with `false` — and implementing one silently rescales by roughly
`(n-1)/n`, which is invisible on a big tensor and enough to move an optical-flow
field by a pixel.

`sx`/`sy` carry the convention so the kernel needs no output extents; the two
expressions are `upsample_bilinear2d!`'s, which is the immediate form of this
same launch.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("upsample_bilinear2d.vec")})
    x = operand(emitctx, op, 1)
    out = dest(emitctx)
    align = Bool(something(get(op.attrs, "arg2", nothing), false))
    sx = align ? (size(out, 1) > 1 ?
                  Float32((size(out, 1) - 1) / max(size(x, 1) - 1, 1)) : 1.0f0) :
                 Float32(size(out, 1) / size(x, 1))
    sy = align ? (size(out, 2) > 1 ?
                  Float32((size(out, 2) - 1) / max(size(x, 2) - 1, 1)) : 1.0f0) :
                 Float32(size(out, 2) / size(x, 2))
    return mapbody!(emitctx, op, upsample_bilinear, out, x, sx, sy, Val(align))
end

"""
`aten::_adaptive_avg_pool2d`, one thread per output element.

The output extents travel as `Val`s so the window arithmetic folds where the
ratio is exact, which it is on every path in these graphs. torch gives the target
as `(H, W)` and the reversed layout reads it as `(y, x)`, so the two `ins` are the
output's second and first extents — but they are also its declared shape, and
that is what this reads: a graph that disagreed with its own buffer would be a
shape error and not something to resolve here.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("_adaptive_avg_pool2d.default")})
    x = operand(emitctx, op, 1)
    out = dest(emitctx)
    return mapbody!(emitctx, op, adaptive_avg_pool, out, x,
                    Val(size(out, 1)), Val(size(out, 2)))
end

"""
`aten::slice_scatter(self, src, dim, start, end, step)`: `self` with one slice
replaced.

Two passes, and the first is the whole of `self`. That copy is not avoidable by
aliasing the output onto the input: the graph may read `self` after this op, so
writing into its bytes would change a value something else still wants. Where it
does NOT — `self` dead after here — `Aliasing` is what notices, because the copy
declares a read of `self` and a write of `out` and their intervals then do not
overlap.

The slice write is `blockcopy!`, which is already "one part into the box of the
output that starts at `off`": a `cat` part, a scattered slice and a pad's
interior are the same write with three reasons. `step` is refused rather than
folded in, since `blockcopy!` walks the output contiguously along each axis and a
step would make that a stride — `runop!` ignored the argument entirely, which is
worse.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("slice_scatter.default")})
    a = operand(emitctx, op, 1)
    src = operand(emitctx, op, 2)
    out = dest(emitctx)
    n = ndims(out)
    d = jdim(Int(get(op.attrs, "arg2", 0)), n)
    lo = Int(get(op.attrs, "arg3", 0))
    step = Int(something(get(op.attrs, "arg5", nothing), 1))
    step == 1 || error(
        "DNNKernels: `slice_scatter` (op $(op.id)) has step $(step), and only " *
        "the unit step is declared -- `blockcopy!` walks the output contiguously " *
        "along the scattered axis. A strided form needs a kernel of its own.")
    od = size(out)
    od == size(a) || error(
        "DNNKernels: `slice_scatter` (op $(op.id)) writes $(od) but its `self` " *
        "is $(size(a)); the result is `self` with a slice replaced, so they are " *
        "the same shape.")
    M.dispatch!(emitctx.g, ew!,
                (out, od, (a,), (bcstrides(od, size(a)),), identity),
                prod(od); name = "$(op.id).self")
    length(src) == 0 && return out
    M.dispatch!(emitctx.g, blockcopy!,
                (out, od, src, size(src), ntuple(k -> k == d ? lo : 0, n)),
                length(src); name = op.id)
    return out
end

"""
`aten::constant_pad_nd`: a fill, then the operand into the interior.

`arg1` is `(lo, hi)` per torch dimension counting from the LAST, so torch's
`-k` is Julia's `k` and the pairs are read in order without reversing. Only the
low pads move the operand; the high ones only make the output bigger, which its
declared shape already says.

Two passes and not one: the border and the interior are disjoint writes, and a
single kernel over the output would branch per element on whether it is inside.
The fill is `fill_kernel!` over the whole output rather than the border only,
because the border is not a box — it is the complement of one.

Into the planned buffer, which is the point. Wan's VAE decoder pads before each
of its 116 3-D convolutions, and at 256x256x9 those temporaries are hundreds of
MB apiece; allocating them outside the plan is what took its decode from a 1.2 GB
slab to a 14.8 GB peak.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("constant_pad_nd.default")})
    a = operand(emitctx, op, 1)
    out = dest(emitctx)
    n = ndims(out)
    # `intlist` and not `ints`: a pad can be SYMBOLIC. MatAnyone's
    # `readout_query` pads by an extent the graph carries as a symbol, and
    # `Int("n")` is a `MethodError` about `Int64` several frames from the pad.
    # Same rule the interpreted path read it with, asked with this path's tables.
    pad = intlist(emitctx.dims, emitctx.res, op.attrs["arg1"])
    los = ntuple(k -> 2k <= length(pad) ? pad[2k - 1] : 0, n)
    his = ntuple(k -> 2k <= length(pad) ? pad[2k] : 0, n)
    for k in 1:n
        size(a, k) + los[k] + his[k] == size(out, k) || error(
            "DNNKernels: `constant_pad_nd` (op $(op.id)) pads axis $k of " *
            "$(size(a)) by ($(los[k]), $(his[k])), which is " *
            "$(size(a, k) + los[k] + his[k]) and its output holds $(size(out, k)).")
    end
    emitfill!(emitctx, op, numattr(emitctx.dims,
                                   something(get(op.attrs, "arg2", nothing), 0));
              name = "$(op.id).border")
    length(a) == 0 && return out
    M.dispatch!(emitctx.g, blockcopy!,
                (out, size(out), a, ntuple(k -> size(a, k), n), los),
                length(a); name = "$(op.id).inner")
    return out
end

"""
`aten::max.dim`, which returns the maximum and WHERE it was.

Two dispatches of one body, `Val(WANTIDX)` choosing which result it stores. Two
rather than one because a kernel writes one element per thread and these are two
tensors; the scan is repeated, which is the same trade `runop!` made and is one
extra read of the reduced axis.

The index is torch's, so 0-based, and it comes back in the VALUES' element type —
that is `maxdim_body`'s choice and it is what the declared buffer's dtype says
too. `destor` for the indices, because a graph that reads only the maximum
declares no buffer for them and the kernel writes them anyway.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("max.dim")})
    a = operand(emitctx, op, 1)
    vals = dest(emitctx, 0)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    sz = ntuple(k -> k == d ? 1 : size(a, k), ndims(a))
    inds = destor(emitctx, 1, eltype(vals), sz)
    mapbody!(emitctx, op, maxdim_body, vals, a, Val(d), Val(false);
             name = "$(op.id).values")
    mapbody!(emitctx, op, maxdim_body, inds, a, Val(d), Val(true);
             name = "$(op.id).indices")
    return (vals, inds)
end

"""
`aten::_softmax` along one axis, as ONE dispatch: one workgroup per slice,
reducing through workgroup memory.

`softmax_kernel!` indexes `a` linearly over a `(pre, n, post)` view, so the
operand has to be dense in that order. A declared operand always is —
`hoistpermutes` resolved the permutes at build and a materialised view is its own
transient — which is why the `materialize` call `runop!` needed here is not
present: there is no wrapper left to collapse.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("_softmax.default")})
    a = operand(emitctx, op, 1)
    out = dest(emitctx)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    pre = prod(ntuple(k -> size(a, k), d - 1); init = 1)
    n = size(a, d)
    post = length(a) ÷ (pre * n)
    M.dispatch!(emitctx.g, softmax_kernel!, (out, a, Val(SOFTMAX_WG), pre, n),
                SOFTMAX_WG * pre * post; group = SOFTMAX_WG, name = op.id)
    return out
end

"""
`aten::cumsum` along one axis, one thread per output element.

Each thread walks the axis from its start, so the work is quadratic in the
scanned extent rather than a parallel scan's linear cost. Deliberate, and the
reason is in `cumsum_body`: the only `cumsum` in any graph here builds SAM 2's
dense positional encoding from a constant 64x64 tensor, 262k adds once per
decoder call. A work-efficient scan belongs here the moment something scans a
long axis, and until then it would be untested code.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("cumsum.default")})
    a = operand(emitctx, op, 1)
    out = dest(emitctx)
    d = jdim(Int(op.attrs["arg1"]), ndims(a))
    return mapbody!(emitctx, op, cumsum_body, out, a, Val(Int(d)))
end

"""
`aten::max_pool2d_with_indices`, whose second result nothing reads.

torch returns `(values, indices)` and every graph here uses only the first, so
the indices are declared as the empty buffer the export gives them. The window,
stride and padding are `Val`-parameters: they are host constants, and in the
kernel's type they make its bounds arithmetic compile-time.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("max_pool2d_with_indices.default")})
    x = operand(emitctx, op, 1)
    k = reverse(ints(op.attrs["arg1"]))
    st = haskey(op.attrs, "arg2") ? reverse(ints(op.attrs["arg2"])) : k
    pd = haskey(op.attrs, "arg3") ? reverse(ints(op.attrs["arg3"])) : [0, 0]
    out = dest(emitctx, 0)
    mapbody!(emitctx, op, maxpool, out, x, Val(k[1]), Val(k[2]),
             Val(st[1]), Val(st[2]), Val(pd[1]), Val(pd[2]))
    return (out, maybedest(emitctx, 1))
end

"""
`aten::mean.dim`, which is `sum.dim_IntList` with the count folded into the
store.

Not a second pass to divide: the count is a host scalar the emit already knows,
so it goes in as `folddims!`'s `post` and multiplies once, inside the
accumulator's type. A pass to scale by a constant is a whole round trip of the
result through memory.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("mean.dim")})
    a = operand(emitctx, op, 1)
    dims = reduceddims(op, ndims(a))
    n = prod(size(a, k) for k in dims)
    A = accum(eltype(dest(emitctx)))
    s = A(1 // n)
    return folddims(emitctx, op, dims, +, zero(A); post = acc -> acc * s)
end

"""
`aten::cat`, as one dispatch per input into its slice of the output.

Each input writes a disjoint region, so nothing orders them against each other:
the walk reports `out` written by each and `Barriers` finds no hazard. An empty
input is skipped rather than dispatched over zero elements -- SAM 2's decoder
concatenates a `(1, 0, 256)` onto the sparse embeddings on the branch with no
boxes.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("cat.default")})
    parts = [operand(emitctx, i) for i in op.ins]
    out = dest(emitctx)
    od = size(out)
    n = length(od)
    d = jdim(Int(get(op.attrs, "arg1", 0)), n)
    total = sum(size(p, d) for p in parts)
    total == od[d] ||
        error("DNNKernels: `cat` (op $(op.id)) joins $(total) along axis $d and " *
              "its output holds $(od[d]).")
    off = 0
    for (j, p) in enumerate(parts)
        len = size(p, d)
        len == 0 && continue
        M.dispatch!(emitctx.g, blockcopy!,
                    (out, od, p, ntuple(k -> size(p, k), n),
                     ntuple(k -> k == d ? off : 0, n)),
                    length(p); name = "$(op.id).$j")
        off += len
    end
    return out
end

"""
`aten::native_layer_norm`, as ONE dispatch.

One workgroup per normalised group, two reduction passes inside the kernel. The
fallback the interpreted path carried -- six broadcast passes for a non-dense or
host operand -- is not here: this path requires the normalised axes to be
leading and dense, which is what the reversed layout gives (torch normalises
over trailing dims, we see them leading), and a graph that violates it gets a
refusal naming the shape rather than a silently slower answer. SAM 2's 96 layer
norms are all this form.

`γ` and `β` are optional and the kernel takes a PLACEHOLDER for a missing one,
with `Val(has)` deciding. Declared, that costs nothing: the walk specialises on
the `Val`, so the branch reading an absent `γ` folds away and the placeholder
comes out `NOTOUCH` rather than declared read.

All three of torch's results are declared -- the output, the mean and the
reciprocal standard deviation. The last two are what a backward pass reads, and
a graph that never reads them still has them placed, which keeps the op's shape
honest for `2 * groups` floats.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("native_layer_norm.default")})
    a = operand(emitctx, op, 1)
    nshape = ints(op.attrs["arg1"])
    eps = Float32(op.attrs["arg4"])
    n = prod(size(a, i) for i in 1:length(nshape))
    γ = length(op.ins) >= 2 ? operand(emitctx, op.ins[2]) : nothing
    β = length(op.ins) >= 3 ? operand(emitctx, op.ins[3]) : nothing
    length(a) % n == 0 || error(
        "DNNKernels: `native_layer_norm` (op $(op.id)) normalises $n elements " *
        "of a $(size(a)) operand, which does not divide it. The normalised axes " *
        "have to be leading and dense here; `hoistpermutes` is what makes them so.")
    out = dest(emitctx, 0)
    groups = length(a) ÷ n
    # The mean and the reciprocal standard deviation are what a backward pass
    # reads, and SAM 2 reads neither -- but the kernel writes them, so they get
    # a destination either way. See `destor`.
    μ = destor(emitctx, 1, Float32, (groups,))
    r = destor(emitctx, 2, Float32, (groups,))
    # The placeholder for an absent operand, and it must be a RESOURCE: the
    # kernel indexes it whether or not the `Val` lets it, so a `nothing` would
    # not compile. `a` is always present and already declared read.
    dummy = γ === nothing ? (β === nothing ? a : β) : γ
    M.dispatch!(emitctx.g, layernorm_kernel!,
                (out, μ, r, a,
                 γ === nothing ? dummy : γ, β === nothing ? dummy : β,
                 Int32(n), eps, Val(γ !== nothing), Val(β !== nothing)),
                groups * LN_WG; group = LN_WG, name = op.id)
    return (out, μ, r)
end

# ── batch norm, in training mode ─────────────────────────────────────────────

"""
`aten::_native_batch_norm_legit.no_stats`, as two passes.

Two dispatches, and a dispatch is a pass: the second reads what the first writes,
and a pass is the unit that may run concurrently, so the dependency between them
is a dependency between passes. Nothing here orders it — `bnstats!` writes `mean`
and `invstd`, `bnapply!` reads them, `argument_usage` reads that off both bodies
and `Barriers` derives the wait.

All three of torch's results are declared: the normalised output, the mean and
the inverse standard deviation. The last two are what the backward pass reads,
and a graph that never reads them still has them placed, which costs `2C`
floats and keeps the op's shape honest.
"""
function emitop!(emitctx::EmitCtx, op::Op,
                 ::Val{Symbol("_native_batch_norm_legit.no_stats")})
    x = operand(emitctx, op.ins[1])
    gamma = operand(emitctx, op.ins[2])
    beta = operand(emitctx, op.ins[3])
    out = dest(emitctx, 0)
    eps = Float32(op.attrs["arg5"])
    id = size(x)
    # torch's channel dim is 1, which in the reversed shape is `ndims - 1`.
    c = length(id) - 1
    C = id[c]
    cstride = prod(ntuple(k -> id[k], c - 1); init = 1)
    nouter = prod(ntuple(k -> id[c + k], length(id) - c); init = 1)
    # `bnstats!` writes both statistics whether or not the graph reads them.
    mean = destor(emitctx, 1, Float32, (C,))
    invstd = destor(emitctx, 2, Float32, (C,))
    M.dispatch!(emitctx.g, bnstats!, (mean, invstd, x, cstride, C, nouter, eps), C;
                name = "$(op.id).stats")
    M.dispatch!(emitctx.g, bnapply!,
                (out, x, mean, invstd, gamma, beta, length(out), cstride, C),
                length(out); name = op.id)
    return (out, mean, invstd)
end

"""
`aten::_native_batch_norm_legit_no_training`, which is inference-mode batch norm:
the statistics are the RUNNING ones, so there is nothing to reduce.

One pass, sharing `bnapply!` with the training form above — the only difference
between them is where `mean` and `invstd` come from, and here they are weights.
`bnapply!` wants the inverse standard deviation while torch stores the variance,
so the reciprocal square root is folded into a `(C,)` scratch by an `ew!` rather
than into the apply kernel: that keeps one kernel for both forms, and `C` is a
few hundred elements.

The driver folds this op into the preceding convolution (`foldbatchnorm`) and so
never asks for it, which is why it had no emit. It is declared anyway: which
passes have run is not something an emit gets to assume, and MatAnyone's graphs
contain it as exported.
"""
function emitop!(emitctx::EmitCtx, op::Op,
                 ::Val{Symbol("_native_batch_norm_legit_no_training.default")})
    x = operand(emitctx, op.ins[1])
    gamma = operand(emitctx, op.ins[2])
    beta = operand(emitctx, op.ins[3])
    rmean = operand(emitctx, op.ins[4])
    rvar = operand(emitctx, op.ins[5])
    out = dest(emitctx, 0)
    eps = Float32(op.attrs["arg6"])
    id = size(x)
    # torch's channel dim is 1, which in the reversed shape is `ndims - 1`.
    c = length(id) - 1
    C = id[c]
    cstride = prod(ntuple(k -> id[k], c - 1); init = 1)
    invstd = scratch(emitctx, Float32, C)
    M.dispatch!(emitctx.g, ew!,
                (invstd, (C,), (rvar,), (bcstrides((C,), (C,)),),
                 v -> inv(sqrt(Float32(v) + eps))), C;
                name = "$(op.id).invstd")
    M.dispatch!(emitctx.g, bnapply!,
                (out, x, rmean, invstd, gamma, beta, length(out), cstride, C),
                length(out); name = op.id)
    # The running statistics are returned in place of the batch ones: torch's
    # no-training form declares the two results empty, and the graph reads
    # neither.
    return (out, rmean, invstd)
end

"""
`aten::embedding(weight, indices)`: a column gather.

The reversed layout puts the embedding dimension FIRST — `weight` is
`(dim, vocab)` and the result is `(dim, indices...)` — so this gathers columns,
and the indices are torch's 0-based ones. `embedding_kernel!` does that
arithmetic and takes the index tensor flattened, since only its linear order
matters.

One dispatch, and the index tensor is an ordinary read operand: nothing comes to
the host. That is what makes an embedding replayable — the interpreted path's
`value` had already kept it on the device, and this keeps it in the plan.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("embedding.default")})
    w = operand(emitctx, op, 1)
    idx = operand(emitctx, op, 2)
    out = dest(emitctx)
    length(out) == 0 && return out
    M.dispatch!(emitctx.g, embedding_kernel!,
                (out, w, M.viewof(idx, (length(idx),)), Int32(length(out))),
                length(out); name = op.id)
    return out
end

"""
`aten::flip` along one or more axes, as one strided read.

The axes are `Val`-parameterised so the per-axis reversal folds; `flip_kernel!`
walks the output flat and `coords4` decomposes it, which is the same shape as
every other rank-4 gather here.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("flip.default")})
    x = operand(emitctx, op, 1)
    out = dest(emitctx)
    length(out) == 0 && return out
    d = ints(op.attrs["arg1"])
    jd = Tuple(jdim(Int(k), ndims(x)) for k in (d isa Integer ? [d] : d))
    nd, W4, H4, C4 = flat4(x)
    M.dispatch!(emitctx.g, flip_kernel!, (out, x, Val(jd), W4, H4, C4), nd;
                group = launchgroup(nd), name = op.id)
    return out
end

"""
`aten::avg_pool2d`: a fixed window and stride, where `_adaptive_avg_pool2d`
derives the window from the output size.

torch orders the window and stride `(H, W)` and Julia's first axis is `W`, so
each pair is read from its ends. The output extents come from the declared
buffer rather than being recomputed — a graph that disagreed with its own shape
is a shape error, not something to resolve here.

`count_include_pad` is not modelled, exactly as in `avg_pool2d!`: every call in
the graphs pools with no padding, where the two agree, and a padded one needs
the divisor to switch between the window area and the in-bounds count. Refused
rather than guessed.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("avg_pool2d.default")})
    x = operand(emitctx, op, 1)
    out = dest(emitctx)
    length(out) == 0 && return out
    pair(v, dflt) = (u = ints(get(op.attrs, v, dflt));
                     u isa Integer ? (u, u) : Tuple(u))
    kk = pair("arg1", nothing)
    ss = let u = get(op.attrs, "arg2", nothing)
        u === nothing || isempty(ints(u)) ? kk : pair("arg2", nothing)
    end
    pp = pair("arg3", [0, 0])
    all(==(0), pp) || error(
        "DNNKernels: `avg_pool2d` (op $(op.id)) pads by $(pp), and the divisor " *
        "then depends on `count_include_pad`, which `avg_pool2d_kernel!` does " *
        "not model. Every call in the graphs pools without padding.")
    nd, W4, H4, C4 = flat4(out)
    M.dispatch!(emitctx.g, avg_pool2d_kernel!,
                (out, x, Int32(kk[end]), Int32(kk[1]), Int32(ss[end]), Int32(ss[1]),
                 Int32(pp[end]), Int32(pp[1]), W4, H4, C4),
                nd; group = launchgroup(nd), name = op.id)
    return out
end

"""
`aten::grid_sampler_2d`: bilinear resampling at coordinates the graph computed,
which is how every optical-flow warp reaches the device.

`grid` is `(2, W, H, N)` after reversal — coordinates first — and the output
takes its spatial extents from the grid and its channels from `x`, which is what
the declared buffer already says.

Both of torch's conventions are read and neither assumed. `align_corners`
changes the coordinate mapping, and `padding_mode` decides what a sample outside
the image is: 0 zeros, 1 border, 2 reflection. BasicVSR++ uses 0 and 1, and
treating border as zeros leaves a dark rim the next warp amplifies. Reflection
is refused rather than approximated by either of the other two.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("grid_sampler_2d.default")})
    x = operand(emitctx, op, 1)
    grid = operand(emitctx, op, 2)
    out = dest(emitctx)
    length(out) == 0 && return out
    align = Bool(something(get(op.attrs, "arg4", nothing), true))
    mode = Int(something(get(op.attrs, "arg3", nothing), 0))
    mode in (0, 1) || error(
        "DNNKernels: `grid_sampler_2d` (op $(op.id)) has padding_mode $(mode), " *
        "and only zeros (0) and border (1) are declared. Reflection is a " *
        "different coordinate fold, not one of these two with a different " *
        "constant.")
    pad = mode == 1 ? :border : :zeros
    nd, W4, H4, C4 = flat4(out)
    M.dispatch!(emitctx.g, grid_sample2d_kernel!,
                (out, x, grid, Val(align), Val(pad), W4, H4, C4), nd;
                group = launchgroup(nd), name = op.id)
    return out
end

"""
`torchvision.deform_conv2d`: modulated deformable convolution v2, which is what
BasicVSR++'s `SecondOrderDeformableAlignment` runs 16 times a clip.

Not an ATen op — it survives `run_decompositions` because it is a registered
custom operator, which is what we want: one graph node instead of a scatter of
index arithmetic to re-fuse.

torch orders the stride, padding and dilation `(h, w)` and Julia's first axis is
`w`, so each pair is read in that order from `arg5` onward. The mask is optional
and a `Val` decides, with the offset standing in for it when absent — a kernel
cannot be handed `nothing` for an array it indexes.
"""
function emitop!(emitctx::EmitCtx, op::Op,
                 ::Val{Symbol("torchvision.deform_conv2d.default")})
    x = operand(emitctx, op.ins[1])
    w = operand(emitctx, op.ins[2])
    offset = operand(emitctx, op.ins[3])
    mask = length(op.ins) >= 4 ? operand(emitctx, op.ins[4]) : nothing
    bias = length(op.ins) >= 5 ? operand(emitctx, op.ins[5]) : nothing
    out = dest(emitctx)
    length(out) == 0 && return out
    a(k, d) = Int(something(get(op.attrs, k, nothing), d))
    nd, W4, H4, C4 = flat4(out)
    M.dispatch!(emitctx.g, deform_conv2d_kernel!,
                (out, x, offset, mask === nothing ? offset : mask, w, bias,
                 Int32(a("arg5", 1)), Int32(a("arg6", 1)),
                 Int32(a("arg7", 0)), Int32(a("arg8", 0)),
                 Int32(a("arg9", 1)), Int32(a("arg10", 1)),
                 Int32(a("arg12", 1)), Int32(a("arg11", 1)),
                 Val(mask !== nothing), W4, H4, C4),
                nd; group = launchgroup(nd), name = op.id)
    return out
end

"""
`aten::_fused_rms_norm`, as ONE dispatch: one workgroup per group, reducing
through workgroup memory.

The normalised axes have to be leading and dense, which is what the reversed
layout gives and what a declared operand always is — `hoistpermutes` resolved
the permutes at build and a materialised view is its own transient. So the
six-pass fallback `runop!` carries for a permuted view has nothing to do here,
and with it goes the `sqaccum` subtlety that fallback needed: the kernel
accumulates in fp32 by construction.

Both results are declared. The reciprocal standard deviation is what a backward
pass reads and the kernel writes it either way; `destor` gives it scratch when
the graph declares none.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("_fused_rms_norm.default")})
    a = operand(emitctx, op.ins[1])
    γ = length(op.ins) >= 2 ? operand(emitctx, op.ins[2]) : nothing
    out = dest(emitctx, 0)
    nshape = ints(op.attrs["arg1"])
    C = prod(size(a, i) for i in 1:length(nshape))
    length(a) % C == 0 || error(
        "DNNKernels: `_fused_rms_norm` (op $(op.id)) normalises $(C) elements " *
        "of a $(size(a)) operand, which does not divide it. The normalised axes " *
        "have to be leading and dense here; `hoistpermutes` is what makes them " *
        "so.")
    groups = length(a) ÷ C
    ε = Float32(something(get(op.attrs, "arg3", nothing), eps(float(eltype(a)))))
    rstd = destor(emitctx, 1, Float32, (groups,))
    M.dispatch!(emitctx.g, rmsnorm_kernel!,
                (out, rstd, a, γ === nothing ? a : γ, Int32(C), ε,
                 Val(γ !== nothing)),
                groups * LN_WG; group = LN_WG, name = op.id)
    return (out, rstd)
end

"""
    resultread(emitctx, i) -> Bool

Does the GRAPH read element `i` of the op being emitted?

`maybedest` answers a different question, whether a resource was declared for
it, and under `keepall` every element has one whether or not anything reads it.
An emit that produces some of a torch schema's results and not others has to ask
about the graph, so it asks the graph.
"""
resultread(emitctx::EmitCtx, i::Integer) =
    any(values(emitctx.aten.buffers)) do b
        b.kind === :view && occursin("getitem", b.viewop) &&
            b.of == emitctx.outid[] && Int(b.attrs["arg1"]) == i
    end

"""
`aten::lstm.input`: a whole recurrent layer as one op, declared.

Three passes per direction and one of them is the recurrence:

  * `w_ih` transposed to `(4H, D)`, so the input projection is a GEMM in the
    layout `gemm!` reads;
  * that projection over the WHOLE sequence at once, with `b_ih` in its store.
    `W_ih x_t` does not depend on `h`, which is what leaves a sequential loop
    small enough to fit in one workgroup;
  * `w_hh` transposed to `(4H, H)`, which is worth 6x inside the loop;
  * `lstm_kernel!`, one workgroup of `4H` threads carrying `h` and `c` in shared
    memory across every timestep.

`kernels/extern/lstm.jl` has the arithmetic and the measurements. What changes
here is only where the intermediates live: `Gx` and the two transposed copies
were `scratch!` from an arena that reset per op, so their bytes could never be
reused by anything else; declared, the placer aliases them against the whole
graph.

The transposes are per REPLAY, not per model, because a weight's layout is not
something an emit may rewrite; `hoistpermutes` is the pass that does that, and
it works on `permute` ops the export produced rather than on the inside of a
composite op. 3.6 MB of copies per layer against a recurrence that reads `WhhT`
once per timestep.

Which configurations are accepted is `lstmconfig`'s, shared with the interpreted
route. The state is NOT returned: `lstm_kernel!` keeps `h` and `c` in shared
memory and never writes them out, so a graph that reads them is refused rather
than handed memory nothing wrote.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("lstm.input")})
    x = operand(emitctx, op.ins[1])                       # (D, T, N)
    hx = [operand(emitctx, String(e)[2:end]) for e in op.attrs["arg1"]]
    ps = [operand(emitctx, String(e)[2:end]) for e in op.attrs["arg2"]]
    # Which configurations are this recurrence is the KERNEL's contract, so it
    # is stated once beside the kernel and both routes read it.
    D, nsteps, H, ndir = lstmconfig(op, x, ps)
    for i in 1:2
        resultread(emitctx, i) || continue
        error("DNNKernels: `lstm` (op $(op.id)) is read for element $(i), its " *
              "final $(i == 1 ? "h" : "c"). `lstm_kernel!` keeps the state in " *
              "shared memory and writes out only the per-step output, so there " *
              "is nothing to return for it.")
    end
    out = dest(emitctx, 0)
    size(out, 1) == ndir * H || error(
        "DNNKernels: `lstm` (op $(op.id)) produces $(ndir * H) features per " *
        "step and its result is declared $(size(out)).")
    size(out, 2) == nsteps || error(
        "DNNKernels: `lstm` (op $(op.id)) reads $(nsteps) steps and its result " *
        "is declared $(size(out)), which holds $(size(out, 2)) of them. The " *
        "export writes a composite op's result shapes from the TRACE; see " *
        "`resultshapes`.")
    outm = M.viewof(out, (ndir * H, nsteps))
    x2 = M.viewof(x, (D, nsteps))
    h0, c0 = hx[1], hx[2]
    for d in 0:(ndir - 1)
        w_ih, w_hh, b_ih, b_hh = ps[4d + 1], ps[4d + 2], ps[4d + 3], ps[4d + 4]
        # `(D, 4H)` -> `(4H, D)` and `(H, 4H)` -> `(4H, H)`: one parent stride
        # per axis of the copy, which is what a transpose is to `stridedcopy!`.
        w_ihT = scratch(emitctx, Float32, 4H, D)
        M.dispatch!(emitctx.g, stridedcopy!, (w_ihT, (4H, D), w_ih, (D, 1), 0),
                    4H * D; name = "$(op.id).ihT$(d)")
        Gx = scratch(emitctx, Float32, 4H, nsteps)
        gemm!(emitctx, op, Gx, w_ihT, x2; bias = b_ih)
        WhhT = scratch(emitctx, Float32, 4H, H)
        M.dispatch!(emitctx.g, stridedcopy!, (WhhT, (4H, H), w_hh, (H, 1), 0),
                    4H * H; name = "$(op.id).hhT$(d)")
        # The initial state is `(H, N, ndir)`, so a direction's is one plane.
        plane = d * size(h0, 1) * size(h0, 2)
        M.dispatch!(emitctx.g, lstm_kernel!,
                    (outm, Gx, WhhT, b_hh,
                     M.viewof(h0, (H,); offset = plane),
                     M.viewof(c0, (H,); offset = plane),
                     nsteps, d * H, Val(H), Val(d == 1)),
                    4H; group = 4H, name = "$(op.id).dir$(d)")
    end
    return (out,)
end

"""
`aten::scatter.src`: `self` with `src` written at the coordinates `index` names
along one axis.

Two passes, and the first is the whole of `self` — the same trade
`slice_scatter` makes and for the same reason: the graph may read `self` after
this op, so writing into its bytes would change a value something else wants.
Where it does not, `Aliasing` notices, because the copy declares a read of
`self` and a write of `out`.

The scatter itself is `ndrange = size(idx)`, one thread per index element, and
the index values are torch's 0-based ones.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("scatter.src")})
    a = operand(emitctx, op, 1)
    idx = operand(emitctx, op, 3)
    src = operand(emitctx, op, 4)
    out = dest(emitctx)
    n = ndims(out)
    d = jdim(Int(op.attrs["arg1"]), n)
    size(idx) == size(src) || error(
        "DNNKernels: `scatter.src` (op $(op.id)) has index $(size(idx)) and " *
        "src $(size(src)); they index the same positions, so they are the same " *
        "shape.")
    ndims(idx) == n || error(
        "DNNKernels: `scatter.src` (op $(op.id)) has a $(ndims(idx))-d index " *
        "and a $(n)-d self.")
    od = size(out)
    M.dispatch!(emitctx.g, ew!,
                (out, od, (a,), (bcstrides(od, size(a)),), identity),
                prod(od); name = "$(op.id).self")
    length(idx) == 0 && return out
    M.dispatch!(emitctx.g, scatter_kernel!, (out, idx, src, Val(d), Val(n)),
                size(idx); name = op.id)
    return out
end

"""
`aten::topk`: the `k` largest along one axis, and where they were.

Two dispatches of one body, `Val(WANTIDX)` choosing which result it stores --
the same shape `max.dim` has, for the same reason: a kernel writes one element
per thread and these are two tensors.

One thread per OUTPUT element, so the thread producing rank `r` re-runs the
selection `r+1` times: O(k*n) per slice with no sort, no shared memory and no
cross-thread agreement. `TOPK_MAX_N` is where that stops being the right trade
and it refuses rather than degrading, as does `largest = false`.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("topk.default")})
    a = operand(emitctx, op.ins[1])
    vals = dest(emitctx, 0)
    k = Int(op.attrs["arg1"])
    d = jdim(Int(get(op.attrs, "arg2", -1)), ndims(a))
    Bool(something(get(op.attrs, "arg3", nothing), true)) || error(
        "DNNKernels: `topk` (op $(op.id)) asks for the SMALLEST k, and " *
        "`topk_body` selects the largest. That is a different comparison, not " *
        "this one with a flag.")
    nn = size(a, d)
    k <= nn || error(
        "DNNKernels: `topk` (op $(op.id)) asks for $(k) of axis $(d), which " *
        "has $(nn).")
    nn <= TOPK_MAX_N || error(
        "DNNKernels: `topk` (op $(op.id)) scans an axis of $(nn) and this " *
        "kernel is O(k*n) per output -- above $(TOPK_MAX_N) it needs a sorting " *
        "network, not this.")
    sz = ntuple(i -> i == d ? k : size(a, i), ndims(a))
    inds = destor(emitctx, 1, eltype(vals), sz)
    mapbody!(emitctx, op, topk_body, vals, a, Val(d), Val(false);
             name = "$(op.id).values")
    mapbody!(emitctx, op, topk_body, inds, a, Val(d), Val(true);
             name = "$(op.id).indices")
    return (vals, inds)
end

"""
    viewowner(r) -> resource

The resource whose bytes `r` names: a view's parent, transitively.

`Mantle.rootresource` answers this for a `BufferRange` and an `Attr` and NOT for
a `ResourceView`, deliberately — it decides which resource a USAGE names, and
folding views into their parent there would change what orders against what for
every graph. The question here is narrower: does writing through `r` reach the
caller's storage, or a copy of it. A `Buffer` is the caller's; a
`TransientBuffer` is the placer's.
"""
viewowner(r) = r
viewowner(r::M.ResourceView) = viewowner(r.parent)

"""
`aten::index_put(self, indices, values)` with ONE index tensor: `self` with the
rows `indices` names along that axis replaced.

This is how a KV cache is written — Whisper's decoder does it eight times a step,
and K2 Horizon 32B sixty-four times a token.

**The index never comes to the host.** `hostidx` in `runop!` downloads it to
build a Julia `view`, and a download is a `flush!` plus `vkWaitSemaphores`: on
Horizon that drained the queue 128 times per token, 1232 of 1458 profile samples
inside `execute!`. A recorded plan could not do it at all — a host read during a
recording sees a buffer that has not been written. Declared, the index is an
ordinary read operand and `indexput_kernel!` does the arithmetic.

Two shapes are refused rather than guessed:

  * `inplace`, which `foldcacheupdate` sets once it has proved that `self` is a
    view of a graph input nothing else reads first. The write then has to land in
    THAT buffer and `op.out` has to be it — not a copy — and a declaration that
    quietly copied instead would leave the caller's cache unwritten with the
    right numbers everywhere this graph looks. `declare!` gives `op.out` storage
    of its own, so making this work means teaching it that an in-place op
    declares none; that is a change to the declaration rule, not to this emit.
  * `accumulate`, and more than one index tensor: `runop!` reaches the host for
    both — a loop for the non-fp32 accumulate, a `view` for advanced indexing.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("index_put.default")})
    a = operand(emitctx, op, 1)
    src = operand(emitctx, op.ins[end])
    n = ndims(a)
    inplace = Bool(something(get(op.attrs, "inplace", nothing), false))
    # IN PLACE: the result IS `self`, so there is no copy and no buffer of its
    # own — `consumedids` gave it none. The write has to land in the caller's
    # bytes, which means `self` must be a DESCRIPTOR over them and not a
    # materialised copy of them; a trailing-axis `select.int` is (see
    # `viewfor`), and anything else is refused rather than written to a copy
    # nobody reads.
    out = if inplace
        viewowner(a) isa M.Buffer || error(
            "DNNKernels: `index_put` (op $(op.id)) is `inplace`, so it writes " *
            "through its `self` — but `self` resolved to a " *
            "$(typeof(viewowner(a))), which is storage of its own rather than a " *
            "window onto the caller's. `viewfor` materialises any view that " *
            "moves its elements, and a write into that copy would leave the " *
            "caller's cache unwritten.")
        emitctx.res[op.out] = a
        a
    else
        dest(emitctx)
    end
    Bool(something(get(op.attrs, "arg3", nothing), false)) && error(
        "DNNKernels: `index_put` (op $(op.id)) accumulates. That is a " *
        "scatter-ADD, which `runop!` does on the device only for fp32 " *
        "(`scatteradd_kernel!`, through `OpAtomicFAdd`) and on the HOST " *
        "otherwise — and a host loop is not something a plan can hold.")
    # The index tensors, by Julia axis. torch lists them from the last dim.
    ids = Tuple{Int,Any}[]
    for (k, e) in enumerate(op.attrs["arg1"])
        e === nothing && continue
        jd = n - k + 1
        1 <= jd <= n || error(
            "DNNKernels: `index_put` (op $(op.id)) indexes torch dim $(k - 1) of " *
            "a $(n)-d input.")
        push!(ids, (jd, operand(emitctx, String(e)[2:end])))
    end
    length(ids) == 1 || error(
        "DNNKernels: `index_put` (op $(op.id)) has $(length(ids)) index tensors, " *
        "which is advanced indexing — `runop!` builds a host `view` for it and " *
        "has never seen one in a profile.")
    d, idx = ids[1]
    ndims(src) == n || error(
        "DNNKernels: `index_put` (op $(op.id)) writes a $(ndims(src))-d source " *
        "into a $(n)-d input; `indexput_kernel!` walks the source's own " *
        "coordinates, so they have the same rank.")
    # `self` into the result first, when there IS a separate result: out of
    # place, so `self` survives for whatever else reads it. `Aliasing` is what
    # notices when nothing does.
    if !inplace
        od = size(out)
        M.dispatch!(emitctx.g, ew!,
                    (out, od, (a,), (bcstrides(od, size(a)),), identity),
                    prod(od); name = "$(op.id).self")
    end
    length(src) == 0 && return out
    M.dispatch!(emitctx.g, indexput_kernel!,
                (out, src, M.viewof(idx, (length(idx),)), Val(d), Val(n),
                 Val(size(src)), Int64(length(src))),
                length(src); group = 256, name = op.id)
    return out
end

"""
`aten::alias.default` as an OP: the result is the operand, and nothing runs.

`foldcacheupdate` produces these — the `cat` that rebuilt a KV cache from its
slices becomes the identity on the tensor the in-place `index_put`s already
wrote. `runop!` returns `value(ctx, op.ins[1])` and this registers the same
resource under the output's id, which `consumedids` left without storage of its
own (see `isaliasing`).

`detach.default` is the same statement about autograd and reaches the emit the
same way.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("alias.default")})
    a = operand(emitctx, op, 1)
    emitctx.res[op.out] = a
    return a
end
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("detach.default")}) =
    emitop!(emitctx, op, Val(Symbol("alias.default")))

# ── attention ────────────────────────────────────────────────────────────────

"""
    sdpaoperand(ctx, op, pos)

Operand `pos` of an attention op, strided where it can be.

`attn_flash_cm!` takes each of q, k and v as a root, a base offset and four
strides, so a permuted or sliced view is something it reads rather than a copy it
is handed. Asked for here and not in `operand` because every other emit takes its
operands dense.
"""
function sdpaoperand(emitctx::EmitCtx, op::Op, pos::Int)
    s = strideview(emitctx, op, pos)
    return s === nothing ? operand(emitctx, op, pos) : s
end

"""
    strideview(ctx, op, pos) -> StridedOperand or nothing

Operand `pos` of `op` as a strided read of a resource, for the emits whose kernel
takes strides. `nothing` for a host scalar, for an operand that is not there, and
for a view [`stridedoperand`](@ref) cannot describe.
"""
function strideview(emitctx::EmitCtx, op::Op, pos::Int)
    haskey(op.attrs, argkey(pos)) && return nothing
    idx = pos - count(p -> haskey(op.attrs, argkey(p)), 1:(pos - 1))
    idx <= length(op.ins) || return nothing
    return stridedoperand(emitctx, op.ins[idx])
end

"""
`aten::_scaled_dot_product_{flash,efficient}_attention`, as declared flash
attention.

Both spellings are the same computation — torch distinguishes which CUDA kernel
it would have used, which says nothing about ours — so they share this body, as
they shared `sdpa` before.

The plan is chosen by the same `flashcm_plan` an immediate call asks, from
`eltype` and `size` alone, which is all a declaration has. `FlashCM2Plan` and
the two fallbacks are REFUSED by name rather than written untested: each has its
own launch shape, and a declared form of one has to be split from its launcher
the way `flash_launches` was. A refusal names the plan, which is what tells the
next person which one to port.

torch returns four results and only the first is read; the export declares the
other three empty, and they are handed back so the tuple's shape is honest.
"""
function emitsdpa!(emitctx::EmitCtx, op::Op; dst = dest(emitctx, 0),
                   defaultscale = nothing)
    q = sdpaoperand(emitctx, op, 1)
    k = sdpaoperand(emitctx, op, 2)
    v = sdpaoperand(emitctx, op, 3)
    bias = length(op.ins) >= 4 ? operand(emitctx, op.ins[4]) : nothing
    sc = get(op.attrs, "scale", nothing)
    scale = sc !== nothing ? Float64(sc) :
            defaultscale !== nothing ? Float64(defaultscale) :
            inv(sqrt(size(q, 1)))
    caps = M.caps(M.backend(emitctx.dev))
    E, Lq, H, B = size(q)
    want = (size(v, 1), Lq, H, B)
    # The kernels read `out` at its rank-4 extents. `fused.sdpa` declares a
    # rank-3 buffer -- `fuseattention` folds the head axis away, which is the
    # same elements in the same order -- so a `viewof` puts the rank back rather
    # than a second launch shape.
    out = size(dst) == want ? dst :
          length(dst) == prod(want) ? M.viewof(dst, want) :
          error("DNNKernels: `$(op.aten)` (op $(op.id)) declares a $(size(dst)) " *
                "result where its operands give $(want).")
    cm2 = flashcm2_plan(caps, q, k, v, bias)
    cm2 isa Decline || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) wants $(cm2), whose launch is " *
        "not split from `sdpaflashcm2!` yet, so it has no declared form. " *
        "`FlashCMPlan` is the one that is ported — see `flash_launches` for the " *
        "shape a port takes.")
    plan = flashcm_plan(caps, q, k, v, bias)
    if plan isa Decline
        cm = coopmat_sdpa_plan(caps, q, k, v, bias)
        cm isa Decline || error(
            "DNNKernels: `$(op.aten)` (op $(op.id)) wants $(cm), whose launch is " *
            "not split from `sdpa_coopmat!` yet, so it has no declared form. " *
            "See `flash_launches` for the shape a port takes.")
        # `threepass!` stages its operands and reads them densely, so a
        # descriptor is materialised here rather than inside it.
        threepass!(emitctx, op, out,
                   q isa StridedOperand ? operand(emitctx, op, 1) : q,
                   k isa StridedOperand ? operand(emitctx, op, 2) : k,
                   v isa StridedOperand ? operand(emitctx, op, 3) : v,
                   bias, scale)
        return sdparesults(emitctx, dst)
    end
    ns = plan.nsplit
    # Flash-decoding scratch, declared rather than bump-allocated: only when the
    # plan splits the key axis, so the single-split path declares nothing extra.
    partial = ns == 1 ? out : scratch(emitctx, Float32, size(v, 1), Lq, H, B, ns)
    ml      = ns == 1 ? out : scratch(emitctx, Float32, Lq, H, B, ns, 2)
    flash_dispatch!(emitctx.g, caps, out, plan, q, k, v, scale, partial, ml; name = op.id)
    return sdparesults(emitctx, dst)
end

"""
What an sdpa op hands back: `dst` alone, or torch's four-tuple.

`_scaled_dot_product_*` returns four values and the graph reads only the first;
the export declares the other three and `maybedest` answers `nothing` for any it
left out, so the tuple's shape stays honest. `fused.sdpa` is one value — the
fusion pass that built it collapsed the rest — and is told apart by whether the
export gave this op a `"#0"` result at all.
"""
sdparesults(emitctx::EmitCtx, dst) =
    maybedest(emitctx, 0) === nothing ? dst :
    (dst, maybedest(emitctx, 1), maybedest(emitctx, 2), maybedest(emitctx, 3))

"""
The three-pass path: always available, always right, and the slowest.

`scores` and `sums` are working storage and are declared transients, so the
placer aliases them against the whole graph — `Workspace` could only ever reuse
them within this one op, and on SAM 2's global blocks the score matrix is the
largest thing the graph asks for.

`q` is transposed to `(L, E, H, B)`, which is the layout `attn_scores` reads;
`k` and `v` need no `densify` because a declared operand is already dense (see
`viewstrides`). The three launches are the same bodies and the same blocked
kernels the immediate path picks, through the selectors that decision was split
into.
"""
function threepass!(emitctx::EmitCtx, op::Op, out, q, k, v, bias, scale)
    E, Lq, H, B = size(q)
    Lk = size(k, 2)
    T = accum(eltype(q))
    qt = scratch(emitctx, eltype(q), Lq, E, H, B)
    transposeLE_dispatch!(emitctx.g, qt, q; name = "$(op.id).toLE")
    ST = eltype(qt)
    scores = scratch(emitctx, ST, Lq, Lk, H, B)
    tk = blockfor(Lk, Lq)
    if tk > 1
        nd = (Lq, Lk ÷ tk, H, B)
        M.dispatch!(emitctx.g, scoresblocked!kernel(tk),
                    (scores, qt, k, bias, T(scale)), nd;
                    group = launchgroup(nd), name = "$(op.id).scores")
    else
        mapbody!(emitctx, op, attn_scores, scores, qt, k, bias, scale;
                 name = "$(op.id).scores")
    end
    # Normalises `scores` IN PLACE and writes the sums, which nothing reads --
    # they are declared because the kernel writes them, not because they are
    # wanted.
    sums = scratch(emitctx, T, Lq, H, B)
    mapbody!(emitctx, op, attn_softmax, sums, scores; name = "$(op.id).softmax")
    tq = blockfor(Lq, Lk)
    if tq > 1
        nd = (size(v, 1), Lq ÷ tq, H, B)
        M.dispatch!(emitctx.g, applyblocked!kernel(tq), (out, scores, v, sums), nd;
                    group = launchgroup(nd), name = "$(op.id).apply")
    else
        mapbody!(emitctx, op, attn_apply, out, scores, v, sums; name = "$(op.id).apply")
    end
    return (out, maybedest(emitctx, 1), maybedest(emitctx, 2), maybedest(emitctx, 3))
end

emitop!(emitctx::EmitCtx, op::Op,
        ::Val{Symbol("_scaled_dot_product_flash_attention.default")}) =
    emitsdpa!(emitctx, op)
emitop!(emitctx::EmitCtx, op::Op,
        ::Val{Symbol("_scaled_dot_product_efficient_attention.default")}) =
    emitsdpa!(emitctx, op)

"""
`fused.sdpa`, which `fuseattention` builds from a bmm/softmax/bmm chain.

The same computation and the same `emitsdpa!`, with two things the fusion pass
decided: the SCALE is already folded into `q`, so it defaults to 1 rather than
`1/sqrt(E)`, and there is ONE result rather than torch's four.
"""
emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("fused.sdpa")}) =
    emitsdpa!(emitctx, op; dst = dest(emitctx), defaultscale = 1.0)

# ── matrix products ──────────────────────────────────────────────────────────

"""
    gemm!(emitctx, op, out, A, B; bias = nothing, epi = identity) -> out

`out = A * B` (+ bias, then `epi`) as declared dispatches, in Mantle's layout.

The operands are already swapped by the caller: torch's `a * b` is `b * a` in the
reversed layout, which is what `runop!` passed `matmul!` too.

`mmplan` decides which path from TYPES and SHAPES — the same function an
immediate `matmul!` asks, so a declared product and an immediate one cannot take
different kernels. Only the cooperative-matrix path is declared here; every other
plan REFUSES by name rather than being written untested. A refusal names the
plan and the shape, which is what tells the next person which one to port and
what to check it against.
"""
function gemm!(emitctx::EmitCtx, op::Op, out, A, B; bias = nothing, epi = identity)
    dev = emitctx.dev
    caps = M.caps(M.backend(dev))
    # `Core.Typeof` of the OPERAND, not `devicetype`: what the plan asks is
    # whether the operand is a dense rank-2 matrix of a given element type, which
    # is a property of the operand and which `densematrix` answers for a resource
    # and for an array alike. `devicetype` answers what the kernel RECEIVES,
    # which is a different question and a different type family.
    plan = mmplan(caps, Core.Typeof(out), Core.Typeof(A), Core.Typeof(B),
                  size(out), size(A), size(B), biasfoldable(bias, size(A, 1)))
    if plan isa Decline
        # Mantle's own scalar GEMM, and NOT a library call: `mul!` on a device
        # array is this backend's kernel, so it is declared like any other
        # dispatch. `astranspose` is not applied because it is the identity for
        # anything but a `PermutedDimsArray`, and a declared resource is never
        # one — `hoistpermutes` resolved those at build.
        Mm, N, K = size(out, 1), size(out, 2), size(A, 2)
        # The split-K GEMV's planes, declared rather than allocated.
        S = N == 1 ? M.gemv_split(Mm, K) : 1
        parts = S > 1 ? scratch(emitctx, Float32, Mm, 1, S) : nothing
        M.scalar_gemm_dispatch!(emitctx.g, out, A, B, Mm, N, K,
                                one(eltype(out)), zero(eltype(out));
                                name = op.id, partials = parts)
        # The scalar path has no epilogue to fold into, so the bias and the
        # activation are passes here — the same ones the graph would have run as
        # their own ops. Folding is an optimisation on the tensor-core path,
        # never a correctness requirement.
        od = size(out)
        bias === nothing || M.dispatch!(emitctx.g, ew!,
            (out, od, (out, bias), (bcstrides(od, od), bcstrides(od, size(bias))), +),
            prod(od); name = "$(op.id).bias")
        epi === identity || M.dispatch!(emitctx.g, ew!,
            (out, od, (out,), (bcstrides(od, od),), epi),
            prod(od); name = "$(op.id).act")
        return out
    end
    if plan isa MMGemvPlan
        # The matrix-VECTOR product: one pass with the bias and the epilogue in
        # its store. `A` is the `(M, K)` matrix and `B` its single column, which
        # is why the immediate path spells this `gemv!(out, B, transpose(A))` --
        # the `Transpose` is that call's dispatch between the two layouts, and a
        # declaration names the layout instead.
        M.gemv_dispatch!(emitctx.g, out, B, A, size(A, 1), size(A, 2);
                         bias, epilogue = epi, name = op.id)
        return out
    end
    plan isa MMCoopMatPlan || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is $(size(A)) * $(size(B)) into " *
        "$(size(out)) and `mmplan` chose $(plan), which has no declared form " *
        "yet. `MMCoopMatPlan`, `MMGemvPlan` and `Decline` are ported; " *
        "`MMInt8Plan` still launches immediately (`matmul!`) and a graph cannot " *
        "hold that. Port the plan rather than widening this branch.")
    Mm, K = size(A)
    N = size(B, 2)
    NP = plan.NP
    # `N` padded up to the kernel's block: `B` is copied into the leading columns
    # of a `K x NP` scratch and the rest zeroed. A declared transient, so the
    # placer aliases it against the whole graph — `Workspace` could only ever
    # reuse it within this one op.
    Bp = B
    if NP != N
        Bp = scratch(emitctx, Float16, K, NP)
        M.dispatch!(emitctx.g, padcols_kernel!, (Bp, B, Val(K), N), (K, NP);
                    name = "$(op.id).padB")
    end
    blk_split = M.coopmat_gemm_shape(Mm, NP, K)
    splitk = blk_split[2]
    if splitk == 1
        # Nothing to reduce: the GEMM starts its accumulators from the bias and
        # converts to `out`'s type as it stores, so there is no fp32 scratch and
        # no second pass. A padded `N` does not force one back either — the
        # destination is column-major, so columns 1..N of an `Mm x NP` buffer are
        # its first `Mm*N` elements contiguously, and the discard is a linear
        # copy rather than a gather.
        dst = NP == N ? out : scratch(emitctx, eltype(out), Mm, NP)
        M.coopmat_gemm_dispatch!(emitctx.g, dst, A, Bp, Mm, NP, K;
                                 blk_split, bias, epilogue = epi, name = op.id)
        # Columns 1..N of the padded buffer ARE its first `Mm*N` elements, so
        # the discard is a linear copy between two views of that shape rather
        # than a gather. `ew!` with `identity` and not `Mantle.copy!`: that pass
        # records `vkCmdCopyImageToBuffer` and wants an image attachment, and a
        # transfer command is not something the access walk can read off a
        # kernel body. A dispatch is.
        if NP != N
            od = size(out)
            src = M.viewof(dst, od)
            M.dispatch!(emitctx.g, ew!,
                        (out, od, (src,), (bcstrides(od, od),), identity),
                        prod(od); name = "$(op.id).unpad")
        end
        return out
    end
    C = scratch(emitctx, Float32, Mm, NP, max(splitk, 1))
    M.coopmat_gemm_dispatch!(emitctx.g, C, A, Bp, Mm, NP, K;
                             blk_split, partials = C, reduce = false, name = op.id)
    M.dispatch!(emitctx.g, mm_epilogue_kernel!,
                (out, C, bias, epi, Val(Mm), Val(splitk), Mm * NP, Mm * N),
                Mm * N; name = "$(op.id).epilogue")
    return out
end

"""`aten::mm(a, b)`, which in the reversed layout is `b * a`."""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("mm.default")})
    a, b = operand(emitctx, op, 1), operand(emitctx, op, 2)
    out = dest(emitctx)
    return gemm!(emitctx, op, out, b, a)
end

"""
`aten::addmm(bias, a, b)` = `bias + a*b`, with the bias and any folded
activation inside the GEMM's store.

`act` is set by `foldrelu`, which deleted the activation op and aliased its
buffer onto this one; `epilogue` is the general form, any unary elementwise
expression as a `FusedOp`. Both are applied in the store, so the fused form
reads and writes the result once instead of three times — and passing the
callable as an argument is the function barrier that keeps it static, since it
comes out of a `Dict{String,Any}`.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("addmm.default")})
    bias = operand(emitctx, op, 1)
    a, b = operand(emitctx, op, 2), operand(emitctx, op, 3)
    out = dest(emitctx)
    epi = get(op.attrs, "epilogue", nothing)
    f = epi === nothing ? actfn(Symbol(get(op.attrs, "act", "none"))) : epi
    return gemm!(emitctx, op, out, b, a; bias, epi = f)
end

"""
`aten::bmm(a, b)`, one declared product per batch plane.

A plane is `Mantle.slice` of the operand, which for the trailing axis is a
contiguous range — so the slice is a descriptor and not a copy, and each plane
gets the same capability dispatch a 2-D product does. `runop!` sliced with
`view` for the same reason and the same result.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("bmm.default")})
    a, b = operand(emitctx, op, 1), operand(emitctx, op, 2)
    out = dest(emitctx)
    nb = size(a, 3)
    nb == size(b, 3) == size(out, 3) || error(
        "DNNKernels: `bmm` (op $(op.id)) has batch extents " *
        "$(size(a, 3)), $(size(b, 3)) and $(size(out, 3)).")
    for i in 1:nb
        gemm!(emitctx, op, planeof(emitctx, out, i),
              planeof(emitctx, b, i), planeof(emitctx, a, i))
    end
    return out
end

"""
One batch plane of a rank-3 resource, as a RANK-2 view at that plane's offset.

`Mantle.slice` would be wrong here even though the elements are the same: it
answers a `BufferRange`, which is rank 1, and `mmplan` asks whether the operand
is a `LavaArray{Float16,2}` — so every plane would decline to a path that has no
declared form. `viewof` keeps the rank, which is what makes a plane get the same
capability dispatch a 2-D product does.

The trailing axis is the batch, so a plane is contiguous and the view is a
descriptor rather than a copy.
"""
planeof(emitctx::EmitCtx, x, i::Integer) =
    M.viewof(x, (size(x, 1), size(x, 2));
             offset = (i - 1) * size(x, 1) * size(x, 2))

# ── convolution ──────────────────────────────────────────────────────────────

"""
`aten::convolution`, forward and dense, as the implicit GEMM plus whatever
split-K needs.

**Everything is decided here, from shapes.** `convtiles`
and `convsplit` read `Cout`, `NPQ`, `CRS` and the shader-core count, all of which
the graph states or the device reports, so the tiling and the split factor are
`Val` parameters of one dispatch rather than a choice made per launch.

Split-K is the interesting half, and it is what `Workspace` was for. Each split
accumulates into the destination atomically, so:

  * the destination must START at the bias rather than being overwritten, which
    is a pass of its own (`ew!` broadcasting the bias along the channel axis, or
    `fill_kernel!` when there is none);
  * a partial sum cannot be clamped, so a fused activation has to wait until the
    splits are summed, and an fp16 output cannot be the accumulator at all
    because Vulkan 1.3 has no fp16 atomic add. Either case accumulates into an
    fp32 SCRATCH and converts once, in a third pass.

That scratch is a declared transient. `Workspace` bump-allocated it from an arena
that reset per op, so its bytes could only ever be reused by the same op; here
its liveness is what its uses say and the placer aliases it against the whole
graph.

**`splitk > 1` makes this convolution non-reproducible**, because the order the
atomics land in is not fixed and float addition is not associative. Measured on
`segment`'s `convolution_21` (fp32, 3x3, 512 -> 768, NPQ = 64, `splitk = 4`): two
runs on identical inputs differ by 3.05e-5 on 9% of elements, carrying to ~5e-7
at the graph's outputs. Not a bug, it is what buys the 4-8x, but it is the floor
for anything measured on a graph containing one.
"""
function emitop!(emitctx::EmitCtx, op::Op, ::Val{Symbol("convolution.default")})
    x = operand(emitctx, op.ins[1])
    w = operand(emitctx, op.ins[2])
    bias = length(op.ins) >= 3 ? operand(emitctx, op.ins[3]) : nothing
    out = dest(emitctx)
    stride = reverse(ints(op.attrs["arg3"]))
    pad = reverse(ints(op.attrs["arg4"]))
    dil = reverse(ints(op.attrs["arg5"]))
    groups = Int(op.attrs["arg8"])
    act = Symbol(get(op.attrs, "act", "none"))

    # A different lowering, not a variant of this one: it branches before any of
    # the shape arithmetic below, which is the forward convolution's.
    get(op.attrs, "arg6", false) == true &&
        return emitconvtranspose!(emitctx, op, x, w, bias, out, stride, pad, dil,
                                  reverse(ints(op.attrs["arg7"])), groups, act)

    # What is declared so far is the dense forward 2-D case. The others are not
    # refusals on principle, they are unported: each has its own kernel in
    # `kernels/extern/` and its own reason to exist, and a graph that needs one
    # should say so here rather than run as something else. `runop!`'s note
    # applies unchanged — a `ConvTranspose2d` taken as an ordinary convolution
    # is a wrong picture with nothing in the numbers to point at it.
    # 1-D is its OWN kernel and not the 2-D one with a degenerate axis: see
    # `conv1d`, whose reason for existing is that the inner loop should not carry
    # a trip count of 1. MatAnyone's mask encoders are 1-D convolutions over a
    # flattened mask, and the layout is `(x, c, n)` with the weight reversed to
    # `(kx, ci, co)`, which is what the export already hands over.
    length(stride) == 1 && return mapbody!(emitctx, op, conv1d, out, x, w, bias,
                                           Val(stride[1]), Val(pad[1]), Val(dil[1]),
                                           Val(groups))
    length(stride) == 2 || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is $(length(stride))-D, and only " *
        "the 1-D and 2-D convolutions are declared. 3-D has `convolution3d!` and " *
        "needs an `emitop!` of its own.")
    groups == 1 || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) has $(groups) groups, and only " *
        "the dense convolution is declared. `convolution_direct!` is the " *
        "grouped kernel; it needs an `emitop!` of its own.")

    KWk, KHk, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = Cin * KHk * KWk
    NPQ = N * OH * OW
    T = eltype(out)
    ACC = accum(eltype(x))
    cores = M.caps(M.backend(emitctx.dev)).cores
    BS_K, BS_NPQ, BS_CRS, WG, TS_K, TS_NPQ = convtiles(Cout, NPQ; cores)
    nbk = cld(Cout, BS_K)
    nbn = cld(NPQ, BS_NPQ)
    splitk = convsplit(nbk, nbn, cld(CRS, BS_CRS); cores)

    # An fp32 destination with no fused activation can take the atomics itself;
    # anything else needs the scratch. Declared, not allocated.
    direct = splitk == 1 || (T === Float32 && act === :none)
    acc = direct ? out : scratch(emitctx, Float32, size(out)...)
    # Fold the activation into the write-back only when there is a single split.
    kact = (splitk == 1 && act === :relu) ? :relu : :none

    if splitk > 1
        od = size(acc)
        if bias === nothing
            M.dispatch!(emitctx.g, M.fill_kernel!, (acc, zero(eltype(acc))),
                        length(acc); name = "$(op.id).prefill")
        else
            # The bias lies along the channel axis, which is the third of four in
            # the reversed layout, so it broadcasts with a zero stride everywhere
            # else. One pass, no materialised copy.
            bd = ntuple(k -> k == 3 ? length(bias) : 1, length(od))
            M.dispatch!(emitctx.g, ew!,
                        (acc, od, (bias,), (bcstrides(od, bd),), identity),
                        prod(od); name = "$(op.id).prefill")
        end
    end

    args = (acc, x, w, bias,
            Val(ACC), Val(splitk), Val(kact),
            Val(BS_K), Val(BS_CRS), Val(BS_NPQ), Val(TS_K), Val(TS_NPQ),
            Val(KWk), Val(KHk),
            Val(stride[1]), Val(stride[2]), Val(pad[1]), Val(pad[2]),
            Val(dil[1]), Val(dil[2]),
            Cin, Cout, Wid, Hei, OW, OH, NPQ, CRS, nbn)
    # `bias` is passed as `nothing` when there is none, rather than omitted: the
    # kernel branches on `bias === nothing` at compile, and `nothing` is a
    # zero-size argument the compiled kernel has no parameter for
    # (`Mantle.NotPassed`), so there is no slot and nothing to declare.
    M.dispatch!(emitctx.g, conv2d_igemm_ki!, args, (nbk * WG, nbn * splitk);
                group = (WG, 1), name = op.id)

    # The third pass: convert the fp32 scratch down, applying the activation
    # once now that the splits have been summed.
    if acc !== out
        od = size(out)
        f = act === :relu ? (v -> max(v, zero(v))) : identity
        M.dispatch!(emitctx.g, ew!, (out, od, (acc,), (bcstrides(od, od),), f),
                    prod(od); name = "$(op.id).reduce")
    elseif splitk > 1 && act === :relu
        # In place: `out` is the destination AND the operand, so the walk reports
        # it read+write and the pass is ordered against the splits that wrote it.
        od = size(out)
        M.dispatch!(emitctx.g, ew!,
                    (out, od, (out,), (bcstrides(od, od),), v -> max(v, zero(v))),
                    prod(od); name = "$(op.id).act")
    end
    return out
end

"""
`aten::convolution` with `transposed = true` -- SAM 2's mask decoder upsamples
its 64x64 embedding to 256x256 with two of these.

Two of the three shapes `convolutiontranspose!` runs are declared, and the choice
between them is [`shufflecase`](@ref)'s, the same predicate the immediate path
asks:

  * **non-overlapping** (stride equal to the kernel, no padding, no dilation, one
    group): the pixel-shuffle identity. One GEMM over `(H*W) x C_in x
    (C_out*S*S)` and one interleave, which on SAM 2's decoder is what turns 3.73
    ms of an 8.44 ms decode into two passes.
  * **everything else**, including grouped: the gather, one thread per output
    element. It is the reference `convtranspose2d` and it is correct for every
    case; it is simply the slow one.

What is NOT declared is `convolutiontranspose_phase!`, the overlapping
decomposition (`K == 2S`, `P == S/2`) that Kokoro's iSTFTNet upsampler and RIFE's
flow decoder need: 137 ms of a 217 ms utterance and 63.0 ms of a 170 ms
interpolation respectively, against the gather. It builds a phase-sliced weight
with `S*J` host-side slice copies, and a declared graph wants that weight folded
at load (`hoistconstants`) rather than rebuilt per call, so it is a port with a
design question in it and not a transcription. Graphs that need it take the
gather meanwhile, which is slow and right.
"""
function emitconvtranspose!(emitctx::EmitCtx, op::Op, x, w, bias, out,
                            stride, pad, dil, outpad, groups, act)
    act === :none || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is transposed and has a fused " *
        "`$(act)`. Neither declared path has an epilogue to fold it into; the " *
        "fusion pass should not have attached one.")
    length(stride) == 2 || error(
        "DNNKernels: `$(op.aten)` (op $(op.id)) is a $(length(stride))-D " *
        "transposed convolution, and only the 2-D form is declared.")

    if shufflecase(w, stride, pad, dil, outpad, groups) && size(x, 4) == 1
        return emitconvtransposeshuffle!(emitctx, op, x, w, bias, out, stride)
    end
    # `output_padding` needs no code here: it only chooses the output SIZE, and
    # the gather computes each output position from whichever inputs reach it --
    # of which the padded positions have none.
    return mapbody!(emitctx, op, convtranspose2d, out, x, w, bias,
                    Val(stride[1]), Val(stride[2]), Val(pad[1]), Val(pad[2]),
                    Val(dil[1]), Val(dil[2]), Val(groups))
end

"""
The non-overlapping transposed convolution: one GEMM and one interleave.

`size(x, 4) == 1` is required because the GEMM flattens `W` and `H` into one
axis, and those are only adjacent in memory within a single batch element.

The weight arrives `(KX, KY, C_out, C_in)` and the GEMM wants
`(C_in, KX*KY*C_out)`. The immediate path writes that as `permutedims(w, (4, 1,
2, 3))` and a `reshape`, and it is exactly the TRANSPOSE of `reshape(w,
KX*KY*C_out, C_in)` -- element `(ci, c)` of the target is `w[c + KX*KY*C_out *
(ci - 1)]` either way -- so it is one `stridedcopy!` and needs no permute kernel.

That transpose is of a graph CONSTANT and belongs at load time, which is
`hoistconstants`' territory; declared per call it is one pass over 2*2*C_out*C_in
elements (65k on SAM 2's decoder, both layers together).
"""
function emitconvtransposeshuffle!(emitctx::EmitCtx, op::Op, x, w, bias, out, stride)
    Wi, Hi, Ci = size(x, 1), size(x, 2), size(x, 3)
    KX, KY, Co = size(w, 1), size(w, 2), size(w, 3)
    ncol = KX * KY * Co
    T = eltype(out)

    wm = scratch(emitctx, eltype(w), Ci, ncol)
    M.dispatch!(emitctx.g, stridedcopy!, (wm, (Ci, ncol), w, (ncol, 1), 0),
                Ci * ncol; name = "$(op.id).weight")
    xm = M.viewof(x, (Wi * Hi, Ci))
    gemmout = scratch(emitctx, T, Wi * Hi, ncol)
    gemm!(emitctx, op, gemmout, xm, wm)
    return mapbody!(emitctx, op, shuffleout, out, gemmout, bias,
                    Val(stride[1]), Val(stride[2]), Int32(Wi))
end
