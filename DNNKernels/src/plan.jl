"""
Static buffer planning.

`torch.export` hands us a lifetime for every transient: `Buffer.live` is the
`[first_op, last_op]` range over which the value is needed. Two buffers whose
lifetimes do not overlap can occupy the same memory, so the whole graph's
scratch space can be laid out *once*, at load time, into a single slab — and
then nothing allocates at run time at all.

That is the point. It is not a cheaper allocator: a pool still makes one
allocation call per op, and this makes none. Measured on this model, the
lifetimes are worth a lot:

    graph                 sum of transients   peak concurrent   reuse
    encode_image                    53.1 MB           4.2 MB    12.6x
    segment                         81.4 MB          25.5 MB     3.2x
    readout_query                    9.8 MB           0.5 MB    18.0x

The graphs run one after another inside a step, so a single slab sized to the
largest peak — 25.5 MB — serves all eight, against 179 MB of separate
allocations.

The second reason to want this is that it pins device addresses: with a fixed
slab the pointer for every op is the same on every step, which is the
precondition for capturing the launch sequence once and replaying it.
"""

"""Round up to a 256-byte boundary — the alignment every GPU backend is happy with."""
alignup(n::Integer, a::Integer = 256) = ((n + a - 1) ÷ a) * a

struct Slab
    offsets::Dict{String,Int}   # buffer id -> byte offset into the slab
    sizes::Dict{String,Int}     # buffer id -> bytes reserved there
    bytes::Int                  # total slab size
end

# `place` for the plan this file produces — see its docstring in execute.jl.
place(p::Slab, ::Nothing, id::AbstractString, ::Type{T}, dims) where {T} = nothing

function place(p::Slab, sl, id::AbstractString, ::Type{T}, dims) where {T}
    off = get(p.offsets, id, nothing)
    # The slot was reserved from the *declared* dtype and shape. A caller asking
    # for a different element type (ops that allocate in `eltype(x)`) could
    # otherwise overrun into the neighbouring buffer, so fall back to a real
    # allocation unless it demonstrably fits.
    (off !== nothing && prod(dims) * sizeof(T) <= get(p.sizes, id, 0)) || return nothing
    # `derive` hands back a real device array sharing the slab's buffer rather
    # than a `reshape(reinterpret(view(...)))` stack. Worth ~7% on Lava (20.6 ->
    # 22.1 steps/s), presumably from the simpler index arithmetic and fewer
    # wrapper layers reaching each kernel.
    #
    # Not, as first assumed, because it switches broadcasts from the Cartesian
    # kernel to the linear one — it does not, and it should not: GPUArrays only
    # uses the linear path when every operand is linearly indexable *and* the
    # shapes match exactly, which is untrue for most ops here.
    # `broadcast_kernel_cartesian` dominating the dispatch count is correct
    # behaviour, not a pathology.
    #
    # Offsets are 256-byte aligned so the element offset is exact.
    slabview(T, sl, dims, off)
end

"""
Buffer ids that outlive the graph: its declared outputs, plus anything they are
views of. A step chains eight graphs through one slab, so a value that escapes
into the next graph must not sit in memory the next graph will overwrite.
"""
function escaping(graph::Graph)
    esc = Set{String}()
    function mark(id)
        id in esc && return
        push!(esc, id)
        b = get(graph.buffers, id, nothing)
        b === nothing && return
        isempty(b.of) || mark(b.of)
    end
    foreach(mark, graph.outputs)
    esc
end

"""
    lifetimes(graph) -> Dict(id => (first_op, last_op))

Lifetimes computed from *our* execution, not from `Buffer.live`.

The exporter's annotation describes torch's own evaluation, which materialises
eagerly. We resolve views lazily in `makeview`, so a parent can still be read
long after torch considers it dead — reusing its memory at the annotated point
corrupts the result. Walking the ops and following view chains to their root
gives the range that is actually safe.
"""
function lifetimes(graph::Graph, lazy = fusableset(graph))
    root(id) = begin
        b = get(graph.buffers, id, nothing)
        (b === nothing || isempty(b.of)) ? id : root(b.of)
    end
    # An op whose result stays lazy (see `fuse.jl`) does not read its operands
    # when it "runs" — the consumer does, later. So an operand's last use is the
    # consumer's index, not this op's, and treating the two as the same lets the
    # planner hand the operand's bytes to another buffer while the unevaluated
    # expression still points at them. That is not a subtle corruption: it moved
    # the fp32 matte from 2.9e-4 to 1.1e-3 against PyTorch, 43 pixels instead of
    # 1. Walk the chain so a run of fused ops carries its operands all the way to
    # whoever finally materialises them.
    index = Dict(o.out => i for (i, o) in enumerate(graph.ops))
    consumer = Dict{String,Int}()
    for (i, o) in enumerate(graph.ops), inp in o.ins
        consumer[inp] = max(get(consumer, inp, 0), i)
    end
    # A lazy value can now reach its consumer through a shape-only view
    # (`lazyreshape`), and a view is a buffer no op names directly, so the direct
    # consumer map reports "nobody reads this" and the chain ends one op too
    # early — the same class of corruption the sink walk exists to prevent.
    viewers = Dict{String,Vector{String}}()
    for (id, b) in graph.buffers
        isempty(b.of) || push!(get!(viewers, b.of, String[]), id)
    end
    function lastread(id, depth = 0)
        depth > 64 && return 0
        m = get(consumer, id, 0)
        for v in get(viewers, id, ())
            m = max(m, lastread(v, depth + 1))
        end
        m
    end
    function sink(id, depth = 0)
        (depth > 64 || !(id in lazy)) && return get(index, id, 0)
        c = lastread(id)
        c == 0 && return get(index, id, 0)
        max(c, sink(graph.ops[c].out, depth + 1))
    end

    first_ = Dict{String,Int}()
    last_ = Dict{String,Int}()
    for (i, o) in enumerate(graph.ops)
        r = root(o.out)
        first_[r] = min(get(first_, r, i), i)
        last_[r] = max(get(last_, r, i), i)
        # If this op's result is deferred, its operands must survive until the
        # expression is finally evaluated.
        upto = o.out in lazy ? max(i, sink(o.out)) : i
        for inp in o.ins
            ri = root(inp)
            last_[ri] = max(get(last_, ri, i), upto)
        end
    end
    Dict(id => (get(first_, id, 0), get(last_, id, 0)) for id in keys(last_))
end

"""
    rootbuffer(graph, id) -> id

Follow a view chain to the buffer that owns the storage. `lifetimes` keys its
ranges by this, since a view has no lifetime of its own.
"""
function rootbuffer(graph::Graph, id::AbstractString, depth::Int = 0)
    depth > 64 && return String(id)
    b = get(graph.buffers, id, nothing)
    (b === nothing || isempty(b.of)) ? String(id) : rootbuffer(graph, b.of, depth + 1)
end

"""View ops that only reinterpret the shape — the same list `makeview` uses."""
const SHAPEONLY_VIEWS = ("view.default", "_unsafe_view.default", "unsqueeze.default",
                         "squeeze.dims", "squeeze.dim")

"""
    materialisedview(graph, b) -> Bool

Whether reading view `b` will force a copy.

`makeview` reshapes a shape-only view of its parent, and Julia cannot reshape a
`PermutedDimsArray` without collapsing it first — so a shape-only view *of a
permute* is materialised by `contiguous`, and every other view stays lazy. That
is decidable from the graph, which is what makes these plannable: on SAM 2's
encoder the predicate names exactly the 51 buffers, totalling 290.2 MB, that the
allocation trace shows `contiguous` allocating at run time.
"""
function materialisedview(graph::Graph, b::Buffer)
    b.kind === :view || return false
    b.viewop in SHAPEONLY_VIEWS || return false
    p = get(graph.buffers, b.of, nothing)
    p === nothing && return false
    p.viewop == "permute.default"
end

"""
    planslab(graph, dims) -> Slab

Lay every transient of `graph` into one slab, reusing memory across
non-overlapping lifetimes.

Greedy by size (largest first, lowest non-conflicting offset), which is what
TFLite and ONNX Runtime use for the same problem: it is not provably optimal but
comes within a few percent, and the plan is computed once at load time so the
cost of computing it does not matter.

Buffers with a tuple shape (`max_pool2d_with_indices`, `native_layer_norm`, …)
carry no single shape and are skipped — those still allocate. So do buffers that
no op produces, which is how a folded-away batch-norm's output looks.

**Fused values get no slab space.** A value `fuse.jl` marks fusable is returned
by `emit` as the `Broadcasted` itself, before `dest` is ever consulted, so it
has no storage to plan; reserving bytes for it reserves them for something that
does not exist. This is not a rounding error — on SAM 2's image encoder it was
**1 334 MB of a 2 020 MB slab**, and dropping it takes the peak to 698 MB
against PyTorch's ~655 MB of activations. It shows up so large because a fused
run reads as a *staircase*: 84 buffers all live at once at the point the chain
finally materialises, each still holding a full 37.7 MB reservation.

Safe in the direction that matters: `lifetimes` still walks the fusion chain, so
the real operands the expression reads stay reserved for as long as it can be
evaluated, and a value that unexpectedly *is* materialised finds no offset and
falls back to `rawalloc` — an allocation, not a corruption.
"""
function planslab(graph::Graph, dims)
    produced = Set(o.out for o in graph.ops)
    esc = escaping(graph)
    lazy = fusableset(graph)
    lt = lifetimes(graph, lazy)
    items = Tuple{String,Int,Int,Int}[]
    # Copies forced by `contiguous`. They are not op outputs, so nothing below
    # would place them and they allocated per call for the life of the model —
    # 290 MB of pool on every SAM 2 encode. They cost nothing to place: given
    # the root's lifetime they slot into holes the greedy already leaves, and
    # the slab does not grow by a byte.
    #
    # The root's lifetime rather than their own is what makes this safe in both
    # directions. It cannot be too short — reading the view is what extends the
    # root's last use, so the root outlives the copy — and it stops the copy
    # from ever being placed on top of the buffer it is a copy *of*, which is
    # the one overlap `permutedims!` cannot survive.
    for (id, b) in graph.buffers
        materialisedview(graph, b) || continue
        id in esc && continue
        r = rootbuffer(graph, id)
        haskey(lt, r) || continue
        isempty(b.shape) && continue
        n = alignup(prod(evalshape(b.shape, dims)) * sizeof(b.dtype))
        push!(items, (id, n, lt[r][1], lt[r][2]))
    end
    for (id, b) in graph.buffers
        b.kind === :transient || continue
        id in produced || continue
        id in esc && continue
        id in lazy && continue
        haskey(lt, id) || continue
        if isempty(b.shape)
            # Multi-output op: place each element under `"<id>.<i>"`, which is the
            # key its handler asks `dest` for. They share the tuple's lifetime,
            # and that lifetime is already correct for them — `lifetimes` walks
            # the view chain, so it ends at the last read of the `getitem` that
            # extracts an element, not at the op that produced the tuple.
            shapes = get(b.attrs, "shapes", nothing)
            shapes === nothing && continue
            dts = get(b.attrs, "dtypes", nothing)
            for (i, s) in enumerate(shapes)
                (s === nothing || isempty(s)) && continue
                dts !== nothing && dts[i] === nothing && continue   # dtype we do not model
                T = dts !== nothing ? dts[i] : b.dtype
                n = alignup(prod(evalshape(s, dims)) * sizeof(T))
                push!(items, ("$(id).$(i - 1)", n, lt[id][1], lt[id][2]))
            end
            continue
        end
        n = alignup(prod(evalshape(b.shape, dims)) * sizeof(b.dtype))
        push!(items, (id, n, lt[id][1], lt[id][2]))
    end
    # largest first: the big buffers are the ones whose placement constrains the
    # slab, so give them the freedom of an empty layout
    sort!(items, by = x -> (-x[2], x[1]))

    placed = Tuple{Int,Int,Int,Int}[]           # offset, bytes, live-from, live-to
    offsets = Dict{String,Int}()
    sizes = Dict{String,Int}()
    total = 0
    for (id, n, ls, le) in items
        off = 0
        while true
            bumped = false
            for (po, pn, pls, ple) in placed
                overlaps_time = !(le < pls || ls > ple)
                overlaps_mem = !(off + n <= po || off >= po + pn)
                if overlaps_time && overlaps_mem
                    off = po + pn                # slide past and rescan
                    bumped = true
                    break
                end
            end
            bumped || break
        end
        push!(placed, (off, n, ls, le))
        offsets[id] = off
        sizes[id] = n
        total = max(total, off + n)
    end
    Slab(offsets, sizes, total)
end

"""
    checkslab(graph, dims, slab) -> (nplanned, nconflicts)

Assert the invariant the plan rests on: no two buffers that are alive at the
same time may share a byte.

Checked against [`lifetimes`], the same ranges `planslab` placed by, and *not*
against `Buffer.live`. The exporter's annotation describes torch's evaluation,
which is shorter than ours wherever a value stays lazy — so a plan that reuses
memory a fused expression can still read looks perfectly legal under `live` and
is exactly the corruption this exists to catch.

Sizes come from `slab.sizes` rather than being recomputed from the shape,
because a multi-output op is planned under `"<id>.<i>"` keys that name no buffer
at all.
"""
function checkslab(graph::Graph, dims, slab::Slab)
    ids = collect(keys(slab.offsets))
    lt = lifetimes(graph, fusableset(graph))
    # `"native_layer_norm_46.2"` is element 2 of the tuple `native_layer_norm_46`,
    # and it is the tuple that carries the lifetime.
    root(id) = (i = findlast('.', id); i === nothing ? id : id[1:(i - 1)])
    bad = 0
    for i in eachindex(ids), j in (i + 1):length(ids)
        a, b = ids[i], ids[j]
        la = get(lt, root(a), nothing); lb = get(lt, root(b), nothing)
        (la === nothing || lb === nothing) && continue
        (la[2] < lb[1] || la[1] > lb[2]) && continue
        oa, ob = slab.offsets[a], slab.offsets[b]
        (oa + slab.sizes[a] <= ob || oa >= ob + slab.sizes[b]) || (bad += 1)
    end
    (length(ids), bad)
end
