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
"""
function planslab(graph::Graph, dims)
    produced = Set(o.out for o in graph.ops)
    esc = escaping(graph)
    lt = lifetimes(graph)
    items = Tuple{String,Int,Int,Int}[]
    for (id, b) in graph.buffers
        b.kind === :transient || continue
        id in produced || continue
        id in esc && continue
        isempty(b.shape) && continue
        haskey(lt, id) || continue
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
same time may share a byte. Cheap enough to run at load time.
"""
function checkslab(graph::Graph, dims, slab::Slab)
    ids = collect(keys(slab.offsets))
    sz = Dict(id => alignup(prod(evalshape(graph.buffers[id].shape, dims)) *
                            sizeof(graph.buffers[id].dtype)) for id in ids)
    bad = 0
    for i in eachindex(ids), j in (i + 1):length(ids)
        a, b = ids[i], ids[j]
        la, lb = graph.buffers[a].live, graph.buffers[b].live
        (la[2] < lb[1] || la[1] > lb[2]) && continue
        oa, ob = slab.offsets[a], slab.offsets[b]
        (oa + sz[a] <= ob || oa >= ob + sz[b]) || (bad += 1)
    end
    (length(ids), bad)
end
