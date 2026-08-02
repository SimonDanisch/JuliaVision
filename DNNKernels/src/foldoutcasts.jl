"""
Fold a narrowing dtype cast into the op that produced its input.

Under autocast the graph is full of `_to_copy` nodes at the precision
boundaries, and `hoistcasts` already evaluates the ones in front of *weights* at
load time. This is the other side: a cast that narrows a value the graph just
computed, where nothing else reads the wide one.

96 of the encoder's 199 runtime casts are exactly that — 51 layer-norm results
and 45 clones, 282 M elements between them. The producer writes four bytes an
element, the cast reads those four and writes two; declaring the producer's
result fp16 in the first place leaves two.

**The values are identical, not merely close.** The producer computes in fp32
either way and the store rounds once to fp16, which is the same rounding the cast
performed. Nothing is reassociated and no intermediate is kept wider than it was.
SAM 2's masks and its worst-case decoder logit come back bit-for-bit unchanged,
which is the check that matters: this pass is only ever allowed to be free.

**Measured 1.4 ms, against about 7 the traffic argument predicts.** Encode,
A/B/A in one session: 135.98 on, 137.38 off, 135.95 on. Removing the ops saves
2.25 GB of reads and writes on paper and delivers a fifth of what that implies,
and none of it is the lazy broadcast fuser having got there first — zero of the
96 were in `fusableset`. So the honest reading is that the serialised per-op
table over-attributes these copies, the same way it inflates every total, and a
pass justified from that table's `_to_copy` row alone would have been oversold.
Kept because 1.4 ms and 96 fewer dispatches for no accuracy cost is still worth
having, but the number in the ledger is the measured one.

Nothing here touches a kernel. `allocate!` takes the dtype from the buffer and
`coerce` forces the declared dtype after every op, so the graph's declaration
*is* the mechanism — retyping the buffer is the whole change.

## Why the readers have to be counted through views

Every one of the 96 reads its input through a view: a layer norm's result is a
`getitem` of the tuple it returns, and a clone's is reshaped before the cast. So
the producer is found by walking the view chain down to a real buffer, and the
reader count has to treat views as transparent — a view is not a use, whoever
reads *it* is. Counting view links as readers makes all 96 look twice-read and
folds nothing, which is the same trap `foldrelu` records for aliases.
"""


"""
Ops whose result may be declared narrower than they would otherwise produce.

Both write every element of their destination through the declared dtype:
`native_layer_norm` allocates with `tupledtype` on both its fused and its
fallback path, and `clone` is an `alloc` plus a broadcast. An op that allocated
its own result and returned it would ignore the declaration and hand back the
wide array, so this list is deliberately short and grows only by checking.
"""
const OUTCAST_PRODUCERS = Set(["native_layer_norm.default", "clone.default"])

"""Buffer ids from `id` down to the first non-view buffer, `id` included."""
function viewchain(g::Graph, id::AbstractString)
    ids = String[id]
    for _ in 1:16
        b = get(g.buffers, ids[end], nothing)
        (b === nothing || b.kind !== :view || isempty(b.of)) && break
        push!(ids, b.of)
    end
    ids
end

"""Which element of a multi-output op this view selects, or `nothing`."""
function tupleindex(b::Buffer)
    b.kind === :view || return nothing
    occursin("getitem", b.viewop) || return nothing
    i = get(b.attrs, "arg1", nothing)
    i isa Integer ? Int(i) : nothing
end

"""
    foldoutcasts(graph) -> (graph, nfolded)

Retype each foldable producer's result and drop the cast.
"""
# Switchable because the fold changes numerics — see the note carried onto
# `foldoutcasts(::AbstractDict)` — and `enabled = false` is how the A/B is run.
function foldoutcasts(g::Graph; enabled::Bool = true)
    enabled || return (g, 0)
    producer = Dict(op.out => op for op in g.ops)
    reads = Dict{String,Int}()
    bump!(id) = (reads[id] = get(reads, id, 0) + 1)
    for op in g.ops, i in op.ins
        bump!(last(viewchain(g, i)))
    end
    for o in g.outputs
        bump!(last(viewchain(g, o)))
    end

    buffers = Dict{String,Buffer}(g.buffers)
    drop = Set{String}()
    folded = 0

    for cast in g.ops
        cast.aten == "_to_copy.default" || continue
        length(cast.ins) == 1 || continue
        chain = viewchain(g, cast.ins[1])
        src = last(chain)
        p = get(producer, src, nothing)
        p === nothing && continue
        p.aten in OUTCAST_PRODUCERS || continue
        get(reads, src, 0) == 1 || continue          # the cast is the only reader
        ob = get(buffers, cast.out, nothing)
        ib = get(buffers, cast.ins[1], nothing)
        (ob === nothing || ib === nothing) && continue
        ob.kind === :transient || continue
        # A cast and nothing else. `_to_copy` can also change layout or device,
        # and one that does is not a dtype declaration.
        ob.shape == ib.shape || continue
        ob.dtype === ib.dtype && continue
        # Narrowing only: widening a producer's result would make it *more*
        # expensive to write, and the fp32 the graph asked for is the one the
        # reference computed in.
        sizeof(ob.dtype) < sizeof(ib.dtype) || continue

        # Retype every view between the producer and the cast, then the
        # producer's own result — which for a multi-output op means one entry of
        # its `dtypes`, not the buffer's `dtype`.
        for id in chain
            b = buffers[id]
            if id == src && haskey(b.attrs, "dtypes")
                ti = tupleindex(buffers[chain[max(1, length(chain) - 1)]])
                ti === nothing && continue
                dts = Any[b.attrs["dtypes"]...]
                length(dts) >= ti + 1 || continue
                dts[ti + 1] = ob.dtype
                attrs = Dict{String,Any}(b.attrs); attrs["dtypes"] = dts
                buffers[id] = Buffer(b.id, b.kind, b.shape, b.dtype, b.key, b.live,
                                     b.of, b.viewop, attrs)
            else
                buffers[id] = Buffer(b.id, b.kind, b.shape, ob.dtype, b.key, b.live,
                                     b.of, b.viewop, b.attrs)
            end
        end

        # The cast's own result becomes an alias of what it used to read.
        buffers[cast.out] = Buffer(cast.out, :view, ob.shape, ob.dtype, "", (0, 0),
                                   cast.ins[1], "alias.default", ob.attrs)
        push!(drop, cast.id)
        folded += 1
    end

    folded == 0 && return (g, 0)
    ops = [o for o in g.ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), folded)
end

"""

`enabled = false` skips the fold, so the two graphs can be compared inside one
session; dropping it changes speed and memory, never values. A keyword rather
than a global, so two comparisons can run at once and a failing test cannot
leave it switched off for everything after.
"""
function foldoutcasts(graphs::AbstractDict; enabled::Bool = true)
    out = Dict{String,Graph}()
    total = 0
    for (n, g) in graphs
        g2, k = foldoutcasts(g; enabled)
        out[n] = g2
        total += k
    end
    (out, total)
end
