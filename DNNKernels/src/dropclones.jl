"""
Drop a clone that copies bytes nothing rewrites.

`aten::clone` is contiguity in torch's terms. A declared graph is SSA — every
buffer is written by exactly one op and only read afterwards — so a clone whose
input and output declare the SAME dtype copies bytes that never change, and its
result can be an alias of its operand instead of a pass over it. SAM 2's encoder
has 45 of those, one per attention output, each feeding a residual add that only
reads it.

A clone that also NARROWS is not this one: `foldoutcasts` retypes 45 more of that
encoder's clones to fp16, and those carry the cast, so they stay a pass.

Refused wherever aliasing would change what somebody else sees: a graph OUTPUT,
an operand that is the graph's own input or a weight, and an output an in-place
op writes ([`isaliasing`](@ref)).

The declared dtype comes through the view chain, because an attention result
reaches the clone as a `getitem` whose dtype is one entry of its parent's
`dtypes` rather than the buffer's own — the same reason `foldoutcasts` walks it.
"""
function dropclones(g::Graph; enabled::Bool = true)
    enabled || return (g, 0)
    buffers = Dict{String,Buffer}(g.buffers)
    outs = Set(g.outputs)
    # Whatever an in-place op writes through is not something to alias into.
    written = Set{String}()
    for o in g.ops
        isaliasing(o) || continue
        for i in o.ins; push!(written, i); end
    end
    drop = String[]
    n = 0
    for o in g.ops
        o.aten == "clone.default" || continue
        length(o.ins) == 1 || continue
        ib = get(buffers, o.ins[1], nothing)
        ob = get(buffers, o.out, nothing)
        (ib === nothing || ob === nothing) && continue
        ob.kind === :transient || continue
        o.out in outs && continue
        o.out in written && continue
        ib.kind in (:external, :weight, :host) && continue
        declareddtype(buffers, ib) === ob.dtype || continue
        buffers[o.out] = Buffer(o.out, :view, ob.shape, ob.dtype, "", (0, 0),
                                o.ins[1], "alias.default", ob.attrs)
        push!(drop, o.id)
        n += 1
    end
    n == 0 && return (g, 0)
    ops = [x for x in g.ops if !(x.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    return (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops,
                  g.fusion), n)
end

"""
The dtype buffer `b` declares, read through a multi-output parent's `dtypes`.

A `getitem` view carries the buffer's own `dtype` field, which for one element of
a tuple is whatever the export wrote for the whole result. The per-element entry
is the one that says what those bytes are.
"""
function declareddtype(buffers::AbstractDict, b::Buffer)
    occursin("getitem", b.viewop) || return b.dtype
    pb = get(buffers, b.of, nothing)
    pb === nothing && return b.dtype
    dts = get(pb.attrs, "dtypes", nothing)
    dts === nothing && return b.dtype
    i = Int(b.attrs["arg1"])
    (length(dts) > i && dts[i + 1] !== nothing) || return b.dtype
    return dts[i + 1]
end

"""
    dropclones(graphs; enabled = true) -> (graphs, n)

Every graph, and how many clones became aliases across all of them.
"""
function dropclones(graphs::AbstractDict; enabled::Bool = true)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = dropclones(g; enabled)
        out[name] = g2
        n += k
    end
    return (out, n)
end
