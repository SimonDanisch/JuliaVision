"""
Compute a constant table in the type its readers get.

An export keeps a double where upstream's Python asked for one. A sine-cosine position table is
the usual case: PP-DocLayoutV3 has 24 `Float64` tensors, from `arange`s through powers to sines,
that take no input and end in one conversion to `Float32`. Nothing reads the double, and Metal
has no `Float64` to compute it in, so the model could not run there at all.

A `Float64` tensor is narrowed only when all of this holds for the tensors around it:

  * it is written from constants, or from other tensors narrowed here;
  * it is not a graph input or output, and not a weight;
  * every reader is narrowed here too, or converts it to `Float32`.

So a graph that really computes in double keeps its `Float64` tensors, and fails clearly on a
device without them. The narrowed values differ from the double's rounding by a few units in the
last place.
"""
function narrowconstants(g::Graph)
    table = Set(id for (id, b) in g.buffers if b.dtype === Float64)
    isempty(table) && return (g, 0)
    writer = Dict(o.out => o for o in g.ops)
    # Refusing one tensor can refuse its neighbours, in either direction, so this runs to a fixed point.
    changed = true
    while changed
        changed = false
        refuse(id) = id in table && (delete!(table, id); changed = true)
        for id in collect(table)
            constantbuffer(g, writer, id, table) || refuse(id)
        end
        for o in g.ops
            o.out in table || convertsto32(g, o) || foreach(refuse, o.ins)
        end
        for (id, b) in g.buffers
            b.kind === :view && !(id in table) && refuse(b.of)
        end
    end
    isempty(table) && return (g, 0)
    buffers = copy(g.buffers)
    for id in table
        b = buffers[id]
        buffers[id] = Buffer(b.id, b.kind, b.shape, Float32, b.key, b.live, b.of, b.viewop, b.attrs)
    end
    ops = map(g.ops) do o
        o.out in table && get(o.attrs, "dtype", nothing) == "float64" || return o
        attrs = Dict{String,Any}(o.attrs)
        attrs["dtype"] = "float32"
        return Op(o.id, o.aten, o.ins, o.out, attrs)
    end
    return (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order, ops, g.fusion), length(table))
end

"Whether `id` is a tensor that only constants and `table` write, and that the graph does not expose."
function constantbuffer(g::Graph, writer::AbstractDict, id::AbstractString, table::AbstractSet)
    (id in g.inputs || id in g.outputs) && return false
    b = g.buffers[id]
    b.kind === :view && return b.of in table
    b.kind === :transient || return false
    o = get(writer, id, nothing)
    return o !== nothing && all(in(table), o.ins)
end

"Whether `o` is a conversion to `Float32`, which is all a reader outside the table may be."
convertsto32(g::Graph, o::Op) = o.aten == "_to_copy.default" && length(o.ins) == 1 && g.buffers[o.out].dtype === Float32

"""
    narrowconstants(graphs) -> (graphs, n)

Every graph, and how many tensors were narrowed across all of them.
"""
function narrowconstants(graphs::AbstractDict)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = narrowconstants(g)
        out[name] = g2
        n += k
    end
    return (out, n)
end
