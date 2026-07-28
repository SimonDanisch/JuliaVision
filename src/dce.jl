"""
Drop ops whose results nothing reads.

`foldbatchnorm` and `hoistcasts` both orphan their inputs — a folded
convolution stops reading the `_to_copy` of its weight, a hoisted cast stops
reading the fp32 original — and `execute!` runs `graph.ops` in order regardless,
so an orphan is a full dispatch (and a slab slot) per step for a value nobody
looks at. Sweeping backwards from the graph outputs removes them all in one
pass, including chains that only became dead because something upstream did.
"""

"""
    dropdead(graph) -> (graph, ndropped)

Remove ops whose output cannot reach a graph output.
"""
function dropdead(g::Graph)
    # A buffer is live if it is a graph output, or feeds a live op, or is the
    # parent of a live view. Views are resolved lazily by `value`, so a view's
    # liveness has to propagate to what it is a view *of*.
    viewsof = Dict{String,Vector{String}}()
    for (id, b) in g.buffers
        b.kind === :view && !isempty(b.of) && push!(get!(viewsof, id, String[]), b.of)
    end
    producer = Dict(op.out => op for op in g.ops)

    live = Set{String}()
    stack = collect(g.outputs)
    while !isempty(stack)
        id = pop!(stack)
        id in live && continue
        push!(live, id)
        for p in get(viewsof, id, ())
            push!(stack, p)
        end
        op = get(producer, id, nothing)
        op === nothing && continue
        append!(stack, op.ins)
    end
    # A view that is itself live keeps its parent live; a view of a dead buffer
    # is dead too, and `order` still mentions it, so filter both lists.
    ops = [o for o in g.ops if o.out in live]
    length(ops) == length(g.ops) && return (g, 0)
    order = [id for id in g.order if !(haskey(producer, id) && !(id in live))]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers, order, ops, g.fusion),
     length(g.ops) - length(ops))
end

function dropdead(graphs::AbstractDict)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = dropdead(g)
        out[name] = g2
        n += k
    end
    (out, n)
end
