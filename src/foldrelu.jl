"""
Fold `relu` into the convolution that feeds it.

86 relus a step, each of them a dispatch plus a full read-modify-write pass over
a convolution output that was just written. The activation is one `max` in the
epilogue the convolution already runs, so folding removes both.

Runs *after* `foldbatchnorm`, which is why the operand has to be resolved
through alias views: folding a batch-norm rewrites its `getitem` readers into
aliases of the convolution's own output, so by the time we get here the relu
reads an alias, not the convolution buffer directly.

The convolution must feed nothing but this relu — otherwise the pre-activation
value is still needed — and the two must agree on dtype, since the relu's output
buffer becomes an alias of the convolution's.
"""

"""Follow `alias`/`detach` views to the buffer that actually holds the data."""
function resolvealias(g::Graph, id::AbstractString)
    for _ in 1:16
        b = get(g.buffers, id, nothing)
        b === nothing && return id
        (b.kind === :view && b.viewop in ("alias.default", "detach.default")) || return id
        id = b.of
    end
    id
end

"""
    foldrelu(graph) -> (graph, nfolded)

Mark each foldable convolution with `attrs["act"] = "relu"` and drop the relu.
"""
function foldrelu(g::Graph)
    producer = Dict(op.out => op for op in g.ops)
    uses = Dict{String,Int}()
    bump!(id) = (uses[id] = get(uses, id, 0) + 1)
    for op in g.ops, i in op.ins
        bump!(resolvealias(g, i))
    end
    for (_, b) in g.buffers
        isempty(b.of) && continue
        # An alias is not itself a use: whoever reads *it* has already been
        # counted, resolved to the same buffer. Counting the link as well made
        # every folded batch-norm's convolution look twice-used — which is
        # exactly the conv->bn->relu chain this pass exists for, so only the 9
        # convolutions with no batch-norm folded.
        b.kind === :view && b.viewop in ("alias.default", "detach.default") && continue
        bump!(resolvealias(g, b.of))
    end
    for o in g.outputs
        bump!(resolvealias(g, o))
    end

    buffers = Dict{String,Buffer}(g.buffers)
    ops = copy(g.ops)
    drop = Set{String}()
    folded = 0

    for relu in g.ops
        relu.aten == "relu.default" || continue
        length(relu.ins) == 1 || continue
        src = resolvealias(g, relu.ins[1])
        conv = get(producer, src, nothing)
        conv === nothing && continue
        # A residual block ends `add(conv, skip) -> relu`, so the relus on the
        # largest feature maps follow an *add*, not a convolution — folding only
        # into convolutions left those 36 behind, and they were the expensive
        # ones. `add.Tensor` already writes every element of its destination, so
        # the activation costs nothing there either.
        conv.aten in ("convolution.default", "add.Tensor") || continue
        get(uses, src, 0) == 1 || continue           # producer feeds only this relu
        haskey(conv.attrs, "act") && continue        # already carrying one
        ob = buffers[relu.out]
        cb = buffers[conv.out]
        ob.dtype === cb.dtype || continue

        ci = findfirst(o -> o.id == conv.id, ops)
        attrs = Dict{String,Any}(conv.attrs)
        attrs["act"] = "relu"
        ops[ci] = Op(conv.id, conv.aten, conv.ins, conv.out, attrs)

        buffers[relu.out] = Buffer(relu.out, :view, ob.shape, ob.dtype, "", (0, 0),
                                   conv.out, "alias.default", ob.attrs)
        push!(drop, relu.id)
        folded += 1
    end

    folded == 0 && return (g, 0)
    ops = [o for o in ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), folded)
end

function foldrelu(graphs::AbstractDict)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = foldrelu(g)
        out[name] = g2
        n += k
    end
    (out, n)
end
