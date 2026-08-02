"""
Fold an activation into the op that feeds it — `relu` into a convolution or a
residual `add`, `gelu` into an `addmm`.

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
        # **This pass runs on every exported graph, so the reach was audited.**
        # Of everything in `gen/graphs`: MatAnyone (all three precisions) and
        # BasicVSR++ contain no `gelu` at all and are untouched; SAM 2's decoder
        # has two and folds neither, they do not meet the conditions below; SAM
        # 2's encoder is the target, 48 of 48. Wan's DiT folds 1 and 3, and has
        # no test in the suite — benign because its graphs are fp32, so
        # `mm_coopmat_applicable` refuses them, `matmul!` takes the scalar path
        # and the epilogue becomes `out .= epi.(out)`: the same computation the
        # deleted op did, one dispatch earlier.
        #
        # `gelu` folds into `addmm` for the same reason `relu` folds into a
        # convolution: the GEMM's store already reads every element of the
        # result, so the activation is free there and a full read-modify-write
        # pass otherwise. SAM 2's encoder has 48 of them and every one has a
        # single reader — 1094.7 MB of output that no longer round-trips.
        #
        # `gelu.default` with `arg1 = "tanh"` is a *different function*, not a
        # faster one, so only the default (exact) form folds; the other keeps its
        # own op.
        act = relu.aten == "relu.default" ? "relu" :
              (relu.aten == "gelu.default" &&
               String(get(relu.attrs, "arg1", "none")) != "tanh") ? "gelu" : ""
        isempty(act) && continue
        length(relu.ins) == 1 || continue
        # A relu reads its convolution directly. A gelu does not: the graph
        # reshapes the `addmm` result first, so the chain is
        # `gelu <- view.default <- addmm` and `resolvealias` — which follows only
        # `alias`/`detach` — stops at the view and finds no producer. Resolving
        # the whole chain is what makes the fold reachable at all; it is also why
        # the alias below targets the *view*, which has the gelu's own shape,
        # rather than the `addmm` buffer, which does not.
        src = act == "gelu" ? rootbuffer(g, relu.ins[1]) : resolvealias(g, relu.ins[1])
        conv = get(producer, src, nothing)
        conv === nothing && continue
        # The reshape has to be the producer's only reader too, or folding the
        # activation in would hand the activated values to whoever else reads it.
        act != "gelu" || get(uses, resolvealias(g, relu.ins[1]), 0) == 1 || continue
        # A residual block ends `add(conv, skip) -> relu`, so the relus on the
        # largest feature maps follow an *add*, not a convolution — folding only
        # into convolutions left those 36 behind, and they were the expensive
        # ones. `add.Tensor` already writes every element of its destination, so
        # the activation costs nothing there either.
        # relu folds into a convolution or the add that ends a residual block;
        # gelu only into `addmm`, which is the only producer whose epilogue can
        # take it and the only one the model actually puts a gelu after.
        (act == "gelu" ? conv.aten == "addmm.default" :
                         conv.aten in ("convolution.default", "add.Tensor")) || continue
        get(uses, src, 0) == 1 || continue           # producer feeds only this relu
        haskey(conv.attrs, "act") && continue        # already carrying one
        ob = buffers[relu.out]
        cb = buffers[conv.out]
        ob.dtype === cb.dtype || continue

        ci = findfirst(o -> o.id == conv.id, ops)
        attrs = Dict{String,Any}(conv.attrs)
        attrs["act"] = act
        ops[ci] = Op(conv.id, conv.aten, conv.ins, conv.out, attrs)

        buffers[relu.out] = Buffer(relu.out, :view, ob.shape, ob.dtype, "", (0, 0),
                                   act == "gelu" ? relu.ins[1] : conv.out,
                                   "alias.default", ob.attrs)
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
