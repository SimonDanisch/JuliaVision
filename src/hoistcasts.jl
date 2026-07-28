"""
Evaluate constant dtype casts at load time.

The autocast export keeps every parameter in fp32 and inserts a `_to_copy` in
front of each use, so a convolution's weight operand is not the weight buffer
but a cast of it. Two consequences, both worth removing before the graph is ever
run:

  * the cast is recomputed on every step even though its input never changes —
    46 casts in `encode_image` alone, each its own dispatch;
  * `foldbatchnorm` looks for a `:weight` buffer at `conv.ins[2]` and finds a
    transient, so under autocast nothing folded at all.

`hoistcasts` replaces each such op with a `:weight` buffer holding the already
converted array. The cast still happens — once, on the host, before upload —
and `weightsource` lets the fold pass reach the fp32 original underneath so the
scale is applied in fp32 and rounded to fp16 exactly once.
"""

"""
    weightsource(g, id) -> Buffer | nothing

Follow `_to_copy`/alias chains from `id` back to the `:weight` buffer that
ultimately feeds it, or `nothing` if anything computed intervenes.
"""
function weightsource(g::Graph, id::AbstractString)
    producer = Dict(op.out => op for op in g.ops)
    seen = 0
    while true
        (seen += 1) > 16 && return nothing
        b = get(g.buffers, id, nothing)
        b === nothing && return nothing
        b.kind === :weight && return b
        if b.kind === :view && b.viewop in ("alias.default", "detach.default")
            id = b.of
            continue
        end
        op = get(producer, id, nothing)
        op === nothing && return nothing
        op.aten in ("_to_copy.default", "clone.default", "detach.default") || return nothing
        length(op.ins) == 1 || return nothing
        id = op.ins[1]
    end
end

"""
    hoistpermutes(graphs, weights) -> (graphs, weights, nhoisted)

Materialise `permute`/`t`/`transpose` *views of weights* into real weight
buffers, laid out the way the consumer wants them.

Every `addmm` in this model receives its weight as a transposed view — 48 a step,
and not one of them reached the cooperative-matrix path, because that kernel
loads with a hardcoded column-major layout and so requires a dense `LavaArray`,
which a `PermutedDimsArray` is not. They all fell back to the scalar kernel at
roughly a twentieth of the throughput.

The transpose is of a *constant*, so it belongs at load time. Doing it here
rather than teaching the GEMM about strided operands also keeps the fast path
free of a runtime shape test.
"""
function hoistpermutes(graphs::AbstractDict, weights::AbstractDict)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = hoistpermutes(g, w)
        out[name] = g2
        n += k
    end
    (out, w, n)
end

function hoistpermutes(g::Graph, weights::Dict{String,Any})
    buffers = Dict{String,Buffer}(g.buffers)
    hoisted = 0
    for (id, b) in g.buffers
        b.kind === :view || continue
        b.viewop in ("permute.default", "t.default", "transpose.int") || continue
        src = weightsource(g, b.of)
        src === nothing && continue
        haskey(weights, src.key) || continue
        W = weights[src.key]
        W isa AbstractArray && ndims(W) >= 2 || continue

        # Same index gymnastics as `makeview`: torch permutes the un-reversed
        # shape, so both the order of the permutation and the indices it names
        # get reversed.
        nd = ndims(W)
        jperm = if b.viewop == "permute.default"
            p = ints(b.attrs["arg1"])
            length(p) == nd || continue
            ntuple(i -> nd - p[nd - i + 1], nd)
        else
            d1 = b.viewop == "t.default" ? 0 : Int(b.attrs["arg1"])
            d2 = b.viewop == "t.default" ? 1 : Int(b.attrs["arg2"])
            pv = collect(1:nd)
            pv[jdim(d1, nd)], pv[jdim(d2, nd)] = pv[jdim(d2, nd)], pv[jdim(d1, nd)]
            Tuple(pv)
        end

        key = g.name * "|" * id * "|hoistperm"
        weights[key] = permutedims(W, jperm)
        buffers[id] = Buffer(id, :weight, b.shape, b.dtype, key, (0, 0), "", "",
                             Dict{String,Any}())
        hoisted += 1
    end
    hoisted == 0 && return (g, 0)
    order = copy(g.order)
    append!(order, [id for id in keys(buffers) if endswith(id, "|hoistperm")])
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, g.ops, g.fusion), hoisted)
end

"""
    hoistconstants(graphs, weights) -> (graphs, weights, nhoisted)

Evaluate ops whose result is a compile-time constant at load time.

`scalar_tensor` and `full` read no tensor at all — the value and the shape are
both in their attrs — yet `runop!` rebuilt them on every step. For the 0-d
`scalar_tensor` that is worse than a wasted dispatch: `planslab` reserves
nothing for a shapeless buffer, so each one also did a real `vkAllocateMemory`
and a matching free every step, churning the allocator for a number that never
changes. It was the only `vkAllocateMemory` left in the steady-state profile.

Only fires when the shape is fully static; `full` with a symbolic extent depends
on `dims` and cannot be known before the first call.
"""
function hoistconstants(graphs::AbstractDict, weights::AbstractDict)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = hoistconstants(g, w)
        out[name] = g2
        n += k
    end
    (out, w, n)
end

function hoistconstants(g::Graph, weights::Dict{String,Any})
    buffers = Dict{String,Buffer}(g.buffers)
    drop = Set{String}()
    hoisted = 0

    for op in g.ops
        isempty(op.ins) || continue
        val = if op.aten == "scalar_tensor.default"
            get(op.attrs, "arg0", nothing)
        elseif op.aten == "full.default"
            get(op.attrs, "arg1", nothing)
        else
            nothing
        end
        val === nothing && continue
        ob = get(buffers, op.out, nothing)
        ob === nothing && continue
        all(s -> s isa Integer, ob.shape) || continue

        key = g.name * "|" * op.id * "|hoistconst"
        weights[key] = fill(convert(ob.dtype, scalar(val)), Tuple(reverse(Int.(ob.shape))))
        buffers[op.out] = Buffer(op.out, :weight, ob.shape, ob.dtype, key, (0, 0),
                                 "", "", Dict{String,Any}())
        push!(drop, op.id)
        hoisted += 1
    end

    hoisted == 0 && return (g, 0)
    # `order` keeps the id: an op and its output buffer share a name here, and
    # `execute!` walks `order` to seed weights — dropping it leaves the hoisted
    # constant unbound ("buffer scalar_tensor of kind weight was never produced").
    ops = [o for o in g.ops if !(o.id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order, ops, g.fusion), hoisted)
end

"""
    hoistcasts(graphs, weights) -> (graphs, weights, nhoisted)

Constant-fold every `_to_copy` whose operand resolves to a weight. Mutates
neither argument; returns fresh graphs and an extended weight table.
"""
function hoistcasts(graphs::AbstractDict, weights::AbstractDict)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = hoistcasts(g, w)
        out[name] = g2
        n += k
    end
    (out, w, n)
end

function hoistcasts(g::Graph, weights::Dict{String,Any})
    buffers = Dict{String,Buffer}(g.buffers)
    drop = Set{String}()
    hoisted = 0

    for op in g.ops
        op.aten == "_to_copy.default" || continue
        length(op.ins) == 1 || continue
        src = weightsource(g, op.ins[1])
        src === nothing && continue
        haskey(weights, src.key) || continue
        ob = get(buffers, op.out, nothing)
        ob === nothing && continue

        # Keys are graph-scoped: op ids repeat across graphs and one weight table
        # serves all of them (the same trap `foldbatchnorm` documents).
        key = g.name * "|" * op.id * "|hoistcast"
        weights[key] = ob.dtype.(weights[src.key])
        buffers[op.out] = Buffer(op.out, :weight, ob.shape, ob.dtype, key, (0, 0),
                                 "", "", Dict{String,Any}())
        push!(drop, op.id)
        hoisted += 1
    end

    hoisted == 0 && return (g, 0)
    ops = [o for o in g.ops if !(o.id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order, ops, g.fusion), hoisted)
end
