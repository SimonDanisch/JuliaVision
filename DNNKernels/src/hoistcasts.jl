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


"""
    hoistconstants(graphs, weights, backend) -> (graphs, weights, nfolded)

The transitive form: fold whole *subgraphs* whose inputs are all weights.

The method above folds an op that reads nothing at all. This one folds an op that
reads only constants, to fixpoint — which on SAM 2's encoder is not a corner case
but **164 of 852 ops**: the bilinear resize of the trunk's position embedding.
That resize does not depend on the image, yet it ran on every encode, and it is
written as sixteen full-size gathers combined by weighted sums. All sixteen
`[1, 144, 256, 256]` fp32 buffers are live at once to feed the final sum, so they
alone were **604 MB of a 698 MB slab**. Folding takes the slab to 274 MB, adds
84 MB of weights, and orphans the 88 MB of position embeddings the subgraph read.

It is also **11.7 ms of a 133 ms encode**, 8.7%, which is worth stating carefully
because the first measurement of it said 0.55 ms and was scoped wrong. Taking the
backward cone from the sixteen `index.Tensor` ops finds the gathers and the index
arithmetic feeding them — 68 ops — and stops exactly where the work is, because
everything that *consumes* the gathers is downstream. Those 68 do cost 0.55 ms,
which is 604 MB at bandwidth and correct as far as it goes. The seventy muls and
adds that then combine the sixteen gathers move ~8 GB and cost the other 11 ms.
The fixpoint set is the right one, and both instruments agree on it: doubling it
says 11.13 ms, deleting it says 11.72 ms.

Two restrictions, both load-bearing:

**Fully concrete shapes only.** A `Model` serves any resolution — `scratchfor`
keys its slab on `dims` — so a value that depends on a symbolic extent is a
constant of *that resolution*, not of the model, and baking it into a weight
would be wrong at the next one. Ops reading a `:host` shape expression are out
for the same reason. What survives is resolution-independent by construction,
which is also why this needs no `dims` in order to run.

**Only when it pays.** Folding trades slab space for resident weights, and a
subgraph whose result is bigger than its intermediates is a loss. The guard
compares the two directly rather than assuming SAM 2's 7:1 ratio generalises.

Unlike every other pass here this one *executes*, so it runs on `backend` after
the upload rather than on the host: one encode's worth of those ops, 11.7 ms of
device time, paid once instead of per frame. The CPU backend was the obvious
alternative and is the wrong one — the same 164 ops take 6.5 s there, 5.5 s of it
JIT for kernels nothing else in the model needs. Its 604 MB of intermediates are
transient in the real sense: Lava returns a freed block to the driver, measured
at `+0 MiB` across an allocate/free of that size, so `nvidia-smi` is unchanged.

`enabled = false` restores the graph as exported, which costs 11.7 ms per encode
and 424 MB of slab, and is worth having for an A/B or when a load that runs no
encode at all is what matters — folding moves the folded ops' JIT to load time.
A keyword rather than a global: it is read in one place, and an A/B that mutates
module state cannot run two ways at once.
"""
function hoistconstants(graphs::AbstractDict, weights::AbstractDict, backend;
                        enabled::Bool = true)
    enabled || return (Dict{String,Graph}(graphs), Dict{String,Any}(weights), 0)
    w = Dict{String,Any}(weights)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = hoistconstants(g, w, backend)
        out[name] = g2
        n += k
    end
    (out, w, n)
end

"""Extents all known now, rather than at a resolution — see `hoistconstants`."""
concreteshape(b::Buffer) = all(x -> x isa Integer, b.shape)

"""
Bytes a buffer occupies. `mapreduce` with an explicit `init` rather than
`prod(Int.(shape))`, because a 0-d buffer's shape is an *empty* `Vector{Any}`:
`Int.` on it stays `Vector{Any}`, and `prod` of that has no identity to return.
MatAnyone's `readout_query` has one, so this is a load-time crash, not a corner.
"""
constbytes(b::Buffer) = mapreduce(Int, *, b.shape; init = 1) * sizeof(b.dtype)

"""
Ops that read only constants, to fixpoint. A weight is constant; so is the output
of an op already in the set. Anything symbolic or host-evaluated is refused.
"""
function constops(g::Graph)
    known = Set{String}(id for (id, b) in g.buffers if b.kind === :weight)
    ops = Set{String}()
    changed = true
    while changed
        changed = false
        for o in g.ops
            o.id in ops && continue
            ob = get(g.buffers, o.out, nothing)
            (ob === nothing && continue)
            concreteshape(ob) || continue
            # `:external` marks a value that leaves the graph, which is most of
            # the encoder's outputs and perfectly foldable. A graph *input* is
            # the caller's, though, and freezing one would answer the wrong
            # question forever. SAM 2 never produces into an input; other graphs
            # are not promised to be so tidy.
            o.out in g.inputs && continue
            all(o.ins) do i
                r = rootbuffer(g, i)
                b = get(g.buffers, r, nothing)
                b !== nothing && b.kind !== :host && concreteshape(b) && r in known
            end || continue
            push!(ops, o.id)
            push!(known, rootbuffer(g, o.out))
            changed = true
        end
    end
    ops
end

"""
Values computed inside the fold set that something outside still reads — either
a surviving op or the graph's own output list. These are what become weights;
everything else in the set disappears with it.
"""
function constescaping(g::Graph, ops::Set{String})
    made = Set{String}(rootbuffer(g, o.out) for o in g.ops if o.id in ops)
    esc = Set{String}()
    for o in g.ops
        o.id in ops && continue
        for i in o.ins
            r = rootbuffer(g, i)
            r in made && push!(esc, r)
        end
    end
    for out in g.outputs
        r = rootbuffer(g, out)
        r in made && push!(esc, r)
    end
    esc
end

"""The fold set on its own, as a graph that can be run once."""
function constsubgraph(g::Graph, ops::Set{String}, outs::Vector{String})
    keep = [o for o in g.ops if o.id in ops]
    need = Set{String}()
    for o in keep
        push!(need, o.out)
        for i in o.ins; push!(need, i); end
    end
    # A view names its parent rather than carrying the storage, so pull the
    # parents in too — repeatedly, since a view of a view is legal.
    for _ in 1:8, id in collect(need)
        b = get(g.buffers, id, nothing)
        b !== nothing && !isempty(b.of) && push!(need, b.of)
    end
    Graph(g.name * "|const", g.symbols, String[], outs,
          Dict{String,Buffer}(id => g.buffers[id] for id in need if haskey(g.buffers, id)),
          [id for id in g.order if id in need], keep, Vector{Vector{String}}())
end

function hoistconstants(g::Graph, weights::Dict{String,Any}, backend)
    ops = constops(g)
    isempty(ops) && return (g, 0)
    esc = constescaping(g, ops)
    isempty(esc) && return (g, 0)

    payload = sum(e -> constbytes(g.buffers[e]), esc; init = 0)
    freed = sum(g.ops; init = 0) do o
        o.id in ops || return 0
        b = g.buffers[rootbuffer(g, o.out)]
        b.kind === :transient ? constbytes(b) : 0
    end
    # Strictly: a subgraph that materialises more than it stops materialising is
    # a loss, and silently taking it would be the pass working against its own
    # reason to exist.
    payload < freed || return (g, 0)

    vals = execute!(constsubgraph(g, ops, sort(collect(esc))), Dict{String,Any}(), weights;
                    dims = NamedTuple(), backend)

    buffers = Dict{String,Buffer}(g.buffers)
    for e in esc
        b = buffers[e]
        key = g.name * "|" * e * "|constfold"
        weights[key] = vals[e]
        buffers[e] = Buffer(b.id, :weight, b.shape, b.dtype, key, (0, 0),
                            "", "", Dict{String,Any}())
    end
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, g.order,
           [o for o in g.ops if !(o.id in ops)], g.fusion), length(ops))
end
