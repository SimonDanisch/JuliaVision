"""
Elementwise fusion, using Julia's own broadcast machinery.

The step is launch-bound, not bandwidth-bound: 2180 dispatches at ~13.2 µs each
account for essentially the whole 28.8 ms, and roughly half of those dispatches
are elementwise broadcasts over tensors small enough (15x8x256) that they run at
20 GB/s — three orders off the card's bandwidth and entirely dominated by the
cost of being a dispatch at all. Making those kernels faster is close to
worthless; not launching them is what pays.

So an elementwise op whose result is consumed exactly once, by another
elementwise op, returns its `Broadcasted` *unmaterialised*. The consumer folds
it into its own expression tree and one kernel does the work of two. Nothing
here re-implements broadcasting — `d .= a .+ b` already builds a lazy tree; this
just declines to call `materialize` in the middle of a chain.

Two guards keep it honest:

  * **One consumer.** A value read twice would be recomputed, and the second
    evaluation is not free. `usecount` counts op operands, view parents and
    graph outputs alike.
  * **Same declared dtype.** Under autocast the graph's dtypes *are* the
    reference's precision policy (`coerce`), and fusing across a boundary would
    silently keep a chain in the wrong precision. The explicit `_to_copy` casts
    stay materialisation points, which is what makes this safe to compare
    against PyTorch at all.
"""

"""ATen ops whose `runop!` is a single broadcast over its operands."""
const FUSABLE = Set([
    "relu.default", "sigmoid.default", "tanh.default", "exp.default",
    "sqrt.default", "rsqrt.default", "neg.default", "abs.default",
    "reciprocal.default", "sin.default", "cos.default", "erf.default",
    "floor.default", "log.default",
    "add.Tensor", "sub.Tensor", "mul.Tensor", "div.Tensor",
    "_to_copy.default",
])

"""
    groupmates(graph) -> Dict{String,Set{String}}

For each op id, the other ops Inductor put in the same generated kernel.
Empty when no plan has been merged (`tools/dump_plan.py`).
"""
function groupmates(g::Graph)
    m = Dict{String,Set{String}}()
    for grp in g.fusion, id in grp
        s = get!(m, id, Set{String}())
        union!(s, grp)
        delete!(s, id)
    end
    m
end

"""
    fusableset(graph) -> Set{String}

Buffer ids that may stay lazy: produced by a fusable op, read exactly once, and
read by a fusable op of the same declared dtype.
"""
const EMPTY = Set{String}()

"""
Let a value with more than one consumer stay lazy, recomputing it per reader.

**Off, because it was measured worth nothing.** Interleaved A/B on the captured
step: 1019 dispatches at 11.060 ms against 1013 at 11.076 ms. The depth-1 guard
below is what limits it — of 67 twice-read elementwise ops, only 6 have fully
materialised operands, and 6 dispatches out of 1019 is under the noise floor.

Kept because the mechanism is right and correct (suite 61/61, e2e unchanged at
2.9077e-4 / 2.7702e-4), and real group codegen needs exactly this — recomputing
a shared subexpression inside each consumer's kernel. It is the guard that has
to go, and that only pays once whole groups are emitted as one kernel rather
than folded pairwise.
"""
const DUPLICATE_FUSION = Ref{Bool}(false)
"""Most consumers a duplicated value may have."""
const DUPLICATE_MAX = Ref{Int}(2)

"""
    finalconsumer(g, readers, id) -> Op | nothing

The op that ultimately reads `id`, looking through any chain of shape-only
views, or `nothing` unless every hop has exactly one reader.

A `view`/`unsqueeze` between two elementwise ops is not a real consumer —
`lazyreshape` carries an unevaluated expression across it — but it *is* a
buffer, so the plain use count saw it as the end of the chain. Following it here
is what turns those 82 blocked candidates into fusable ones.
"""
function finalconsumer(g::Graph, readers::Dict{String,Vector{Any}}, id, depth = 0)
    depth > 8 && return nothing
    rs = get(readers, id, nothing)
    (rs === nothing || length(rs) != 1) && return nothing
    kind, obj = rs[1]
    kind === :op && return obj::Op
    kind === :view && obj.viewop in SHAPEONLY && return finalconsumer(g, readers, obj.id, depth + 1)
    nothing
end

function fusableset(g::Graph)
    mates = groupmates(g)
    readers = Dict{String,Vector{Any}}()
    note!(id, r) = push!(get!(readers, id, Any[]), r)
    # Readers are resolved through `alias`/`detach` views, because
    # `foldbatchnorm` and `foldrelu` rewrite their consumers into aliases of the
    # producer's own buffer. Counting the alias as the use instead of the op
    # behind it hid the real consumer and left most candidates unfusable.
    for op in g.ops, i in op.ins
        note!(resolvealias(g, i), (:op, op))
    end
    for (_, b) in g.buffers
        isempty(b.of) && continue
        # An alias is transparent and already counted above; any other view is a
        # reader, though a shape-only one can still be walked through.
        b.kind === :view && b.viewop in ("alias.default", "detach.default") && continue
        note!(resolvealias(g, b.of), (:view, b))
    end
    for o in g.outputs
        note!(resolvealias(g, o), (:out, o))
    end

    out = Set{String}()
    # Second pass below adds values read *twice*. Inductor does the same thing —
    # it recomputes rather than materialising — and the trade is favourable here
    # for the same reason fusion is at all: what a dispatch costs is being a
    # dispatch, so evaluating a small expression twice inside two kernels beats
    # evaluating it once and paying a launch plus a barrier to hand it over.
    for op in g.ops
        op.aten in FUSABLE || continue
        c = finalconsumer(g, readers, op.out)
        c === nothing && continue
        c.aten in FUSABLE || continue
        # An `act` attr means the consumer was already folded into something
        # else (see `foldrelu`); leave those alone.
        haskey(c.attrs, "act") && continue
        pb = get(g.buffers, op.out, nothing)
        cb = get(g.buffers, c.out, nothing)
        (pb === nothing || cb === nothing) && continue
        # Dtype boundaries are normally where fusion stops, because under
        # autocast the declared dtypes *are* the reference's precision policy.
        # Inductor's plan overrides that for the pairs it actually fused: those
        # are what PyTorch itself runs when compiled, so following them here
        # matches the reference more closely rather than less.
        if pb.dtype !== cb.dtype
            (c.out in get(mates, op.out, EMPTY)) || continue
        end
        push!(out, op.out)
    end

    # Duplication pass, strictly depth-1: only ops whose own operands are all
    # materialised may be read twice. Without that guard a chain of duplicated
    # values would fan out exponentially — each extra consumer multiplying the
    # work of everything upstream — and the win here is bounded (one dispatch and
    # one barrier per value) while the blowup would not be.
    if DUPLICATE_FUSION[]
        for op in g.ops
            op.aten in FUSABLE || continue
            op.out in out && continue
            any(i -> resolvealias(g, i) in out, op.ins) && continue   # depth-1 only
            rs = get(readers, op.out, nothing)
            rs === nothing && continue
            2 <= length(rs) <= DUPLICATE_MAX[] || continue
            all(r -> r[1] === :op && r[2].aten in FUSABLE && !haskey(r[2].attrs, "act"), rs) || continue
            pb = get(g.buffers, op.out, nothing)
            pb === nothing && continue
            all(r -> (cb = get(g.buffers, r[2].out, nothing);
                      cb !== nothing && cb.dtype === pb.dtype), rs) || continue
            push!(out, op.out)
        end
    end
    out
end

fusablesets(graphs::AbstractDict) =
    Dict{String,Set{String}}(n => fusableset(g) for (n, g) in graphs)
