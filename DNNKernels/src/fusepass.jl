"""
Elementwise fusion as a graph rewrite: `op{a} -> op{b} -> op{c}` becomes one op
carrying a [`FusedOp`](@ref).

## Why a rewrite and not the lazy version

`fuse.jl` fuses at *run time*, by returning a `Base.Broadcasted` unmaterialised
and letting the consumer's broadcast nest it. It works, and the graph never finds
out — which is why `lifetimes` walks fusion chains with depth guards to discover
when an operand is really last read, `planslab` special-cases values that have no
storage, and discovery has to see through the same laziness. Each of those
reconstructs, later and approximately, something the fusion knew exactly.

Here the fusion is a fact about the graph. A fused chain is one op with declared
inputs and one output, so `lifetimes` is the graph's, placement is the graph's,
and the intermediates simply are not there to need special-casing. Mantle's
`Liveness`/`Place` see one pass with one interval, like any other.

It also makes the kernel set knowable. A nested `Broadcasted`'s type encodes the
whole tree and only exists once execution has nested it, so it cannot be
enumerated ahead of time — which is why the frozen kernel cache has to be filled
by running a workload. A `FusedOp` is built by this pass, at load, from the
graph; the set of kernels is a property of the model.

## What it buys in dispatches, measured

`fuse.jl` computes the set of values it would keep lazy, so the two are directly
comparable rather than estimated:

```
                  lazy defers   this pass removes   still lazy after
sam2_encoder            1          1  (1 fused)            0
sam2_decoder            6         17  (6 fused, 11 epi)     0
MatAnyone (8)          73         67  (63 fused, 4 premap)  5
```

It captures nearly all of what the lazy version did and adds the two axes that
version cannot reach at all: [`foldepilogue`](@ref) into the GEMM's write-out
(11 in SAM 2's decoder) and [`foldpremap`](@ref) into a reduction's map step (4).
Deferring a broadcast can only ever fuse an elementwise op into another
elementwise op.

85 ops of 1409 is still modest, and the reason this pass exists is the section
above rather than the count. But the earlier claim here — that the ceiling was
84 and the lazy version already reached 82, so a rewrite gained two ops — came
from a table with two different measurement bases in it and does not hold.

The second is small for a structural reason, measured rather than guessed: of
136 reductions in these ten graphs, 111 read a value that something else also
reads, and 114 of those producers are the `add.Tensor` of a residual stream,
which has to exist as a tensor regardless. See `foldpremap` for the breakdown.

## Conservative by construction

An op joins a group only if it is exactly a function of its operands, with no
attribute this pass models incompletely. `add.Tensor` with a non-unit `alpha`, or
carrying an `act` from `foldrelu`, is left alone rather than half-understood: a
fusion that gets the arithmetic subtly wrong is a wrong tensor, not an error.

Worth knowing that this costs nothing: across both models, the number of
elementwise ops those refusals keep out of a group *that would otherwise have
joined one* is **zero**. What separates this pass from `fuse.jl`'s reach is not
the guards but the single-reader rule — the lazy version lets a multi-consumer
value stay lazy and recomputes it per reader (137 edges, 107 of them with more
than two readers), where this one materialises it once.
"""

"""
    fusedfunc(g, op) -> (f, arity) | nothing

The function this op computes, or `nothing` when it is not expressible as one.
"""
function fusedfunc(g::Graph, op::Op)
    # `act` means `foldrelu` already folded an activation in here; the op is no
    # longer just its own arithmetic.
    haskey(op.attrs, "act") && return nothing
    a = op.aten
    if a == "add.Tensor"
        alpha(op) == 1 || return nothing
        return (+, 2)
    elseif a == "sub.Tensor"
        alpha(op) == 1 || return nothing
        return (-, 2)
    elseif a == "mul.Tensor"
        return (*, 2)
    elseif a == "div.Tensor"
        return (/, 2)
    elseif a == "_to_copy.default"
        b = get(g.buffers, op.out, nothing)
        b === nothing && return nothing
        # torch's float -> integer cast saturates and truncates toward zero, which
        # is `safetrunc`, not `T(x)`. Rather than reimplement it here and risk the
        # two drifting, that case simply does not fuse.
        if b.dtype <: Integer
            src = get(g.buffers, resolvealias(g, first(op.ins)), nothing)
            (src === nothing || !(src.dtype <: Integer)) && return nothing
        end
        return (Cast(b.dtype), 1)
    end
    f = get(UNARY_FUSED, a, nothing)
    f === nothing ? nothing : (f, 1)
end

"""
    operandrefs(g, op, arity, slot, ins) -> Tuple | nothing

The `arity` operands of `op` as `In`/`Tmp`/`Konst`, resolved the way
[`operand`](@ref) resolves them at run time.

**A position is a scalar attribute or a tensor input, and it is not enough to
walk `op.ins`.** `add.Tensor(x, 1e-6)` carries the scalar in `attrs["arg1"]` and
has one entry in `ins`; building the arguments from `ins` alone produced `+(x)`,
which is *not an error* — it is unary plus, the identity. That shipped a decoder
whose mask logits were off by 1.96 and looked like a precision bug. `-(x)` from
the same mistake is negation rather than subtraction, which is worse for being
plausible.

Returns `nothing` when the operand cannot be resolved at pass time: a symbolic
scalar needs `evalexpr` against `dims`, and the rewrite passes run at `Model`
construction, before any resolution is known.
"""
function operandrefs(g::Graph, op::Op, arity::Int, slot::Dict{String,Any}, ins::Vector{String})
    refs = Any[]
    for pos in 1:arity
        key = "arg$(pos - 1)"
        if haskey(op.attrs, key)
            v = scalar(op.attrs[key])
            v isa Number || return nothing
            push!(refs, Konst(v))
        else
            idx = pos - count(p -> haskey(op.attrs, "arg$(p - 1)"), 1:(pos - 1))
            idx <= length(op.ins) || return nothing
            id = resolvealias(g, op.ins[idx])
            push!(refs, get!(slot, id) do
                push!(ins, id)
                In(length(ins))
            end)
        end
    end
    Tuple(refs)
end

"""Whether this op can be put in a group at all — it has a function, and every
operand that function needs can be named without running anything."""
function fusable(g::Graph, op::Op)
    fa = fusedfunc(g, op)
    fa === nothing && return false
    operandrefs(g, op, fa[2], Dict{String,Any}(), String[]) !== nothing
end

"""
    fusegroups(g) -> Vector{Vector{Op}}

Maximal runs of fusable ops, each in topological order and ending at the op whose
result leaves the group.

A group grows backwards from its final op: an operand joins when it is fusable,
is produced inside this graph, has exactly one reader, and shares the group's
declared dtype. The dtype rule is the one from `fuse.jl` and it is load-bearing —
under autocast the graph's dtypes *are* the reference's precision policy, and a
chain fused across a boundary keeps intermediates in registers where the
reference rounds. That is measurable: widening the old pass moved SAM 2's IoU
from 1.00000/0.99972/0.97727 to 0.98750/0.99978/0.95556.
"""
function fusegroups(g::Graph)
    producer = Dict(o.out => o for o in g.ops)
    readers = Dict{String,Int}()
    for o in g.ops, i in o.ins
        id = resolvealias(g, i)
        readers[id] = get(readers, id, 0) + 1
    end
    for (_, b) in g.buffers
        isempty(b.of) && continue
        id = resolvealias(g, b.of)
        readers[id] = get(readers, id, 0) + 1
    end
    for o in g.outputs
        id = resolvealias(g, o)
        readers[id] = get(readers, id, 0) + 1
    end

    taken = Set{String}()
    groups = Vector{Vector{Op}}()
    # Backwards, so a chain is claimed by its LAST op and grows towards its
    # inputs. Forwards would claim the head first and then have to decide which
    # of several consumers it belongs to.
    for op in reverse(g.ops)
        op.id in taken && continue
        fusable(g, op) || continue
        dt = g.buffers[op.out].dtype
        oshape = g.buffers[op.out].shape
        members = Op[op]
        push!(taken, op.id)
        # Breadth-first over operands; `members` stays in reverse-topological
        # order and is flipped at the end.
        k = 1
        while k <= length(members)
            for i in members[k].ins
                id = resolvealias(g, i)
                p = get(producer, id, nothing)
                p === nothing && continue
                p.id in taken && continue
                get(readers, id, 0) == 1 || continue
                fusable(g, p) || continue
                pb = get(g.buffers, p.out, nothing)
                (pb === nothing || pb.dtype !== dt) && continue
                # **Do not absorb an op that is SMALLER than the group's output.**
                # A fused expression is evaluated once per output element, so an
                # operand the output broadcasts over gets its producer recomputed
                # for every element that shares it. SAM 2's decoder has the
                # decomposed layer norm
                #
                #     mean_1 (128,128,1,1) -> add -> sqrt -> div (128,128,64,1)
                #
                # where `add` and `sqrt` run on 16384 elements and the consumer on
                # 1048576. Absorbing them multiplies that `sqrt` by 64 — the length
                # of the normalised axis. Nothing bounds that factor in general.
                #
                # Compared structurally, not by element count: these passes run at
                # `Model` construction, where a shape may still be symbolic in `h`
                # and `w`. Equal shapes cannot expand; anything else is refused
                # rather than resolved, which also declines some same-size reshapes.
                pb.shape == oshape || continue
                push!(members, p)
                push!(taken, p.id)
            end
            k += 1
        end
        length(members) > 1 && push!(groups, reverse(members))
    end
    groups
end

"""
    recipe(g, group) -> (FusedOp, Vector{String})

The fused callable and the external buffer ids it reads, in `In` order.
"""
function recipe(g::Graph, group::Vector{Op})
    slot = Dict{String,Any}()             # buffer id -> In(i) or Tmp(k)
    ins = String[]
    funcs = Any[]
    args = Any[]
    for (k, op) in enumerate(group)
        f, arity = fusedfunc(g, op)
        refs = operandrefs(g, op, arity, slot, ins)
        refs === nothing && error(
            "fuseops: $(op.aten) ($(op.id)) reached `recipe` with an operand that " *
            "cannot be named at pass time; `fusable` should have excluded it")
        # Belt and braces, because the failure this guards is silent: a binary op
        # handed one argument is `+(x)` or `-(x)`, both of which run and both of
        # which are the wrong function.
        length(refs) == arity || error(
            "fuseops: $(op.aten) ($(op.id)) needs $arity operands, resolved $(length(refs))")
        # Round to the declared dtype, because that is what materialising this
        # op's result would have done. See `Rounded`; without it a Float64 scalar
        # attribute promotes the rest of the chain.
        dt = g.buffers[op.out].dtype
        push!(funcs, f isa Cast ? f : Rounded(dt, f))
        push!(args, refs)
        slot[resolvealias(g, op.out)] = Tmp(k)
    end
    (FusedOp(Tuple(funcs), Tuple(args)), ins)
end

"""
    fuseops(graph) -> (graph, nfused)

Replace each fusable group with one `fused.elementwise` op. `nfused` counts the
ops that disappeared, not the groups.
"""
function fuseops(g::Graph)
    groups = fusegroups(g)
    isempty(groups) && return (g, 0)
    buffers = Dict{String,Buffer}(g.buffers)
    ops = copy(g.ops)
    drop = Set{String}()
    fused = 0
    for grp in groups
        fo, ins = recipe(g, grp)
        last_ = grp[end]
        attrs = Dict{String,Any}(last_.attrs)
        attrs["fused"] = fo
        idx = findfirst(o -> o.id == last_.id, ops)
        ops[idx] = Op(last_.id, "fused.elementwise", ins, last_.out, attrs)
        for o in grp[1:(end - 1)]
            push!(drop, o.id)
            fused += 1
        end
    end
    ops = [o for o in ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), fused)
end

"""
    unaryfused(g, op) -> (FusedOp, inid) | nothing

This op as a function of **one** tensor, plus the id of that tensor.

That is the shape an epilogue can take: `matmul!`'s `epi` is applied to a single
value inside the GEMM's store, so it may close over constants but cannot read a
second tensor. `mul.Tensor(x, 2)` qualifies and `mul.Tensor(x, y)` does not. It
is also the shape a reduction's map step can take — see [`foldpremap`](@ref),
which is why the operand id comes back too: folding into a *consumer* leaves
that consumer reading the operand where it used to read this op's result.

Handles an already-fused op too, so a chain `fuseops` collapsed can be absorbed
whole rather than only its last link.

**`Rounded`, like `recipe`.** Folding removes the store this op would have done,
and with it the rounding to its declared dtype. For an epilogue that costs
nothing — the GEMM's own store rounds to the same type immediately after — but a
reduction consumes the value with no store in between, so a Float64 scalar
attribute would otherwise carry the whole reduction into Float64.
"""
function unaryfused(g::Graph, op::Op)
    if op.aten == "fused.elementwise"
        fo = get(op.attrs, "fused", nothing)
        (fo isa FusedOp && nargs(fo) == 1 && length(op.ins) == 1) || return nothing
        return (fo, op.ins[1])          # `recipe` already wrapped every step
    end
    fa = fusedfunc(g, op)
    fa === nothing && return nothing
    f, arity = fa
    ntensor = arity - count(p -> haskey(op.attrs, "arg$(p - 1)"), 1:arity)
    ntensor == 1 || return nothing
    ins = String[]
    refs = operandrefs(g, op, arity, Dict{String,Any}(), ins)
    refs === nothing && return nothing
    length(ins) == 1 || return nothing
    b = get(g.buffers, op.out, nothing)
    b === nothing && return nothing
    (FusedOp((f isa Cast ? f : Rounded(b.dtype, f),), (refs,)), ins[1])
end

"""
    foldepilogue(graph) -> (graph, n)

Fold a unary elementwise op into the `addmm` that produces it.

The GEMM's store already reads and writes every element of its result, so an
activation applied there is free, and a separate op is a full read-modify-write
pass over the same memory. `foldrelu` does exactly this for `relu` and `gelu`
by name; this generalises it to any unary expression by carrying a `FusedOp`
instead of a symbol.

**`addmm` only, for now.** It is the one producer whose handler takes `epi` as a
*function* (`matmul!(…; epi)`). Convolution's epilogue dispatches on
`Val(act::Symbol)` inside `conv_epilogue_kernel!`, so generalising it means
changing the kernel's signature rather than the caller's — a bigger change, and
SAM 2 has one candidate for it against eleven for `addmm`.
"""
function foldepilogue(g::Graph)
    producer = Dict(o.out => o for o in g.ops)
    readers = Dict{String,Int}()
    for o in g.ops, i in o.ins
        id = resolvealias(g, i); readers[id] = get(readers, id, 0) + 1
    end
    for (_, b) in g.buffers
        isempty(b.of) && continue
        id = resolvealias(g, b.of); readers[id] = get(readers, id, 0) + 1
    end
    for o in g.outputs
        id = resolvealias(g, o); readers[id] = get(readers, id, 0) + 1
    end

    buffers = Dict{String,Buffer}(g.buffers)
    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for op in g.ops
        uf = unaryfused(g, op)
        uf === nothing && continue
        fo, _ = uf
        isempty(op.ins) && continue
        src = resolvealias(g, first(op.ins))
        p = get(producer, src, nothing)
        (p === nothing || p.aten != "addmm.default") && continue
        get(readers, src, 0) == 1 || continue        # the GEMM feeds only this op
        (haskey(p.attrs, "act") || haskey(p.attrs, "epilogue")) && continue
        p.id in drop && continue
        ob = get(buffers, op.out, nothing); pb = get(buffers, p.out, nothing)
        (ob === nothing || pb === nothing || ob.dtype !== pb.dtype) && continue

        pi_ = findfirst(o -> o.id == p.id, ops)
        attrs = Dict{String,Any}(p.attrs); attrs["epilogue"] = fo
        ops[pi_] = Op(p.id, p.aten, p.ins, p.out, attrs)
        # The consumed op's buffer becomes an alias of the GEMM's, exactly as
        # `foldrelu` does — whoever read it still finds the same numbers.
        buffers[op.out] = Buffer(op.out, :view, ob.shape, ob.dtype, "", (0, 0),
                                 p.out, "alias.default", ob.attrs)
        push!(drop, op.id)
        n += 1
    end
    n == 0 && return (g, 0)
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers,
           [id for id in g.order if !(id in drop)],
           [o for o in ops if !(o.id in drop)], g.fusion), n)
end

"""
Reductions a mapping function can be pushed into.

`sum(f, a; dims)` *is* `mapreduce(f, +, a; dims)` and `prod(f, a; dims)` is the
same with `*`, so these three need no new kernel — only the function passed one
level down. `any`/`all` are deliberately absent: premapping into them changes
what is being asked, not how it is computed.
"""
const PREMAPPABLE = ("sum.dim_IntList", "mean.dim", "prod.dim_int")

"""
    foldpremap(graph) -> (graph, n)

Fold a unary elementwise op into the reduction that consumes it:
`sum(f.(x))` becomes `mapreduce(f, +, x)`.

The mirror image of [`foldepilogue`](@ref) — that one folds into the op that
*produces* the value, this one into the op that *consumes* it — and it saves the
same thing: a full read-modify-write pass over a tensor whose elements the
reduction was going to touch anyway. The reduction reads the operand where it
used to read the mapped result, and the mapped op disappears.

**Where this actually fires is narrow, and the reason is structural.** Across
SAM 2 and MatAnyone, of 136 reductions:

```
producer has >1 reader              111
producer not unary-fusable           16
no producer (a graph input)           5
candidate                             4
```

114 of those producers are `add.Tensor`, which is the residual stream: in
`x = x + sublayer(x)` followed by `norm(x)`, the sum feeds both the norm and the
next skip connection, so it has to exist as a tensor regardless. That is a
property of transformers, not of this pass, and no amount of widening the
predicate reaches it.

The four that do fire are all `prod(1 - x)` in MatAnyone's `readout_query` and
`segment`. Small, and the pass is ~40 lines that reuse `unaryfused` — but the
alternative reading, that reductions are simply not worth premapping, is wrong
for a reason worth having written down.

`native_layer_norm` and `_softmax` are excluded, and the reason is **not** that
they compute their reductions in hand-written kernels — that is true and it is
beside the point. Broken out per reduction:

```
                            total   single-reader src   premappable
native_layer_norm.default     114          12                0
mean.dim                       11           1                0
sum.dim_IntList                 4           0                0
prod.dim_int                    4           4                4
_softmax.default                3           3                0
```

All 15 of those single-reader producers are **binary**: 12 are `add.Tensor` with
two tensor operands and 3 are `bmm.default`. A map step is applied per element
to one value, so a two-tensor add cannot be one whatever the kernel signature
says. Changing those kernels to accept an `f` would fold zero ops on either
model.
"""
function foldpremap(g::Graph)
    producer = Dict(o.out => o for o in g.ops)
    readers = Dict{String,Int}()
    for o in g.ops, i in o.ins
        id = resolvealias(g, i); readers[id] = get(readers, id, 0) + 1
    end
    for (_, b) in g.buffers
        isempty(b.of) && continue
        id = resolvealias(g, b.of); readers[id] = get(readers, id, 0) + 1
    end
    for o in g.outputs
        id = resolvealias(g, o); readers[id] = get(readers, id, 0) + 1
    end

    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for op in g.ops
        op.aten in PREMAPPABLE || continue
        haskey(op.attrs, "premap") && continue
        isempty(op.ins) && continue
        src = resolvealias(g, first(op.ins))
        p = get(producer, src, nothing)
        p === nothing && continue
        get(readers, src, 0) == 1 || continue      # the mapped value feeds only this
        p.id in drop && continue
        uf = unaryfused(g, p)
        uf === nothing && continue
        fo, inid = uf
        # The reduction reduces over the operand's extents now, so the shapes it
        # derives from its input have to be unchanged by the map. An elementwise
        # op does not change them, but a `Cast` between differently-shaped views
        # would, and that is cheaper to refuse than to reason about.
        pb = get(g.buffers, p.out, nothing); ib = get(g.buffers, resolvealias(g, inid), nothing)
        (pb === nothing || ib === nothing || pb.shape != ib.shape) && continue

        oi = findfirst(o -> o.id == op.id, ops)
        attrs = Dict{String,Any}(op.attrs); attrs["premap"] = fo
        ops[oi] = Op(op.id, op.aten, [inid; op.ins[2:end]], op.out, attrs)
        push!(drop, p.id)
        n += 1
    end
    n == 0 && return (g, 0)
    (Graph(g.name, g.symbols, g.inputs, g.outputs, Dict{String,Buffer}(g.buffers),
           [id for id in g.order if !(id in drop)],
           [o for o in ops if !(o.id in drop)], g.fusion), n)
end

function foldpremap(graphs::AbstractDict)
    out = Dict{String,Graph}(); n = 0
    for (name, g) in graphs
        g2, k = foldpremap(g); out[name] = g2; n += k
    end
    (out, n)
end

function foldepilogue(graphs::AbstractDict)
    out = Dict{String,Graph}(); n = 0
    for (name, g) in graphs
        g2, k = foldepilogue(g); out[name] = g2; n += k
    end
    (out, n)
end

function fuseops(graphs::AbstractDict)
    out = Dict{String,Graph}()
    n = 0
    for (name, g) in graphs
        g2, k = fuseops(g)
        out[name] = g2
        n += k
    end
    (out, n)
end

"""
One fused elementwise op.

The `FusedOp` is built by the pass and carried in the attributes, so it is
constructed once at load rather than per call. Fetching it out of an
`Dict{String,Any}` is a dynamic lookup, which is why the broadcast is behind a
function barrier: `fusedemit!` specialises on the concrete `FusedOp` type, so the
kernel it launches is as concrete as any other, and the one dynamic dispatch is
per op per run — the same cost `runop!`'s own `Val` dispatch already pays.
"""
runop!(ctx::Ctx, op::Op, ::Val{Symbol("fused.elementwise")}) =
    fusedemit!(ctx, op.attrs["fused"], ntuple(i -> value(ctx, op.ins[i]), length(op.ins)))

@inline fusedemit!(ctx::Ctx, fo::FusedOp, args::Tuple) =
    emit(ctx, Base.broadcasted(fo, args...))
