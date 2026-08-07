"""
Fold a widening dtype cast into the op that reads it.

The mirror of `foldoutcasts`. That pass removes a cast that *narrows* a value the
graph just computed, by declaring the producer's result narrow. This one removes
a cast that *widens* a value before a reduction reads it, by letting the
reduction read the narrow one — because its accumulator was already the wide
type and the fp32 tensor told it nothing the accumulation did not already do.

91 of the encoder's 97 runtime casts are one shape: `add` writes fp16, a
`_to_copy` widens it to fp32, and a `native_layer_norm` reduces over the result.
The cast reads two bytes an element and writes four; the norm then reads those
four. Reading the two directly leaves the norm's arithmetic untouched and its
input traffic halved.

**The values are identical, not merely close.** fp16 -> fp32 is exact — every
fp16 is an fp32 — and `native_layer_norm` sums in `Float32` on both of its
paths: the fused kernel takes the accumulator as `Float32` outright, and the
fallback takes `accum(eltype(a))`, which is `Float32` for `Float16` as well as
for `Float32`. So the summation sees the same values in the same order and
rounds the same way.

## What it is measured to do, and what is not measured yet

Structural, and exact: it folds **91 of the encoder's 97** casts, taking the
graph from 640 ops to 549, and SAM 2's six encoder outputs come back at the same
rms against the PyTorch references to every digit — `8.89e-5`, `4.04e-4`,
`4.15e-3` and three exact zeros, unchanged.

On the kernel pair alone, at the real norm shape (144 x 65536), the two routes
agree to `max|difference| = 0.0` and cost 0.452 ms against 0.778 — 41.9% — the
cast gone and the norm reading half the bytes.

End to end the honest figure is **not yet taken**: both arms were measured on a
loaded laptop where the same graph swings 272-431 ms, so an interleaved paired
A/B (25 rounds, two models in one process) gives the direction — 20 of 25 rounds
favour the fold — but a median difference of ~5 ms that the machine's noise
swamps. Take that number again on a quiet box before quoting it. It is the same
trap `foldoutcasts` records: the serialised per-op table predicted ~10 ms here,
and per-op tables over-attribute copies.

**It does not save memory, which was the prediction and it was wrong.** The 91
fp32 temporaries disappear and `planslab`'s buffer count drops 981 -> 890, but
the peak is 261.25 MB either way: those temporaries were never what set it.

The norm's *output* dtype is unaffected: it comes from the graph's declaration
via `tupledtype`, which is `Float32` for all 91, not from `eltype(a)`. That
matters — the handler falls back to `eltype(a)` when the graph declares nothing,
and folding a cast in front of such an op would silently narrow its result.
`WIDECAST_READERS` is therefore a list of ops verified to declare, not a
category anyone should extend by analogy.

## Why only a reduction

An elementwise consumer needs no help: `fuse.jl` already folds a cast into the
op that reads it when both are elementwise, which is why `add.Tensor` costs
nothing measurable. A reduction is what that rule cannot cross, and it is also
the only consumer for which the wide input is provably redundant — it is the
accumulator, not the operand, that decides a sum's precision.

## Why the reader has to be counted through views

Same trap `foldoutcasts` and `foldrelu` record: the norm reads the cast's result
through a view chain, and a view is not a use — whoever reads *it* is. Counting
view links as readers makes every candidate look twice-read and folds nothing.
"""

"""
Ops that may read a narrower input than the graph declared for them.

Both of `native_layer_norm`'s paths accumulate in `Float32` for either input
dtype, and it takes its result's dtype from the graph rather than from its
input. An op that does neither is not a candidate: the first makes the fold
change the arithmetic, the second makes it change the result's type.
"""
const WIDECAST_READERS = Set(["native_layer_norm.default"])

"""Whether every value of `from` is exactly representable in `to`."""
exactwidening(from::Type, to::Type) =
    from === Float16 && (to === Float32 || to === Float64)

function foldincasts(g::Graph; enabled::Bool = true)
    enabled || return (g, 0)
    readers = Dict{String,Vector{Op}}()
    for op in g.ops, i in op.ins
        push!(get!(readers, last(viewchain(g, i)), Op[]), op)
    end
    isoutput = Set(last(viewchain(g, o)) for o in g.outputs)

    buffers = Dict{String,Buffer}(g.buffers)
    drop = Set{String}()
    folded = 0

    for cast in g.ops
        cast.aten == "_to_copy.default" || continue
        length(cast.ins) == 1 || continue
        ob = get(buffers, cast.out, nothing)
        ib = get(buffers, cast.ins[1], nothing)
        (ob === nothing || ib === nothing) && continue
        ob.kind === :transient || continue
        # A cast and nothing else, in the widening direction, and exact.
        ob.shape == ib.shape || continue
        exactwidening(ib.dtype, ob.dtype) || continue
        # One reader, and one this is allowed for. The wide value must reach
        # nothing else: another reader would silently be handed the narrow one.
        cast.out in isoutput && continue
        rs = get(readers, cast.out, Op[])
        length(rs) == 1 || continue
        only(rs).aten in WIDECAST_READERS || continue

        # The cast's result becomes an alias of what it used to read, declared
        # at the narrow dtype — the reader resolves through it and gets the
        # buffer the producer already wrote.
        buffers[cast.out] = Buffer(cast.out, :view, ib.shape, ib.dtype, "", (0, 0),
                                   cast.ins[1], "alias.default", ob.attrs)
        push!(drop, cast.id)
        folded += 1
    end

    folded == 0 && return (g, 0)
    ops = [o for o in g.ops if !(o.id in drop)]
    order = [id for id in g.order if !(id in drop)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, buffers, order, ops, g.fusion), folded)
end

function foldincasts(graphs::AbstractDict; enabled::Bool = true)
    out = Dict{String,Graph}()
    total = 0
    for (n, g) in graphs
        g2, k = foldincasts(g; enabled)
        out[n] = g2
        total += k
    end
    (out, total)
end
