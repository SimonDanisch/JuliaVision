"""
Rewrite an attention written as separate ops into one `fused.sdpa`.

Not every export produces `_scaled_dot_product_*`. DepthAnything's lowers
attention to `mul -> bmm -> softmax -> clone -> bmm`, so it could never reach
`sdpa` and paid **21.7 ms of its 37.1 ms**:

    bmm  QK^T    (6,1370,64)x(6,64,1370)    6.07 ms   2.85 TF/s
    bmm  attn@V  (6,1370,1370)x(6,1370,64) 12.83      1.35 TF/s
    softmax                                  2.76
    clone of the (1,6,1370,1370) scores      22.5 MB per layer, 270 MB over 12

`sdpa` at the same shapes is **0.183 ms/layer against 1.805 — 9.88x** — because
the scores are never materialised at all.

The pattern, and what each operand is:

    mul.Tensor     q * scale            -> the fused op's q, with scale folded in
    bmm.default    q @ k^T              k is reached through a permute of the
    _softmax                              last two axes
    clone.default  (optional)           a copy of the scores; fusing deletes it
    bmm.default    attn @ v             -> v

**Finding the operands.** Each `bmm` input is a chain of views over the QKV
projection — `view -> expand -> select -> permute -> view -> addmm` — so the
operands cannot be read off the `bmm` directly. Walking the `of` chain and taking
the **last buffer whose shape is `(1, H, S, D)`** picks all three correctly:

    q   view_4 -> expand_1 -> mul           both 4-D match; `mul` is last
    v   view_8 -> expand_4 -> select_2      stops before `permute_2` goes 5-D
    k   view_5 -> expand_2 -> permute_3     those are (1,H,D,S) and do NOT match,
                              -> select_1   so the transposed pre-image is taken

**This was verified numerically before it was written**, because a mis-traced
transpose here produces a plausible wrong image rather than an error: against the
unfused chain on this model's own tensors, `rel 2.4e-3` — fp16 attention accuracy.

The first attempt at that check compared against a graph run WITH the slab plan,
where transient buffers are reused, so `mul` no longer held layer 1's values by
the time it was read: it reported `rel 1.1` and looked like a broken fusion. Run
the comparison with `plan = nothing` or the operands are not what they are named.
"""

"""
    lastmatching(g, id, shape) -> String

The last buffer in `id`'s view chain whose shape equals `shape`. See the note
above on why this is the rule that finds q, k and v uniformly.
"""
function lastmatching(g::Graph, id::AbstractString, shape)
    found = nothing
    cur = id
    while true
        bb = g.buffers[cur]
        bb.shape == shape && (found = cur)
        (bb.kind === :view && !isempty(string(bb.of)) && haskey(g.buffers, bb.of)) || break
        cur = bb.of
    end
    found
end

"""
    viewroot(g, id) -> String

Follow `of` until the buffer is not a view — the op output the chain reads from.
"""
function viewroot(g::Graph, id::AbstractString)
    cur = id
    while true
        bb = g.buffers[cur]
        (bb.kind === :view && !isempty(string(bb.of)) && haskey(g.buffers, bb.of)) || return cur
        cur = bb.of
    end
end

"""
    attentiongroup(g, producer, b2) -> (q, k, v) | nothing

The three operands if `b2` is the second `bmm` of an attention, else `nothing`.
Every shape relation is checked, so a `bmm` pair that merely looks similar is
declined rather than silently rewritten.
"""
function attentiongroup(g::Graph, producer::AbstractDict, b2::Op)
    # The scores reach the second `bmm` through a run of copies: a `clone`, and
    # autocast's `_to_copy`, which under this export is a REAL Float32 -> Float16
    # cast because the softmax runs in fp32.
    #
    # Skipping a dtype change would normally be wrong. It is safe here for one
    # specific reason: **fusing deletes the scores**, so nothing downstream can
    # observe their precision. `sdpa` computes its softmax in fp32 regardless,
    # and the fused op's output dtype comes from the declared output buffer, not
    # from anything on this path. That is the claim the numeric check in this
    # file's header tests end to end, and it holds at rel 2.4e-3.
    passthrough = Set(["clone.default", "_to_copy.default", "detach.default", "alias.default"])
    skipped = String[]
    sc = get(producer, viewroot(g, b2.ins[1]), nothing)
    while sc !== nothing && sc.aten in passthrough
        push!(skipped, sc.id)
        sc = get(producer, viewroot(g, sc.ins[1]), nothing)
    end
    sc === nothing && return nothing
    sc.aten == "_softmax.default" || return nothing
    b1 = get(producer, viewroot(g, sc.ins[1]), nothing)
    (b1 === nothing || b1.aten != "bmm.default") && return nothing

    # q from the first bmm; its 4-D form fixes (H, S, D)
    qc = g.buffers[b1.ins[1]].shape          # (H, S, D)
    length(qc) == 3 || return nothing
    H, S, D = qc
    want = Any[1, H, S, D]
    q = lastmatching(g, b1.ins[1], want); q === nothing && return nothing
    k = lastmatching(g, b1.ins[2], want); k === nothing && return nothing
    v = lastmatching(g, b2.ins[2], want); v === nothing && return nothing
    # the scores must really be (H, S, S), and the output (H, S, D)
    g.buffers[b1.out].shape == Any[H, S, S] || return nothing
    g.buffers[b2.out].shape == Any[H, S, D] || return nothing
    (q, k, v, b1, sc, b2, skipped)
end

"""
    fuseattention(g) -> (g, n)

Replace each `bmm -> softmax -> [clone] -> bmm` with one `fused.sdpa`.

The scale stays where it is: the `mul` that produced q is left in the graph and
the fused op runs with `scale = 1`, so nothing here has to know how the exporter
spelled the scaling.
"""
function fuseattention(g::Graph)
    producer = Dict{String,Op}()
    for op in g.ops
        isempty(string(op.out)) || (producer[op.out] = op)
    end
    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for (i, op) in enumerate(ops)
        op.aten == "bmm.default" || continue
        op.id in drop && continue
        got = attentiongroup(g, producer, op)
        got === nothing && continue
        q, k, v, b1, sc, b2, skipped = got
        # Nothing OUTSIDE the group may read a buffer the group stops producing,
        # or that op loses its input. The group's own members are exempt — the
        # softmax reads the scores by construction, and forgetting to exempt it
        # is what made this decline every block it had correctly matched.
        group = Set{String}([b1.id, sc.id, b2.id]); union!(group, skipped)
        # `gone` is the DROPPED ops' outputs. `b2.out` is not among them — the
        # fused op inherits it — and including it made the very next op, which
        # legitimately reads the attention result, look like an outside consumer.
        dropping = Set{String}([b1.id, sc.id]); union!(dropping, skipped)
        gone = Set{String}(o.out for o in g.ops if o.id in dropping)
        any(o -> !(o.id in group) && any(x -> viewroot(g, x) in gone, o.ins), g.ops) && continue
        ops[i] = Op(b2.id, "fused.sdpa", [q, k, v], b2.out, Dict{String,Any}(b2.attrs))
        push!(drop, b1.id); push!(drop, sc.id)
        for id in skipped; push!(drop, id); end
        n += 1
    end
    n == 0 && return (g, 0)
    keep = [o for o in ops if !(o.id in drop)]
    dropped = Set(o.out for o in g.ops if o.id in drop)
    order = [id for id in g.order if !(id in dropped)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, Dict{String,Buffer}(g.buffers),
           order, keep, g.fusion), n)
end

"""
    FUSEATTENTION[] = true

**OFF BY DEFAULT — the rewrite is not correct yet.** It matches and rewrites all
12 of DepthAnything's blocks, and the operands it picks are the right ones
(verified against the unfused chain at `rel 2.4e-3`), but the resulting model
produces a **constant** output. The fault is upstream of `runop!`: swapping the
`out=` destination for `sdpa`'s return value changed nothing, and the wrong value
is bit-identical across both, so the fused op's result is not reaching the graph
at all rather than being computed wrongly.

Left in, off, because the matcher and the numeric groundwork are the expensive
half and both are done. What is NOT yet checked: whether `fuseops` — which runs
after this — absorbs the `mul.Tensor` that produces q into an elementwise group
and retires the buffer this op names as its input, and whether `g.order` needs
more than the dropped outputs removed.

Also the A/B switch: comparing fused against unfused needs two driver runs on one
model, because the raw graph cannot be run against a driver-built model's weights.
"""
const FUSEATTENTION = Ref(false)

function fuseattention(graphs::AbstractDict)
    FUSEATTENTION[] || return (graphs, 0)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fuseattention(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end
