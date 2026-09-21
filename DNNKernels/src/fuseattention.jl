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

**Verify this numerically**, because a mis-traced transpose here produces a
plausible wrong image rather than an error: against the unfused chain on this
model's own tensors, `rel 2.4e-3`, which is fp16 attention accuracy.

Run that comparison with `plan = nothing`. Against a graph run WITH a plan the
transients are reused, so `mul` no longer holds layer 1's values by the time it
is read, and the check reports `rel 1.1` for a correct fusion.
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
    safesoftmaxtail(g, producer, wh) -> (softmax, ops, scores) | nothing

torch's `_safe_softmax`, which is what `scaled_dot_product_attention` becomes
when it is *exported* rather than lowered to a kernel:

    attn = softmax(scores, -1)
    attn = where(!any(scores != -inf, -1), zeros_like(attn), attn)

The second line is the guard for a query row whose every key is masked out:
`softmax` of an all `-inf` row is `NaN`, and the reference returns zeros there.

`wh` is the `where`. The return is the softmax it guards, the ops the guard is
made of, and the scores buffer its `eq` tested — the caller matches that against
the `bmm` it finds, because dropping the guard is only sound when the scores are
the product itself. A product of finite operands has no `-inf` in it, so the
guard selects the softmax at every position and deleting it changes nothing; put
an additive mask between the two and that stops being true, which is why the
scores identity is checked rather than assumed.
"""
function safesoftmaxtail(g::Graph, producer::AbstractDict, wh::Op)
    length(wh.ins) == 3 || return nothing
    sm = get(producer, viewroot(g, wh.ins[3]), nothing)
    sm === nothing && return nothing
    sm.aten == "_softmax.default" || return nothing
    get(sm.attrs, "arg1", 0) == -1 || return nothing

    zero = get(producer, viewroot(g, wh.ins[2]), nothing)
    zero === nothing && return nothing
    zero.aten == "full_like.default" || return nothing
    viewroot(g, zero.ins[1]) == sm.out || return nothing
    iszero(scalar(get(zero.attrs, "arg1", 1))) || return nothing

    allmasked = get(producer, viewroot(g, wh.ins[1]), nothing)
    (allmasked === nothing || allmasked.aten != "logical_not.default") && return nothing
    anykey = get(producer, viewroot(g, allmasked.ins[1]), nothing)
    (anykey === nothing || anykey.aten != "any.dim") && return nothing
    get(anykey.attrs, "arg1", 0) == -1 && get(anykey.attrs, "arg2", false) === true ||
        return nothing
    unmasked = get(producer, viewroot(g, anykey.ins[1]), nothing)
    (unmasked === nothing || unmasked.aten != "logical_not.default") && return nothing
    isinf = get(producer, viewroot(g, unmasked.ins[1]), nothing)
    (isinf === nothing || isinf.aten != "eq.Scalar") && return nothing
    scalar(get(isinf.attrs, "arg1", 0)) == -Inf || return nothing

    (softmax = sm, ops = [wh.id, zero.id, allmasked.id, anykey.id, unmasked.id, isinf.id],
     scores = isinf.ins[1])
end

"""
    flashoperand(g, producer, id, want; allowscale) -> (id, scale) | nothing

One attention operand as fp16, with any scaling on the way to it.

`lastmatching` alone finds the operand the `bmm` reads, and under this export
that is an fp32 copy of an fp16 tensor: torch's decomposition scales q and k by
`scale^0.5` each and runs the product in fp32. Fed that, `flashcm_plan` declines
on `:eltype` and a 4096-query attention falls back to materialising its scores —
2.12 GB per layer, which is what made this worth writing.

So the walk continues through `mul.Scalar` and the widening `_to_copy` to the
fp16 tensor underneath, multiplying the scalars out as it goes; the fused op
carries their product as its `scale` and the `mul`s are left for `dropdead`.

`allowscale = false` for v, where a scalar is NOT the attention's scale but a
factor on the result, and folding it into `scale` would be wrong.

Returns the fp32 operand unchanged, with `scale = 1`, when no fp16 pre-image is
reachable: that is the operand the pass took before and the graphs that relied
on it still get it.
"""
function flashoperand(g::Graph, producer::AbstractDict, id::AbstractString, want;
                      allowscale::Bool = true)
    direct = lastmatching(g, id, want)
    direct !== nothing && g.buffers[direct].dtype === Float16 && return (direct, 1.0)
    cur, scale = id, 1.0
    for _ in 1:8
        cand = lastmatching(g, cur, want)
        cand !== nothing && g.buffers[cand].dtype === Float16 && return (cand, scale)
        op = get(producer, viewroot(g, cand === nothing ? cur : cand), nothing)
        op === nothing && break
        if allowscale && op.aten == "mul.Scalar" && length(op.ins) == 1 &&
                get(op.attrs, "arg1", nothing) isa Real
            scale *= Float64(op.attrs["arg1"])
            cur = op.ins[1]
        elseif op.aten == "_to_copy.default" && length(op.ins) == 1
            cur = op.ins[1]
        else
            break
        end
    end
    direct === nothing ? nothing : (direct, 1.0)
end

"""
    attentiongroup(g, producer, b2) -> (q, k, v, scale, ...) | nothing

The operands if `b2` is the second `bmm` of an attention, else `nothing`.
Every shape relation is checked, so a `bmm` pair that merely looks similar is
declined rather than silently rewritten.

`scale` is `nothing` when the scaling is left in the graph — the case this pass
was written for — and a number when [`flashoperand`](@ref) folded it in order to
reach fp16 operands.
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
    guarded = nothing
    if sc.aten == "where.self"
        tail = safesoftmaxtail(g, producer, sc)
        tail === nothing && return nothing
        append!(skipped, tail.ops)
        guarded = tail.scores
        sc = tail.softmax
    end
    sc.aten == "_softmax.default" || return nothing
    b1 = get(producer, viewroot(g, sc.ins[1]), nothing)
    (b1 === nothing || b1.aten != "bmm.default") && return nothing
    # The guard is only a no-op over the product itself — see `safesoftmaxtail`.
    guarded === nothing || viewroot(g, guarded) == b1.out || return nothing

    # q fixes (H, Lq, D); k arrives transposed and fixes Lk. Lq and Lk are not
    # the same length in a joint attention — Qwen-Image's image stream queries
    # 4096 positions over 4352 keys — so they are tracked separately and every
    # shape below is checked against both.
    qc = g.buffers[b1.ins[1]].shape          # (H, Lq, D)
    kc = g.buffers[b1.ins[2]].shape          # (H, D, Lk)
    length(qc) == 3 && length(kc) == 3 || return nothing
    H, Lq, D = qc
    Hk, Dk, Lk = kc
    H == Hk && D == Dk || return nothing
    # the scores must really be (H, Lq, Lk), and the output (H, Lq, D)
    g.buffers[b1.out].shape == Any[H, Lq, Lk] || return nothing
    g.buffers[b2.ins[2]].shape == Any[H, Lk, D] || return nothing
    g.buffers[b2.out].shape == Any[H, Lq, D] || return nothing

    wantq = Any[1, H, Lq, D]
    wantkv = Any[1, H, Lk, D]
    q = flashoperand(g, producer, b1.ins[1], wantq); q === nothing && return nothing
    k = flashoperand(g, producer, b1.ins[2], wantkv); k === nothing && return nothing
    v = flashoperand(g, producer, b2.ins[2], wantkv; allowscale = false)
    v === nothing && return nothing
    # All three or none: a fused op with an fp16 q and an fp32 k has no plan at
    # all, and half a rewrite is worse than the graph it replaced.
    fp16 = all(x -> g.buffers[x[1]].dtype === Float16, (q, k, v))
    scale = fp16 ? q[2] * k[2] * v[2] : nothing
    (q[1], k[1], v[1], scale, b1, sc, b2, skipped)
end

"""
    fuseattention(g) -> (g, n)

Replace each `bmm -> softmax -> [clone | safe-softmax guard] -> bmm` with one
`fused.sdpa`.

The scale stays where it is whenever the operands are already fp16: the `mul`
that produced q is left in the graph and the fused op runs with `scale = 1`, so
nothing here has to know how the exporter spelled the scaling. An export that
upcasts to fp32 around the product leaves no fp16 operand to fuse over, and
there `flashoperand` folds the scaling into the op instead — see its docstring.
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
        n >= FUSEATTENTIONLIMIT[] && break
        got = attentiongroup(g, producer, op)
        got === nothing && continue
        q, k, v, scale, b1, sc, b2, skipped = got
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
        # The graph's own outputs count as readers. `fusemaskedattention` has
        # always checked this and this one did not: a graph asked for its scores
        # or its attention weights — which is how a reference dump is taken —
        # lost the buffer that produced them.
        any(id -> viewroot(g, id) in gone, g.outputs) && continue
        any(o -> !(o.id in group) && any(x -> viewroot(g, x) in gone, o.ins), g.ops) && continue
        attrs = Dict{String,Any}(b2.attrs)
        scale === nothing || (attrs["scale"] = scale)
        ops[i] = Op(b2.id, "fused.sdpa", [q, k, v], b2.out, attrs)
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
    FUSEATTENTION[] = false

Turn the rewrite off, for A/B-ing it against the unfused graph on one model.
On by default: it is worth **2.53x** on DepthAnything (37.4 -> 14.8 ms, 35% ->
88% of PyTorch) with the output matching the unfused model at rel 9.4e-4 and
correlation 0.9999977.
"""
const FUSEATTENTION = Ref(true)

"""
    FUSEATTENTIONLIMIT[] = 1

Fuse at most this many blocks. For bisecting: fusing one block at a time and
watching where the model's output first diverges separates "every block is
slightly off and it compounds" from "one block resolves an operand differently
from the one that was checked by hand".
"""
const FUSEATTENTIONLIMIT = Ref(typemax(Int))

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
