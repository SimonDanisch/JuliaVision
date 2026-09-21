"""
Collapse an exported rotary embedding into one op.

`apply_rotary_pos_emb` is written in every HF model as

    x1  = x[..., :H/2]
    x2  = x[..., H/2:]
    rot = cat(-x2, x1)
    y   = x * cos + rot * sin

and `run_decompositions()` keeps it that way: a `neg`, a `cat`, two `mul`s and an
`add` per projection, twice per layer. On K2 Horizon 32B that is ~380 dispatches
of a decode step for what is an elementwise function of one tensor — the
concatenated half is never needed as a buffer, since `rot[j]` is `-x[j + H/2]`
below the midpoint and `x[j - H/2]` above it.

Runs before `fuseops`, which would otherwise absorb the two `mul`s and the `add`
into a `FusedOp` and leave the `neg`/`cat` stranded in front of it.
"""

"""
    FUSEROPE[] = true

Turn the rewrite off, to A/B it against the unfused graph.
"""
const FUSEROPE = Ref(true)

"""
    fuserope(graphs) -> (graphs, n)

Rewrite each `cat(-x[H/2:], x[:H/2]) * sin + x * cos` into one `fused.rope`.
"""
function fuserope(graphs::AbstractDict)
    FUSEROPE[] || return (Dict{String,Any}(graphs), 0)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fuserope(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

# A slice of `id` along its last axis starting at `lo`, or `nothing`.
function lastslice(g::Graph, id::AbstractString)
    b = get(g.buffers, id, nothing)
    (b === nothing || b.kind !== :view || b.viewop != "slice.Tensor") && return nothing
    d = Int(get(b.attrs, "arg1", 0))
    nd = length(b.shape)
    (d == -1 || d == nd - 1) || return nothing
    (String(b.of), Int(get(b.attrs, "arg2", 0)), Int(b.shape[end]))
end

function fuserope(g::Graph)
    producer = Dict{String,Op}()
    byin = Dict{String,Vector{Op}}()
    for op in g.ops
        isempty(string(op.out)) || (producer[op.out] = op)
        for id in op.ins
            push!(get!(byin, id, Op[]), op)
            r = viewroot(g, id)
            r == id || push!(get!(byin, r, Op[]), op)
        end
    end
    sole(id) = (rs = get(byin, id, nothing); rs !== nothing && length(rs) == 1 ? rs[1] : nothing)

    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for c in g.ops
        c.aten == "cat.default" || continue
        length(c.ins) == 2 || continue
        Int(get(c.attrs, "arg1", 0)) == -1 || continue

        ng = get(producer, c.ins[1], nothing)
        (ng === nothing || ng.aten != "neg.default") && continue
        hi = lastslice(g, ng.ins[1]); hi === nothing && continue
        lo = lastslice(g, c.ins[2]);  lo === nothing && continue
        hi[1] == lo[1] || continue                       # same source tensor
        lo[2] == 0 && hi[2] == lo[3] && hi[3] == lo[3] || continue
        X = hi[1]
        H = Int(g.buffers[X].shape[end])
        H == 2 * lo[3] || continue

        m1 = sole(c.out)
        (m1 === nothing || m1.aten != "mul.Tensor") && continue
        sinid = viewroot(g, m1.ins[1]) == c.out ? m1.ins[2] : m1.ins[1]

        a = sole(m1.out)
        (a === nothing || a.aten != "add.Tensor") && continue
        otherid = viewroot(g, a.ins[1]) == viewroot(g, m1.out) ? a.ins[2] : a.ins[1]
        m2 = get(producer, viewroot(g, otherid), nothing)
        (m2 === nothing || m2.aten != "mul.Tensor") && continue
        # The other product must be `x * cos` over the very tensor that was
        # rotated. Compared by ID first: `X` is itself a view here — the export
        # rotates `permute_2`, not the buffer underneath it — so asking for its
        # root and comparing that matches nothing.
        same(id) = id == X || viewroot(g, id) == viewroot(g, X)
        cosid = if same(m2.ins[1])
            m2.ins[2]
        elseif same(m2.ins[2])
            m2.ins[1]
        else
            continue
        end

        # If the ONLY thing the rotated tensor feeds is an in-place cache write,
        # store into the cache directly and drop that write. `fuserope` runs
        # after `foldcacheupdate`, so the `index_put` is already in place and its
        # target is the caller's own buffer.
        # Nothing may still name the rotated tensor once it stops existing: not a
        # declared output, and not the parent of a surviving view.
        escapes = a.out in g.outputs || any(b -> String(b.of) == a.out, values(g.buffers))
        put = escapes ? nothing : sole(a.out)
        if put !== nothing && put.aten == "index_put.default" &&
           Bool(something(get(put.attrs, "inplace", nothing), false)) &&
           length(put.ins) == 3 && viewroot(g, put.ins[end]) == viewroot(g, a.out)
            # Into the WRITE's slot, not the rotation's: the fused op reads the
            # cache view the write targeted, so it has to sit where that write
            # sat. Replacing the rotation's slot instead leaves the original
            # write in the list, still reading a tensor that no longer exists.
            ops[findfirst(==(put), ops)] =
                Op(put.id, "fused.ropecache", [X, cosid, sinid, put.ins[1], put.ins[2]],
                   put.out, Dict{String,Any}())
            for o in (c, ng, m1, m2, a); push!(drop, o.id); end
            n += 1
            continue
        end
        ops[findfirst(==(a), ops)] = Op(a.id, "fused.rope", [X, cosid, sinid], a.out,
                                        Dict{String,Any}())
        for o in (c, ng, m1, m2); push!(drop, o.id); end
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
    pairview(g, id) -> (x, P) | nothing

The `(..., P, 2)` view of an `(..., 2P)` tensor that an INTERLEAVED rotary
splits into its even and odd components.
"""
function pairview(g::Graph, id::AbstractString)
    b = get(g.buffers, id, nothing)
    (b === nothing || b.kind !== :view || b.viewop != "view.default") && return nothing
    sh = b.shape
    length(sh) >= 2 || return nothing
    (sh[end] isa Integer && Int(sh[end]) == 2) || return nothing
    sh[end - 1] isa Integer || return nothing
    p = get(g.buffers, b.of, nothing)
    p === nothing && return nothing
    ps = p.shape
    length(ps) == length(sh) - 1 || return nothing
    (ps[end] isa Integer && Int(ps[end]) == 2 * Int(sh[end - 1])) || return nothing
    all(k -> ps[k] == sh[k], 1:(length(ps) - 1)) || return nothing
    (String(b.of), Int(sh[end - 1]))
end

"""
    pairhalf(g, id) -> (pairview, which) | nothing

`squeeze(slice(V, -1, w, w + 1))` — one of the two components of a pair view,
and which of them it is.
"""
function pairhalf(g::Graph, id::AbstractString)
    b = get(g.buffers, id, nothing)
    (b === nothing || b.kind !== :view) && return nothing
    b.viewop in ("squeeze.dims", "squeeze.dim") || return nothing
    sl = get(g.buffers, b.of, nothing)
    (sl === nothing || sl.kind !== :view || sl.viewop != "slice.Tensor") && return nothing
    nd = length(sl.shape)
    d = Int(get(sl.attrs, "arg1", 0))
    (d == -1 || d == nd - 1) || return nothing
    lo = Int(get(sl.attrs, "arg2", 0))
    hi = Int(get(sl.attrs, "arg3", 0))
    (hi == lo + 1 && sl.shape[end] isa Integer && Int(sl.shape[end]) == 1) || return nothing
    (String(sl.of), lo)
end

"""
    fusepairrope(graphs) -> (graphs, n)

The OTHER rotary spelling: a rotation of adjacent PAIRS rather than of halves.

`fuserope` above collapses `cat(-x[H/2:], x[:H/2]) * sin + x * cos`, which is
how the HF models write it. Qwen-Image 2.1 writes the complex form instead —
view the head as `H/2` pairs, rotate each one, interleave the results back:

    xr = x.view(..., H/2, 2)
    a, b = xr[..., 0], xr[..., 1]
    out  = stack((a*cos - b*sin, a*sin + b*cos), -1).flatten(-2)

Same function, different layout, and nothing matched it. What that costs is not
the arithmetic — `fuseops` collapses each of the two halves into one pass — it
is the LAYOUT: the two components are a stride-2 read each, so both are
materialised, and the interleave is another pass over the result. Measured on
one layer at 1024²: 1.95 ms of squeezes, 1.16 of interleave, 1.84 of arithmetic
and ~1.1 of the view and cast around them, against 1.6 ms for both rotations as
one kernel.

The pattern is spelled out rather than generalised, on purpose: it is eight ops
in a fixed arrangement over one pair view, and a looser match on a rewrite that
silently changes what a model computes is not worth the reach.
"""
function fusepairrope(graphs::AbstractDict)
    FUSEROPE[] || return (Dict{String,Any}(graphs), 0)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fusepairrope(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

function fusepairrope(g::Graph)
    producer = Dict{String,Op}()
    byin = Dict{String,Vector{Op}}()
    for op in g.ops
        isempty(string(op.out)) || (producer[op.out] = op)
        for id in op.ins
            push!(get!(byin, id, Op[]), op)
            r = viewroot(g, id)
            r == id || push!(get!(byin, r, Op[]), op)
        end
    end
    sole(id) = (rs = get(byin, id, nothing); rs !== nothing && length(rs) == 1 ? rs[1] : nothing)
    # One product of a component and a table, split into which component it is
    # and what it was multiplied by.
    function split1(m::Op)
        h1 = pairhalf(g, m.ins[1])
        h2 = pairhalf(g, m.ins[2])
        h1 !== nothing && h2 === nothing && return (h1, m.ins[2])
        h2 !== nothing && h1 === nothing && return (h2, m.ins[1])
        return nothing
    end

    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for c in g.ops
        c.aten == "cat.default" || continue
        length(c.ins) == 2 || continue
        Int(get(c.attrs, "arg1", 0)) == -1 || continue
        us = [get(g.buffers, i, nothing) for i in c.ins]
        all(u -> u !== nothing && u.kind === :view && u.viewop == "unsqueeze.default", us) || continue

        A = get(producer, String(us[1].of), nothing)      # a*cos - b*sin
        B = get(producer, String(us[2].of), nothing)      # a*sin + b*cos
        (A === nothing || B === nothing) && continue
        (A.aten == "sub.Tensor" && B.aten == "add.Tensor") || continue
        length(A.ins) == 2 && length(B.ins) == 2 || continue

        ms = [get(producer, id, nothing) for id in (A.ins[1], A.ins[2], B.ins[1], B.ins[2])]
        any(m -> m === nothing || m.aten != "mul.Tensor" || length(m.ins) != 2, ms) && continue
        ps = map(split1, ms)
        any(isnothing, ps) && continue
        # All four read the same pair view, and the subtraction is the one whose
        # order is fixed: `a*cos` minus `b*sin`.
        V = ps[1][1][1]
        all(q -> q[1][1] == V, ps) || continue
        (ps[1][1][2] == 0 && ps[2][1][2] == 1) || continue
        cosid, sinid = ps[1][2], ps[2][2]
        # The addition may come either way round.
        i3, i4 = ps[3][1][2], ps[4][1][2]
        (bc, as) = i3 == 1 && i4 == 0 ? (ps[3], ps[4]) :
                   i4 == 1 && i3 == 0 ? (ps[4], ps[3]) : (nothing, nothing)
        bc === nothing && continue
        viewroot(g, bc[2]) == viewroot(g, cosid) || continue
        viewroot(g, as[2]) == viewroot(g, sinid) || continue

        pv = pairview(g, V)
        pv === nothing && continue
        X, P = pv

        # Nothing outside the group may read what the group is about to delete.
        all(m -> sole(m.out) !== nothing && sole(m.out).id in (A.id, B.id), ms) || continue
        (sole(A.out) === c && sole(B.out) === c) || continue

        # The narrowing cast on the way out, which the kernel does in its store.
        last = c
        w = sole(c.out)
        if w !== nothing && w.aten == "_to_copy.default" &&
           haskey(g.buffers, w.out) && haskey(g.buffers, c.out) &&
           sizeof(g.buffers[w.out].dtype) <= sizeof(g.buffers[c.out].dtype)
            last = w
        end

        ops[findfirst(==(last), ops)] =
            Op(last.id, "fused.pairrope", [X, cosid, sinid], last.out,
               Dict{String,Any}("P" => P))
        for o in (ms[1], ms[2], ms[3], ms[4], A, B, c)
            o.id == last.id || push!(drop, o.id)
        end
        n += 1
    end
    n == 0 && return (g, 0)
    keep = [o for o in ops if !(o.id in drop)]
    dropped = Set(o.out for o in g.ops if o.id in drop)
    order = [id for id in g.order if !(id in dropped)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, Dict{String,Buffer}(g.buffers),
           order, keep, g.fusion), n)
end
