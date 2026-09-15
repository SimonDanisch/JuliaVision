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
