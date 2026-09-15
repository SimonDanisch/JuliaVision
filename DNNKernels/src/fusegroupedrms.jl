"""
Collapse an exported grouped RMS norm into one op.

`nn.RMSNorm` survives `run_decompositions()` whole and reaches
`_fused_rms_norm.default`. A GROUPED one — `layernorm_num_groups = 4`, where the
mean square is taken over a quarter of the hidden dim while the gain spans all of
it — does not, and comes out as

    xf   = _to_copy(x, fp32)
    sq   = pow(xf, 2)
    ms   = mean(sq, [-1], keepdim = true)
    e    = add(ms, eps)
    r    = rsqrt(e)
    y    = mul(xf, r)
    z    = mul(gamma, view(y))
    out  = _to_copy(z, fp16)

eight dispatches over ~20 KB, twice per layer. On K2 Horizon 32B that is 129
norms and roughly 10 ms of a 182 ms token — not arithmetic, just dispatch
latency, which on this device is ~13 us no matter how little the kernel does.

Runs BEFORE `fuseops`, which would otherwise collapse `add`+`rsqrt` into a
`FusedOp` whose epsilon this pass would then have to parse back out of an
expression tree.
"""

"""
    fusegroupedrms(graphs) -> (graphs, n)

Rewrite each decomposed grouped RMS norm into one `fused.groupedrms`.
"""
function fusegroupedrms(graphs::AbstractDict)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fusegroupedrms(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

# The single op reading `id` (through views), or `nothing` if there is not
# exactly one. "Exactly one" is what makes dropping the intermediate safe.
function solereader(g::Graph, byin::AbstractDict, id::AbstractString)
    rs = get(byin, id, nothing)
    rs === nothing && return nothing
    length(rs) == 1 ? rs[1] : nothing
end

function fusegroupedrms(g::Graph)
    producer = Dict{String,Op}()
    byin = Dict{String,Vector{Op}}()
    for op in g.ops
        isempty(string(op.out)) || (producer[op.out] = op)
        for id in op.ins
            push!(get!(byin, viewroot(g, id), Op[]), op)
            id == viewroot(g, id) || push!(get!(byin, id, Op[]), op)
        end
    end

    ops = copy(g.ops)
    drop = Set{String}()
    n = 0
    for p in g.ops
        p.aten == "pow.Tensor_Scalar" || continue
        Int(get(p.attrs, "arg1", 0)) == 2 || continue
        p.id in drop && continue

        m = solereader(g, byin, p.out)
        (m === nothing || m.aten != "mean.dim") && continue
        ints(get(m.attrs, "arg1", Int[])) == [-1] || continue
        Bool(something(get(m.attrs, "arg2", nothing), false)) || continue

        a = solereader(g, byin, m.out)
        (a === nothing || a.aten != "add.Tensor") && continue
        epsv = get(a.attrs, "arg1", nothing)
        epsv isa Real || continue

        r = solereader(g, byin, a.out)
        (r === nothing || r.aten != "rsqrt.default") && continue

        u = solereader(g, byin, r.out)
        (u === nothing || u.aten != "mul.Tensor") && continue
        # The other operand of the scale must be the very tensor that was squared.
        xid = p.ins[1]
        viewroot(g, u.ins[1]) == viewroot(g, xid) ||
            viewroot(g, u.ins[2]) == viewroot(g, xid) || continue

        v = solereader(g, byin, u.out)
        (v === nothing || v.aten != "mul.Tensor") && continue
        gid = viewroot(g, v.ins[1]) == viewroot(g, u.out) ? v.ins[2] : v.ins[1]
        gb = get(g.buffers, gid, nothing)
        (gb === nothing || gb.kind !== :weight) && continue

        # An optional narrowing cast on the way out, which the kernel does in its
        # store anyway.
        last = v
        w = solereader(g, byin, v.out)
        if w !== nothing && w.aten == "_to_copy.default"
            last = w
        end

        # The input: skip a widening cast in front of the square, since the
        # kernel reads whatever it is given and accumulates in fp32.
        src = xid
        pc = get(producer, viewroot(g, xid), nothing)
        if pc !== nothing && pc.aten == "_to_copy.default"
            # Every reader of the widened value must be inside this group — and
            # there are always TWO of them, the square and the scale, so asking
            # for a sole reader here rejects every norm there is.
            rs = get(byin, pc.out, Op[])
            !isempty(rs) && all(o -> o.id in (p.id, u.id), rs) && (src = pc.ins[1])
        end

        # From the VIEW that was squared, not from its root. The root is the flat
        # `(1, 1, 5120)` hidden state; the view is the `(1, 1, 4, 1280)` reshape
        # that makes the grouping, and the grouping is the whole point.
        sh = g.buffers[xid].shape
        length(sh) >= 2 || continue
        C = sh[end]; NG = sh[end-1]
        (C isa Integer && NG isa Integer) || continue
        # The gain spans the row, so it must cover every group of it.
        length(g.buffers[gid].shape) >= 1 &&
            Int(g.buffers[gid].shape[end]) == Int(C) * Int(NG) || continue

        i = findfirst(o -> o.id == last.id, ops)
        ops[i] = Op(last.id, "fused.groupedrms", [src, gid], last.out,
                    Dict{String,Any}("C" => Int(C), "ng" => Int(NG),
                                     "eps" => Float64(epsv)))
        for o in (p, m, a, r, u, v)
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
