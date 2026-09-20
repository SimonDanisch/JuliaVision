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
        # The other operand of the scale must be the very tensor that was
        # squared -- or the value that tensor is a widening cast OF. Both spell
        # the same multiply: `pow` reads `fp32(x)` while the scale reads `x` and
        # widens in the multiply, which is what Qwen-Image 2.1's q/k norm
        # exports as, and squaring an fp16 in fp32 is exact either way.
        xid = p.ins[1]
        pcast = get(producer, viewroot(g, xid), nothing)
        precast = (pcast !== nothing && pcast.aten == "_to_copy.default") ?
                  pcast.ins[1] : ""
        samex(id) = viewroot(g, id) == viewroot(g, xid) ||
                    (!isempty(precast) && viewroot(g, id) == viewroot(g, precast))
        samex(u.ins[1]) || samex(u.ins[2]) || continue

        # An optional narrowing cast between the normalisation and the gain.
        # See `MIDROUND` in `groupedrms_kernel!`: it is a rounding the fused
        # kernel has to perform, not one it may skip.
        mid = solereader(g, byin, u.out)
        (mid === nothing) && continue
        midround = mid.aten == "_to_copy.default"
        v = midround ? solereader(g, byin, mid.out) : mid
        (v === nothing || v.aten != "mul.Tensor") && continue
        # Against what the gain multiply actually READS, which is the cast's
        # result where there is one and the scale's where there is not.
        normed = midround ? mid.out : u.out
        gid = viewroot(g, v.ins[1]) == viewroot(g, normed) ? v.ins[2] : v.ins[1]
        gb = get(g.buffers, gid, nothing)
        (gb === nothing || gb.kind !== :weight) && continue

        # An optional NARROWING cast on the way out, which the kernel does in its
        # store anyway. A widening one is not the same op: the chain rounds to
        # the gain multiply's own type first, and absorbing it would drop that
        # rounding and write a value the graph never had.
        last = v
        w = solereader(g, byin, v.out)
        if w !== nothing && w.aten == "_to_copy.default" &&
           haskey(g.buffers, w.out) && haskey(g.buffers, v.out) &&
           sizeof(g.buffers[w.out].dtype) <= sizeof(g.buffers[v.out].dtype)
            last = w
        end

        # The input: skip a widening cast in front of the square, since the
        # kernel reads whatever it is given and accumulates in fp32.
        src = xid
        if pcast !== nothing && pcast.aten == "_to_copy.default"
            # Every reader of the widened value must be inside this group — and
            # there are always TWO of them, the square and the scale, so asking
            # for a sole reader here rejects every norm there is. Where the scale
            # reads the pre-cast value instead, the square is the only reader.
            rs = get(byin, pcast.out, Op[])
            !isempty(rs) && all(o -> o.id in (p.id, u.id), rs) && (src = pcast.ins[1])
        end

        # From the VIEW that was squared, not from its root. The root is the flat
        # `(1, 1, 5120)` hidden state; the view is the `(1, 1, 4, 1280)` reshape
        # that makes the grouping, and the grouping is the whole point.
        sh = g.buffers[xid].shape
        length(sh) >= 1 || continue
        C = sh[end]
        C isa Integer || continue
        # How far the gain reaches decides `ng`, and the two cases the kernel
        # already covers are the two that occur:
        #
        #   * it spans EVERY group of the row (`C * ng` values, `ng = sh[end-1]`),
        #     which is the grouped norm this pass was written for;
        #   * it spans ONE group (`C` values) and repeats, which is what a
        #     per-head q/k norm is -- Qwen-Image 2.1 normalises each 128-wide
        #     head of a `(1, L, 32, 128)` state under a single 128-long gain.
        #     That is `ng = 1`, and `gof = (g % 1) * C` is then zero for every
        #     group, so the same kernel runs it.
        #
        # Without the second case those two norms stayed eight dispatches each,
        # and the `mean.dim` alone was 5.35 ms of a 223 ms layer: one thread per
        # row reading 128 contiguous floats, which is a cache line per lane.
        length(g.buffers[gid].shape) >= 1 || continue
        gl = Int(g.buffers[gid].shape[end])
        NG = if gl == Int(C)
            1
        elseif length(sh) >= 2 && sh[end-1] isa Integer && gl == Int(C) * Int(sh[end-1])
            Int(sh[end-1])
        else
            0
        end
        NG == 0 && continue

        i = findfirst(o -> o.id == last.id, ops)
        ops[i] = Op(last.id, "fused.groupedrms", [src, gid], last.out,
                    Dict{String,Any}("C" => Int(C), "ng" => Int(NG),
                                     "eps" => Float64(epsv),
                                     "midround" => midround))
        for o in (p, m, a, r, u, mid, v)
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
