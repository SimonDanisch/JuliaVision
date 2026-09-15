"""
Collapse an exported SwiGLU into one op.

`silu(gate) * up` comes out of `run_decompositions()` under autocast as

    gf  = _to_copy(gate, fp32)
    sg  = sigmoid(gf)
    sl  = mul(gf, sg)
    sh  = _to_copy(sl, fp16)
    out = mul(sh, up)

five dispatches over a 26624-wide tensor per layer, two of which exist only to
move it between fp16 and fp32. The kernel evaluates the sigmoid in fp32 and
rounds to fp16 at the same point the graph does, so this changes how many
dispatches run and not what they compute.

Runs before `fuseops`, which collapses the chain into two `FusedOp`s with the
narrowing cast stranded between them.
"""

"""
    FUSESWIGLU[] = true

Turn the rewrite off, to A/B it against the unfused graph.
"""
const FUSESWIGLU = Ref(true)

"""
    fuseswiglu(graphs) -> (graphs, n)

Rewrite each `silu(gate) * up` into one `fused.swiglu`.
"""
function fuseswiglu(graphs::AbstractDict)
    FUSESWIGLU[] || return (Dict{String,Any}(graphs), 0)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fuseswiglu(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

function fuseswiglu(g::Graph)
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
    for sg in g.ops
        sg.aten == "sigmoid.default" || continue
        gf = sg.ins[1]
        # x * sigmoid(x), in either operand order.
        sl = sole(sg.out)
        (sl === nothing || sl.aten != "mul.Tensor") && continue
        (viewroot(g, sl.ins[1]) == viewroot(g, gf) ||
         viewroot(g, sl.ins[2]) == viewroot(g, gf)) || continue

        nxt = sole(sl.out)
        nxt === nothing && continue
        cast = nothing
        if nxt.aten == "_to_copy.default"
            cast = nxt
            nxt = sole(cast.out)
            nxt === nothing && continue
        end
        nxt.aten == "mul.Tensor" || continue
        silu = cast === nothing ? sl.out : cast.out
        upid = viewroot(g, nxt.ins[1]) == viewroot(g, silu) ? nxt.ins[2] : nxt.ins[1]
        viewroot(g, upid) == viewroot(g, silu) && continue

        # The gate before it was widened, so the op reads what the projection
        # wrote rather than a second copy of it.
        src = gf
        pc = get(producer, viewroot(g, gf), nothing)
        if pc !== nothing && pc.aten == "_to_copy.default"
            rs = get(byin, pc.out, Op[])
            !isempty(rs) && all(o -> o.id in (sg.id, sl.id), rs) && (src = pc.ins[1])
        end

        ops[findfirst(==(nxt), ops)] =
            Op(nxt.id, "fused.swiglu", [src, upid], nxt.out, Dict{String,Any}())
        for o in (sg, sl); push!(drop, o.id); end
        cast === nothing || push!(drop, cast.id)
        n += 1
    end
    n == 0 && return (g, 0)
    keep = [o for o in ops if !(o.id in drop)]
    dropped = Set(o.out for o in g.ops if o.id in drop)
    order = [id for id in g.order if !(id in dropped)]
    (Graph(g.name, g.symbols, g.inputs, g.outputs, Dict{String,Buffer}(g.buffers),
           order, keep, g.fusion), n)
end
