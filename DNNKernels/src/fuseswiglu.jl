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

"""
    FUSESWIGLUMM[] = true

Turn `fuseswiglumm` off, to A/B it against the SwiGLU and its product apart.
"""
const FUSESWIGLUMM = Ref(true)

"""
    fuseswiglumm(graphs) -> (graphs, n)

Fold each `fused.swiglu` whose only reader is `mm(swiglu, w)`, with `w` a
weight, into that product: `fused.swiglumm(gate, up, w)`.

What it buys is in the emit. A ConvRot weight rotates the product's input before
the GEMM, and the rotation's first pass can compute the SwiGLU as it reads, so
the SwiGLU's result (101 MB a layer at Qwen-Image 2.1's `12288 x 4118`) is never
written or read back. Any other weight gets the same two ops as before.

The product may see the SwiGLU through reshapes and nothing else, since the fused
op stands for the SwiGLU's own dense layout.
"""
function fuseswiglumm(graphs::AbstractDict)
    FUSESWIGLUMM[] || return (Dict{String,Any}(graphs), 0)
    out = Dict{String,Any}()
    total = 0
    for (k, g) in graphs
        if g isa Graph
            g2, n = fuseswiglumm(g)
            out[k] = g2; total += n
        else
            out[k] = g
        end
    end
    (out, total)
end

function fuseswiglumm(g::Graph)
    readers = Dict{String,Vector{Op}}()
    for op in g.ops, id in op.ins
        push!(get!(readers, viewroot(g, id), Op[]), op)
    end
    escaping = Set(viewroot(g, id) for id in g.outputs)
    ops = copy(g.ops)
    drop = Set{String}()
    for sw in g.ops
        sw.aten == "fused.swiglu" || continue
        sw.out in escaping && continue
        rs = get(readers, sw.out, Op[])
        length(rs) == 1 || continue
        mm = only(rs)
        mm.aten == "mm.default" && reshapeof(g, mm.ins[1], sw.out) || continue
        wb = get(g.buffers, mm.ins[2], nothing)
        (wb === nothing || wb.kind !== :weight) && continue
        ops[findfirst(==(mm), ops)] =
            Op(mm.id, "fused.swiglumm", [sw.ins[1], sw.ins[2], mm.ins[2]], mm.out,
               Dict{String,Any}("dtype" => g.buffers[sw.out].dtype))
        push!(drop, sw.id)
    end
    isempty(drop) && return (g, 0)
    dropped = Set(o.out for o in g.ops if o.id in drop)
    (Graph(g.name, g.symbols, g.inputs, g.outputs, Dict{String,Buffer}(g.buffers),
           [id for id in g.order if !(id in dropped)],
           [o for o in ops if !(o.id in drop)], g.fusion), length(drop))
end

"""Whether `id` is `root` or a chain of reshapes of it."""
function reshapeof(g::Graph, id::AbstractString, root::AbstractString)
    cur = id
    while cur != root
        b = g.buffers[cur]
        (b.kind === :view && b.viewop in SHAPEONLY_VIEWS) || return false
        cur = b.of
    end
    true
end
