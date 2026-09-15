# The exported masked attention has a rounded scale/add between QK and softmax.
function fusemaskedattention(g::Graph)
    producer = Dict(o.out=>o for o in g.ops)
    drops = Set{String}()
    ops = copy(g.ops)
    count = 0
    for (i,b2) in enumerate(ops)
        count >= FUSEATTENTIONLIMIT[] && break
        b2.aten == "bmm.default" || continue
        sc = get(producer,viewroot(g,b2.ins[1]),nothing)
        sc === nothing && continue
        sc.aten == "_softmax.default" || continue
        get(sc.attrs,"arg1",0) == -1 || continue
        scores = get(producer,viewroot(g,sc.ins[1]),nothing)
        scores === nothing && continue
        scores.aten == "fused.elementwise" && length(scores.ins)==2 || continue
        f = get(scores.attrs,"fused",nothing)
        f isa FusedOp || continue
        f.funcs isa Tuple{Rounded{Float16,typeof(*)},Rounded{Float16,typeof(+)}} || continue
        f.args isa Tuple{Tuple{In{1},Konst{Float64}},Tuple{Tmp{1},In{2}}} || continue
        scale = f.args[1][2].v
        isfinite(scale) && scale > 0 || continue
        b1 = get(producer,viewroot(g,scores.ins[1]),nothing)
        b1 === nothing && continue
        b1.aten == "bmm.default" || continue
        all(id -> g.buffers[id].dtype === Float16,
            (b1.ins..., b1.out, sc.out, b2.ins[2], b2.out)) || continue
        qs = g.buffers[b1.ins[1]].shape
        ks = g.buffers[b1.ins[2]].shape
        length(qs)==3 && length(ks)==3 || continue
        h,lq,d = qs; hk,dk,lk = ks
        h==hk && d==dk || continue
        g.buffers[b1.out].shape == Any[h,lq,lk] || continue
        g.buffers[b2.ins[2]].shape == Any[h,lk,d] || continue
        g.buffers[b2.out].shape == Any[h,lq,d] || continue
        mask = scores.ins[2]
        g.buffers[mask].shape == Any[1,1,lq,lk] || continue
        q = lastmatching(g,b1.ins[1],Any[1,h,lq,d])
        k = lastmatching(g,b1.ins[2],Any[1,h,lk,d])
        v = lastmatching(g,b2.ins[2],Any[1,h,lk,d])
        any(isnothing,(q,k,v)) && continue
        group = Set((b1.id,scores.id,sc.id,b2.id))
        gone = Set((b1.out,scores.out,sc.out))
        any(id -> viewroot(g,id) in gone,g.outputs) && continue
        any(o -> !(o.id in group) && any(x->viewroot(g,x) in gone,o.ins),g.ops) && continue
        ops[i] = Op(b2.id,"fused.maskedattention",[q,k,v,mask],b2.out,Dict{String,Any}("scale"=>scale))
        union!(drops,(b1.id,scores.id,sc.id)); count += 1
    end
    count==0 && return g,0
    keep = [o for o in ops if !(o.id in drops)]
    gone = Set(o.out for o in g.ops if o.id in drops)
    Graph(g.name,g.symbols,g.inputs,g.outputs,copy(g.buffers),
          [id for id in g.order if !(id in gone)],keep,g.fusion),count
end

function fusemaskedattention(graphs::AbstractDict)
    FUSEATTENTION[] || return (graphs, 0)
    out = copy(graphs); n=0
    for (name,g) in graphs
        out[name],count = fusemaskedattention(g); n+=count
    end
    out,n
end
