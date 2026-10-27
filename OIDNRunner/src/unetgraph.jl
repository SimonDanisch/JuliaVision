# OIDN's packed HWC convolution is a semantic graph operation, not an opaque
# whole-network callback. The graph names every activation, weight and bias;
# generic fusion can inspect it before its specialized layout is lowered.
struct UNetGraphPlan
    graph::DNNKernels.Graph
    fusion::NamedTuple
    plan::Mantle.Plan
end
Mantle.free!(p::UNetGraphPlan)=Mantle.free!(p.plan)

function unetgraph(net::UNet,t::TileBuffers,backend)
    dk=DNNKernels
    buffers=Dict{String,dk.Buffer}();order=String[];ops=dk.Op[]
    bindings=Dict{String,Any}();weights=Dict{String,Any}()
    current=IdDict{Any,String}()
    function tensor(x,kind;key="")
        id="v$(length(order)+1)"
        buffers[id]=dk.Buffer(id,kind,Any[length(x)],eltype(x),key,(0,0),"","",Dict{String,Any}())
        push!(order,id);bindings[id]=x;id
    end
    input=tensor(t.input,:external);current[t.input]=input
    function weight(x)
        id=get(current,x,nothing)
        id===nothing || return id
        key="w$(length(weights)+1)";weights[key]=x
        id=tensor(x,:weight;key);current[x]=id;id
    end
    function convop(be,t,L,Y,X0,X1,level)
        ins=String[current[X0]]
        X1===nothing || push!(ins,current[X1])
        append!(ins,(weight(L.Wt),weight(L.bias)))
        id=tensor(Y,:transient)
        attrs=Dict{String,Any}("width"=>t.tw>>level,"height"=>t.th>>level,
            "capacity"=>capacity(t.tw,t.th,level),"c0"=>L.C0,"c1"=>L.C1,
            "up"=>L.up,"cout"=>L.CO,"k"=>L.K)
        push!(ops,dk.Op(id,"oidn.hwc_convolution",ins,id,attrs))
        current[Y]=id;Y
    end
    function poolop(be,t,Y,X,C,level)
        id=tensor(Y,:transient)
        push!(ops,dk.Op(id,"oidn.hwc_maxpool",[current[X]],id,
            Dict{String,Any}("width"=>t.tw>>level,"height"=>t.th>>level,"channels"=>C)))
        current[Y]=id;Y
    end
    unet_stages!(convop,poolop,backend,net,t)
    output=current[t.output]
    graph=dk.Graph("oidn_hwc_unet",String[],[input],[output],buffers,order,ops)
    optimized,report=dk.genericfusion(graph)
    g=Mantle.Graph(Mantle.todevice(backend))
    dk.emitgraph!(g,optimized,weights,(;);inputs=bindings,outputs=Dict(output=>t.output))
    UNetGraphPlan(optimized,report,Mantle.record!(Mantle.Plan(g)))
end

function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{Symbol("oidn.hwc_convolution")})
    dk=DNNKernels;a=op.attrs;be=Mantle.backend(ctx.dev)
    Y=dk.dest(ctx);X0=dk.operand(ctx,op.ins[1])
    X1=a["c1"]==0 ? X0 : dk.operand(ctx,op.ins[2])
    Wt=dk.operand(ctx,op.ins[end-1]);bias=dk.operand(ctx,op.ins[end])
    S=stagevec(be);tiling=convtiling(a["cout"]);BO,BP=tiling;nt=convthreads(tiling)
    kernels=convmatrixtile(be)==8 ? CONV_KERNELS_8 : CONV_KERNELS
    nblk=(a["cout"]÷BO)*(a["capacity"]÷BP)
    Mantle.dispatch!(ctx.g,kernels[tiling],
        (Y,reinterpret(S,X0),reinterpret(S,X1),reinterpret(S,Wt),bias,
         Val(halves(S)),Val(a["width"]),Val(a["height"]),Val(a["c0"]),Val(a["c1"]),
         Val(a["up"]),Val(a["cout"]),Val(a["k"]),Val(true),prefetch(be)),nblk*nt;
         group=nt,name=op.id)
    Y
end
function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{Symbol("oidn.hwc_maxpool")})
    dk=DNNKernels;a=op.attrs;V=halves(poolvec(Mantle.backend(ctx.dev)))
    Y=dk.dest(ctx);X=dk.operand(ctx,op.ins[1]);P=poolvec(Mantle.backend(ctx.dev))
    n=(a["channels"]÷V)*(a["width"]÷2)*(a["height"]÷2)
    Mantle.dispatch!(ctx.g,maxpool_kernel!,
        (reinterpret(P,Y),reinterpret(P,X),Val(V),Val(a["width"]),Val(a["height"]),Val(a["channels"])),
        cld(n,256)*256;group=256,name=op.id)
    Y
end

"""Run the optimized semantic DNN graph; input/output stay in the caller's buffers."""
function forward!(backend,net::UNet,t::TileBuffers)
    p=get!(t.plans,net) do
        unetgraph(net,t,backend)
    end
    Mantle.run!(p.plan)
    t.output
end
