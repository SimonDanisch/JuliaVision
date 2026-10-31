module SemanticGraphTests
using Test,DNNKernels
import Mantle as M
import KernelInterface as KI

function state_pair!(y,state,x,w,r,n)
    i=KI.get_global_id().x
    if i<=n
        @inbounds y[i]=x[i]+state[i]+w[i]*Float32(r[1])
        @inbounds state[i]+=1f0
    end
    nothing
end
function copy_part!(out,x,n)
    i=KI.get_global_id().x
    i<=n && (@inbounds out[i]=x[i])
    nothing
end
for (tag,kernel) in ((Symbol("test.semantic_state"),state_pair!),
                     (Symbol("test.semantic_copy"),copy_part!))
    @eval function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,::Val{$(QuoteNode(tag))})
        args=DNNKernels.semanticargument(ctx,op,op.attrs["arguments"])
        M.dispatch!(ctx.g,$kernel,args,16;name=op.id)
        nothing
    end
end

function run(dev)
    x=M.Buffer(dev,ones(Float32,16));w=M.Buffer(dev,ones(Float32,16))
    state=M.Buffer(dev,zeros(Float32,16));y=M.Buffer(dev,Float32,16);out=M.Buffer(dev,Float32,16)
    r=M.GPURef(dev,Int32(2));plan=nothing
    @testset "Shared semantic graph frontend" begin
        try
            g=SemanticGraph(dev;name="state_and_views")
            semanticweights!(g,Dict("scale"=>w))
            semanticop!(g,"test.semantic_state",(y,state,x,w,r,Int32(16));writes=(1,2))
            for offset in (0,8)
                semanticop!(g,"test.semantic_copy",(M.viewof(out,(8,);offset),
                    M.viewof(y,(8,);offset),Int32(8)))
            end
            ir=semanticgraph(g)
            @test length(ir.ops)==3
            @test haskey(ir.buffers[first(ir.ops).out].attrs,"shapes")
            @test count(b->b.kind===:weight,values(ir.buffers))==1
            # Second half consumes the allocation version produced by first half.
            @test ir.ops[2].out in ir.ops[3].ins
            emitted=emitsemantic!(g)
            @test emitted.graph isa DNNKernels.Graph
            @test emitted.mantle===g.mantle
            plan=M.record!(M.Plan(emitted.mantle))
            for (value,ref,expected,iteration) in ((1f0,2,3f0,1),(3f0,4,8f0,2))
                fill!(M.storage(x),value);r[]=Int32(ref)
                M.run!(plan);M.waitfor!(plan)
                @test all(==(expected),Array(M.storage(y)))
                @test all(==(expected),Array(M.storage(out)))
                @test all(==(Float32(iteration)),Array(M.storage(state)))
            end
            @test_throws ErrorException emitsemantic!(g)
            @test_throws ErrorException semanticop!(g,"test.semantic_copy",(out,x,Int32(16)))
            fresh=SemanticGraph(dev)
            semanticweights!(fresh,Dict("scale"=>w))
            @test_throws ArgumentError semanticop!(fresh,"test.semantic_copy",(w,x,Int32(16)))
            @test_throws ArgumentError semanticop!(fresh,"test.semantic_copy",(out,x,Int32(16));writes=())
            @test_throws ArgumentError semanticop!(fresh,"test.semantic_copy",(out,ones(Float32,16),Int32(16)))
            @test_throws ArgumentError semanticop!(fresh,"test.semantic_copy",(out,x,Int32(16));attrs=Dict("hidden"=>x))
            alias=SemanticGraph(dev)
            id=semanticop!(alias,"test.semantic_copy",(out,x,Int32(16));writes=(1,1))
            @test !haskey(alias.buffers[id].attrs,"shapes")
        finally
            M.waitidle(dev)
            plan===nothing || M.free!(plan)
            foreach(M.free!,(x,w,state,y,out,r))
        end
    end
end

function activation(dev)
    DK=DNNKernels
    gate=M.Buffer(dev,Float32,(256,1));up=M.Buffer(dev,Float32,(256,1))
    owned=Any[gate,up];plans=Any[];records=Any[]
    allocate(T,dims...)=begin b=M.Buffer(dev,T,dims);push!(owned,b);b;end
    @testset "Semantic activation preparation" begin
        try
            for mapped in (false,true)
                empty!(records)
                for semantic in (false,true)
                    g=semantic ? SemanticGraph(dev) : M.Graph(dev)
                    out=allocate(Float32,256,1)
                    cache=ActivationPrep(g)
                    p=if mapped
                        declare_mapped_q8!(cache,:value,out,DK.SwiGLUMap{Float32}(),(gate,up))
                    else
                        declare_q8!(cache,:value,gate)
                    end
                    again=declare_q8!(cache,:value,mapped ? out : gate)
                    @test again===p
                    @test activation_report(cache).prepared==1
                    @test activation_report(cache).reused==1
                    q=allocate(DK.UVec4,16,1);ds=allocate(Float32,16,1)
                    for (dst,src) in ((q,p.q),(ds,p.scales))
                        if semantic
                            semanticop!(g,"test.semantic_copy",(dst,src,Int32(16)))
                        else
                            M.dispatch!(g,copy_part!,(dst,src,Int32(16)),16;name="quant.copy")
                        end
                    end
                    if semantic
                        @test startswith(first(g.ops).aten,"dnn.")
                        @test length(g.ops)==3
                    end
                    mg=semantic ? emitsemantic!(g).mantle : g
                    plan=M.record!(M.Plan(mg));push!(plans,plan)
                    push!(records,(;plan,q,ds,out))
                end
                for input in (0.2f0,-0.4f0)
                    fill!(M.storage(gate),input);fill!(M.storage(up),1.3f0)
                    for r in records
                        M.run!(r.plan);M.waitfor!(r.plan)
                    end
                    a,b=records
                    @test Array(M.storage(a.q))==Array(M.storage(b.q))
                    @test reinterpret(UInt32,vec(Array(M.storage(a.ds))))==reinterpret(UInt32,vec(Array(M.storage(b.ds))))
                    if mapped
                        @test reinterpret(UInt32,vec(Array(M.storage(a.out))))==reinterpret(UInt32,vec(Array(M.storage(b.out))))
                    end
                end
            end
        finally
            M.waitidle(dev);foreach(M.free!,plans);foreach(M.free!,owned)
        end
    end
end
end
