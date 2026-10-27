module BoundTupleTests
using Test, DNNKernels
import Mantle
import KernelInterface as KI

function pair_kernel!(y,z,x,n)
    i=KI.get_global_id().x
    if i<=n
        v=@inbounds x[i]
        @inbounds y[i]=v+1f0
        @inbounds z[i]=2f0*v
    end
    nothing
end
function DNNKernels.emitop!(ctx::DNNKernels.EmitCtx,op::DNNKernels.Op,
                           ::Val{Symbol("test.bound_pair")})
    y,z=DNNKernels.dest(ctx,0),DNNKernels.dest(ctx,1)
    x=DNNKernels.operand(ctx,op.ins[1])
    Mantle.dispatch!(ctx.g,pair_kernel!,(y,z,x,length(x)),length(x);name=op.id)
    nothing
end

function graph(;meta=16)
    DK=DNNKernels;buffers=Dict{String,DK.Buffer}()
    buffers["x"]=DK.Buffer("x",:external,Any[16],Float32,"",(0,0),"","",Dict{String,Any}())
    buffers["pair"]=DK.Buffer("pair",:transient,Any[],Float32,"",(0,0),"","",
        Dict{String,Any}("shapes"=>[Any[meta],Any[meta],nothing],"dtypes"=>[Float32,Float32,nothing]))
    for (i,id) in enumerate(("y","z"))
        buffers[id]=DK.Buffer(id,:view,Any[16],Float32,"",(0,0),"pair","getitem",
            Dict{String,Any}("arg1"=>i-1))
    end
    DK.Graph("bound_tuple",String[],["x"],["y","z"],buffers,["x","pair","y","z"],
        [DK.Op("pair","test.bound_pair",["x"],"pair",Dict{String,Any}())])
end

function test_bound_tuple(device)
    M=Mantle;DK=DNNKernels
    x=M.Buffer(device,ones(Float32,16));y=M.Buffer(device,Float32,16);z=M.Buffer(device,Float32,16)
    @testset "Caller-bound multi-output graph" begin
        try
            for alias in (false,true)
                output=alias ? x : y
                g=M.Graph(device)
                # getitem shapes are authoritative, just as for unbound tuple
                # results (traced metadata can contain a less general extent).
                ir=graph(;meta=alias ? 8 : 16)
                result=DK.emitgraph!(g,first(DK.genericfusion(ir)),Dict(),(;);
                    inputs=Dict("x"=>x,"pair"=>(output,z,nothing)))
                @test result["y"]===output
                @test result["z"]===z
                plan=M.record!(M.Plan(g))
                try
                    for value in (1f0,3f0)
                        fill!(M.storage(x),value)
                        M.run!(plan);M.waitfor!(plan)
                        @test all(==(value+1f0),Array(M.storage(output)))
                        @test all(==(2f0*value),Array(M.storage(z)))
                    end
                finally
                    M.free!(plan)
                end
            end
            ir=graph()
            ctx=DK.emitinto!(M.Graph(device),ir,Dict(),(;),DK.consumedids(ir),Set{String}(),
                Dict("x"=>x,"pair"=>(y,z,nothing)))
            # Bound results keep sharing on; a write to bound storage drops the
            # representations prepared from it (test_semanticstandard.jl).
            @test ctx.activations.enabled
            for bad in ((y,z),(y,z,y),(y,nothing,nothing),[y,z,nothing])
                @test_throws ArgumentError DK.emitgraph!(M.Graph(device),graph(),Dict(),(;);
                    inputs=Dict("x"=>x,"pair"=>bad))
            end
        finally
            M.waitidle(device);M.free!(x);M.free!(y);M.free!(z)
        end
    end
end
end
