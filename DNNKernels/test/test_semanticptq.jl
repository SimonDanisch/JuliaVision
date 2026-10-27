module SemanticPTQTests
using Test,Random,DNNKernels
import Mantle as M

function run(dev)
    DK=DNNKernels;rng=MersenneTwister(411)
    rows=64;k=128
    bytes=rand(rng,UInt8(0):UInt8(242),rows*28)
    scale=collect(reinterpret(UInt8,[Float16(0.1)]))
    for row in 0:rows-1
        bytes[28row+27:28row+28]=scale
    end
    A=DK.ptq1matrix(M.backend(dev),bytes,rows,k)
    owned=Any[A.data];plans=Any[]
    alloc(T,dims)=begin b=M.Buffer(dev,T,dims);push!(owned,b);b;end
    @testset "Runner-independent semantic PTQ1" begin
        try
            @test which(DK.ptq1_rows!,Tuple{DK.SemanticGraph,M.Buffer,DK.PTQ1Matrix,M.Buffer}).module===DK
            for (columns,T) in ((1,Float32),(1,Float16),(3,Float16),(64,Float16))
                x=alloc(T,(k,columns));outputs=Any[]
                for semantic in (false,true)
                    g=semantic ? DK.SemanticGraph(dev) : M.Graph(dev)
                    semantic && DK.semanticweights!(g,Dict("weight"=>A.data))
                    y=alloc(Float32,(rows,columns));z=alloc(Float32,(rows,columns))
                    if columns==1 && T===Float32
                        DK.ptq1_rows!(g,y,A,x;name="rows")
                        DK.ptq1_rows2!(g,y,A,z,A,x;name="pair")
                    else
                        cover=DK.ptq1_native_gemm!(g,y,A,x,columns;name="native")
                        cover<columns && DK.ptq1_cols!(g,y,A,x,cover,columns;name="columns")
                    end
                    if semantic
                        @test all(startswith(op.aten,"dnn.ptq1_") for op in g.ops)
                    end
                    mg=semantic ? DK.emitsemantic!(g).mantle : g
                    p=M.record!(M.Plan(mg));push!(plans,p);push!(outputs,(;y,z,p))
                end
                for input in (Float16(0.125),Float16(-0.25),Float16(0))
                    fill!(M.storage(x),input)
                    for o in outputs
                        M.run!(o.p);M.waitfor!(o.p)
                    end
                    a,b=outputs
                    @test reinterpret(UInt32,vec(Array(M.storage(a.y))))==reinterpret(UInt32,vec(Array(M.storage(b.y))))
                    @test all(isfinite,Array(M.storage(b.y)))
                    if columns==1 && T===Float32
                        @test reinterpret(UInt32,vec(Array(M.storage(a.z))))==reinterpret(UInt32,vec(Array(M.storage(b.z))))
                    end
                    if iszero(input)
                        # Known answer, independent of either implementation:
                        # a bias-free product of finite weights and zero is zero.
                        @test all(iszero,Array(M.storage(b.y)))
                        if columns==1 && T===Float32
                            @test all(iszero,Array(M.storage(b.z)))
                        end
                    end
                end
            end
            bad=alloc(Float16,(k,1));out=alloc(Float32,(rows,1))
            for g in (M.Graph(dev),DK.SemanticGraph(dev))
                @test_throws ArgumentError DK.ptq1_rows!(g,out,A,bad;name="invalid")
                @test_throws ArgumentError DK.ptq1_rows2!(g,out,A,out,A,bad;name="invalid_pair")
            end
        finally
            M.waitidle(dev);foreach(M.free!,plans);foreach(M.free!,owned)
        end
    end
end
end
