using Test, Random

# GPU-to-GPU qualification: identical initialized weights and topology, existing
# direct Mantle declarations versus the semantic DNNKernels route. No CPU neural
# implementation or external checkpoint is needed for this integration fixture.
function test_sparse_decoder_graph(TR,device)
    M=TR.Mantle;DK=TR.DNNKernels
    rng=MersenneTwister(751)
    weights=Dict{String,M.Buffer}()
    function weight(name,dims...;value=nothing)
        a=value===nothing ? 0.02f0 .* randn(rng,Float32,dims...) : fill(Float32(value),dims...)
        weights[name]=M.Buffer(device,Float16.(a))
    end
    C=32
    weight("from_latent.weight",C,32);weight("from_latent.bias",C)
    p="blocks.0.0"
    weight("$p.conv.weight",C,27C);weight("$p.conv.bias",C)
    weight("$p.norm.weight",C;value=1);weight("$p.norm.bias",C;value=0)
    weight("$p.mlp.0.weight",4C,C);weight("$p.mlp.0.bias",4C)
    weight("$p.mlp.2.weight",C,4C);weight("$p.mlp.2.bias",C)
    p="blocks.0.1"
    weight("$p.to_subdiv.weight",16,C;value=0)
    weight("$p.to_subdiv.bias",16;value=1) # all eight children, deterministically
    weight("$p.norm1.weight",C;value=1);weight("$p.norm1.bias",C;value=0)
    weight("$p.conv1.weight",8C,27C);weight("$p.conv1.bias",8C)
    weight("$p.conv2.weight",C,27C);weight("$p.conv2.bias",C)
    weight("output_layer.weight",7,C);weight("output_layer.bias",7)
    dec=TR.SparseDecoder(device,weights,[C,C],[1,0],7,true)
    coords=Int32[1 2 4; 1 2 3; 1 1 2]
    slat=randn(rng,Float32,32,3)
    declarations=Any[]
    references=Any[]
    reference_factory(dev)=begin
        g=M.Graph(dev);push!(references,g);g
    end
    factory(dev)=begin
        g=TR.SparseDecoderGraph(dev);push!(declarations,g);g
    end
    records=Any[]
    @testset "TRELLIS sparse semantic decoder" begin
        try
            # Two independent changed inputs catch frozen/captured activations.
            for repeat in 1:2
                input=slat .* Float32(repeat)
                old=TR.decodelevels(dec,input,coords,2;res=8,graph_factory=reference_factory)
                new=TR.decodelevels(dec,input,coords,2;res=8,graph_factory=factory)
                @test old[2]==new[2]
                @test map(l->l.coords,old[1])==map(l->l.coords,new[1])
                @test all(isfinite,new[3])
                @test reinterpret(UInt32,vec(old[3]))==reinterpret(UInt32,vec(new[3]))
            end
            guide=TR.DecodeResult(zeros(Float32,7,24),
                [TR.Level(coords,8),TR.children(TR.Level(coords,8),trues(8,3))[1]],
                [trues(8,3)])
            guided=TR.SparseDecoder(device,weights,[C,C],[1,0],7,false)
            old=TR.decodelevels(guided,slat,coords,2;res=8,guide,graph_factory=reference_factory)
            new=TR.decodelevels(guided,slat,coords,2;res=8,guide,graph_factory=factory)
            @test old[2]==new[2]==guide.masks
            @test reinterpret(UInt32,vec(old[3]))==reinterpret(UInt32,vec(new[3]))
            @test length(declarations)==6
            for g in declarations
                ir=TR.decodergraph(g)
                @test ir isa DK.Graph
                @test g.emitted
                @test all(startswith(op.aten,"trellis.") for op in ir.ops)
                @test any(op.aten=="trellis.partial_matmul" for op in ir.ops)
                @test any(b.kind===:weight for b in values(ir.buffers))
                @test all(all(haskey(ir.buffers,id) for id in op.ins) for op in ir.ops)
                @test length(unique(op.out for op in ir.ops))==length(ir.ops)
                # Grid scatter writes its cells (argument 2), NOT the brick table.
                scatters=filter(op->op.aten=="trellis.voxel_scatter",ir.ops)
                @test !isempty(scatters)
                @test all(op.attrs["arguments"][2].input==0 for op in scatters)
                @test all(op.attrs["arguments"][1].input>0 for op in scatters)
                @test_throws ErrorException TR.decoderrunonce!(g)
                push!(records,(ops=length(ir.ops),kinds=sort!(unique(op.aten for op in ir.ops)),
                    weights=count(b->b.kind===:weight,values(ir.buffers))))
            end
            @test_throws ErrorException TR.decoderdispatch!(TR.SparseDecoderGraph(device),identity,(),1;name="unknown")
            @test_throws ArgumentError TR.decoderarg!(TR.SparseDecoderGraph(device),ones(Float32,2),String[])
        finally
            M.waitidle(device)
            owned=IdDict{Any,Nothing}()
            for g in declarations,r in values(g.bindings)
                r isa M.Buffer && (owned[r]=nothing)
            end
            for g in references,r in values(g.ids.by_id)
                r isa M.Buffer && (owned[r]=nothing)
            end
            for r in values(weights)
                delete!(owned,r)
            end
            foreach(M.free!,keys(owned))
            foreach(M.free!,values(weights))
        end
    end
    records
end
