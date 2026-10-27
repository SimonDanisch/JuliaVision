isdefined(@__MODULE__,:PublicDocumentRetrievalGPU)||include("public_document_retrieval_gpu.jl")
using Test
function test_public_document_retrieval_gpu(device,index)
    R=PublicDocumentRetrieval;G=PublicDocumentRetrievalGPU
    @testset "Shared sparse BM25 software/device arithmetic" begin
        buffers=Any[]
        upload(x)=(b=Mantle.Buffer(device,x);push!(buffers,b);b)
        try
            offsets=upload(index.offsets);terms=upload(index.terms);counts=upload(index.counts)
            lengths=upload(index.lengths);idf=upload(index.idf)
            n=length(index.names);out=upload(fill(-999f0,n+7))
            for text in ("save figure PNG image","scatter points histogram bins","JSON parse dictionary","nonexistentwordxyz")
                queryids=R.query_ids(index,text)
                query=upload(isempty(queryids) ? Int32[0] : queryids)
                plan=nothing
                try
                    graph=Mantle.Graph(device)
                    Mantle.dispatch!(graph,G.score_documents!,
                        (out,offsets,terms,counts,lengths,idf,query,Int32(length(queryids)),
                         index.average_length,1.2f0,0.75f0,Int32(n)),n;group=64)
                    plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                    for repeat in 1:2
                        out[:]=fill(-999f0,n+7)
                        submissions=Mantle.submissions(device);Mantle.run!(plan)
                        @test Mantle.submissions(device)-submissions==1
                        actual=Array(out)
                        expected=[R.score(index.offsets,index.terms,index.counts,index.lengths,index.idf,
                            queryids,Int32(d),index.average_length,1.2f0,0.75f0) for d in 1:n]
                        @test all(isfinite,actual[1:n])
                        @test isapprox(actual[1:n],expected;rtol=2e-6,atol=2e-6)
                        @test actual[n+1:end]==fill(-999f0,7)
                    end
                finally
                    plan===nothing||Mantle.free!(plan)
                end
            end
        finally
            foreach(Mantle.free!,reverse(buffers))
        end
    end
end
