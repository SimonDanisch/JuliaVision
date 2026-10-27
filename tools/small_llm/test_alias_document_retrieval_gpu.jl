isdefined(@__MODULE__,:AliasDocumentRetrievalGPU)||include("alias_document_retrieval_gpu.jl")
import Mantle
using Test
function test_alias_document_retrieval_gpu(device,index)
    A=AliasDocumentRetrieval;P=PublicDocumentRetrieval;G=AliasDocumentRetrievalGPU
    @testset "Shared alias/two-field score CPU/device arithmetic" begin
        buffers=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(buffers,b);b)
        try
            body=index.body;titles=index.titles
            bo=upload(body.offsets);bt=upload(body.terms);bc=upload(body.counts);bl=upload(body.lengths);bi=upload(body.idf)
            to=upload(titles.offsets);tt=upload(titles.terms);tc=upload(titles.counts);tl=upload(titles.lengths);ti=upload(titles.idf)
            n=length(body.names);out=upload(fill(-999f0,n+7))
            for text in ("scatter points histogram bins","save figure PNG image","JSON parse dictionary","nonexistentwordxyz")
                bq=P.query_ids(body,text);tq=P.query_ids(titles,text)
                bqd=upload(isempty(bq) ? Int32[0] : bq);tqd=upload(isempty(tq) ? Int32[0] : tq)
                for weight in (0f0,2f0,16f0)
                    plan=nothing
                    try
                        graph=Mantle.Graph(device)
                        Mantle.dispatch!(graph,G.score_documents!,
                            (out,bo,bt,bc,bl,bi,bqd,body.average_length,to,tt,tc,tl,ti,tqd,titles.average_length,
                             Int32(length(bq)),Int32(length(tq)),weight,1.2f0,0.75f0,Int32(n)),n;group=64)
                        plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                        expected=[A.score(body.offsets,body.terms,body.counts,body.lengths,body.idf,bq,body.average_length,
                            titles.offsets,titles.terms,titles.counts,titles.lengths,titles.idf,tq,titles.average_length,
                            Int32(i),weight,1.2f0,0.75f0) for i in 1:n]
                        for repeat in 1:2
                            out[:]=fill(-999f0,n+7);before=Mantle.submissions(device);Mantle.run!(plan)
                            @test Mantle.submissions(device)-before==1
                            actual=Array(out)
                            @test all(isfinite,actual[1:n])
                            @test isapprox(actual[1:n],expected;atol=2f-6,rtol=2f-6)
                            @test actual[n+1:end]==fill(-999f0,7)
                        end
                    finally
                        plan===nothing||Mantle.free!(plan)
                    end
                end
            end
        finally
            foreach(Mantle.free!,reverse(buffers))
        end
    end
end
