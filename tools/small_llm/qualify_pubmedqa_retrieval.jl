include("pubmedqa_retrieval_protocol.jl")
isdefined(@__MODULE__,:PublicDocumentRetrievalGPU)||include("public_document_retrieval_gpu.jl")
using Test,JSON3,SHA
import Mantle
function qualify_pubmedqa_retrieval(device;snapshot="gen/pubmedqa-retrieval-20261025",
        destination=joinpath(@__DIR__,"pubmedqa_retrieval_numeric_validation.json"))
    isfile(destination)&&error("Preserve qualification")
    occursin("llvmpipe",lowercase(string(device)))||error("ROOT software-only")
    corpuspath=joinpath(snapshot,"public/corpus.json");taskpath=joinpath(snapshot,"public/tasks.json")
    corpus=JSON3.read(read(corpuspath,String));tasks=JSON3.read(read(taskpath,String))
    index=PublicDocumentRetrieval.Index(corpus);owned=Any[]
    upload(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    checks=@testset "Actual biomedical corpus uses the same shared BM25 arithmetic" begin
        @test length(index.names)==1000
        @test length(tasks)==16
        try
            offsets=upload(index.offsets);terms=upload(index.terms);counts=upload(index.counts)
            lengths=upload(index.lengths);idf=upload(index.idf);out=upload(fill(-999f0,1007))
            for task in tasks
                queryids=PublicDocumentRetrieval.query_ids(index,String(task.question))
                query=upload(isempty(queryids) ? Int32[0] : queryids);plan=nothing
                try
                    graph=Mantle.Graph(device)
                    Mantle.dispatch!(graph,PublicDocumentRetrievalGPU.score_documents!,
                        (out,offsets,terms,counts,lengths,idf,query,Int32(length(queryids)),index.average_length,1.2f0,0.75f0,Int32(1000)),1000;group=64)
                    plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                    expected=[PublicDocumentRetrieval.score(index.offsets,index.terms,index.counts,index.lengths,index.idf,
                        queryids,Int32(d),index.average_length,1.2f0,0.75f0) for d in 1:1000]
                    for replay in 1:2
                        out[:]=fill(-999f0,1007);before=Mantle.submissions(device);Mantle.run!(plan)
                        after_dispatch=Mantle.submissions(device)
                        actual=Array(out) # separate transfer submission, not part of replay
                        @test after_dispatch-before==1
                        @test actual[1:1000]≈expected rtol=2e-6 atol=2e-6
                        @test actual[1001:end]==fill(-999f0,7)
                        @test first(sortperm(1:1000;by=i->(-actual[i],index.names[i])))==first(sortperm(1:1000;by=i->(-expected[i],index.names[i])))
                    end
                finally
                    plan===nothing||Mantle.free!(plan)
                end
            end
        finally
            foreach(Mantle.free!,reverse(owned))
        end
    end
    count=Test.get_test_counts(checks)
    result=(;complete=true,software_checks=count.passes+count.cumulative_passes,
        scope="CPU/software Vulkan BM25 math/ranking on actual unlabelled corpus, not medical answer quality or physical GPU retrieval timing",
        corpus_sha256=bytes2hex(sha256(read(corpuspath))),tasks_sha256=bytes2hex(sha256(read(taskpath))),
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("qualify_pubmedqa_retrieval.jl","public_document_retrieval.jl","public_document_retrieval_gpu.jl","pubmedqa_retrieval_protocol.jl")))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
