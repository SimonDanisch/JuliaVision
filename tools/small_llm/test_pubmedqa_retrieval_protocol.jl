include("pubmedqa_retrieval_protocol.jl")
using Test
function test_pubmedqa_retrieval_protocol()
    tasks=[(;pmid=string(i),question="topic$(i) effect",abstract_context=["topic$(i) effect result"]) for i in 1:4]
    corpus=[(;qualified=t.pmid,text=only(t.abstract_context)) for t in tasks]
    @testset "Body-only retrieval is automatic, bounded and counterbalanced" begin
        rows=PubMedQARetrievalProtocol.jobs(tasks,corpus)
        @test length(rows)==16
        @test length(unique(r.id for r in rows))==16
        @test Set(first(rows[4i-3:4i]).arm for i in 1:4)==Set(PubMedQARetrievalProtocol.ARMS)
        for task in tasks
            retrieved=only(filter(r->r.task==task.pmid&&r.arm=="retrieved_context",rows))
            oracle=only(filter(r->r.task==task.pmid&&r.arm=="corresponding_context",rows))
            @test retrieved.context_pmid==task.pmid
            @test retrieved.messages==oracle.messages
            @test retrieved.retrieval_found===true
            @test retrieved.retrieval_score>0
        end
        missing=[(;t...,question="qqqqqqunmatched") for t in tasks]
        rows=PubMedQARetrievalProtocol.jobs(missing,corpus)
        @test count(r->r.arm=="retrieved_context"&&r.retrieval_found===false,rows)==4
        @test all(r->r.arm!="retrieved_context"||r.context_pmid===nothing,rows)
        @test_throws ArgumentError PubMedQARetrievalProtocol.jobs(tasks,corpus;characters=0)
        @test_throws ArgumentError PubMedQARetrievalProtocol.jobs(tasks,corpus;characters=true)
        bad=Any[corpus...];bad[1]=(;corpus[1]...,final_decision="PRIVATE")
        @test_throws ArgumentError PubMedQARetrievalProtocol.jobs(tasks,bad)
        badtasks=Any[tasks...];badtasks[1]=(;tasks[1]...,LONG_ANSWER="PRIVATE")
        @test_throws ArgumentError PubMedQARetrievalProtocol.jobs(badtasks,corpus)
        clipped=PubMedQARetrievalProtocol.jobs(tasks,corpus;characters=3)
        @test all(r->r.arm!="retrieved_context"||!occursin("effect result",last(r.messages[2])),clipped)
    end
end
