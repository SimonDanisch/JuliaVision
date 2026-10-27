include("pubmedqa_reading_protocol.jl")
using Test
function test_pubmedqa_reading_protocol()
    tasks=[(;pmid=string(i),question="QUESTION_"*string(i),abstract_context=["CONTEXT_"*string(i)]) for i in 1:6]
    @testset "Source-only biomedical reading controls, no clinical claims" begin
        rows=PubMedQAReadingProtocol.jobs(tasks)
        @test length(rows)==18
        @test length(unique(r.id for r in rows))==18
        @test all(arm->count(r->r.arm==arm,rows)==6,PubMedQAReadingProtocol.ARMS)
        @test Set(first(rows[3i-2:3i]).arm for i in 1:3)==Set(PubMedQAReadingProtocol.ARMS)
        for (i,task) in enumerate(tasks)
            matched=only(filter(r->r.task==task.pmid&&r.arm=="corresponding_context",rows))
            wrong=only(filter(r->r.task==task.pmid&&r.arm=="mismatched_context",rows))
            none=only(filter(r->r.task==task.pmid&&r.arm=="question_only",rows))
            @test matched.context_pmid==task.pmid
            @test wrong.context_pmid!=task.pmid
            @test none.context_pmid===nothing
            @test occursin(only(task.abstract_context),last(matched.messages[2]))
            @test !occursin(only(task.abstract_context),last(wrong.messages[2]))
            @test !occursin("CONTEXT_",last(none.messages[2]))
            @test matched.messages[1]==wrong.messages[1]==none.messages[1]
        end
        @test_throws ArgumentError PubMedQAReadingProtocol.jobs([tasks[1]])
        @test_throws ArgumentError PubMedQAReadingProtocol.jobs([tasks[1],tasks[1]])
        bad=Any[tasks...];bad[1]=(;tasks[1]...,final_decision="PRIVATE_GOLD")
        @test_throws ArgumentError PubMedQAReadingProtocol.jobs(bad)
        bad[1]=(;tasks[1]...,LONG_ANSWER="PRIVATE_CONCLUSION")
        @test_throws ArgumentError PubMedQAReadingProtocol.jobs(bad)
        for (source,expected) in (("yes","yes"),(" NO \n","no"),("Maybe","maybe"),
                ("yes.","invalid"),("\"yes\"","invalid"),("yes because...","invalid"),
                ("","invalid"),("```yes```","invalid"))
            @test PubMedQAReadingProtocol.decision(source)==expected
        end
        @test_throws ArgumentError PubMedQAReadingProtocol.decision(repeat("x",65537))
    end
end
