include("run_pubmedqa_retrieval.jl")
using Test
function test_pubmedqa_retrieval_runner()
    @testset "Four-arm host selector preserves immutable attempts" begin
        mktempdir() do directory
            jobs=[(;id="pubmed_1_"*arm) for arm in ("question_only","corresponding_context","mismatched_context","retrieved_context")]
            @test pubmedqa_retrieval_pending(directory,jobs)==jobs
            write(joinpath(directory,jobs[4].id*".json"),"{}")
            @test length(pubmedqa_retrieval_pending(directory,jobs))==3
            write(joinpath(directory,jobs[1].id*"_attempt.json"),"{}")
            @test_throws ErrorException pubmedqa_retrieval_pending(directory,jobs)
            @test_throws ArgumentError pubmedqa_retrieval_pending(directory,[(;id="../../private")])
            @test_throws ArgumentError pubmedqa_retrieval_pending(directory,vcat(jobs,jobs[1:1]))
        end
        @test_throws ArgumentError run_pubmedqa_retrieval(;seconds=0)
        @test_throws ArgumentError run_pubmedqa_retrieval(;max_jobs=5)
        @test normpath(pwd())!="/tmp/small-llm-bosgame-20261025"
        @test_throws ErrorException run_pubmedqa_retrieval()
    end
end
