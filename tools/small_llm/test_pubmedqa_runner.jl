include("run_pubmedqa_reading.jl")
using Test
function test_pubmedqa_runner()
    @testset "Biomedical runner refuses retries and unintended hardware" begin
        mktempdir() do directory
            jobs=[(;id="pubmed_1_question_only"),(;id="pubmed_1_corresponding_context")]
            @test pubmedqa_pending(directory,jobs)==jobs
            write(joinpath(directory,jobs[1].id*".json"),"{}")
            write(joinpath(directory,jobs[1].id*"_attempt.json"),"{}")
            @test pubmedqa_pending(directory,jobs)==[jobs[2]]
            write(joinpath(directory,jobs[2].id*"_attempt.json"),"{}")
            @test_throws ErrorException pubmedqa_pending(directory,jobs)
            @test_throws ArgumentError pubmedqa_pending(directory,[(;id="../outside")])
            @test_throws ArgumentError pubmedqa_pending(directory,[jobs[1],jobs[1]])
        end
        @test_throws ArgumentError run_pubmedqa_reading(;seconds=0)
        @test_throws ArgumentError run_pubmedqa_reading(;max_jobs=5)
        # This host test is ROOT-only. The guard precedes state/model creation.
        @test normpath(pwd())!="/tmp/small-llm-bosgame-20261025"
        @test_throws ErrorException run_pubmedqa_reading()
    end
end
