include("humaneval_jet_policy.jl")
using Test
@testset "External candidate identity" begin
    for task in load_frozen_humaneval_tasks()
        path=joinpath(@__DIR__,"humaneval_subset_results",task.id*"_baseline.json")
        isfile(path) || continue
        candidate=JSON3.read(read(path,String))
        @test validate_external_candidate(task,candidate)===nothing
        for key in (:task,:contract_sha256,:conditions_sha256)
            mutated=Dict(pairs(candidate));mutated[key]="invalid"
            @test_throws ErrorException validate_external_candidate(task,JSON3.read(JSON3.write(mutated)))
        end
    end
    for job in external_repair_jobs()
        @test job.arm in ("self_review","jet_feedback")
        @test job.messages[2].second==job.task.prompt
        baseline=JSON3.read(read(joinpath(@__DIR__,"humaneval_subset_results",job.task.id*"_baseline.json"),String))
        @test job.messages[3].second==baseline.answer
        @test !isfile(job.output)
        @test !occursin(job.task.tests,last(job.messages).second)
    end
end
