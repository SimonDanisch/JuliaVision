include("rpn_public_example_tasks.jl")
isdefined(@__MODULE__,:RPNExampleSemantics)||include("rpn_example_semantics.jl")
using Test
function test_rpn_public_example_tasks()
    @testset "Identical public examples and prompts in both numeric-mask arms" begin
        tasks=rpn_public_example_tasks();jobs=rpn_public_example_jobs()
        @test length(tasks)==4 && length(jobs)==8
        @test length(unique(t.id for t in tasks))==4
        @test first(jobs).arm=="disabled" && jobs[3].arm=="enabled"
        for task in tasks
            pair=filter(j->j.task==task.id,jobs)
            @test length(pair)==2
            @test pair[1].messages==pair[2].messages
            @test pair[1].public_cases==pair[2].public_cases==task.public_cases
            @test Set(j.arm for j in pair)==Set(("enabled","disabled"))
            @test all(j->j.enabled==(j.arm=="enabled"),pair)
            @test occursin(JSON3.write(task.public_cases),last(last(pair[1].messages)))
            bank=RPNExampleSemantics.Bank(task.public_cases)
            @test length(bank.expected)==5 && bank.case_offsets==Int32[1,5,6]
            @test all(isfinite,bank.expected)
        end
    end
end
