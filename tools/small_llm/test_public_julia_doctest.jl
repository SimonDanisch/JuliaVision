include("public_julia_doctest.jl")
isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || include("humaneval_subset.jl")
using Test
@testset "Public doctest qualification" begin
    for task in load_frozen_humaneval_tasks()
        result=diagnose_public_doctest(task,task.reference)
        @test result.analysis_completed
        @test all(row->row.completed&&row.matches_public_example,result.payload)
    end
    task=(;id="public_doctest_boundary",prompt="Implement function.\n\n f(x::Int)::Int\n\n```jldoctest\njulia> f(1)\n1\n```\n")
    wrong=diagnose_public_doctest(task,"f(x)=2")
    @test wrong.analysis_completed
    @test only(wrong.payload).completed
    @test !only(wrong.payload).matches_public_example
    expected="run(`id`)"
    changed=merge(task,(;prompt=replace(task.prompt,"\n1\n```"=>"\n"*expected*"\n```")))
    result=diagnose_public_doctest(changed,"f(x)=2")
    @test result.analysis_completed
    @test only(result.payload).expected==expected
    @test !only(result.payload).matches_public_example
end
