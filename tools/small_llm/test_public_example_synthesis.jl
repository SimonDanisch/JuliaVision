include("public_example_synthesis.jl")
isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
using Test
function test_public_example_synthesis()
    S=PublicExampleSynthesis;B=RPNExampleSemantics.Bank
    @testset "Task-blind bounded search with shared numeric evaluator" begin
        square=B([(;input=Float32[-2,0,3],output=Float32[4,0,9])])
        result=S.solve(square;max_cost=2,seconds=10)
        @test result.found && result.operations==2
        @test RPNPointwiseInterface.parse_program(result.program).accepted
        @test result.candidates<=result.candidate_budget
        @test result.unique_states<=result.state_budget
        repeat=S.solve(square;max_cost=2,seconds=10)
        @test result.program==repeat.program && result.candidates==repeat.candidates
        affine=B([(;input=Float32[-1,0,2],output=Float32[-1,1,5])])
        result=S.solve(affine;max_cost=5,candidate_budget=500_000,seconds=10)
        @test result.found
        @test RPNPointwiseInterface.parse_program(result.program).accepted
        impossible=B([(;input=Float32[0,0],output=Float32[1,2])])
        exhausted=S.solve(impossible;max_cost=3)
        @test !exhausted.found && exhausted.reason=="cost_exhausted"
        limited=S.solve(impossible;candidate_budget=1)
        @test !limited.found && limited.reason=="candidate_budget" && limited.candidates==1
        limited=S.solve(impossible;state_budget=1)
        @test !limited.found && limited.reason=="state_budget" && limited.unique_states==1
        @test S.solve(square;seconds=1e-12).reason=="deadline"
        for kwargs in ((;max_cost=0),(;max_cost=true),(;candidate_budget=0),(;candidate_budget=true),
                (;state_budget=0),(;state_budget=true),(;seconds=0),(;seconds=NaN),(;seconds=true))
            @test_throws ArgumentError S.solve(square;kwargs...)
        end
        approximate=B([(;input=Float32[1],output=Float32[1])];atol=1f-3)
        @test_throws ArgumentError S.solve(approximate)
    end
end
