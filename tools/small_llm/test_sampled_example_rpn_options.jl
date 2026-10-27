include("sampled_example_rpn_program.jl")
using Test
function test_sampled_example_rpn_options()
    bank=RPNExampleSemantics.Bank([(;input=Float32[1],output=Float32[1])])
    @testset "Example generator rejects bad options before touching a model/device" begin
        for kwargs in ((;enabled=1),(;max_tokens=0),(;max_tokens=true),(;max_tokens=8193),
                (;operation_budget=0),(;operation_budget=true),(;operation_budget=1025),
                (;output_chunk=0),(;output_chunk=true))
            @test_throws ArgumentError sampled_example_rpn_program(nothing,nothing,nothing,bank;kwargs...)
        end
    end
end
