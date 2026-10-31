include("sampled_rpn_program.jl")
using Test
function test_sampled_rpn_options()
    @testset "Invalid RPN options rejected before model/device access" begin
        for options in ((;enabled=1),(;max_tokens=0),(;max_tokens=true),(;max_tokens=8193),
                        (;operation_budget=0),(;operation_budget=true),(;operation_budget=1025),
                        (;max_depth=1),(;max_depth=true),(;max_depth=65))
            @test_throws ArgumentError sampled_rpn_program(nothing,nothing,[];options...)
        end
    end
end
