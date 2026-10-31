include("rpn_stencil_contract_v2.jl")
isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
using Test
function test_rpn_stencil_contract_v2()
    @testset "NEW prompt fixes closure wording without editing frozen original" begin
        example="two\none\nsub\nhalf\nmul\nfinish"
        @test RPNPointwiseInterface.parse_program(example).accepted
        for task in rpn_stencil_tasks()
            original=rpn_stencil_messages(task);revised=rpn_stencil_messages_v2(task)
            @test first(original)==first(revised)
            @test endswith(last(last(revised)),task.specification)
            @test occursin("maximum32 non-finish operations.",last(last(original)))
            @test !occursin("maximum32 non-finish operations.",last(last(revised)))
            @test occursin("total to39 non-finish operations",last(last(revised)))
            @test occursin(example,last(last(revised)))
        end
    end
end
