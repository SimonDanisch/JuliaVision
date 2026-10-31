include("head_input_tap.jl")
using Test

function test_head_input_tap()
    @testset "Head tap host ownership and restoration" begin
        model=(;config=(;hidden_size=3),weights=Dict{String,Any}("lm_head"=>:original))
        query=zeros(Float32,3)
        result=with_head_input_tap(model,query) do
            @test model.weights["lm_head"] isa HeadInputTap
            @test model.weights["lm_head"].weight===:original
            @test model.weights["lm_head"].query===query
            @test_throws ArgumentError with_head_input_tap(()->nothing,model,query)
            :result
        end
        @test result===:result
        @test model.weights["lm_head"]===:original
        @test_throws ErrorException with_head_input_tap(()->error("expected"),model,query)
        @test model.weights["lm_head"]===:original
        @test_throws ArgumentError with_head_input_tap(()->nothing,model,zeros(Float32,2))
        @test_throws ArgumentError with_head_input_tap(()->nothing,model,zeros(Float64,3))
        with_head_input_tap(()->empty!(model.weights),model,query)
        @test isempty(model.weights)
    end
end
