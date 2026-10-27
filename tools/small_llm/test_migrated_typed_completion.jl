include("migrated_typed_completion.jl")
using Test
function test_migrated_typed_choice()
    @testset "Compiler completion inspects source, not scores/tasks" begin
        rejected=migrated_typed_choice("not a valid program (")
        @test !rejected.lowered
        @test rejected.code=="not a valid program ("
        code="function kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32);for i=1:n;out[i]=2f0*x[i];end;end"
        selected=migrated_typed_choice(code)
        @test selected.lowered
        @test occursin("KI.get_global_id",selected.code)
        @test occursin("2.0f0",selected.code)||occursin("2.0f",selected.code)
        @test !migrated_typed_choice(replace(code,"n::Int32"=>"n::Float32")).lowered
    end
end
