isdefined(@__MODULE__,:PublicJuliaSignatures) || include("public_julia_signatures.jl")
isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || include("humaneval_subset.jl")
using Test, JSON3
function test_public_julia_signatures()
    P=PublicJuliaSignatures
    @testset "public type signatures, no eval" begin
        @test P.from_declaration("solve(x::Vector{Float64},w::Int)::Float64")==
            (;name="solve",types="[Tuple{Vector{Float64},Int}]")
        @test P.from_declaration("solve!(out,x)")==
            (;name="solve!",types="[Tuple{Any,Any}]")
        @test P.from_declaration("empty()::Nothing").types=="[Tuple{}]"
        @test_throws ArgumentError P.from_declaration("solve(x::typeof(run(`true`)))")
        @test_throws ArgumentError P.from_declaration("solve(x=run(`true`))")
        @test_throws ArgumentError P.from_declaration("Base.solve(x::Int)")
        rows=Any[]
        for task in load_frozen_humaneval_tasks()
            spec=P.from_task_prompt(task.prompt)
            @test !isempty(spec.name)
            @test startswith(spec.types,"[Tuple{")
            push!(rows,(;task=task.id,spec...))
        end
        report=(;scope="public docstring-to-isolated-JET input metadata only",passed=true,rows)
        open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"humaneval_subset_results/public_signatures.json"),"w")
    end
    nothing
end
