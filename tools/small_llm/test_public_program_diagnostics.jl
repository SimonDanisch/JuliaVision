include("public_program_diagnostics.jl")
using Test,JSON3,SHA
function test_public_program_diagnostics()
    P=PublicProgramDiagnostics
    tests=@testset "Public syntax/import diagnostics, no generated execution" begin
        @test isempty(P.analyze("using CairoMakie, Base64; println(1)").outside_published_contract)
        @test P.analyze("using DataFrames: DataFrame").outside_published_contract==["DataFrames"]
        @test P.analyze("import CSV as C").outside_published_contract==["CSV"]
        @test P.analyze("using Statistics: mean").imports==["Statistics"]
        @test isempty(P.analyze("using .LocalModule").outside_published_contract)
        @test isempty(P.analyze("quote;using SecretModule;end").imports)
        @test isempty(P.analyze("@eval using SecretModule").imports)
        @test !isempty(P.analyze("function broken(").parse_errors)
        @test occursin("broken",first(P.analyze("function broken(").parse_errors))
        @test isempty(P.analyze("error(\"MUST NEVER EXECUTE\")").parse_errors)
        @test occursin("NO benchmark tests",P.render(P.analyze("using CSV")))
        @test_throws ArgumentError P.analyze(repeat("x",262145))
    end
    c=Test.get_test_counts(tests);report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="Public AST/dependency analyzer qualification only, not model reasoning gains",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("public_program_diagnostics.jl","test_public_program_diagnostics.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"public_program_diagnostics_validation.json"),"w");report
end
