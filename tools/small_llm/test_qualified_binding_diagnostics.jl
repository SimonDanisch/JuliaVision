include("qualified_binding_diagnostics.jl")
using Test,JSON3,SHA
function test_qualified_binding_diagnostics()
    Q=QualifiedBindingDiagnostics;b=Dict("Makie"=>["Figure","Axis"],"JSON3"=>["read","write"])
    tests=@testset "Public namespace-name diagnostics, no execution" begin
        @test Q.analyze("Makie.backend();JSON3.read(\"{}\")",b).references_to_unlisted_bindings==["Makie.backend"]
        @test isempty(Q.analyze("Makie.Figure()",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze(":(Makie.backend())",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("const Makie=another;Makie.backend()",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("function f(Makie);Makie.backend();end",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("(Makie,x)->Makie.backend()",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("(Makie,x)=pair;Makie.backend()",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("function f(;Makie=other);Makie.backend();end",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("function f(Makie::Module);Makie.backend();end",b).references_to_unlisted_bindings)
        @test isempty(Q.analyze("module Makie;end;Makie.backend()",b).references_to_unlisted_bindings)
        @test Q.analyze("error(\"must not execute\");Makie.backend()",b).references_to_unlisted_bindings==["Makie.backend"]
        @test occursin("no replacement",Q.render(Q.analyze("Makie.backend()",b)))
        inventory=JSON3.read(read(joinpath(@__DIR__,"julia_namespace_contract.json"),String))
        @test inventory.complete
        @test isempty(Q.analyze("CairoMakie.activate!();Makie.Figure()",inventory.bindings).references_to_unlisted_bindings)
    end
    c=Test.get_test_counts(tests);report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="Public name-inventory analyzer qualification only, NOT model-quality evidence",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("qualified_binding_diagnostics.jl","test_qualified_binding_diagnostics.jl","julia_namespace_contract.json")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"qualified_binding_diagnostics_validation.json"),"w");report
end
