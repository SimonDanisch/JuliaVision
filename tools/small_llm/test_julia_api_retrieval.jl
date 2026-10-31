include("julia_docs_retrieval.jl")
using .JuliaDocsRetrieval, Test, JSON3

function test_api_retrieval()
    index=JuliaDocsRetrieval.build_index()
    prompt="Using Makie (not GLMakie), plot a semi-transparent band between mu .- sigma and mu .+ sigma, then lines and axislegend(ax,position=:lt)."
    chunks=JuliaDocsRetrieval.retrieve_api(index,prompt)
    @testset "API retrieval scope and coverage" begin
        @test length(chunks)==3
        @test any(c->endswith(c.source,"/band.md"),chunks)
        @test any(c->endswith(c.source,"/legend.md"),chunks)
        @test all(c->startswith(c.source,"dev/Makie/"),chunks)
        @test all(c->!occursin("GLMakie",c.source),chunks)
        @test length(unique(c.source for c in chunks))==length(chunks)
        @test all(c->length(c.body)<=3500,chunks)
    end
    row=(;scope="CPU retrieval-policy qualification, not model accuracy",chunks)
    open(io->JSON3.pretty(io,row),joinpath(@__DIR__,"julia_api_retrieval_validation.json"),"w")
    row
end
