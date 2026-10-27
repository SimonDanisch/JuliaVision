include("julia_method_cards.jl")
using Test
module MethodCardFixture
    square(x::Int)=x*x
end
@testset "Trusted source-card boundary" begin
    method=only(methods(MethodCardFixture.square))
    rows=JuliaMethodCards.cards([method];root=@__DIR__,context_lines=2)
    @test length(rows)==1
    @test occursin("square",only(rows).text)
    @test only(rows).last_line-only(rows).first_line<=1
    @test length(only(rows).source_sha256)==64
    @test only(rows).bounded_context_not_complete_body
    @test occursin("Bounded source context",JuliaMethodCards.format_cards(rows))
    @test_throws ArgumentError JuliaMethodCards.cards([method];root=joinpath(@__DIR__,"reference"))
    @test_throws ArgumentError JuliaMethodCards.cards([method];root=@__DIR__,max_file_bytes=1)
    @test_throws ArgumentError JuliaMethodCards.cards(fill(method,9);root=@__DIR__)
    @test_throws ArgumentError JuliaMethodCards.cards(["candidate expression"];root=@__DIR__)
    @test_throws ArgumentError JuliaMethodCards.cards([method];root=@__DIR__,context_lines=0)
end
