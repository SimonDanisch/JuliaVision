include("julia_api_cards.jl")
using .JuliaAPICards, Test, SHA
module CardTestAPI
"A trusted documented function, without a candidate body."
example(x)=x
undocumented(x)=x
end

function test_api_cards()
    @testset "runtime API cards" begin
        rows=JuliaAPICards.cards(CardTestAPI,[:example,:undocumented,:not_defined,:example])
        @test length(rows)==3
        @test rows[1].defined && rows[1].documented
        @test occursin("trusted documented",rows[1].text)
        @test rows[1].full_document_sha256==bytes2hex(sha256(rows[1].text))
        @test rows[2].defined && !rows[2].documented
        @test isempty(rows[2].text)
        @test !rows[3].defined && !rows[3].documented
        @test isempty(rows[3].text)
        @test !rows[1].truncated
        @test occursin("not_defined",JuliaAPICards.format_cards(rows))
        @test_throws ArgumentError JuliaAPICards.cards(CardTestAPI,[:example];chars=10)
        @test_throws ArgumentError JuliaAPICards.cards(CardTestAPI,fill(:example,17))
    end
    (;passed=true,checks=12)
end
