include("public_document_context.jl")
using Test
function test_public_document_context()
    R=PublicDocumentRetrieval;C=PublicDocumentContext
    @testset "Automatic public documentation context, no model query" begin
        rows=[(;qualified="Plot.scatter!",text="scatter points "*repeat("α",3000)),
              (;qualified="JSON.read",text="parse JSON input string")]
        index=R.Index(rows)
        pack=C.build(index,"scatter points";documents=1,characters_per_document=20)
        @test pack.query=="scatter points"
        @test length(pack.rows)==1
        @test only(pack.rows).qualified=="Plot.scatter!"
        @test length(only(pack.rows).text)==20
        @test only(pack.rows).truncated
        @test occursin(JSON3.write(pack.rows),pack.text)
        @test isempty(C.build(index,"neverpresentword").rows)
        @test isempty(C.build(index,"scatter";documents=0).rows)
        @test_throws ArgumentError C.build(index,"x";documents=true)
        @test_throws ArgumentError C.build(index,"x";documents=17)
        @test_throws ArgumentError C.build(index,"x";characters_per_document=0)
        @test_throws ArgumentError C.build(index,"x";characters_per_document=true)
        @test_throws ArgumentError C.build(index,"x";characters_per_document=8001)
    end
end
