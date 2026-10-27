include("alias_document_retrieval.jl")
using Test

function test_alias_document_retrieval()
    A=AliasDocumentRetrieval
    @testset "Actual-identity alias groups, not same-text inference" begin
        rows=[(;qualified="Makie.scatter",text="plot positions",binding_id=1),
            (;qualified="CairoMakie.scatter",text="plot positions",binding_id=1),
            (;qualified="Other.scatter",text="plot positions",binding_id=2),
            (;qualified="Makie.points",text="different binding-specific documentation",binding_id=1),
            (;qualified="Makie.save",text="write a figure to disk",binding_id=3)]
        index=A.Index(rows)
        @test length(index.body.names)==4
        @test index.body.names==A.Index(reverse(rows)).body.names
        @test sort(length.(index.aliases))==[1,1,1,2]
        selected=A.retrieve(index,"scatter";limit=4)
        @test length(selected)==2
        @test any(r->Set(r.aliases)==Set(("Makie.scatter","CairoMakie.scatter")),selected)
        @test any(r->r.aliases==["Other.scatter"],selected)
        @test isempty(A.retrieve(index,"unfindable_token"))
        @test isempty(A.retrieve(index,"scatter";limit=0))
        @test A.retrieve(index,"scatter";name_weight=0)!=A.retrieve(index,"scatter";name_weight=2)
        @test all(r->isfinite(r.score),selected)
        for kwargs in ((;limit=-1),(;limit=true),(;name_weight=NaN),(;name_weight=-1),(;name_weight=17),(;k1=0),(;b=2))
            @test_throws ArgumentError A.retrieve(index,"scatter";kwargs...)
        end
        @test_throws ArgumentError A.Index(Any[])
        @test_throws ArgumentError A.Index(vcat(rows,[first(rows)]))
        for id in (0,-1,true,1.0)
            @test_throws ArgumentError A.Index([(;qualified="A.f",text="doc",binding_id=id)])
        end
        # A blank doc shared by two DIFFERENT objects must not collapse them.
        blank=A.Index([(;qualified="A.f",text="",binding_id=1),(;qualified="A.g",text="",binding_id=2)])
        @test length(blank.body.names)==2
    end
end
