include("public_document_retrieval.jl")
using Test
function test_public_document_retrieval()
    R=PublicDocumentRetrieval
    @testset "Public sparse retrieval arithmetic, not model intelligence" begin
        rows=[(;qualified="Plot.histogram!",text="Histogram bins frequency counts"),
              (;qualified="Plot.scatter!",text="Scatter markers points points"),
              (;qualified="JSON.read",text="Parse a JSON string dictionary")]
        index=R.Index(rows)
        @test first(R.retrieve(index,"histogram bins")).qualified=="Plot.histogram!"
        @test first(R.retrieve(index,"scatter points")).qualified=="Plot.scatter!"
        @test isempty(R.retrieve(index,"nonexistentword"))
        @test isempty(R.retrieve(index,"points";limit=0))
        @test R.query_ids(index,"points points")==R.query_ids(index,"points")
        @test R.retrieve(index,"points points")==R.retrieve(index,"points")
        @test R.words("CairoMakie FigureSize")==["cairo","makie","figure","size"]
        # Independent dense formula for EVERY sparse document/query term.
        for query in ("points bins","JSON dictionary","scatter markers frequency")
            ids=R.query_ids(index,query)
            for d in 1:3
                expected=0.0
                docwords=R.words(rows[d].qualified*"\n"*rows[d].text)
                for q in ids
                    word=only(w for (w,i) in index.vocabulary if i==q)
                    countword=count(==(word),docwords)
                    expected+=index.idf[q]*countword*2.2/(countword+1.2*(0.25+0.75*length(docwords)/index.average_length))
                end
                actual=R.score(index.offsets,index.terms,index.counts,index.lengths,index.idf,ids,Int32(d),index.average_length,1.2f0,0.75f0)
                @test isapprox(actual,expected;atol=1e-6,rtol=1e-6)
            end
        end
        @test_throws ArgumentError R.Index([])
        @test_throws ArgumentError R.Index([rows[1],rows[1]])
        @test_throws ArgumentError R.retrieve(index,"points";limit=-1)
        @test_throws ArgumentError R.retrieve(index,"points";k1=NaN)
        @test_throws ArgumentError R.retrieve(index,"points";b=2)
    end
end
