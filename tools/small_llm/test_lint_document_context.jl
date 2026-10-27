include("lint_document_context.jl")
using Test
function test_lint_document_context()
    rows=[(;qualified="Alpha.read",text="Read and parse JSON values from text",binding_id=1),
        (;qualified="Alias.read",text="Read and parse JSON values from text",binding_id=1),
        (;qualified="Alpha.write",text="Serialize JSON values to text",binding_id=2),
        (;qualified="Beta.parse",text="A generic parse helper",binding_id=3)]
    index=AliasDocumentRetrieval.Index(rows)
    inventory=Dict("Alpha"=>["read","write"],"Beta"=>["parse"])
    @testset "Linter-derived retrieval is source-only and bounded" begin
        result=LintDocumentContext.build(index,inventory,"decode JSON","error(\"not executed\");Alpha.parse(x)")
        @test result.diagnostic.source_parse_ok
        @test result.diagnostic.unlisted_qualified_names==["Alpha.parse"]
        @test occursin("Alpha.parse",result.query)
        @test !occursin("not executed",result.query)
        @test all(r->r.qualified in [x.qualified for x in rows],result.rows)
        @test occursin("not a runtime/type proof",result.diagnostic_text)
        scoped=LintDocumentContext.build(index,inventory,"decode JSON","f(Alpha)=Alpha.parse(x)")
        @test isempty(scoped.diagnostic.unlisted_qualified_names)
        @test scoped.query=="decode JSON"
        @test "Alpha" in scoped.diagnostic.skipped_modules
        limited=LintDocumentContext.build(index,inventory,"JSON","Alpha.unknown(x);Beta.unknown(x)";max_references=1,characters_per_document=12)
        @test length(limited.diagnostic.unlisted_qualified_names)==1
        @test limited.diagnostic.omitted_reference_count==1
        @test all(r->length(r.text)<=12,limited.rows)
        @test isempty(LintDocumentContext.build(index,inventory,"JSON","Alpha.parse(x)";documents=0).rows)
        @test !LintDocumentContext.build(index,inventory,"JSON","function (").diagnostic.source_parse_ok
        @test_throws ArgumentError LintDocumentContext.build(index,inventory,"JSON","x";max_references=true)
        @test_throws ArgumentError LintDocumentContext.build(index,inventory,repeat("x",65537),"x")
    end
end
