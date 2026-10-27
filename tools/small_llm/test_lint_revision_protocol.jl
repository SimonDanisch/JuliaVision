include("lint_revision_protocol.jl")
using Test
function test_lint_revision_protocol()
    index=AliasDocumentRetrieval.Index([(;qualified="Alpha.read",text="parse JSON text",binding_id=1)])
    bindings=Dict("Alpha"=>["read"])
    original=[Dict("system"=>"Julia contract"),Dict("user"=>"Read public JSON input")]
    code="Alpha.parse(x)"
    @testset "Paired revisions use same public task and original code" begin
        control=lint_revision_messages(original,code,:self_review,index,bindings)
        feedback=lint_revision_messages(original,code,:lint_docs,index,bindings)
        @test length(control.messages)==length(feedback.messages)==4
        @test control.messages[1:3]==feedback.messages[1:3]
        @test control.messages[3]==("assistant"=>code)
        @test control.context===nothing
        @test feedback.context.diagnostic.unlisted_qualified_names==["Alpha.parse"]
        @test occursin("Alpha.read",last(feedback.messages[4]))
        @test occursin("no tool observations",last(control.messages[4]))
        @test original[2]["user"]=="Read public JSON input"
        @test_throws ArgumentError lint_revision_messages(original,code,:unknown,index,bindings)
        @test_throws ArgumentError lint_revision_messages([Dict("assistant"=>"not a task")],code,:self_review,index,bindings)
    end
end
