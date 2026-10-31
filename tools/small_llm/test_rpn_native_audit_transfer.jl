include("rpn_native_audit_transfer.jl")
using Test
function test_rpn_native_audit_transfer()
    @testset "Grading helper deployment is sources-only, no private cases" begin
        payload=rpn_native_audit_source_payload()
        @test Set(keys(payload.updates))==Set(RPN_NATIVE_AUDIT_SOURCES)
        @test length(payload.updates)==9
        @test all(name->endswith(name,".jl")||endswith(name,".py"),keys(payload.updates))
        @test !haskey(payload.updates,"prepare_rpn_native_audit.jl")
        @test !haskey(payload.updates,"qualify_rpn_stencil_tasks.jl")
        @test !haskey(payload.updates,"native_audit_input_PRIVATE.json")
        @test all(name->!occursin("PRIVATE",name),keys(payload.updates))
        @test validate_document_study_payload(payload)===nothing
    end
end
