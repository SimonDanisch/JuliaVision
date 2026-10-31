include("api_surface_audit.jl")
using Test

function test_api_surface_audit()
    @testset "Public API surface observations" begin
        @test api_surface_audit("ki_launch","KI.launch(kernel; ndrange=4)").requires_review
        @test api_surface_audit("ki_launch","KernelInterface.launch(k,(1,1,1),(64,1,1),(out,x))").requires_review
        @test !api_surface_audit("ki_launch","kernel(out,x; workgroupsize=64,ndrange=length(x))").requires_review
        @test !api_surface_audit("ki_launch","text=\"KI.launch(kernel)\"").requires_review
        @test api_surface_audit("ki_launch","function incomplete(").parse_problem
        @test !api_surface_audit("unrelated_task","KI.launch(k,groups,items,args)").requires_review
    end
end
