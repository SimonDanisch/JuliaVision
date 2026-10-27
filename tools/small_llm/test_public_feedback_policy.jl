include("humaneval_public_feedback.jl")
using Test
@testset "Public-feedback trigger, no hidden correctness selection" begin
    @test !public_smoke_trigger((;analysis_completed=true,payload=[(;completed=true,value="false")]))
    @test public_smoke_trigger((;analysis_completed=true,payload=[(;completed=false,error="empty reduction")]))
    @test public_smoke_trigger((;analysis_completed=true,payload=[(;completed=true,matches_public_example=false)]))
    @test !public_smoke_trigger((;analysis_completed=true,payload=[(;completed=true,matches_public_example=true)]))
    @test public_smoke_trigger((;analysis_completed=false,loading_or_harness_error="ERROR: Package NotInstalled; /input/candidate.jl:1"))
    @test !public_smoke_trigger((;analysis_completed=false,loading_or_harness_error="timeout"))
    @test !public_smoke_trigger((;analysis_completed=false,loading_or_harness_error="ERROR: /input/runner.jl:1"))
    @test !public_smoke_trigger((;analysis_completed=false,loading_or_harness_error=nothing))
end
