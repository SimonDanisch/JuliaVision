include("missing_outcome_bounds.jl")
using Test
function test_missing_outcome_bounds()
    @testset "Unknown outcomes are not silently failures" begin
        a=missing_outcome_bounds(2,7,8)
        @test a.complete_case_rate==2/7&&a.registered_cohort_lower==0.25&&a.registered_cohort_upper==0.375
        @test missing_outcome_bounds(0,0,8).complete_case_rate===nothing
        @test missing_outcome_bounds(0,0,8).registered_cohort_upper==1.
        @test missing_outcome_bounds(8,8,8).registered_cohort_lower==1.
        @test_throws ArgumentError missing_outcome_bounds(true,7,8)
        @test_throws ArgumentError missing_outcome_bounds(8,7,8)
        @test_throws ArgumentError missing_outcome_bounds(0,0,0)
    end
end
