include("score_bayes_cpu_reference.jl")
using Test
function test_bayes_cpu_reference_score()
    @testset "Native diagnostic keeps truncation/format failures" begin
        good=(;id="mock",answer="{\"probability\":0.25}",eos_completed=true,truncated_or_time_limited=false)
        @test bayes_cpu_diagnostic_row(good,0.25).status=="passed"
        @test bayes_cpu_diagnostic_row(good,0.5).status=="failed"
        @test !bayes_cpu_diagnostic_row((;good...,eos_completed=false,truncated_or_time_limited=true),0.25).admitted
        @test bayes_cpu_diagnostic_row((;good...,answer="not JSON"),0.25).status=="failed"
        @test bayes_cpu_diagnostic_row((;id="mock",inference_error="synthetic"),0.25).inference_error
        @test bayes_cpu_diagnostic_row((;good...,answer="{\"probability\":true}"),0.25).status=="failed"
    end
end
