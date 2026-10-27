using Test

function test_attention_logit_metrics()
    @testset "Distribution-drift scoring" begin
        a=Float64.(log.([0.2,0.3,0.5]));b=Float64.(log.([0.4,0.1,0.5]))
        same=attention_logit_metrics(a,a,1)
        @test same.kl ≈ 0 atol=1e-14
        @test same.candidate_nll ≈ -log(0.3)
        @test same.top1_agrees
        different=attention_logit_metrics(a,b,1)
        @test different.kl ≈ 0.2log(0.5)+0.3log(3) atol=1e-14
        @test different.candidate_nll-different.reference_nll ≈ log(3)
        shifted=attention_logit_metrics(a.+1000,b.-1000,1)
        @test shifted.kl ≈ different.kl atol=1e-12
        extreme=attention_logit_metrics([0.0,-1000.0],[-1000.0,0.0],0)
        @test extreme.kl ≈ 1000
        @test !extreme.top1_agrees
        @test_throws ArgumentError attention_logit_metrics(a,b,3)
        @test_throws ArgumentError attention_logit_metrics([NaN,0.0],[0.0,0.0],0)
        @test_throws DimensionMismatch attention_logit_metrics(a,[0.0],0)
    end
end
