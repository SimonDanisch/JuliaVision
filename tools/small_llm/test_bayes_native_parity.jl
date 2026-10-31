include("bayes_native_parity.jl")
using Test,JSON3
function test_bayes_native_parity(device)
    @testset "Shared Bayesian snapshot/parity harness" begin
        @test bayes_parity_float("-Inf")==-Inf32
        @test bayes_parity_float("+Inf")==Inf32
        @test bayes_parity_encode(-Inf32)=="-Inf"
        @test_throws ArgumentError bayes_parity_float(true)
        @test_throws ArgumentError bayes_parity_float("NaN")
        @test_throws ArgumentError bayes_parity_encode(NaN32)
        @test_throws ArgumentError bayes_parity_float(floatmax(Float64))
        @test_throws ArgumentError bayes_parity_encode(true)
        @test bayes_parity_encode(0.25f0)==0.25f0
        network=JSON3.write((;nodes=[(;name="x",states=["no","yes"],parents=String[],rows=[(;given=String[],p=[1.,0.])])],query="x",observed=Dict("x"=>"yes")))
        expected=(;status=1,prior=Float32[1,0],posterior=Float32[0,0],log_likelihood=["-Inf","-Inf"],defined=[1,0],log_evidence="-Inf")
        result=audit_bayes_native_case(device,network,expected)
        @test result.passed
        @test result.first_submissions==result.replay_submissions==1
        @test result.bitwise_replay
        wrong=(;expected...,status=0)
        @test !audit_bayes_native_case(device,network,wrong).passed
    end
end
