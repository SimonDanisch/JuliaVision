using Test, Mantle
import KernelAbstractions
using .KnowledgeBias

function test_knowledge_bias(device)
    prior=Float32[0.7,0.2,0.1]
    logits=Mantle.Buffer(device,log.(prior));best=Mantle.Buffer(device,Int32,(1,))
    decoder=nothing
    try
        decoder=KnowledgeBias.Decoder((;logits,best,model=(;device)))
        @testset "External evidence changes probabilities before selection" begin
            likelihood=Float32[0.01,1,1]
            KnowledgeBias.set_evidence!(decoder,Dict(i-1=>log(likelihood[i]) for i in 1:3))
            KnowledgeBias.apply!(decoder)
            conditioned=exp.(Array(logits));conditioned./=sum(conditioned)
            exact=prior.*likelihood;exact./=sum(exact)
            @test conditioned ≈ exact atol=2f-7
            @test only(Array(best))==1       # zero-based second hypothesis
            Mantle.update!(logits,log.(prior))
            KnowledgeBias.set_evidence!(decoder,Dict(2=>0f0);default=-Inf32)
            KnowledgeBias.apply!(decoder)
            @test only(Array(best))==2
            @test Array(logits)[1:2]==fill(-Inf32,2)
            @test_throws ArgumentError KnowledgeBias.set_evidence!(decoder,Dict();default=-Inf32)
            @test_throws ArgumentError KnowledgeBias.set_evidence!(decoder,Dict(3=>0f0))
            @test_throws ArgumentError KnowledgeBias.set_evidence!(decoder,Dict(0=>NaN32))
        end
    finally
        decoder===nothing || close(decoder)
        Mantle.free!(logits);Mantle.free!(best)
    end
    nothing
end

function test_knowledge_bias_decode(model)
    evidence=KnowledgeBias.Evidence(model.device,Int(model.config.vocab_size))
    s=nothing
    try
        s=Qwen35Experiment.session(model;capacity=16,
            logit_hook=(g,logits)->KnowledgeBias.declare!(g,logits,evidence))
        @testset "Evidence, selection and feedback share one submission" begin
            KnowledgeBias.set_evidence!(evidence,Dict(7=>0f0);default=-Inf32)
            Qwen35Experiment.step!(s,42)
            Qwen35Experiment.decode!(s,3)
            @test Array(s.predictions)[1:4]==fill(Int32(7),4)
            KnowledgeBias.set_evidence!(evidence,Dict(11=>0f0);default=-Inf32)
            before=Mantle.submissions(model.device)
            Qwen35Experiment.decode!(s,2)
            KernelAbstractions.synchronize(model.device)
            @test Mantle.submissions(model.device)-before==2
            @test Array(s.predictions)[5:6]==fill(Int32(11),2)
            @test only(Array(s.best))==11
        end
    finally
        s===nothing || close(s)
        close(evidence)
    end
    nothing
end
