using Test
using .StructuredEvidence
const SE=StructuredEvidence

function test_structured_evidence()
    @testset "Exact finite Bayesian evidence" begin
        # Axis order is (rain, sprinkler, wet). State 1 is false, 2 true.
        wet=zeros(2,2,2)
        wet[:,:,2]=[0.01 0.8;0.9 0.99]
        wet[:,:,1]=1 .- wet[:,:,2]
        factors=[SE.Factor([:rain],[0.8,0.2]),SE.Factor([:sprinkler],[0.9,0.1]),
                 SE.Factor([:rain,:sprinkler,:wet],wet)]
        r=SE.infer(factors,:rain,Dict(:wet=>2))
        @test r.prior ≈ [0.8,0.2]
        @test exp(r.log_evidence) ≈ 0.253
        @test exp.(r.log_likelihood) ≈ [0.089,0.909]
        @test r.posterior ≈ [0.0712,0.1818]./0.253
        @test r.enumerated_states==8
        scores=SE.token_likelihood_scores(r,[11,22])
        updated=r.prior.*exp.([scores[11],scores[22]])
        @test updated./sum(updated) ≈ r.posterior rtol=1e-6
        @test_throws ArgumentError SE.token_likelihood_scores(r,[11,11])
        @test_throws ArgumentError SE.infer(factors,:rain;max_states=4)
        @test_throws ArgumentError SE.infer(factors,:rain,Dict(:wet=>3))
        @test_throws ArgumentError SE.infer(factors,:unknown)
        @test_throws ArgumentError SE.Factor([:x],[-0.1,1.1])
        @test_throws ArgumentError SE.Factor([:x,:x],ones(2,2))
        impossible=SE.infer([SE.Factor([:x],[1.0,0.0])],:x,Dict(:x=>2))
        @test impossible.posterior===nothing
        @test ismissing(impossible.log_likelihood[2])
        @test_throws ArgumentError SE.token_likelihood_scores(impossible,[11,22])
        # Tiny unnormalized factors must stay meaningful in log space.
        tiny=SE.infer([SE.Factor([:x],[1e-250,2e-250]),SE.Factor([:x],[1e-250,1e-250])],:x)
        @test tiny.prior ≈ [1/3,2/3]
    end
    @testset "Open-world Horn proof state" begin
        rules=[SE.Rule(:b,[:a]),SE.Rule(:c,[:b,:enabled]),SE.Rule(:a,[:c])]
        r=SE.prove([:a,:enabled],rules,:c)
        @test r.status===:entailed
        @test r.proof[:c]==2
        @test SE.prove([:a],rules,:c).status===:unknown
        @test SE.prove([:a],rules,:c;negative_facts=[:c]).status===:refuted
        @test SE.prove([:a,:enabled],rules,:c;negative_facts=[:c]).status===:inconsistent
        @test SE.prove(Symbol[],rules,:c).derived==Set{Symbol}()
        @test SE.prove(Symbol[],[SE.Rule(:a,Symbol[])],:a).status===:entailed
        @test SE.proof_token_scores(r,11,22)==Dict(11=>0f0)
        @test SE.proof_token_scores(SE.prove([:a],rules,:c),11,22)===nothing
        @test SE.proof_token_scores(SE.prove([:a],rules,:c;negative_facts=[:c]),11,22)==Dict(22=>0f0)
        @test SE.proof_token_scores(SE.prove([:a,:enabled],rules,:c;negative_facts=[:c]),11,22)===nothing
        @test_throws ArgumentError SE.proof_token_scores(r,11,11)
    end
    nothing
end
