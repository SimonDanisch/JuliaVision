using Test, Random
import Mantle, KernelAbstractions
const PE=PortableEvidence

function test_portable_evidence(device)
    snapshots=Any[]
    @testset "Shared finite-factor kernels" begin
        wet=zeros(2,2,2);wet[:,:,2]=[0.01 0.8;0.9 0.99];wet[:,:,1]=1 .- wet[:,:,2]
        cases=[
            ([StructuredEvidence.Factor([:rain],[0.8,0.2]),StructuredEvidence.Factor([:sprinkler],[0.9,0.1]),
              StructuredEvidence.Factor([:rain,:sprinkler,:wet],wet)],:rain,Dict(:wet=>2)),
            ([StructuredEvidence.Factor([:x],[1.0,0.0])],:x,Dict(:x=>2)),
            ([StructuredEvidence.Factor([:x],[1e-250,2e-250]),StructuredEvidence.Factor([:x],[1e-250,1e-250])],:x,Dict{Symbol,Int}()),
            ([StructuredEvidence.Factor([:x],[0.0,0.0])],:x,Dict{Symbol,Int}())]
        for (factors,query,observations) in cases
            e=PE.prepare_bayes(device,factors,query,observations)
            try
                before=Mantle.submissions(device)
                PE.run!(e);KernelAbstractions.synchronize(device)
                @test Mantle.submissions(device)-before==1
                status=only(Array(e.outputs.status))
                if all(iszero,factors[1].values)
                    @test status==2
                    @test all(iszero,Array(e.outputs.prior))
                else
                    ref=StructuredEvidence.infer(factors,query,observations)
                    @test Array(e.outputs.prior) ≈ ref.prior atol=3e-5 rtol=3e-5
                    @test only(Array(e.outputs.log_evidence)) ≈ ref.log_evidence atol=3e-5 rtol=3e-5
                    @test status==(ref.posterior===nothing ? 1 : 0)
                    if ref.posterior!==nothing
                        @test Array(e.outputs.posterior) ≈ ref.posterior atol=3e-5 rtol=3e-5
                    end
                    for (i,likelihood) in enumerate(ref.log_likelihood)
                        @test Array(e.outputs.defined)[i]==(ismissing(likelihood) ? 0 : 1)
                        if !ismissing(likelihood)
                            @test Array(e.outputs.log_likelihood)[i] ≈ likelihood atol=3e-5 rtol=3e-5
                        end
                    end
                end
                push!(snapshots,(kind=:bayes,status,prior=Array(e.outputs.prior),posterior=Array(e.outputs.posterior)))
            finally
                close(e)
            end
        end
        rng=MersenneTwister(402)
        factors=[StructuredEvidence.Factor([:b,:a],rand(rng,3,2)),
                 StructuredEvidence.Factor([:c,:b],rand(rng,2,3))]
        e=PE.prepare_bayes(device,factors,:b,Dict(:c=>2))
        try
            PE.run!(e)
            @test Array(e.outputs.posterior) ≈ StructuredEvidence.infer(factors,:b,Dict(:c=>2)).posterior rtol=3e-5
            # A queued observation update uses the same recording.
            observed=zeros(Int32,length(e.metadata.names))
            observed[findfirst(==(:a),e.metadata.names)]=1
            e.inputs.observed[:]=observed
            PE.run!(e)
            @test Array(e.outputs.posterior) ≈ StructuredEvidence.infer(factors,:b,Dict(:a=>1)).posterior rtol=3e-5
        finally
            close(e)
        end
    end
    @testset "Shared Horn closure and truth-mask kernels" begin
        rules=[StructuredEvidence.Rule(:b,[:a]),StructuredEvidence.Rule(:c,[:b,:enabled]),
               StructuredEvidence.Rule(:a,[:c]),StructuredEvidence.Rule(:ungrounded,[:ungrounded])]
        for (facts,negative) in (([:a,:enabled],Symbol[]),([:a],[:c]),([:a],Symbol[]),([:a,:enabled],[:c]))
            e=PE.prepare_horn(device,facts,rules,:c;negative_facts=negative)
            bias=Mantle.Buffer(device,zeros(Float32,4));plan=nothing
            status_bias=Mantle.Buffer(device,zeros(Float32,4))
            enabled=Mantle.GPURef(device,Int32(0))
            try
                # Re-declare the solver and mask into ONE graph, not two replays.
                g=Mantle.Graph(device)
                PE.declare_horn!(g,e.inputs,e.outputs,e.metadata)
                PE.declare_truth_mask!(g,bias,e,1,2)
                PE.declare_status_mask!(g,status_bias,e,(0,1,2,3),enabled)
                plan=Mantle.Plan(g;budget=Inf);Mantle.record!(plan)
                before=Mantle.submissions(device)
                Mantle.run!(plan);KernelAbstractions.synchronize(device)
                @test Mantle.submissions(device)-before==1
                reference=StructuredEvidence.prove(facts,rules,:c;negative_facts=negative)
                status=Array(e.outputs.status)[e.metadata.query]
                codes=Dict(:unknown=>0,:entailed=>1,:refuted=>2,:inconsistent=>3)
                @test status==codes[reference.status]
                known=Array(e.outputs.positive);origins=Array(e.outputs.provenance)
                @test Set(e.metadata.names[findall(==(1),known)])==reference.derived
                for (i,origin) in enumerate(origins)
                    if origin>0
                        @test rules[origin].head==e.metadata.names[i]
                        @test all(p->p in reference.derived,rules[origin].premises)
                    end
                end
                expected=status==1 ? Float32[-Inf,0,-Inf,-Inf] :
                         status==2 ? Float32[-Inf,-Inf,0,-Inf] : zeros(Float32,4)
                @test Array(bias)==expected
                @test Array(status_bias)==zeros(Float32,4)
                enabled[]=Int32(1)
                before=Mantle.submissions(device)
                Mantle.run!(plan);KernelAbstractions.synchronize(device)
                @test Mantle.submissions(device)-before==1
                target=fill(-Inf32,4);target[Int(status)+1]=0f0
                @test Array(status_bias)==target
                push!(snapshots,(kind=:horn,status,known,origins))
            finally
                plan===nothing || Mantle.free!(plan)
                Mantle.free!(bias);Mantle.free!(status_bias);Mantle.free!(enabled);close(e)
            end
        end
        for rules in (StructuredEvidence.Rule[],[StructuredEvidence.Rule(:a,Symbol[])])
            e=PE.prepare_horn(device,Symbol[],rules,:a)
            try
                PE.run!(e)
                @test only(Array(e.outputs.status))==(isempty(rules) ? 0 : 1)
            finally
                close(e)
            end
        end
    end
    test_portable_bayes_readout(device)
    snapshots
end

function test_portable_bayes_readout(device)
    factors=[StructuredEvidence.Factor([:rain],[0.8,0.2]),
             StructuredEvidence.Factor([:rain,:wet],[0.911 0.089 0;0.091 0.909 0])]
    e=PE.prepare_bayes(device,factors,:rain,Dict(:wet=>2))
    base=Float32[log(0.8),log(0.2),0,0]
    logits=Mantle.Buffer(device,base);tokens=Mantle.Buffer(device,Int32[0,1]);plan=nothing
    try
        g=Mantle.Graph(device)
        PE.declare_bayes!(g,e.inputs,e.outputs,e.metadata)
        PE.declare_bayesian_logits!(g,logits,e,tokens;restrict_candidates=true)
        plan=Mantle.Plan(g;budget=Inf);Mantle.record!(plan)
        @testset "Computed likelihoods update logits in one submission" begin
            before=Mantle.submissions(device)
            Mantle.run!(plan);KernelAbstractions.synchronize(device)
            @test Mantle.submissions(device)-before==1
            probabilities=exp.(Array(logits));probabilities./=sum(probabilities)
            @test probabilities ≈ Float32[0.0712/0.253,0.1818/0.253,0,0] atol=3f-5
            @test only(Array(e.outputs.status))==0
            e.inputs.observed[:]=zeros(Int32,length(e.metadata.names))
            logits[:]=copy(base)
            Mantle.run!(plan)
            probabilities=exp.(Array(logits));probabilities./=sum(probabilities)
            @test probabilities ≈ Float32[0.8,0.2,0,0] atol=3f-5
            observed=zeros(Int32,length(e.metadata.names))
            observed[findfirst(==(:wet),e.metadata.names)]=3
            e.inputs.observed[:]=observed;logits[:]=copy(base)
            Mantle.run!(plan)
            @test only(Array(e.outputs.status))==1
            @test Array(logits)==base # Impossible evidence must not force an answer.
        end
    finally
        plan===nothing || Mantle.free!(plan)
        Mantle.free!(logits);Mantle.free!(tokens);close(e)
    end
    nothing
end
