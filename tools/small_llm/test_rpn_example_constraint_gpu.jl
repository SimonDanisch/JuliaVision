isdefined(@__MODULE__,:RPNExampleConstraintGPU)||include("rpn_example_constraint_gpu.jl")
using Test
import Mantle
function test_rpn_example_constraint_gpu(device)
    G=RPNPrefixGrammar;D=RPNPrefixGrammarGPU;E=RPNExampleConstraintGPU;S=GPUSampling
    @testset "Resident public-example finish constraint, synthetic logits" begin
        grammar=G.Grammar(s->Int32.(codeunits(s)),512,511;max_depth=3,budget=5)
        for mode in (:valid,:disabled_wrong,:forced_wrong,:masked_nan,:inactive)
            state=nothing;examples=nothing;sampler=nothing;plan=nothing
            logits=Mantle.Buffer(device,zeros(Float32,512))
            try
                target=mode in (:disabled_wrong,:forced_wrong) ? Float32[0,0] : Float32[0.5,0.5]
                bank=RPNExampleSemantics.Bank([(;input=Float32[3,4],output=target)];max_depth=3)
                state=D.State(device,grammar)
                examples=E.State(device,grammar,bank;enabled=mode!=:disabled_wrong)
                sampler=S.State(device,512;topk=1,temperature=1f0,presence=0f0)
                mode==:inactive&&S.activate!(sampler,false)
                graph=Mantle.Graph(device);E.declare!(graph,logits,state,examples,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                tokens=reduce(vcat,grammar.sequences[[8,7,13,9,14,17,18]])
                expected=G.initial(grammar);failed=false
                for token in tokens
                    raw=fill(-5f0,512);raw[501]=99f0;raw[Int(token)+1]=2f0
                    mode==:masked_nan&&(raw[501]=NaN32)
                    logits[:]=raw;before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    if mode==:inactive
                        @test Array(logits)==raw
                        @test only(Array(sampler.counter))==0
                        @test Array(examples.stack)==zeros(Float32,6)
                        break
                    elseif mode==:masked_nan || (mode==:forced_wrong && only(Array(sampler.status))!=0)
                        @test only(Array(sampler.status))==1
                        @test only(Array(state.rejected))==1
                        @test only(Array(state.done))==0
                        @test only(Array(examples.raw_status))==(mode==:masked_nan ? 1 : 0)
                        failed=true;break
                    end
                    @test only(Array(sampler.status))==0
                    @test only(Array(sampler.selected))==token
                    expected=G.transition(grammar,expected,token)
                    observed=(only(Array(state.node)),only(Array(state.depth)),only(Array(state.operations)),
                        only(Array(state.done)),only(Array(state.rejected)))
                    @test observed==expected
                end
                if mode in (:valid,:disabled_wrong)
                    @test only(Array(state.done))==1
                    @test Array(examples.stack)[[1,4]]==Float32[0.5,0.5]
                    @test all(iszero,Array(examples.status))
                    previous=Array(examples.stack)
                    for extra in 1:3
                        raw=fill(-5f0,512);raw[501]=99f0;logits[:]=raw
                        before=Mantle.submissions(device);Mantle.run!(plan)
                        @test Mantle.submissions(device)-before==1
                        @test only(Array(sampler.status))==0
                        @test only(Array(sampler.selected))==500
                        @test only(Array(state.done))==1
                        @test Array(examples.stack)==previous
                    end
                elseif mode in (:masked_nan,:forced_wrong)
                    @test failed
                end
            finally
                plan===nothing||Mantle.free!(plan)
                sampler===nothing||close(sampler)
                examples===nothing||close(examples)
                state===nothing||close(state)
                Mantle.free!(logits)
            end
        end
    end
end
