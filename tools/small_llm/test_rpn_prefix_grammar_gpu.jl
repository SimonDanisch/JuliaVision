isdefined(@__MODULE__,:RPNPrefixGrammarGPU)||include("rpn_prefix_grammar_gpu.jl")
using Test
function test_rpn_prefix_grammar_gpu(device)
    G=RPNPrefixGrammar;D=RPNPrefixGrammarGPU;S=GPUSampling
    @testset "Shared postfix grammar + device sampler, trusted logits only" begin
        grammar=G.Grammar(s->Int32.(codeunits(s)),512,511;max_depth=3,budget=3)
        for mode in (:valid,:masked_nan,:masked_inf,:all_masked,:inactive,:disabled)
            state=nothing;sampler=nothing;plan=nothing
            logits=Mantle.Buffer(device,zeros(Float32,512))
            try
                state=D.State(device,grammar;enabled=mode!=:disabled)
                sampler=S.State(device,512;topk=1,temperature=1f0,presence=0f0)
                mode==:inactive&&S.activate!(sampler,false)
                graph=Mantle.Graph(device);D.declare!(graph,logits,state,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                # Two pushes + square exhaust budget; sub must then reduce,
                # followed by finish/EOS. No task solution is tested here.
                tokens=reduce(vcat,grammar.sequences[[1,2,11,13,17,18]])
                expected=G.initial(grammar)
                for token in tokens
                    raw=fill(-5f0,512);raw[501]=99f0;raw[Int(token)+1]=2f0
                    mode==:masked_nan&&(raw[501]=NaN32)
                    mode==:masked_inf&&(raw[501]=Inf32)
                    if mode==:all_masked;fill!(raw,-Inf32);raw[501]=99f0;end
                    logits[:]=raw
                    before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    if mode in (:masked_nan,:masked_inf,:all_masked)
                        @test only(Array(sampler.status))==1
                        @test only(Array(state.rejected))==1
                        @test only(Array(state.node))==-1
                        break
                    elseif mode==:inactive
                        @test Array(logits)==raw
                        @test only(Array(state.node))==expected[1]
                        @test only(Array(sampler.counter))==0
                        break
                    elseif mode==:disabled
                        @test only(Array(sampler.selected))==500
                        @test only(Array(sampler.status))==0
                        @test only(Array(state.node))==expected[1]
                        @test only(Array(state.operations))==0
                        break
                    end
                    @test only(Array(sampler.selected))==token
                    @test only(Array(sampler.status))==0
                    expected=G.transition(grammar,expected,token)
                    observed=(only(Array(state.node)),only(Array(state.depth)),only(Array(state.operations)),
                        only(Array(state.done)),only(Array(state.rejected)))
                    @test observed==expected
                end
                mode==:valid&&(@test only(Array(state.done))==1)
            finally
                plan===nothing||Mantle.free!(plan)
                sampler===nothing||close(sampler)
                state===nothing||close(state)
                Mantle.free!(logits)
            end
        end
    end
end
