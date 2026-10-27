include("finite_choice_constraint_gpu.jl")
using Test
import Mantle

function test_finite_choice_constraint_gpu(device)
    G=FiniteChoiceConstraint;D=FiniteChoiceConstraintGPU;S=GPUSampling
    g=G.Grammar(s->Int32.(codeunits(s)),ids->String(UInt8.(ids)),["a","ab"],512,511)
    @testset "Finite choices: shared kernels and one recorded submission" begin
        for mode in (:valid,:masked_nan,:masked_inf,:all_masked,:inactive,:disabled)
            logits=Mantle.Buffer(device,zeros(Float32,512));state=nothing;sampler=nothing;plan=nothing
            try
                state=D.State(device,g;enabled=mode!=:disabled)
                sampler=S.State(device,512;topk=1,temperature=1f0,presence=0f0,fallback=511)
                mode==:inactive&&S.activate!(sampler,false)
                graph=Mantle.Graph(device);D.declare!(graph,logits,state,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                expected=(Int32(1),Int32(0),Int32(0))
                for token in vcat(g.sequences[2],Int32[511,511])
                    raw=fill(-5f0,512);raw[501]=99f0;raw[Int(token)+1]=2f0
                    mode==:masked_nan&&(raw[501]=NaN32)
                    mode==:masked_inf&&(raw[501]=Inf32)
                    if mode==:all_masked;fill!(raw,-Inf32);raw[501]=99f0;end
                    logits[:]=raw;before=Mantle.submissions(device);Mantle.run!(plan)
                    @test Mantle.submissions(device)-before==1
                    if mode in (:masked_nan,:masked_inf,:all_masked)
                        @test only(Array(sampler.status))==1
                        @test only(Array(state.rejected))==1
                        @test only(Array(state.node))==-1
                        # Valid later input must not clear rejection.
                        logits[:]=zeros(Float32,512);Mantle.run!(plan)
                        @test only(Array(sampler.status))==1
                        @test only(Array(state.rejected))==1
                        break
                    elseif mode==:inactive
                        @test Array(logits)==raw
                        @test only(Array(sampler.counter))==0
                        @test only(Array(state.node))==1
                        break
                    elseif mode==:disabled
                        @test only(Array(sampler.selected))==500
                        @test only(Array(sampler.status))==0
                        @test only(Array(state.node))==1
                        break
                    end
                    @test only(Array(sampler.selected))==token
                    @test only(Array(sampler.status))==0
                    expected=G.transition(g,expected,token)
                    @test (only(Array(state.node)),only(Array(state.done)),only(Array(state.rejected)))==expected
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
