isdefined(@__MODULE__,:RPNPrefixGrammarGPU)||include("rpn_prefix_grammar_gpu.jl")
using Test
function test_rpn_disabled_sampler_identity(device)
    G=RPNPrefixGrammar;D=RPNPrefixGrammarGPU;S=GPUSampling
    grammar=G.Grammar(s->Int32.(codeunits(s)),512,511)
    @testset "Disabled grammar versus ordinary stochastic sampler" begin
        for presence in (0f0,1.5f0),seed in (UInt32(1),UInt32(20261024))
            a=nothing;b=nothing;state=nothing;pa=nothing;pb=nothing
            la=Mantle.Buffer(device,zeros(Float32,512));lb=Mantle.Buffer(device,zeros(Float32,512))
            try
                a=S.State(device,512;topk=20,top_p=0.95f0,temperature=0.6f0,presence,seed,fallback=511)
                b=S.State(device,512;topk=20,top_p=0.95f0,temperature=0.6f0,presence,seed,fallback=511)
                state=D.State(device,grammar;enabled=false)
                ga=Mantle.Graph(device);S.declare!(ga,la,a)
                gb=Mantle.Graph(device);D.declare!(gb,lb,state,b)
                pa=Mantle.record!(Mantle.Plan(ga;budget=Inf));pb=Mantle.record!(Mantle.Plan(gb;budget=Inf))
                for draw in 0:7
                    raw=Float32[sin(Float32(i)*0.17f0)+cos(Float32(i+draw)*0.031f0) for i in 1:512]
                    la[:]=raw;lb[:]=raw
                    before=Mantle.submissions(device);Mantle.run!(pa)
                    @test Mantle.submissions(device)-before==1
                    before=Mantle.submissions(device);Mantle.run!(pb)
                    @test Mantle.submissions(device)-before==1
                    @test Array(a.selected)==Array(b.selected)
                    @test Array(a.counter)==Array(b.counter)
                    @test Array(a.status)==Array(b.status)==Int32[0]
                    @test Array(a.seen)==Array(b.seen)
                    @test isequal(Array(la),Array(lb))
                    @test only(Array(state.node))==grammar.roots[1]
                    @test only(Array(state.operations))==0
                    @test only(Array(state.rejected))==0
                end
            finally
                pa===nothing||Mantle.free!(pa);pb===nothing||Mantle.free!(pb)
                a===nothing||close(a);b===nothing||close(b);state===nothing||close(state)
                Mantle.free!(la);Mantle.free!(lb)
            end
        end
    end
end
