isdefined(@__MODULE__,:RPNPrefixGrammarGPU)||include("rpn_prefix_grammar_gpu.jl")
using Test
function test_rpn_native_tokens_gpu(device,grammar)
    G=RPNPrefixGrammar;D=RPNPrefixGrammarGPU;S=GPUSampling
    n=Int(grammar.vocabulary)
    labelset=Set(grammar.labels)
    illegal=findfirst(i->!(Int32(i-1) in labelset),1:n)
    illegal===nothing&&error("fixture needs a token outside the grammar")
    @testset "Full-vocabulary native-token grammar; trusted logits only" begin
        logits=Mantle.Buffer(device,zeros(Float32,n));state=nothing;sampler=nothing;plan=nothing
        try
            state=D.State(device,grammar)
            sampler=S.State(device,n;topk=1,temperature=1f0,presence=0f0)
            graph=Mantle.Graph(device);D.declare!(graph,logits,state,sampler)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
            tokens=reduce(vcat,grammar.sequences[[1,2,11,13,17,18]])
            expected=G.initial(grammar)
            for token in tokens
                raw=fill(-80f0,n);raw[illegal]=50f0;raw[Int(token)+1]=1f0
                logits[:]=raw
                before=Mantle.submissions(device);Mantle.run!(plan)
                @test Mantle.submissions(device)-before==1
                @test only(Array(sampler.selected))==token
                @test only(Array(sampler.status))==0
                expected=G.transition(grammar,expected,token)
                observed=(only(Array(state.node)),only(Array(state.depth)),only(Array(state.operations)),
                    only(Array(state.done)),only(Array(state.rejected)))
                @test observed==expected
            end
            @test only(Array(state.done))==1
            @test only(Array(state.depth))==1
            @test only(Array(sampler.counter))==length(tokens)
        finally
            plan===nothing||Mantle.free!(plan)
            sampler===nothing||close(sampler)
            state===nothing||close(state)
            Mantle.free!(logits)
        end
    end
end
