isdefined(@__MODULE__,:TokenPrefixTrieGPU) || include("token_prefix_trie_gpu.jl")
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
using Test
function test_token_prefix_trie_gpu(device)
    T=TokenPrefixTrie;G=TokenPrefixTrieGPU;S=GPUSampling
    @testset "Recorded sparse prefix sampler integration" begin
        for raw in (Float32[1,9,3,2,8,7,6,5],
                    Float32[1,NaN,3,2,8,7,6,5],
                    Float32[1,Inf,3,2,8,7,6,5],fill(-Inf32,8),
                    Float32[-Inf,9,3,2,8,-Inf,6,5])
            trie=T.Trie([[0,2,3],[0,2,4],[5,1]],8)
            logits=Mantle.Buffer(device,raw);state=nothing;sampler=nothing;plan=nothing
            try
                state=G.State(device,trie)
                sampler=S.State(device,8;topk=1,temperature=1f0,presence=0f0)
                graph=Mantle.Graph(device)
                G.declare_before!(graph,logits,state,sampler)
                S.declare!(graph,logits,sampler)
                G.declare_after!(graph,state,sampler)
                plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
                expected=Int32(1)
                for step in 1:4
                    logits[:]=raw
                    before_submission=Mantle.submissions(device)
                    Mantle.run!(plan)
                    @test Mantle.submissions(device)-before_submission==1
                    token=only(Array(sampler.selected))
                    status=only(Array(sampler.status))
                    if any(isnan,raw)||any(==(Inf32),raw)||all(==(-Inf32),raw)||
                            (raw[1]==-Inf32 && raw[6]==-Inf32)
                        @test status==1
                        @test only(Array(state.node)) == -1
                    else
                        scores=[T.constrained_logit(trie.offsets,trie.labels,trie.targets,
                            expected,Int32(i-1),raw[i]) for i in 1:8]
                        reference=S.reference(scores;topk=1,temperature=1f0,presence=0f0)
                        @test token==reference.selected
                        @test status==0
                        expected=T.next_node(trie,expected,token)
                        @test only(Array(state.node))==expected
                    end
                end
                S.activate!(sampler,false)
                before=only(Array(state.node));logits[:]=ones(Float32,8)
                Mantle.run!(plan)
                @test only(Array(state.node))==before
                @test Array(logits)==ones(Float32,8)
            finally
                plan===nothing||Mantle.free!(plan)
                sampler===nothing||close(sampler)
                state===nothing||close(state)
                Mantle.free!(logits)
            end
        end
    end
end
