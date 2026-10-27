isdefined(@__MODULE__,:TokenPrefixTrieGPU) || include("token_prefix_trie_gpu.jl")
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
using Test

"""Trusted full-vocabulary fixture; readbacks here verify, not drive, transitions."""
function test_callable_prefix_graph(device,catalog)
    T=TokenPrefixTrie;G=TokenPrefixTrieGPU;S=GPUSampling
    trie=catalog.trie;n=Int(trie.vocabulary)
    raw=Float32[Float32(mod(i*104729,65521))/65521f0 for i in 1:n]
    logits=Mantle.Buffer(device,raw);state=nothing;sampler=nothing;plan=nothing
    @testset "Full public API catalog recorded constraint" begin
        try
            state=G.State(device,trie)
            sampler=S.State(device,n;topk=1,temperature=1f0,presence=0f0)
            graph=Mantle.Graph(device)
            G.declare_before!(graph,logits,state,sampler)
            S.declare!(graph,logits,sampler)
            G.declare_after!(graph,state,sampler)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
            expected=Int32(1);tokens=Int32[]
            longest=maximum(length,catalog.sequences)
            for step in 1:longest+1
                logits[:]=raw
                submissions=Mantle.submissions(device)
                Mantle.run!(plan)
                @test Mantle.submissions(device)-submissions==1
                token=only(Array(sampler.selected))
                @test only(Array(sampler.status))==0
                if expected != 0
                    # At most the sparse outgoing edges are admissible. The
                    # oracle deliberately scans them, not the device search.
                    edges=trie.offsets[expected]:trie.offsets[expected+1]-1
                    scores=[raw[Int(trie.labels[e])+1] for e in edges]
                    best=maximum(scores)
                    labels=[trie.labels[e] for e in edges if raw[Int(trie.labels[e])+1]==best]
                    @test token==minimum(labels)
                    push!(tokens,token)
                    expected=T.next_node(trie,expected,token)
                    @test only(Array(state.node))==expected
                else
                    @test token==Int32(argmax(raw)-1)
                    @test only(Array(state.node))==0
                    break
                end
            end
            @test expected==0
            @test tokens in catalog.sequences
        finally
            plan===nothing||Mantle.free!(plan)
            sampler===nothing||close(sampler)
            state===nothing||close(state)
            Mantle.free!(logits)
        end
    end
end
