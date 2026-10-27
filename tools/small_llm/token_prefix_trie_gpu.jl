isdefined(@__MODULE__, :TokenPrefixTrie) || include("token_prefix_trie.jl")
isdefined(@__MODULE__, :SafeReasoningBudget) || include("safe_reasoning_budget.jl")
module TokenPrefixTrieGPU
import ..TokenPrefixTrie
import ..SafeReasoningBudget
import Mantle
import KernelInterface as KI

struct State
    node
    offsets
    labels
    targets
    terminal
    vocabulary::Int
end

function State(device, trie::TokenPrefixTrie.Trie)
    buffers = Any[]
    upload(x) = (b = Mantle.Buffer(device, x); push!(buffers,b); b)
    try
        State(upload(Int32[1]),upload(trie.offsets),upload(trie.labels),
            upload(trie.targets),upload(trie.terminal),Int(trie.vocabulary))
    catch
        foreach(Mantle.free!,buffers);rethrow()
    end
end
Base.close(s::State) = (foreach(Mantle.free!, (s.node,s.offsets,s.labels,s.targets,s.terminal));nothing)

function mask!(logits,node,offsets,labels,targets,active,n::Int32)
    i = Int32(KI.get_global_id().x)
    if i <= n && active[1] != 0
        @inbounds logits[i] = TokenPrefixTrie.constrained_logit(
            offsets,labels,targets,node[1],i-Int32(1),logits[i])
    end
    nothing
end

function advance!(node,offsets,labels,targets,terminal,active,status,selected)
    if KI.get_global_id().x == 1 && active[1] != 0
        @inbounds node[1] = status[1] == 0 ?
            TokenPrefixTrie.next_node(offsets,labels,targets,terminal,node[1],selected[1]) : Int32(-1)
    end
    nothing
end

"""Place before the sampler. Preserve raw validity before masking can hide it.

Call declare_after! after that SAME sampler. Replays must restore model logits
before this mask, as for ordinary sampling. No CPU token/state polling.
"""
function declare_before!(graph,logits,state::State,sampler)
    length(logits) == state.vocabulary == sampler.vocabulary || throw(DimensionMismatch("trie vocabulary"))
    Mantle.dispatch!(graph,SafeReasoningBudget.raw_validity!,
        (sampler.status,logits,Int32(state.vocabulary)),256;group=256,name="token_prefix.raw_validity")
    Mantle.dispatch!(graph,mask!,
        (logits,state.node,state.offsets,state.labels,state.targets,sampler.active,Int32(state.vocabulary)),
        state.vocabulary;group=256,name="token_prefix.mask")
    nothing
end
function declare_after!(graph,state::State,sampler)
    state.vocabulary == sampler.vocabulary || throw(DimensionMismatch("trie vocabulary"))
    Mantle.dispatch!(graph,advance!,
        (state.node,state.offsets,state.labels,state.targets,state.terminal,
         sampler.active,sampler.status,sampler.selected),1;group=1,name="token_prefix.advance")
    nothing
end
end
