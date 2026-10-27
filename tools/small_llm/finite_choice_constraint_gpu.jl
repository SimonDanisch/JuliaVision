isdefined(@__MODULE__,:FiniteChoiceConstraint)||include("finite_choice_constraint.jl")
isdefined(@__MODULE__,:SafeReasoningBudget)||include("safe_reasoning_budget.jl")
isdefined(@__MODULE__,:GPUSampling)||include("gpu_sampling.jl")

module FiniteChoiceConstraintGPU
import ..FiniteChoiceConstraint, ..SafeReasoningBudget, ..GPUSampling
import Mantle
import KernelInterface as KI

struct State
    enabled
    node
    done
    rejected
    offsets
    labels
    targets
    terminal
    vocabulary::Int
    eos::Int32
end

function State(device,g::FiniteChoiceConstraint.Grammar;enabled=true)
    enabled isa Bool||throw(ArgumentError("Boolean control"))
    owned=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        State(upload(Int32[enabled]),upload(Int32[1]),upload(Int32[0]),upload(Int32[0]),
            upload(g.trie.offsets),upload(g.trie.labels),upload(g.trie.targets),upload(g.trie.terminal),Int(g.trie.vocabulary),g.eos)
    catch
        foreach(Mantle.free!,reverse(owned));rethrow()
    end
end
Base.close(s::State)=(foreach(Mantle.free!,(s.enabled,s.node,s.done,s.rejected,s.offsets,s.labels,s.targets,s.terminal));nothing)

function mask!(logits,enabled,node,done,offsets,labels,targets,active,n::Int32,eos::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n&&active[1]!=0&&enabled[1]!=0
        @inbounds logits[i]=FiniteChoiceConstraint.constrained_value(offsets,labels,targets,node[1],done[1],eos,i-Int32(1),logits[i])
    end
    nothing
end

function advance!(enabled,node,done,rejected,offsets,labels,targets,terminal,active,status,selected,eos::Int32)
    if KI.get_global_id().x==1&&active[1]!=0&&enabled[1]!=0
        @inbounds if status[1]!=0
            node[1]=Int32(-1);rejected[1]=Int32(1)
        else
            a,b,c=FiniteChoiceConstraint.transition(offsets,labels,targets,terminal,node[1],done[1],rejected[1],eos,selected[1])
            node[1]=a;done[1]=b;rejected[1]=c;status[1]=max(status[1],c)
        end
    end
    nothing
end

"""Entire sparse mask/sample/transition inside the existing decode recording.
No host token feedback. Raw validity is checked BEFORE masking; errors stick.
This enforces canonical finite support only, not correct choice or Julia code.
"""
function declare!(graph,logits,s::State,sampler)
    length(logits)==s.vocabulary==sampler.vocabulary||throw(DimensionMismatch("Vocabulary"))
    Mantle.dispatch!(graph,SafeReasoningBudget.raw_validity!,
        (sampler.status,logits,Int32(s.vocabulary)),256;group=256,name="choice.raw_validity")
    Mantle.dispatch!(graph,mask!,
        (logits,s.enabled,s.node,s.done,s.offsets,s.labels,s.targets,sampler.active,Int32(s.vocabulary),s.eos),
        s.vocabulary;group=256,name="choice.mask")
    GPUSampling.declare!(graph,logits,sampler)
    Mantle.dispatch!(graph,advance!,
        (s.enabled,s.node,s.done,s.rejected,s.offsets,s.labels,s.targets,s.terminal,sampler.active,sampler.status,sampler.selected,s.eos),
        1;group=1,name="choice.advance")
    nothing
end
end
