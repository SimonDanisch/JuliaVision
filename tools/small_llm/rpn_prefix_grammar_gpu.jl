isdefined(@__MODULE__,:RPNPrefixGrammar)||include("rpn_prefix_grammar.jl")
isdefined(@__MODULE__,:SafeReasoningBudget)||include("safe_reasoning_budget.jl")
isdefined(@__MODULE__,:GPUSampling)||include("gpu_sampling.jl")
module RPNPrefixGrammarGPU
import ..RPNPrefixGrammar
import ..TokenPrefixTrie
import ..SafeReasoningBudget
import ..GPUSampling
import Mantle
import KernelInterface as KI

struct State
    enabled
    node
    depth
    operations
    done
    rejected
    offsets
    labels
    targets
    terminal
    roots
    arities
    vocabulary::Int
    max_depth::Int32
    budget::Int32
end

function State(device,g::RPNPrefixGrammar.Grammar;enabled=true)
    enabled isa Bool||throw(ArgumentError("Boolean grammar control required"))
    owned=Any[]
    upload(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        State(upload(Int32[enabled]),upload(Int32[g.roots[1]]),upload(Int32[0]),upload(Int32[0]),upload(Int32[0]),upload(Int32[0]),
            upload(g.offsets),upload(g.labels),upload(g.targets),upload(g.terminal_action),upload(g.roots),upload(g.arities),
            Int(g.vocabulary),g.max_depth,g.budget)
    catch
        foreach(Mantle.free!,reverse(owned));rethrow()
    end
end
Base.close(s::State)=(foreach(Mantle.free!,(s.enabled,s.node,s.depth,s.operations,s.done,s.rejected,
    s.offsets,s.labels,s.targets,s.terminal,s.roots,s.arities));nothing)

function mask!(logits,enabled,node,offsets,labels,targets,active,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n && active[1]!=0 && enabled[1]!=0
        @inbounds logits[i]=TokenPrefixTrie.constrained_logit(offsets,labels,targets,node[1],i-Int32(1),logits[i])
    end
    nothing
end

function advance!(enabled,node,depth,operations,done,rejected,offsets,labels,targets,terminal,roots,arities,
        active,status,selected,max_depth::Int32,budget::Int32)
    if KI.get_global_id().x==1 && active[1]!=0 && enabled[1]!=0
        @inbounds begin
            if status[1]!=0
                node[1]=Int32(-1);rejected[1]=Int32(1)
            else
                a,b,c,d,e=RPNPrefixGrammar.transition(offsets,labels,targets,terminal,roots,arities,
                    node[1],depth[1],operations[1],done[1],rejected[1],selected[1],max_depth,budget)
                node[1]=a;depth[1]=b;operations[1]=c;done[1]=d;rejected[1]=e
                status[1]=max(status[1],e) # one invocation/writer, not a reduction
            end
        end
    end
    nothing
end

"""Declare the ENTIRE grammar/sampler transition inside the existing decode graph.

Raw neural validity is checked before masking. Grammar state advances from the
device-selected token after sampling, with no CPU token/state feedback.
"""
function declare!(graph,logits,s::State,sampler)
    length(logits)==s.vocabulary==sampler.vocabulary||throw(DimensionMismatch("grammar vocabulary"))
    Mantle.dispatch!(graph,SafeReasoningBudget.raw_validity!,
        (sampler.status,logits,Int32(s.vocabulary)),256;group=256,name="rpn.raw_validity")
    Mantle.dispatch!(graph,mask!,
        (logits,s.enabled,s.node,s.offsets,s.labels,s.targets,sampler.active,Int32(s.vocabulary)),
        s.vocabulary;group=256,name="rpn.mask")
    GPUSampling.declare!(graph,logits,sampler)
    Mantle.dispatch!(graph,advance!,
        (s.enabled,s.node,s.depth,s.operations,s.done,s.rejected,s.offsets,s.labels,s.targets,s.terminal,s.roots,s.arities,
         sampler.active,sampler.status,sampler.selected,s.max_depth,s.budget),1;group=1,name="rpn.advance")
    nothing
end
end
