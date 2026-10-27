isdefined(@__MODULE__,:RPNExampleConstraint)||include("rpn_example_constraint.jl")
isdefined(@__MODULE__,:RPNPrefixGrammarGPU)||include("rpn_prefix_grammar_gpu.jl")
module RPNExampleConstraintGPU
import ..RPNExampleConstraint
import ..RPNExampleSemantics
import ..RPNPrefixGrammarGPU
import ..RPNPrefixGrammar
import ..SafeReasoningBudget
import ..GPUSampling
import Mantle
import KernelInterface as KI

struct State
    enabled
    stack
    features
    expected
    status
    allowed
    nonfinish
    raw_status
    examples::Int32
    max_depth::Int32
    atol::Float32
    rtol::Float32
end

function State(device,grammar::RPNPrefixGrammar.Grammar,bank::RPNExampleSemantics.Bank;enabled=true)
    enabled isa Bool||throw(ArgumentError("Boolean example control required"))
    bank.max_depth==grammar.max_depth||throw(ArgumentError("stack bound mismatch"))
    n=length(bank.expected);0<n<=65536||throw(ArgumentError("invalid example count"))
    owned=Any[];upload(x)=(b=Mantle.Buffer(device,x);push!(owned,b);b)
    try
        State(upload(Int32[enabled]),upload(zeros(Float32,Int(bank.max_depth)*n)),
            upload(bank.features),upload(bank.expected),upload(zeros(Int32,n)),
            upload(Int32[0]),upload(RPNExampleConstraint.nonfinish_nodes(grammar)),upload(Int32[0]),
            Int32(n),bank.max_depth,bank.atol,bank.rtol)
    catch
        foreach(Mantle.free!,reverse(owned));rethrow()
    end
end
Base.close(s::State)=(foreach(Mantle.free!,(s.enabled,s.stack,s.features,s.expected,s.status,s.allowed,s.nonfinish,s.raw_status));nothing)

function propagate_raw!(status,raw_status)
    if KI.get_global_id().x==1
        @inbounds status[1]=max(status[1],raw_status[1])
    end
    nothing
end

function check_finish!(allowed,stack,expected,status,depth,active,max_depth::Int32,n::Int32,atol::Float32,rtol::Float32)
    if KI.get_global_id().x==1
        @inbounds allowed[1]=active[1]!=0 && RPNExampleConstraint.finish_allowed(
            stack,expected,status,depth[1],max_depth,n,atol,rtol) ? Int32(1) : Int32(0)
    end
    nothing
end

function mask!(logits,enabled,grammar_enabled,node,offsets,labels,targets,nonfinish,allowed,active,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds if active[1]!=0 && enabled[1]!=0 && grammar_enabled[1]!=0
            RPNExampleConstraint.token_allowed(offsets,labels,targets,nonfinish,node[1],i-Int32(1),allowed[1]!=0)||
                (logits[i]=-Inf32)
        end
    end
    nothing
end

function advance_examples!(stack,features,status,node,depth,offsets,labels,targets,terminal,active,grammar_enabled,
        sampler_status,selected,max_depth::Int32,n::Int32)
    example=Int32(KI.get_global_id().x)
    if example<=n
        @inbounds if active[1]!=0 && grammar_enabled[1]!=0 && sampler_status[1]==0 && status[example]==0
            action=RPNExampleConstraint.committed_action(offsets,labels,targets,terminal,node[1],selected[1])
            if 1<=action<=RPNPrefixGrammar.FINISH
                status[example]=RPNExampleSemantics.step!(stack,features,example,action,depth[1],max_depth)
            end
        end
    end
    nothing
end

"""One resident graph: raw validation, both masks, sampling, example + grammar state.

No host token feedback. Disabled-example control uses identical declarations.
Example disagreement at a forced finish is a recorded failed attempt; the
caller must not force EOS or repair arithmetic to claim a success.
"""
function declare!(graph,logits,g::RPNPrefixGrammarGPU.State,s::State,sampler)
    length(logits)==g.vocabulary==sampler.vocabulary||throw(DimensionMismatch("vocabulary"))
    g.max_depth==s.max_depth||throw(DimensionMismatch("example stack"))
    Mantle.dispatch!(graph,SafeReasoningBudget.raw_validity!,
        (s.raw_status,logits,Int32(g.vocabulary)),256;group=256,name="example.raw_validity")
    Mantle.dispatch!(graph,propagate_raw!,
        (sampler.status,s.raw_status),1;group=1,name="example.propagate_raw_validity")
    Mantle.dispatch!(graph,RPNPrefixGrammarGPU.mask!,
        (logits,g.enabled,g.node,g.offsets,g.labels,g.targets,sampler.active,Int32(g.vocabulary)),
        g.vocabulary;group=256,name="example.grammar_mask")
    Mantle.dispatch!(graph,check_finish!,
        (s.allowed,s.stack,s.expected,s.status,g.depth,sampler.active,s.max_depth,s.examples,s.atol,s.rtol),1;group=1,name="example.finish_check")
    Mantle.dispatch!(graph,mask!,
        (logits,s.enabled,g.enabled,g.node,g.offsets,g.labels,g.targets,s.nonfinish,s.allowed,sampler.active,Int32(g.vocabulary)),
        g.vocabulary;group=256,name="example.finish_mask")
    GPUSampling.declare!(graph,logits,sampler)
    Mantle.dispatch!(graph,advance_examples!,
        (s.stack,s.features,s.status,g.node,g.depth,g.offsets,g.labels,g.targets,g.terminal,sampler.active,g.enabled,
         sampler.status,sampler.selected,s.max_depth,s.examples),s.examples;group=64,name="example.advance")
    Mantle.dispatch!(graph,RPNPrefixGrammarGPU.advance!,
        (g.enabled,g.node,g.depth,g.operations,g.done,g.rejected,g.offsets,g.labels,g.targets,g.terminal,g.roots,g.arities,
         sampler.active,sampler.status,sampler.selected,g.max_depth,g.budget),1;group=1,name="example.grammar_advance")
    nothing
end
end
