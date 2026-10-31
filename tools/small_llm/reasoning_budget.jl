"""GPU-resident reasoning-budget controller; steering, NOT factual evidence."""
module ReasoningBudget
import Mantle
import KernelInterface as KI

struct State
    done
    force
    forced_count
    vocabulary::Int
    end_id::Int32
    limit::UInt32
end

function State(device,vocabulary,end_id;limit=1024)
    vocabulary isa Integer && !(vocabulary isa Bool) && 0<vocabulary<=typemax(Int32) &&
        end_id isa Integer && !(end_id isa Bool) && 0<=end_id<vocabulary || throw(ArgumentError("single-token closing marker required"))
    limit isa Integer && !(limit isa Bool) && 0<=limit<=typemax(UInt32) || throw(ArgumentError("invalid thinking budget"))
    owned=Any[]
    buffer()=begin b=Mantle.Buffer(device,Int32[0]);push!(owned,b);b end
    try
        State(buffer(),buffer(),buffer(),Int(vocabulary),Int32(end_id),UInt32(limit))
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end
Base.close(s::State)=(foreach(Mantle.free!,(s.done,s.force,s.forced_count));nothing)

function decide!(done,force,count,active,draws,selected,limit::UInt32,end_id::Int32)
    if KI.get_global_id().x==1
        @inbounds begin
            force[1]=Int32(0)
            if active[1]!=0
                draws[1]>0 && selected[1]==end_id && (done[1]=Int32(1))
                if done[1]==0 && draws[1]>=limit
                    force[1]=Int32(1);count[1]+=Int32(1)
                end
            end
        end
    end
    nothing
end

function close_thinking!(logits,force,end_id::Int32,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n && force[1]!=0
        @inbounds logits[i]=i-Int32(1)==end_id ? 0f0 : -Inf32
    end
    nothing
end

"""Declare BEFORE the existing sampler in the same graph.

The sampler's previous selected token detects natural closure; its draw counter
excludes intermediate prompt steps. On budget exhaustion only the known closing
token is allowed for ONE draw. The ordinary sampler still consumes that draw,
and subsequent tokens are unconstrained. No CPU token polling/feedback here.
"""
function declare!(graph,logits,state::State,sampler)
    length(logits)==state.vocabulary==sampler.vocabulary || throw(DimensionMismatch("budget vocabulary"))
    Mantle.dispatch!(graph,decide!,
        (state.done,state.force,state.forced_count,sampler.active,sampler.counter,sampler.selected,state.limit,state.end_id),
        1;group=1,name="reasoning_budget.decide")
    Mantle.dispatch!(graph,close_thinking!,(logits,state.force,state.end_id,Int32(state.vocabulary)),
        state.vocabulary;group=256,name="reasoning_budget.close")
    nothing
end
end
