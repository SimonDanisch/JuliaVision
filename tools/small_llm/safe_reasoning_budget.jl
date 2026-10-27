isdefined(@__MODULE__,:ReasoningBudget) || include("reasoning_budget.jl")
module SafeReasoningBudget
import ..ReasoningBudget
import Mantle
import KernelInterface as KI
const State=ReasoningBudget.State

# Preserve the existing sampler's sticky rejection flag BEFORE steering can
# overwrite NaN/+Inf/all-masked raw logits. One workgroup, one status writer.
function raw_validity!(status,logits,n::Int32)
    lane=Int32(KI.get_local_id().x)-Int32(1)
    partial=KI.localmemory(Int32,Val((512,)),Val(1))
    bad=Int32(0);finite=Int32(0);i=lane+Int32(1)
    while i<=n
        @inbounds x=logits[i]
        bad+=Int32(isnan(x)||x==Inf32);finite+=Int32(isfinite(x));i+=Int32(256)
    end
    @inbounds partial[lane+Int32(1)]=bad
    @inbounds partial[lane+Int32(257)]=finite
    KI.barrier();stride=Int32(128)
    while stride>0
        if lane<stride
            @inbounds partial[lane+Int32(1)]+=partial[lane+stride+Int32(1)]
            @inbounds partial[lane+Int32(257)]+=partial[lane+stride+Int32(257)]
        end
        KI.barrier();stride>>=1
    end
    if lane==0
        @inbounds status[1]=max(status[1],Int32(partial[1]>0 || partial[257]==0))
    end
    nothing
end

function declare!(graph,logits,state::State,sampler)
    length(logits)==state.vocabulary==sampler.vocabulary || throw(DimensionMismatch("budget vocabulary"))
    Mantle.dispatch!(graph,raw_validity!,(sampler.status,logits,Int32(state.vocabulary)),256;group=256,name="reasoning_budget.raw_validity")
    ReasoningBudget.declare!(graph,logits,state,sampler)
    nothing
end
end
