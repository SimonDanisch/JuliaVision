# Experimental alternative, NOT used by any frozen quality study.
# Exact block top-k followed by a sorted-list merge: four graph dispatches
# including the unchanged RNG/nucleus selection and score feedback.
isdefined(@__MODULE__,:GPUSampling) || include("gpu_sampling.jl")
module TiledGPUSampling
import Mantle, BonsaiRunner
import KernelInterface as KI
import ..GPUSampling

function partial_topk!(values,indices,logits,seen,presence::Float32,n::Int32,k::Int32)
    vals=KI.localmemory(Float32,Val((256,)),Val(1))
    ids=KI.localmemory(Int32,Val((256,)),Val(2))
    t=Int32(KI.get_local_id().x-1);block=Int32(KI.get_group_id().x-1)
    i=block*Int32(256)+t
    v=-Inf32;index=typemax(Int32)
    if i<n
        @inbounds v=logits[i+Int32(1)]-(seen[i+Int32(1)]>0 ? presence : 0f0)
        index=i
    end
    @inbounds vals[t+Int32(1)]=v
    @inbounds ids[t+Int32(1)]=index
    KI.barrier()
    # Bitonic network. Lower slots finish in descending Julia argmax order,
    # including NaNs and deterministic lowest-token-id ties.
    width=Int32(2)
    while width<=Int32(256)
        distance=width>>Int32(1)
        while distance>Int32(0)
            peer=t ⊻ distance
            if t<peer
                @inbounds begin
                    a=vals[t+Int32(1)];ia=ids[t+Int32(1)]
                    b=vals[peer+Int32(1)];ib=ids[peer+Int32(1)]
                    high_first=(t & width)==Int32(0)
                    swap=high_first ? BonsaiRunner._better(a,ia,b,ib) : BonsaiRunner._better(b,ib,a,ia)
                    if swap
                        vals[t+Int32(1)]=b;ids[t+Int32(1)]=ib
                        vals[peer+Int32(1)]=a;ids[peer+Int32(1)]=ia
                    end
                end
            end
            KI.barrier()
            distance>>=Int32(1)
        end
        width<<=Int32(1)
    end
    if t<k
        slot=block*k+t+Int32(1)
        @inbounds values[slot]=vals[t+Int32(1)]
        @inbounds indices[slot]=ids[t+Int32(1)]
    end
    nothing
end

function merge_topk!(values,indices,partial_values,partial_indices,blocks::Int32,k::Int32)
    vals=KI.localmemory(Float32,Val((256,)),Val(1))
    ids=KI.localmemory(Int32,Val((256,)),Val(2))
    cursors=KI.localmemory(Int32,Val((2048,)),Val(3))
    t=Int32(KI.get_local_id().x-1)
    block=t
    while block<blocks
        @inbounds cursors[block+Int32(1)]=Int32(0)
        block+=Int32(256)
    end
    # Cursors are workgroup LOCAL memory, not storage-buffer memory requiring
    # a wider memory-semantics fence. No atomic store/load idiom is used.
    KI.barrier()
    rank=Int32(1)
    while rank<=k
        best=-Inf32;index=typemax(Int32);block=t
        while block<blocks
            cursor=@inbounds cursors[block+Int32(1)]
            if cursor<k
                slot=block*k+cursor+Int32(1)
                value=@inbounds partial_values[slot]
                candidate=@inbounds partial_indices[slot]
                if BonsaiRunner._better(best,index,value,candidate)
                    best=value;index=candidate
                end
            end
            block+=Int32(256)
        end
        best,index=BonsaiRunner._argmax_group(vals,ids,t,best,index)
        if t==Int32(0)
            @inbounds begin
                values[rank]=best;indices[rank]=index
                owner=index÷Int32(256)+Int32(1)
                cursors[owner]+=Int32(1)
            end
        end
        KI.barrier()
        rank+=Int32(1)
    end
    nothing
end

# Library-owned resource contracts, identical for every backend. Shared-memory
# implementation details do not touch graph resources. Some Julia 1.12 LLVM
# shared-pointer helpers have no reflective IR; do not guess their effects or
# monkeypatch the engine. State the actual public arguments' effects here.
Mantle.argument_usage(::Type{typeof(partial_topk!)},
    ::Type{<:Tuple{Any,Any,Any,Any,Float32,Int32,Int32}})=
    (Mantle.WRITE,Mantle.WRITE,Mantle.READ,Mantle.READ,Mantle.NOTOUCH,Mantle.NOTOUCH,Mantle.NOTOUCH)
Mantle.argument_usage(::Type{typeof(merge_topk!)},
    ::Type{<:Tuple{Any,Any,Any,Any,Int32,Int32}})=
    (Mantle.WRITE,Mantle.WRITE,Mantle.READ,Mantle.READ,Mantle.NOTOUCH,Mantle.NOTOUCH)

function declare!(graph,logits,state::GPUSampling.State)
    length(logits)==state.vocabulary || throw(DimensionMismatch("sampler vocabulary"))
    n=state.vocabulary;k=state.topk;blocks=cld(n,256)
    blocks<=2048 || throw(ArgumentError("tiled sampler shared-memory limit: vocabulary <=524288"))
    values=Mantle.Transient.Buffer(graph,Float32,(k,))
    indices=Mantle.Transient.Buffer(graph,Int32,(k,))
    partial_values=Mantle.Transient.Buffer(graph,Float32,(blocks*k,))
    partial_indices=Mantle.Transient.Buffer(graph,Int32,(blocks*k,))
    Mantle.dispatch!(graph,partial_topk!,
        (partial_values,partial_indices,logits,state.seen,state.presence,Int32(n),Int32(k)),
        blocks*256;group=256,name="sampling_tiled.partial")
    Mantle.dispatch!(graph,merge_topk!,
        (values,indices,partial_values,partial_indices,Int32(blocks),Int32(k)),
        256;group=256,name="sampling_tiled.merge")
    Mantle.dispatch!(graph,GPUSampling.sample_topk!,
        (state.selected,state.status,state.seen,state.counter,state.active,values,indices,
            Int32(k),state.temperature,state.top_p,state.seed,state.fallback),
        1;group=1,name="sampling_tiled.select")
    Mantle.dispatch!(graph,GPUSampling.force_selection!,
        (logits,state.selected,state.active,Int32(n)),n;group=256,name="sampling_tiled.feedback")
    nothing
end
end
