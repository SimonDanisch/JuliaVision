"""Experimental exact top-k/nucleus sampler, declared inside the decode graph.

Uses the existing portable argmax reductions. No token/logit readback or host RNG
occurs between tokens. This hook replaces logits with one-hot selection scores;
do NOT use those scores for likelihood evaluation. It is opt-in and does not
change the runner's default greedy path.
"""
module GPUSampling
import Mantle, BonsaiRunner
import KernelInterface as KI

struct State
    active
    counter
    seen
    selected
    status
    vocabulary::Int
    topk::Int
    temperature::Float32
    top_p::Float32
    presence::Float32
    seed::UInt32
    fallback::Int32
end

function State(device,n;topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,seed=UInt32(1),fallback=0)
    temperature=Float32(temperature);top_p=Float32(top_p);presence=Float32(presence)
    0<n<=typemax(Int32) || throw(ArgumentError("invalid vocabulary"))
    1<=topk<=min(n,64) || throw(ArgumentError("topk must be <=64 and <=vocabulary"))
    isfinite(temperature)&&temperature>0 || throw(ArgumentError("temperature must be positive and finite"))
    isfinite(top_p)&&0<top_p<=1 || throw(ArgumentError("top_p must be in (0,1]"))
    isfinite(presence)&&presence>=0 || throw(ArgumentError("presence must be finite and nonnegative"))
    0<=fallback<n || throw(ArgumentError("fallback outside vocabulary"))
    owned=Any[]
    buffer(x)=begin b=Mantle.Buffer(device,x);push!(owned,b);b;end
    try
        State(buffer(Int32[1]),buffer(UInt32[0]),buffer(zeros(Int32,n)),
            buffer(Int32[0]),buffer(Int32[0]),Int(n),Int(topk),Float32(temperature),
            Float32(top_p),Float32(presence),UInt32(seed),Int32(fallback))
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end

Base.close(s::State)=(foreach(Mantle.free!,(s.active,s.counter,s.seen,s.selected,s.status));nothing)
activate!(s::State,active=true)=(s.active[:]=Int32[active];s)

@inline function uniform23(seed::UInt32,counter::UInt32)
    x=seed ⊻ (counter+UInt32(0x9e3779b9))
    x=(x ⊻ (x>>16))*UInt32(0x21f0aaad)
    x=(x ⊻ (x>>15))*UInt32(0x735a2d97)
    x=x ⊻ (x>>15)
    (Float32(x>>9)+0.5f0)*Float32(1/8388608)
end

function copy_adjusted!(scratch,logits,seen,presence::Float32,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds scratch[i]=logits[i]-(seen[i]>0 ? presence : 0f0))
    nothing
end

function save_and_mask!(values,indices,scratch,rank::Int32)
    if KI.get_global_id().x==1
        @inbounds begin
            i=indices[rank]+Int32(1)
            values[rank]=scratch[i]
            scratch[i]=-Inf32
        end
    end
    nothing
end

function sample_topk!(selected,status,seen,counter,active,values,indices,k::Int32,
                      temperature::Float32,top_p::Float32,seed::UInt32,fallback::Int32)
    if KI.get_global_id().x==1 && active[1]!=0
        @inbounds begin
            m=values[1];invalid=!isfinite(m);total=0f0
            for j in Int32(1):k
                invalid|=isnan(values[j]) || values[j]==Inf32
                total+=exp((values[j]-m)/temperature)
            end
            chosen=fallback
            if invalid || !(total>0f0) || !isfinite(total)
                status[1]=Int32(1) # sticky: callers must reject this generation
            else
                mass=0f0;keep=Int32(1)
                for j in Int32(1):k
                    mass+=exp((values[j]-m)/temperature);keep=j
                    mass>=top_p*total && break
                end
                target=uniform23(seed,counter[1])*mass
                cdf=0f0;chosen=indices[keep]
                for j in Int32(1):keep
                    cdf+=exp((values[j]-m)/temperature)
                    if target<cdf;chosen=indices[j];break;end
                end
            end
            selected[1]=chosen;seen[chosen+Int32(1)]+=Int32(1)
            counter[1]+=UInt32(1)
        end
    end
    nothing
end

function force_selection!(logits,selected,active,n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n && active[1]!=0
        @inbounds logits[i]=i-Int32(1)==selected[1] ? 0f0 : -Inf32
    end
    nothing
end

function declare!(g,logits,s::State)
    length(logits)==s.vocabulary || throw(DimensionMismatch("sampler vocabulary"))
    scratch=Mantle.Transient.Buffer(g,Float32,(s.vocabulary,))
    values=Mantle.Transient.Buffer(g,Float32,(s.topk,))
    indices=Mantle.Transient.Buffer(g,Int32,(s.topk,))
    Mantle.dispatch!(g,copy_adjusted!,(scratch,logits,s.seen,s.presence,Int32(s.vocabulary)),
        s.vocabulary;group=256,name="sampling.copy_adjusted")
    for rank in 1:s.topk
        best=Mantle.viewof(indices,(1,);offset=rank-1)
        BonsaiRunner.declare_argmax!(g,best,scratch;name="sampling.top_$rank")
        Mantle.dispatch!(g,save_and_mask!,(values,indices,scratch,Int32(rank)),1;group=1,name="sampling.mask_$rank")
    end
    Mantle.dispatch!(g,sample_topk!,(s.selected,s.status,s.seen,s.counter,s.active,values,indices,
        Int32(s.topk),s.temperature,s.top_p,s.seed,s.fallback),1;group=1,name="sampling.select")
    Mantle.dispatch!(g,force_selection!,(logits,s.selected,s.active,Int32(s.vocabulary)),
        s.vocabulary;group=256,name="sampling.feedback_scores")
    nothing
end

# The draws of `count[]` sampled tokens at `tokens[first+1 ...]` taken back, as
# `fixdraws!` does on the host: the counter as if they were never drawn and
# their presence counts removed. Only an active sampler drew them.
function undo_draws!(counter,seen,active,tokens,first::Int32,count)
    if KI.get_global_id().x==1 && active[1]!=0
        @inbounds begin
            n=count[1]
            counter[1]-=UInt32(n)
            for i in Int32(0):n-Int32(1)
                seen[tokens[first+i+Int32(1)]+Int32(1)]-=Int32(1)
            end
        end
    end
    nothing
end

"""DECLARE the reversal of the draws of `count[]` (a `GPURef{Int32}`) sampled
tokens at `tokens[first+1 ...]`, for a speculative verify whose later columns
were refused."""
declare_undo!(g,s::State,tokens,first::Integer,count)=
    Mantle.dispatch!(g,undo_draws!,(s.counter,s.seen,s.active,tokens,Int32(first),count),1;group=1,name="sampling.undo")

"""Independent host reference for arithmetic qualification, not model decoding."""
function reference(logits;topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,
                   seen=zeros(Int32,length(logits)),seed=UInt32(1),counter=UInt32(0),fallback=0)
    adjusted=Float32.(logits).-ifelse.(seen.>0,Float32(presence),0f0)
    (any(isnan,adjusted)||any(==(Inf32),adjusted)||all(==(-Inf32),adjusted)) &&
        return (;selected=Int32(fallback),status=1)
    order=sortperm(eachindex(adjusted);by=i->(-adjusted[i],i))[1:min(topk,length(adjusted))]
    v=adjusted[order];m=first(v)
    probabilities=exp.((v.-m)./Float32(temperature))
    total=0f0
    for p in probabilities;total+=p;end
    mass=0f0;cut=length(v)
    for j in eachindex(probabilities)
        mass+=probabilities[j]
        if mass>=Float32(top_p)*total;cut=j;break;end
    end
    u=uniform23(UInt32(seed),UInt32(counter));cdf=0f0
    selected=order[cut]-1
    for j in 1:cut
        cdf+=probabilities[j]
        if u*mass<cdf;selected=order[j]-1;break;end
    end
    (;selected=Int32(selected),status=0)
end
end
