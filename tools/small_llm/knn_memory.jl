"""Experimental representation-keyed token memory, using shared KI kernels.

Interpolates normalized neural and retrieved token distributions; it is not a
Bayesian posterior and does not establish that retrieved facts are true. Keys
must come from the same model/layer representation as the query. Model wiring,
real datastore construction and task-level benefits require separate tests.
"""
module KNNMemory
import Mantle, BonsaiRunner
import KernelInterface as KI

struct Memory
    keys
    labels
    status
    width::Int
    records::Int
    vocabulary::Int
    topk::Int
    temperature::Float32
    weight::Float32
end

function Memory(device,keys::AbstractMatrix,labels,vocabulary;topk=8,temperature=1f0,weight=0.25f0)
    width,records=size(keys)
    0<width<=typemax(Int32)-256 && records>0 && length(keys)<=typemax(Int32) || throw(ArgumentError("memory dimensions"))
    0<vocabulary<=typemax(Int32)-256 && 1<=topk<=min(records,64) || throw(ArgumentError("memory vocabulary/top-k"))
    eltype(keys) in (Float16,Float32) && all(isfinite,keys) || throw(ArgumentError("finite Float16/32 keys required"))
    length(labels)==records && all(id->id isa Integer && !(id isa Bool) && 0<=id<vocabulary,labels) ||
        throw(ArgumentError("zero-based token labels required"))
    temperature=Float32(temperature);weight=Float32(weight)
    isfinite(temperature)&&temperature>0 && isfinite(weight)&&0<=weight<=1 || throw(ArgumentError("memory mixture settings"))
    owned=Any[]
    upload(x)=begin buffer=Mantle.Buffer(device,x);push!(owned,buffer);buffer;end
    try
        Memory(upload(keys),upload(Int32.(labels)),upload(Int32[0]),width,records,Int(vocabulary),Int(topk),temperature,weight)
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end
Base.close(memory::Memory)=(foreach(Mantle.free!,(memory.keys,memory.labels,memory.status));nothing)

function distances!(scores,keys,query,width::Int32)
    lane=Int32(KI.get_local_id().x)-Int32(1);record=Int32(KI.get_group_id().x)-Int32(1)
    partial=KI.localmemory(Float32,Val((256,)),Val(1));value=0f0
    d=lane
    while d<width
        @inbounds delta=Float32(keys[record*width+d+Int32(1)])-Float32(query[d+Int32(1)])
        value=muladd(delta,delta,value);d+=Int32(256)
    end
    @inbounds partial[lane+Int32(1)]=value
    KI.barrier();stride=Int32(128)
    while stride>0
        lane<stride && (@inbounds partial[lane+Int32(1)]+=partial[lane+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    if lane==0
        value=partial[1]
        @inbounds scores[record+Int32(1)]=isfinite(value) ? -value : -Inf32
    end
    nothing
end

function select_neighbor!(values,tokens,indices,scores,labels,rank::Int32)
    if KI.get_global_id().x==1
        @inbounds begin
            i=indices[rank]+Int32(1)
            values[rank]=scores[i];tokens[rank]=labels[i];scores[i]=-Inf32
        end
    end
    nothing
end

function normalize!(stats,status,logits,values,n::Int32,k::Int32,temperature::Float32)
    lane=Int32(KI.get_local_id().x)-Int32(1)
    partial=KI.localmemory(Float32,Val((256,)),Val(1))
    invalid=KI.localmemory(Int32,Val((256,)),Val(2))
    maximum=-Inf32;bad=Int32(0);i=lane+Int32(1)
    while i<=n
        @inbounds x=logits[i]
        bad+=Int32(isnan(x)||x==Inf32);maximum=max(maximum,x);i+=Int32(256)
    end
    @inbounds begin partial[lane+Int32(1)]=maximum;invalid[lane+Int32(1)]=bad;end
    KI.barrier();stride=Int32(128)
    while stride>0
        if lane<stride
            @inbounds begin
                partial[lane+Int32(1)]=max(partial[lane+Int32(1)],partial[lane+stride+Int32(1)])
                invalid[lane+Int32(1)]+=invalid[lane+stride+Int32(1)]
            end
        end
        KI.barrier();stride>>=1
    end
    maximum=partial[1];valid=invalid[1]==0&&isfinite(maximum)
    lane==0 && (@inbounds stats[1]=maximum)
    total=0f0;i=lane+Int32(1)
    while i<=n
        valid && (@inbounds total+=exp(logits[i]-maximum));i+=Int32(256)
    end
    # Every lane finished reading the maximum before shared scratch is reused.
    KI.barrier()
    @inbounds partial[lane+Int32(1)]=total
    KI.barrier();stride=Int32(128)
    while stride>0
        lane<stride && (@inbounds partial[lane+Int32(1)]+=partial[lane+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    if lane==0
        @inbounds best=values[1]
        mass=0f0
        if isfinite(best)
            for j in Int32(1):k
                @inbounds mass+=exp((values[j]-best)/temperature)
            end
        end
        @inbounds begin
            stats[2]=partial[1];stats[3]=mass;stats[4]=best
            status[1]=!valid || !(partial[1]>0f0) ? Int32(1) : !(mass>0f0) ? Int32(2) : Int32(0)
        end
    end
    nothing
end

function mix!(logits,stats,status,values,tokens,n::Int32,k::Int32,temperature::Float32,weight::Float32)
    i=Int32(KI.get_global_id().x)
    if i<=n && status[1]==0 && weight!=0f0
        best=-Inf32
        for j in Int32(1):k
            @inbounds tokens[j]==i-Int32(1) && (best=max(best,values[j]))
        end
        mass=0f0
        if isfinite(best)
            for j in Int32(1):k
                @inbounds tokens[j]==i-Int32(1) && (mass+=exp((values[j]-best)/temperature))
            end
        end
        # Log-space interpolation retains tiny neural probabilities instead of
        # turning all logits below the exp underflow threshold into hard zeros.
        @inbounds begin
            left=weight<1f0 ? log1p(-weight)+logits[i]-stats[1]-log(stats[2]) : -Inf32
            right=mass>0f0 ? log(weight)+(best-stats[4])/temperature+log(mass)-log(stats[3]) : -Inf32
            maximum=max(left,right)
            logits[i]=maximum==-Inf32 ? -Inf32 : maximum+log(exp(left-maximum)+exp(right-maximum))
        end
    end
    nothing
end

function declare!(graph,logits,query,memory::Memory)
    length(logits)==memory.vocabulary && length(query)==memory.width || throw(DimensionMismatch("memory query/logits"))
    eltype(logits)===Float32 && eltype(query)===Float32 || throw(ArgumentError("Float32 query/logits required"))
    scratch(dims)=Mantle.Transient.Buffer(graph,Float32,dims)
    scores=scratch((memory.records,));values=scratch((memory.topk,));stats=scratch((4,))
    indices=Mantle.Transient.Buffer(graph,Int32,(memory.topk,))
    tokens=Mantle.Transient.Buffer(graph,Int32,(memory.topk,))
    Mantle.dispatch!(graph,distances!,(scores,memory.keys,query,Int32(memory.width)),memory.records*256;group=256,name="memory.distance")
    for rank in 1:memory.topk
        best=Mantle.viewof(indices,(1,);offset=rank-1)
        BonsaiRunner.declare_argmax!(graph,best,scores;name="memory.neighbor_$rank")
        Mantle.dispatch!(graph,select_neighbor!,(values,tokens,indices,scores,memory.labels,Int32(rank)),1;group=1,name="memory.take_$rank")
    end
    Mantle.dispatch!(graph,normalize!,(stats,memory.status,logits,values,Int32(memory.vocabulary),Int32(memory.topk),memory.temperature),256;group=256,name="memory.normalize")
    Mantle.dispatch!(graph,mix!,(logits,stats,memory.status,values,tokens,Int32(memory.vocabulary),Int32(memory.topk),memory.temperature,memory.weight),memory.vocabulary;group=256,name="memory.interpolate")
    nothing
end
end
