"""Model-specific output-projection features, not arbitrary database embeddings.

Uses a fixed subset of native logits as a linear projection of the final model
representation, followed by per-vector centering/RMS normalization. This can
use the existing logit hook without changing the qualified Qwen runner. It is
not the original hidden-state kNN-LM metric or a trained latent-memory bridge.
"""
module LogitFeatures
import Mantle
import KernelInterface as KI

struct Features
    ids
    vocabulary::Int
    width::Int
end
function Features(device,vocabulary,ids)
    0<vocabulary<=typemax(Int32)-256 && 1<=length(ids)<=65536 || throw(ArgumentError("feature dimensions"))
    all(id->id isa Integer && !(id isa Bool) && 0<=id<vocabulary,ids) && length(unique(ids))==length(ids) ||
        throw(ArgumentError("unique zero-based vocabulary IDs required"))
    Features(Mantle.Buffer(device,Int32.(ids)),Int(vocabulary),length(ids))
end
Base.close(features::Features)=(Mantle.free!(features.ids);nothing)

function project!(query,logits,ids,width::Int32)
    lane=Int32(KI.get_local_id().x)-Int32(1)
    partial=KI.localmemory(Float32,Val((256,)),Val(1))
    total=0f0;i=lane+Int32(1)
    while i<=width
        @inbounds total+=logits[ids[i]+Int32(1)];i+=Int32(256)
    end
    @inbounds partial[lane+Int32(1)]=total
    KI.barrier();stride=Int32(128)
    while stride>0
        lane<stride && (@inbounds partial[lane+Int32(1)]+=partial[lane+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    mean=partial[1]/Float32(width);square=0f0;i=lane+Int32(1)
    while i<=width
        @inbounds delta=logits[ids[i]+Int32(1)]-mean
        square=muladd(delta,delta,square);i+=Int32(256)
    end
    KI.barrier() # all lanes consumed the shared mean
    @inbounds partial[lane+Int32(1)]=square
    KI.barrier();stride=Int32(128)
    while stride>0
        lane<stride && (@inbounds partial[lane+Int32(1)]+=partial[lane+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    variance=partial[1]/Float32(width)
    scale=isfinite(mean)&&isfinite(variance)&&variance>1f-12*max(1f0,mean*mean) ? inv(sqrt(variance)) : NaN32
    i=lane+Int32(1)
    while i<=width
        @inbounds query[i]=(logits[ids[i]+Int32(1)]-mean)*scale
        i+=Int32(256)
    end
    nothing
end

function declare!(graph,query,logits,features::Features)
    length(query)==features.width && length(logits)==features.vocabulary || throw(DimensionMismatch("logit features"))
    eltype(query)===Float32 && eltype(logits)===Float32 || throw(ArgumentError("Float32 features required"))
    Mantle.dispatch!(graph,project!,(query,logits,features.ids,Int32(features.width)),256;group=256,name="memory.logit_features")
    nothing
end

struct Capture
    history
    counter
    status
    active
    width::Int
    capacity::Int
end
function Capture(device,width,capacity)
    0<width<=65536 && 0<capacity<typemax(Int32) && width*capacity<=typemax(Int32) || throw(ArgumentError("capture dimensions"))
    owned=Any[]
    upload(x)=begin buffer=Mantle.Buffer(device,x);push!(owned,buffer);buffer;end
    try
        Capture(upload(zeros(Float32,width,capacity)),upload(Int32[0]),upload(Int32[0]),upload(Int32[0]),width,capacity)
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end
Base.close(capture::Capture)=(foreach(Mantle.free!,(capture.history,capture.counter,capture.status,capture.active));nothing)
activate!(capture::Capture,active=true)=(capture.active[:]=Int32[active];capture)
reset!(capture::Capture)=(capture.counter[:]=Int32[0];capture.status[:]=Int32[0];capture)

function capture!(history,counter,status,active,query,width::Int32,capacity::Int32)
    i=Int32(KI.get_global_id().x);column=counter[1]
    if active[1]!=0 && i<=width
        if column<capacity
            @inbounds history[column*width+i]=query[i]
        elseif i==1
            @inbounds status[1]=Int32(1)
        end
    end
    nothing
end
function advance!(counter,active,capacity::Int32)
    if KI.get_global_id().x==1 && active[1]!=0
        @inbounds counter[1]=min(counter[1]+Int32(1),capacity)
    end
    nothing
end
function declare_capture!(graph,query,capture::Capture)
    length(query)==capture.width || throw(DimensionMismatch("capture query"))
    Mantle.dispatch!(graph,capture!,
        (capture.history,capture.counter,capture.status,capture.active,query,Int32(capture.width),Int32(capture.capacity)),
        capture.width;group=256,name="memory.capture")
    Mantle.dispatch!(graph,advance!,(capture.counter,capture.active,Int32(capture.capacity)),1;group=1,name="memory.capture_advance")
    nothing
end
end
