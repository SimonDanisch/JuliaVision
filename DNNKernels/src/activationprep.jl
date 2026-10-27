"""Packed ggml Q8_1 activations, with Float32 scales and block sums.

These are graph resources, not cached device values. Every replay recomputes
them. A representation is valid only for the logical input value named by its
caller (including its write version), layout and quantization policy.
"""
struct PreparedQ8{Q,S}
    q::Q
    scales::S
    k::Int
    columns::Int
end

"""Declaration-local activation sharing for one declared graph.

Works with a Mantle graph or the semantic DNN frontend; the latter retains the
preparation as IR nodes until the shared fusion/lowering pipeline runs.

`declare_q8!(cache, value, input)` shares preparation for the same immutable
logical `value`. Handwritten graphs must use a new value/version after a write,
or call `invalidate_q8!`. Invalidate on entry/exit of loops or conditionals too:
a representation declared outside a loop cannot stand for its updated value
inside. Never retain this cache between graph declarations.
`enabled=false` disables sharing for diagnostic comparisons, not by backend.
"""
mutable struct ActivationPrep{G}
    graph::G
    enabled::Bool
    entries::Dict{Any,Any}
    prepared::Int
    reused::Int
    fused::Int
end
ActivationPrep(g; enabled=true) = ActivationPrep(g, enabled, Dict{Any,Any}(), 0, 0, 0)
invalidate_q8!(c::ActivationPrep) = (empty!(c.entries); c)
"""Drop the representations (of those in `among`) prepared from `written`'s storage."""
function invalidate_q8!(c::ActivationPrep, written; among=collect(keys(c.entries)))
    root=activationroot(written)
    for key in among
        activationroot(key[2][1])===root && delete!(c.entries,key)
    end
    c
end
"""Declaration statistics; no GPU synchronization or host readback."""
activation_report(c::ActivationPrep) =
    (;prepared=c.prepared,reused=c.reused,fused=c.fused,live_representations=length(c.entries))

# Semantic frontends specialize these declaration verbs, never the arithmetic.
activationbuffer(g,T,dims)=M.Transient.Buffer(g,T,dims)
activationdispatch!(g,k,args,n;kw...)=M.dispatch!(g,k,args,n;kw...)

activationlayout(x)=(x,0,size(x))
activationlayout(x::M.ResourceView)=(x.parent,x.offset,size(x))
q8key(value,input,k,n)=(value,activationlayout(input),k,n,eltype(input),:q8_1_f32)

function q8extent(input)
    ndims(input)==2 || throw(DimensionMismatch("Q8 preparation requires a dense K x N value"))
    k,n=size(input)
    k>0 && k%QK_K==0 && n>0 || throw(DimensionMismatch("Q8 preparation extent"))
    k*n<=typemax(Int32) || throw(ArgumentError("Q8 preparation exceeds Int32 addressing"))
    (k,n)
end

function declare_q8!(c::ActivationPrep, value, input; name="activation.q8")
    k,n=q8extent(input)
    key=q8key(value,input,k,n)
    if c.enabled && haskey(c.entries,key)
        c.reused+=1
        return c.entries[key]
    end
    q=activationbuffer(c.graph,UVec4,(k÷16,n))
    ds=activationbuffer(c.graph,Float32,(k÷16,n))
    activationdispatch!(c.graph,q8act_kernel!, (q,ds,input,Int32(k),Int32(n*(k÷32))),
                n*(k÷32);group=256,name)
    p=PreparedQ8(q,ds,k,n)
    c.prepared+=1
    c.enabled && (c.entries[key]=p)
    p
end

# Built INSIDE the device kernel. Each block reads each element exactly once,
# so the optional materialized result is stored with the original dtype round.
struct Q8MappedInput{O,F,A}
    out::O
    f::F
    inputs::A
end
@inline function Base.getindex(x::Q8MappedInput,i::Integer)
    v=eltype(x.out)(x.f(map(a->(@inbounds a[i]),x.inputs)...))
    @inbounds x.out[i]=v
    v
end

activationroot(x)=x
activationroot(x::Union{M.ResourceView,M.BufferRange})=activationroot(x.parent)
function mapped_q8_kernel!(q,ds,out,f,inputs,k::Int32,n::Int32)
    q8act_kernel!(q,ds,Q8MappedInput(out,f,inputs),k,n)
end

# One value per lane, but the exact same 32-value quantization policy as the
# serial block producer. Integer sums are exact; the max, scale and round are
# unchanged. Each subgroup can contain several independent 32-lane blocks.
function mapped_q8_parallel!(q,ds,out,f,inputs,k::Int32,columns::Int32,::Val{SG}) where SG
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<k*columns
        v=Float32(Q8MappedInput(out,f,inputs)[i+Int32(1)])
        lane=i&Int32(31);base=(i%Int32(SG))&Int32(-32)
        amax=abs(v);shift=Int32(16)
        while shift>0
            amax=max(amax,KI.shfl(amax,base+(lane⊻shift)+Int32(1)))
            shift>>=1
        end
        d=amax/127f0;inverse=d==0f0 ? 0f0 : 1f0/d
        quant=q4k_round(v*inverse);sum=quant;shift=Int32(16)
        while shift>0
            sum+=KI.shfl(sum,base+(lane⊻shift)+Int32(1));shift>>=1
        end
        quad=base+(lane&Int32(-4))+Int32(1)
        a=KI.shfl(quant,quad);b=KI.shfl(quant,quad+Int32(1))
        c=KI.shfl(quant,quad+Int32(2));e=KI.shfl(quant,quad+Int32(3))
        word=q4k_bytes(a%UInt8,b%UInt8,c%UInt8,e%UInt8)
        half=base+(lane&Int32(-16))+Int32(1)
        w0=KI.shfl(word,half);w1=KI.shfl(word,half+Int32(4))
        w2=KI.shfl(word,half+Int32(8));w3=KI.shfl(word,half+Int32(12))
        block=i÷Int32(32)
        if lane%Int32(16)==0
            @inbounds q[block*Int32(2)+lane÷Int32(16)+Int32(1)]=
                (VecElement(w0),VecElement(w1),VecElement(w2),VecElement(w3))
        end
        if lane==0
            @inbounds ds[block*Int32(2)+Int32(1)]=d
            @inbounds ds[block*Int32(2)+Int32(2)]=d*Float32(sum)
        end
    end
    nothing
end

"""Fuse a dense scalar expression's store and Q8 preparation.

`f` is a concrete callable (including `FusedOp`). Inputs must have exactly the
output extent: no implicit broadcasting, reshaping, alias writes or reductions.
The materialized output is retained for observable/multiple float consumers.
"""
function declare_mapped_q8!(c::ActivationPrep,value,out,f,inputs::Tuple;name="expression.q8")
    k,n=q8extent(out)
    all(a->size(a)==size(out),inputs) || throw(DimensionMismatch("mapped Q8 inputs"))
    isempty(inputs) && throw(ArgumentError("mapped Q8 requires inputs"))
    any(a->activationroot(a)===activationroot(out),inputs) &&
        throw(ArgumentError("mapped Q8 refuses input/output aliasing"))
    # A producer is a WRITE, even if its logical id was used before.
    invalidate_q8!(c)
    q=activationbuffer(c.graph,UVec4,(k÷16,n))
    ds=activationbuffer(c.graph,Float32,(k÷16,n))
    sg=M.caps(c.graph.dev).subgroup
    if sg>=64 && sg<=256 && sg%32==0
        activationdispatch!(c.graph,mapped_q8_parallel!,
            (q,ds,out,f,inputs,Int32(k),Int32(n),Val(sg)),k*n;group=256,name)
    else
        activationdispatch!(c.graph,mapped_q8_kernel!,
            (q,ds,out,f,inputs,Int32(k),Int32(n*(k÷32))),n*(k÷32);group=256,name)
    end
    p=PreparedQ8(q,ds,k,n)
    c.prepared+=1;c.fused+=1
    c.enabled && (c.entries[q8key(value,out,k,n)]=p)
    p
end

"""SwiGLU's declared intermediate rounding, shared by all fused-Q8 consumers."""
struct SwiGLUMap{T} end
@inline function (::SwiGLUMap{T})(gate,up) where T
    x=Float32(gate)
    T(Float32(T(x/(1f0+exp(-x))))*Float32(up))
end
