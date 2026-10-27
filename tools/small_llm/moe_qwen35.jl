module Qwen35MoE
import Mantle
import KernelInterface as KI
import BonsaiRunner
import DNNKernels

# Opaque kernels, declared into a plain Mantle graph or a DNNKernels semantic graph.
kerneldispatch!(g::Mantle.Graph,f,args,n;kw...)=Mantle.dispatch!(g,f,args,n;kw...)
kerneldispatch!(g::DNNKernels.SemanticGraph,f,args,n;kw...)=DNNKernels.semantickernel!(g,f,args,n;kw...)

# One 256-work-item group per token. Match the native router's FP32 softmax,
# top-k probabilities and selected-weight renormalization. Exact ties use the
# lower expert ID; torch.topk does not promise the same tie ordering.
function router!(selected,weights,status,logits,ne::Int32,topk::Int32)
    prob=KI.localmemory(Float32,Val((256,)),Val(1))
    values=KI.localmemory(Float32,Val((256,)),Val(2))
    indices=KI.localmemory(Int32,Val((256,)),Val(3))
    chosen=KI.localmemory(Float32,Val((256,)),Val(4))
    stats=KI.localmemory(Float32,Val((3,)),Val(5))
    t=Int32(KI.get_local_id().x)-Int32(1)
    token=Int32(KI.get_group_id().x)-Int32(1)
    x=t<ne ? Float32(@inbounds logits[token*ne+t+Int32(1)]) : -Inf32
    @inbounds begin
        prob[t+Int32(1)]=x
        indices[t+Int32(1)]=t<ne && !isfinite(x) ? Int32(1) : Int32(0)
        values[t+Int32(1)]=x
    end
    KI.barrier()
    stride=Int32(128)
    while stride>0
        if t<stride
            @inbounds begin
                indices[t+Int32(1)]+=indices[t+stride+Int32(1)]
                values[t+Int32(1)]=max(values[t+Int32(1)],values[t+stride+Int32(1)])
            end
        end
        KI.barrier();stride>>=1
    end
    # Uniform early return: invalid logits cannot produce unsafe expert IDs.
    if indices[1]!=0
        t==0 && (@inbounds status[token+Int32(1)]=Int32(1))
        if t<topk
            @inbounds begin selected[token*topk+t+Int32(1)]=Int32(-1);weights[token*topk+t+Int32(1)]=0f0;end
        end
        return nothing
    end
    t==0 && (@inbounds begin stats[1]=values[1];status[token+Int32(1)]=Int32(0);end)
    KI.barrier()
    e=t<ne ? exp(x-stats[1]) : 0f0
    @inbounds values[t+Int32(1)]=e
    KI.barrier()
    stride=Int32(128)
    while stride>0
        t<stride && (@inbounds values[t+Int32(1)]+=values[t+stride+Int32(1)])
        KI.barrier();stride>>=1
    end
    t==0 && (@inbounds stats[2]=values[1])
    KI.barrier()
    @inbounds prob[t+Int32(1)]=t<ne ? e/stats[2] : -Inf32
    KI.barrier()
    for rank in Int32(0):topk-Int32(1)
        @inbounds begin values[t+Int32(1)]=prob[t+Int32(1)];indices[t+Int32(1)]=t;end
        KI.barrier()
        stride=Int32(128)
        while stride>0
            if t<stride
                a=t+Int32(1);b=a+stride
                @inbounds if values[b]>values[a] || (values[b]==values[a] && indices[b]<indices[a])
                    values[a]=values[b];indices[a]=indices[b]
                end
            end
            KI.barrier();stride>>=1
        end
        winner=indices[1]
        if t==0
            @inbounds begin
                selected[token*topk+rank+Int32(1)]=winner
                chosen[rank+Int32(1)]=values[1]
            end
        end
        t==winner && (@inbounds prob[t+Int32(1)]=-Inf32)
        KI.barrier()
    end
    if t==0
        total=0f0
        for j in Int32(1):topk;total+=chosen[j];end
        @inbounds stats[3]=total
    end
    KI.barrier()
    t<topk && (@inbounds weights[token*topk+t+Int32(1)]=chosen[t+Int32(1)]/stats[3])
    nothing
end

function declare_router!(g,selected,weights,status,logits;name="moe.router")
    ne,tokens=size(logits);topk=size(selected,1)
    1<=topk<=ne<=256 || throw(ArgumentError("router needs 1 <= top-k <= experts <= 256"))
    size(selected)==size(weights)==(topk,tokens) || throw(DimensionMismatch("router output shape"))
    length(status)==tokens || throw(DimensionMismatch("router status shape"))
    eltype(selected)===Int32 && eltype(status)===Int32 || throw(ArgumentError("router indices/status must be Int32"))
    eltype(weights)===Float32 || throw(ArgumentError("router weights must be Float32"))
    kerneldispatch!(g,router!,
        (selected,weights,status,logits,Int32(ne),Int32(topk)),tokens*256;group=256,name)
end

# Routed outputs are [width, selected experts, tokens]; no atomic reduction.
# Downstream expert kernels must skip all work when the router status is bad.
function combine!(out,routed,shared,shared_gate,weights,status,width::Int32,topk::Int32,total::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<total
        token=i÷width;d=i%width
        if status[token+Int32(1)]!=0
            @inbounds out[i+Int32(1)]=NaN32
            return nothing
        end
        value=0f0
        for j in Int32(0):topk-Int32(1)
            @inbounds value=muladd(weights[token*topk+j+Int32(1)],
                routed[(token*topk+j)*width+d+Int32(1)],value)
        end
        gate=@inbounds shared_gate[token+Int32(1)]
        # Stable sigmoid avoids overflow for very negative gates.
        a=exp(-abs(gate));sig=gate>=0f0 ? inv(1f0+a) : a/(1f0+a)
        @inbounds out[i+Int32(1)]=muladd(sig,shared[i+Int32(1)],value)
    end
    nothing
end

function declare_combine!(g,out,routed,shared,shared_gate,weights,status;name="moe.combine")
    width,tokens=size(out);topk=size(weights,1)
    size(routed)==(width,topk,tokens) || throw(DimensionMismatch("routed expert output shape"))
    size(shared)==size(out) && length(shared_gate)==tokens && size(weights,2)==tokens &&
        length(status)==tokens || throw(DimensionMismatch("MoE combine shape"))
    kerneldispatch!(g,combine!,
        (out,routed,shared,shared_gate,weights,status,Int32(width),Int32(topk),Int32(width*tokens)),
        width*tokens;group=256,name)
end

# Floating-weight qualification path. IDs are read inside the kernel; only the
# chosen experts' weights are touched. Quantized expert GEMV is a separate path.
function routed_gemv!(out,W,x,selected,status,k::Int32,m::Int32,ne::Int32,topk::Int32,
                      rows::Int32,::Val{PER_EXPERT},::Val{SG}) where {PER_EXPERT,SG}
    t=Int32(KI.get_local_id().x)-Int32(1)
    row=(Int32(KI.get_group_id().x)-Int32(1))*Int32(256÷SG)+t÷Int32(SG)
    row>=rows && return nothing
    lane=t%Int32(SG);slot=row÷m;d=row%m;token=slot÷topk
    if status[token+Int32(1)]!=0
        lane==0 && (@inbounds out[row+Int32(1)]=0f0)
        return nothing
    end
    expert=@inbounds selected[slot+Int32(1)]
    if expert<0 || expert>=ne
        lane==0 && (@inbounds out[row+Int32(1)]=NaN32)
        return nothing
    end
    wc=(expert*m+d)*k;xc=(PER_EXPERT ? slot : token)*k
    value=0f0
    j=lane
    while j<k
        @inbounds value=muladd(Float32(W[wc+j+Int32(1)]),Float32(x[xc+j+Int32(1)]),value)
        j+=Int32(SG)
    end
    value=KI.sub_group_reduce_add(value)
    lane==0 && (@inbounds out[row+Int32(1)]=value)
    nothing
end

function declare_routed_linear!(g,out,W,x,selected,status;per_expert=false,name="moe.expert_linear")
    k,m,ne=size(W);topk,tokens=size(selected)
    size(out)==(m,topk,tokens) || throw(DimensionMismatch("expert projection output"))
    expected=per_expert ? (k,topk,tokens) : (k,tokens)
    size(x)==expected && length(status)==tokens || throw(DimensionMismatch("expert projection input"))
    prod(size(W))<=typemax(Int32) && length(out)<=typemax(Int32) ||
        throw(ArgumentError("expert projection exceeds 32-bit addressing"))
    sg=Mantle.caps(g.dev).subgroup
    kerneldispatch!(g,routed_gemv!,
        (out,W,x,selected,status,Int32(k),Int32(m),Int32(ne),Int32(topk),Int32(length(out)),Val(per_expert),Val(sg)),
        cld(length(out),256÷sg)*256;group=256,name)
end

function swiglu_packed!(out,gateup,width::Int32,total::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<total
        slot=i÷width;d=i%width;base=slot*width*Int32(2)+d+Int32(1)
        @inbounds begin
            gate=Float32(gateup[base]);up=Float32(gateup[base+width])
            a=exp(-abs(gate));sig=gate>=0f0 ? inv(1f0+a) : a/(1f0+a)
            out[i+Int32(1)]=(gate*sig)*up
        end
    end
    nothing
end

function declare_dense!(g,out,W,x;name)
    k,m=size(W);tokens=size(x,2)
    size(x,1)==k && size(out)==(m,tokens) || throw(DimensionMismatch("dense MoE projection"))
    kerneldispatch!(g,BonsaiRunner.dense_gemv_batch_kernel!,
        (out,W,x,Int32(k),Int32(m),Int32(cld(m,4)),Val(Mantle.caps(g.dev).subgroup)),
        cld(m,4)*tokens*256;group=256,name)
end

"""Declare the floating-weight MoE FFN into its caller's graph.

Weights use native logical layout, with contiguous input dimension in Julia:
router [hidden, experts], gate_up [hidden, 2*ffn, experts], down [ffn, hidden,
experts], shared_gate_up [hidden, 2*shared_ffn], shared_down [shared_ffn, hidden],
shared_gate [hidden, 1]. Routing, selected expert products and shared gating
stay in one recording, without host-selected IDs or per-expert submissions.
"""
function declare_block!(g,out,x,W,selected,routing,status;name="moe")
    width,tokens=size(x);topk=size(selected,1);ne=size(W.router,2)
    f=size(W.gate_up,2)÷2;shared_f=size(W.shared_gate_up,2)÷2
    size(W.router)==(width,ne) && size(W.gate_up)==(width,2f,ne) &&
        size(W.down)==(f,width,ne) && size(W.shared_gate_up)==(width,2shared_f) &&
        size(W.shared_down)==(shared_f,width) && size(W.shared_gate)==(width,1) &&
        size(out)==size(x) || throw(DimensionMismatch("MoE block weights"))
    scratch(dims)=Mantle.Transient.Buffer(g,Float32,dims)
    logits=scratch((ne,tokens));gu=scratch((2f,topk,tokens));act=scratch((f,topk,tokens))
    routed=scratch((width,topk,tokens));sgu=scratch((2shared_f,tokens))
    sact=scratch((shared_f,tokens));shared=scratch((width,tokens));gate=scratch((1,tokens))
    declare_dense!(g,logits,W.router,x;name=name*".gate_logits")
    declare_router!(g,selected,routing,status,logits;name=name*".route")
    declare_routed_linear!(g,gu,W.gate_up,x,selected,status;name=name*".gate_up")
    kerneldispatch!(g,swiglu_packed!,(act,gu,Int32(f),Int32(length(act))),length(act);group=256,name=name*".swiglu")
    declare_routed_linear!(g,routed,W.down,act,selected,status;per_expert=true,name=name*".down")
    declare_dense!(g,sgu,W.shared_gate_up,x;name=name*".shared_gate_up")
    kerneldispatch!(g,swiglu_packed!,(sact,sgu,Int32(shared_f),Int32(length(sact))),length(sact);group=256,name=name*".shared_swiglu")
    declare_dense!(g,shared,W.shared_down,sact;name=name*".shared_down")
    declare_dense!(g,gate,W.shared_gate,x;name=name*".shared_gate")
    declare_combine!(g,out,routed,shared,gate,routing,status;name=name*".combine")
end

include("moe_quant_qwen35.jl")

for (kernel,writes) in ((router!,(1,2,3)),(combine!,(1,)),(routed_gemv!,(1,)),(swiglu_packed!,(1,)),
                        (checked_q8act!,(1,2)),(routed_q4_gemv!,(1,)))
    @eval DNNKernels.kernelresults(::typeof($kernel))=$writes
end
end
