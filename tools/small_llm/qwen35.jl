# Experimental text-only Qwen3.5 runner. Uses native HF safetensors and the
# existing DNNKernels arithmetic. The network is a DNNKernels semantic graph:
# projections, SwiGLU and residuals are standard IR operations, so the shared
# pattern/fusion passes and Q8 activation sharing apply; the recurrent, cache and
# attention kernels are opaque operations with declared results.
module Qwen35Experiment
using JSON3, DNNKernels, KernelAbstractions
import Mantle, BonsaiRunner
import KernelInterface as KI
const DK = DNNKernels
const BR = BonsaiRunner
include("qwen_weight_cache.jl")

struct DenseWeight{A}
    data::A                         # contiguous input dimension, HF row layout
    m::Int
    k::Int
end
mutable struct Model
    device
    config
    weights::Dict{String,Any}
    owned::Vector{Any}
    quantize::Symbol
end
mutable struct Session
    model::Model
    capacity::Int
    position::Int
    token
    posref
    logits
    best
    predictions
    plan
    owned::Vector{Any}
    layer_outputs::Vector{Any}
    recurrent::Dict{Int,Any}
    kv::Dict{Int,Any}
    prefills::Dict{Int,Any}
    logit_hook
    attention::Symbol
    fusion      # the shared passes' report for the decode graph
end
# Sessions assembled from fixtures carry no fusion report.
Session(model,capacity,position,token,posref,logits,best,predictions,plan,owned,layer_outputs,
        recurrent,kv,prefills,logit_hook,attention)=Session(model,capacity,position,token,posref,
    logits,best,predictions,plan,owned,layer_outputs,recurrent,kv,prefills,logit_hook,attention,nothing)

function advance_decode!(token,position,predictions,best)
    if KI.get_global_id().x==1
        @inbounds begin
            predictions[position[1]+Int32(1)]=best[1]
            token[1]=best[1]
            position[1]+=Int32(1)
        end
    end
    nothing
end

function embedding!(out, weight, token, width::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=width
        @inbounds out[i]=Float32(weight[Int32(token[1])*width+i])
    end
    nothing
end
function add!(x, branch, n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds x[i]+=branch[i])
    nothing
end
function swiglu!(out, gate, up, n::Int32)
    i=Int32(KI.get_global_id().x)
    if i<=n
        @inbounds begin
            a=Float32(gate[i]); out[i]=a/(1f0+exp(-a))*Float32(up[i])
        end
    end
    nothing
end
# HF repeats each Q/K head consecutively. Bonsai's converted checkpoint uses
# cyclic repetition instead. Expand explicitly so the existing GDN kernel sees
# one Q/K vector per value head, without changing Bonsai's layout or arithmetic.
function repeat_qk!(q, k, conv, heads::Int32, keyheads::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<128heads
        h=i÷Int32(128); d=i%Int32(128)
        source=(h÷(heads÷keyheads))*Int32(128)+d+Int32(1)
        @inbounds begin q[i+Int32(1)]=conv[source]; k[i+Int32(1)]=conv[128keyheads+source]; end
    end
    nothing
end
function prepare_kv!(kc, vc, k, v, norm, position, nk::Int32, theta::Float32, eps::Float32)
    sh=KI.localmemory(Float32,Val((512,)),Val(1))
    d=Int32(KI.get_local_id().x)-Int32(1)
    h=Int32(KI.get_group_id().x)-Int32(1)
    base=h*Int32(256)
    @inbounds begin
        value=Float32(k[base+d+Int32(1)])
        sh[d+Int32(1)]=value; sh[256+d+Int32(1)]=value*value
    end
    KI.barrier()
    stride=Int32(128)
    while stride>0
        d<stride && (sh[256+d+Int32(1)]+=sh[256+d+stride+Int32(1)])
        KI.barrier(); stride>>=1
    end
    scale=inv(sqrt(sh[257]/256f0+eps))
    @inbounds sh[d+Int32(1)]*=scale*norm[d+Int32(1)]
    KI.barrier()
    if d<32
        angle=Float32(position[1])*exp(-log(theta)*Float32(2d)/64f0)
        co,si=cos(angle),sin(angle)
        a,b=sh[d+Int32(1)],sh[d+Int32(33)]
        sh[d+Int32(1)]=a*co-b*si; sh[d+Int32(33)]=a*si+b*co
    end
    KI.barrier()
    at=Int32(position[1])*nk*Int32(256)+base+d+Int32(1)
    @inbounds begin
        kc[at]=eltype(kc)(sh[d+Int32(1)])
        vc[at]=eltype(vc)(v[base+d+Int32(1)])
    end
    nothing
end

# Split the cache into 64-position chunks. A subgroup owns a score, then the
# workgroup forms a stable local softmax and accumulates each value dimension.
# No sequence-square buffer, host readback, or allocations occur per token.
function attention_partial!(pm, ps, pa, q, kc, vc, position, nk::Int32,
                            ratio::Int32, chunks::Int32, ::Val{SG}) where SG
    w=KI.localmemory(Float32,Val((256,)),Val(1))
    t=Int32(KI.get_local_id().x)-Int32(1)
    wg=Int32(KI.get_group_id().x)-Int32(1)
    h=wg÷chunks; chunk=wg%chunks; p0=chunk*Int32(64)
    p0>Int32(position[1]) && return nothing
    n=min(Int32(64),Int32(position[1])-p0+Int32(1))
    lane=t%Int32(SG); subgroup=t÷Int32(SG); kh=h÷ratio
    p=subgroup
    while p<64
        dot=0f0
        if p<n
            d=lane
            while d<256
                @inbounds dot=muladd(q[h*Int32(256)+d+Int32(1)],
                    Float32(kc[((p0+p)*nk+kh)*Int32(256)+d+Int32(1)]),dot)
                d+=Int32(SG)
            end
        end
        dot=KI.sub_group_reduce_add(dot)
        lane==0 && (@inbounds w[p+Int32(1)]=p<n ? dot*0.0625f0 : -Inf32)
        p+=Int32(256÷SG)
    end
    t>=64 && (@inbounds w[t+Int32(1)]=-Inf32)
    KI.barrier()
    # Preserve scores while reducing the maximum in the upper half.
    t<64 && (@inbounds w[t+Int32(129)]=w[t+Int32(1)])
    KI.barrier()
    stride=Int32(32)
    while stride>0
        t<stride && (@inbounds w[t+Int32(129)]=max(w[t+Int32(129)],w[t+stride+Int32(129)]))
        KI.barrier(); stride>>=1
    end
    m=w[129]
    t<64 && (@inbounds w[t+Int32(1)]=exp(w[t+Int32(1)]-m))
    KI.barrier()
    value=0f0
    for j in Int32(0):n-Int32(1)
        @inbounds value=muladd(w[j+Int32(1)],Float32(vc[((p0+j)*nk+kh)*Int32(256)+t+Int32(1)]),value)
    end
    @inbounds pa[wg*Int32(256)+t+Int32(1)]=value
    if t==0
        total=0f0
        for j in Int32(1):Int32(64); total+=w[j]; end
        @inbounds begin pm[wg+Int32(1)]=m; ps[wg+Int32(1)]=total; end
    end
    nothing
end

include("grouped_attention_qwen35.jl")

dispatch(g::Mantle.Graph,f,args,n;group=256,name=string(nameof(f))) =
    Mantle.dispatch!(g,f,args,n;group,name)
dispatch(g::DK.SemanticGraph,f,args,n;group=256,name=string(nameof(f))) =
    DK.semantickernel!(g,f,args,n;group,name)

for (kernel,writes) in ((advance_decode!,(1,2,3)),(embedding!,(1,)),(add!,(1,)),(swiglu!,(1,)),
                        (repeat_qk!,(1,2)),(prepare_kv!,(1,2)),(attention_partial!,(1,2,3)),
                        (attention_grouped!,(1,2,3)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end

# Weight objects a model hands to declared operations, by key.
semanticweights(key,w)=(key=>w,)
semanticweights(key,w::DenseWeight)=(key=>w.data,)
neuralweights(m)=Dict{String,Any}(p for (k,v) in m.weights for p in semanticweights(k,v))

matrix(x)=ndims(x)==2 ? x : Mantle.viewof(x,(length(x),1))

function linear!(g,out,A::DenseWeight,x;name,prepared=nothing)
    sg=Mantle.caps(g.dev).subgroup
    dispatch(g,BR.dense_gemv_batch_kernel!,
        (out,A.data,x,Int32(A.k),Int32(A.m),Int32(cld(A.m,4)),Val(sg)),
        cld(A.m,4)*256;name)
end
# A standard product: activation preparation, its sharing and any producer
# fusion are the shared emitter's capability-selected decisions.
linear!(g::DK.SemanticGraph,out,A::DK.KQuantMatrix,x;name)=
    DK.standardop!(g,"mm.default",(matrix(x),A);out=matrix(out),name)
function linear!(g::Mantle.Graph,out,A::DK.KQuantMatrix,x;name,prepared=nothing)
    m,k=size(A)
    p=prepared===nothing ? DK.declare_q8!(DK.ActivationPrep(g),nothing,
        Mantle.viewof(x,(k,1));name=name*".q8") : prepared
    (p.k,p.columns)==(k,1) || throw(DimensionMismatch("linear prepared activation"))
    xq,xds=p.q,p.scales
    r=DK.q4krows(1)
    dispatch(g,DK.kqgemv_kernels(A)[(1,r)],
        (out,DK.kqarrays(A)...,xq,xds,out,identity,Int32(m),Int32(k÷256),Val(false)),
        cld(cld(m,r),DK.Q4K_WG÷32)*DK.Q4K_WG;group=DK.Q4K_WG,name)
end

function loadmodel(dir;quantize=:none,device=Mantle.Device("radeon"),quantcache=nothing)
    quantize in (:none,:q4k) || throw(ArgumentError("quantize must be :none or :q4k"))
    quantcache===nothing || quantize===:q4k || throw(ArgumentError("disk cache is for Q4_K only"))
    c=JSON3.read(read(joinpath(dir,"config.json"),String)).text_config
    @assert c.head_dim==256 && c.linear_key_head_dim==128 && c.linear_value_head_dim==128
    @assert c.linear_conv_kernel_dim==4 && c.tie_word_embeddings
    @assert c.rope_parameters.rope_type=="default" && c.rope_parameters.partial_rotary_factor==0.25
    model=Model(device,c,Dict{String,Any}(),Any[],quantize)
    cache=nothing;cached_weights=0
    upload(x) = begin b=Mantle.Buffer(device,x);push!(model.owned,b);b;end
    try
        if quantcache!==nothing
            opened=QwenWeightCache.open_cache(dir,quantcache,device)
            cache=opened.cache
            println("Qwen quantcache: ",opened.directory);flush(stdout)
        end
        index=JSON3.read(read(joinpath(dir,"model.safetensors.index.json"),String))
        for file in sort!(unique(String.(collect(values(index.weight_map)))))
            tensors=DK.readsafetensors(joinpath(dir,file);mmap=true)
            for key in sort!(collect(keys(tensors)))
                startswith(key,"model.language_model.") || continue
                name=replace(key,"model.language_model."=>"";count=1)
                raw=tensors[key]
                if cache!==nothing && ndims(raw)==2 && name!="embed_tokens.weight"
                    k,m=size(raw)
                    if m>=128 && k%256==0 && DK.iscached(cache,name,(m,k),:q4k)
                        model.weights[name]=QwenWeightCache.load_owned!(device,cache,name,model.owned)
                        cached_weights+=1;continue
                    end
                end
                # BF16 is read as raw UInt16 by DNNKernels' file reader.
                x=eltype(raw)===UInt16 ? reinterpret.(Float32,UInt32.(raw).<<16) : Float32.(raw)
                if ndims(x)==2 && name!="embed_tokens.weight"
                    k,m=size(x)
                    if quantize===:q4k && m>=128 && k%256==0
                        dense=upload(permutedims(x))
                        a=DK.quantizeq4k(device,dense)
                        push!(model.owned,a.q); Mantle.free!(dense); popat!(model.owned,length(model.owned)-1)
                        model.weights[name]=a
                        cache===nothing || DK.cachestore!(cache,name,(m,k),a)
                    else
                        model.weights[name]=DenseWeight(upload(x),m,k)
                    end
                else
                    if endswith(name,"input_layernorm.weight") || endswith(name,"post_attention_layernorm.weight") ||
                       name=="norm.weight" || endswith(name,"q_norm.weight") || endswith(name,"k_norm.weight")
                        x .+= 1f0
                    elseif endswith(name,"A_log")
                        x = .-exp.(x)
                    end
                    model.weights[name]=upload(x)
                end
            end
        end
        embedding=model.weights["embed_tokens.weight"]
        if quantize===:q4k
            shape=(Int(c.vocab_size),Int(c.hidden_size))
            if cache!==nothing && DK.iscached(cache,"lm_head",shape,:q6k)
                a=QwenWeightCache.load_owned!(device,cache,"lm_head",model.owned);cached_weights+=1
            else
                dense=upload(permutedims(Array(embedding)))
                a=DK.quantizeq6k(device,dense)
                append!(model.owned,DK.kqarrays(a)); Mantle.free!(dense)
                deleteat!(model.owned,findfirst(x->x===dense,model.owned))
                cache===nothing || DK.cachestore!(cache,"lm_head",shape,a)
            end
            model.weights["lm_head"]=a
        else
            model.weights["lm_head"]=DenseWeight(embedding,Int(c.vocab_size),Int(c.hidden_size))
        end
        println("Loaded Qwen3.5: ",c.hidden_size," wide, ",c.num_hidden_layers," layers, ",quantize,"; cached=",cached_weights)
        model
    catch
        close(model); rethrow()
    finally
        cache===nothing || DK.closecache!(cache)
    end
end
function Base.close(m::Model)
    foreach(Mantle.free!,m.owned); empty!(m.owned); empty!(m.weights);nothing
end

function session(m::Model;capacity=4096,debug_layers=false,logit_hook=nothing,profile=false,
                 kv_type=Float32,attention=:split)
    c=m.config; 0<capacity<=c.max_position_embeddings || throw(ArgumentError("invalid capacity"))
    kv_type in (Float32,Float16) || throw(ArgumentError("KV storage must be Float32 or Float16"))
    attention in (:split,:grouped) || throw(ArgumentError("unknown attention implementation"))
    ratio=Int(c.num_attention_heads)÷Int(c.num_key_value_heads)
    attention===:grouped && !(ratio in (1,2,4,8)) &&
        throw(ArgumentError("grouped attention supports one, two, four or eight queries per KV head"))
    owned=Any[]
    persistent(T,dims)=begin a=Mantle.Buffer(m.device,T,dims);fill!(a,zero(T));push!(owned,a);a;end
    token=Mantle.GPURef(m.device,Int32(0));pos=Mantle.GPURef(m.device,Int32(0))
    push!(owned,token,pos)
    logits=persistent(Float32,(Int(c.vocab_size),));best=persistent(Int32,(1,))
    predictions=persistent(Int32,(capacity,))
    s=Session(m,capacity,0,token,pos,logits,best,predictions,nothing,owned,Any[],
              Dict{Int,Any}(),Dict{Int,Any}(),Dict{Int,Any}(),logit_hook,attention,nothing)
    try
        g=DK.SemanticGraph(m.device;name="qwen35_decode")
        DK.semanticweights!(g,neuralweights(m))
        scratch(n)=Mantle.Transient.Buffer(g,Float32,(Int(n),))
        w=Int(c.hidden_size);f=Int(c.intermediate_size);nh=Int(c.linear_num_value_heads)
        nkh=Int(c.linear_num_key_heads);kd=128nkh;vd=128nh
        nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads);chunks=cld(capacity,64)
        x=scratch(w);norm=scratch(w);branch=scratch(w)
        gate=scratch(f);up=scratch(f)
        qkv=scratch(2kd+vd);z=scratch(vd);conv=scratch(2kd+vd)
        rq=scratch(vd);rk=scratch(vd);alpha=scratch(nh);beta=scratch(nh);gdn=scratch(vd)
        qfull=scratch(512nqh);kproj=scratch(256nkv);vproj=scratch(256nkv)
        aq=scratch(256nqh);qgate=scratch(256nqh);attn=scratch(256nqh)
        pm=scratch(nqh*chunks);ps=scratch(nqh*chunks);pa=scratch(nqh*chunks*256)
        wt(name)=m.weights[name]
        eps=Float32(c.rms_norm_eps);theta=Float32(c.rope_parameters.rope_theta)
        dispatch(g,embedding!,(x,wt("embed_tokens.weight"),token,Int32(w)),w;name="embedding")
        for il in 0:Int(c.num_hidden_layers)-1
            p="layers.$il."
            dispatch(g,BR.rmsnorm_batch_kernel!,
                (norm,x,wt(p*"input_layernorm.weight"),Int32(w),eps),256;name=p*"input_norm")
            if String(c.layer_types[il+1])=="linear_attention"
                a=p*"linear_attn."
                linear!(g,qkv,wt(a*"in_proj_qkv.weight"),norm;name=a*"qkv")
                linear!(g,z,wt(a*"in_proj_z.weight"),norm;name=a*"z")
                linear!(g,alpha,wt(a*"in_proj_a.weight"),norm;name=a*"alpha")
                linear!(g,beta,wt(a*"in_proj_b.weight"),norm;name=a*"beta")
                hist=persistent(Float32,(3,2kd+vd));state=persistent(Float32,(128,128,nh))
                s.recurrent[il]=(hist,state)
                dispatch(g,DK.depthwise_conv4_kernel!,
                    (conv,hist,qkv,wt(a*"conv1d.weight"),Int32(2kd+vd)),2kd+vd;name=a*"conv")
                dispatch(g,repeat_qk!,(rq,rk,conv,Int32(nh),Int32(nkh)),vd;name=a*"repeat_qk")
                v=Mantle.viewof(conv,(vd,);offset=2kd)
                dispatch(g,DK.gated_delta_net_kernel!,
                    (gdn,state,rq,rk,v,alpha,beta,wt(a*"dt_bias"),wt(a*"A_log"),z,
                     wt(a*"norm.weight"),eps,Int32(nh),Int32(nh),Val(DK.GDN_LANES)),
                    nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=a*"gdn")
                linear!(g,branch,wt(a*"out_proj.weight"),gdn;name=a*"out")
            else
                a=p*"self_attn."
                linear!(g,qfull,wt(a*"q_proj.weight"),norm;name=a*"q")
                linear!(g,kproj,wt(a*"k_proj.weight"),norm;name=a*"k")
                linear!(g,vproj,wt(a*"v_proj.weight"),norm;name=a*"v")
                dispatch(g,BR.prepare_q_kernel!,
                    (aq,qgate,qfull,wt(a*"q_norm.weight"),pos,theta,eps),nqh*256;name=a*"prepare_q")
                kc=persistent(kv_type,(256,nkv,capacity));vc=persistent(kv_type,(256,nkv,capacity))
                s.kv[il]=(kc,vc)
                dispatch(g,prepare_kv!,
                    (kc,vc,kproj,vproj,wt(a*"k_norm.weight"),pos,Int32(nkv),theta,eps),nkv*256;name=a*"prepare_kv")
                declare_attention!(g,s.attention,pm,ps,pa,aq,kc,vc,pos,nkv,nqh,chunks;name=a*"attention")
                dispatch(g,BR.attention_combine_kernel!,
                    (attn,pm,ps,pa,qgate,pos,Int32(chunks)),nqh*256;name=a*"combine")
                linear!(g,branch,wt(a*"o_proj.weight"),attn;name=a*"out")
            end
            dispatch(g,BR.add_rmsnorm_batch_kernel!,
                (norm,x,branch,wt(p*"post_attention_layernorm.weight"),Int32(w),eps,
                 Val(Mantle.caps(m.device).subgroup)),256;name=p*"residual_norm")
            mlp=declare_mlp!(g,m,branch,norm,gate,up,p;columns=1)
            residual=Mantle.viewof(x,(w,1))
            DK.standardop!(g,"add.Tensor",(residual,mlp);out=residual,name=p*"residual")
            if debug_layers
                dest=persistent(Float32,(w,));push!(s.layer_outputs,dest)
                dispatch(g,BR.copy_kernel!,(dest,x,Int32(w)),w;name=p*"debug_copy")
            end
        end
        dispatch(g,BR.rmsnorm_batch_kernel!,
            (norm,x,wt("norm.weight"),Int32(w),eps),256;name="final_norm")
        linear!(g,logits,wt("lm_head"),norm;name="lm_head")
        # Selection and token feedback follow the network in the same Mantle
        # graph. The hook declares GPU work before selection; its resources must
        # outlive the session.
        neural=DK.emitsemantic!(g);mg=neural.mantle;s.fusion=neural.report
        logit_hook===nothing || logit_hook(mg,logits)
        BR.declare_argmax!(mg,best,logits;name="argmax")
        dispatch(mg,advance_decode!,(token,pos,predictions,best),1;group=1,name="advance_decode")
        # Keep this linear decode graph in one submission batch.
        s.plan=Mantle.record!(Mantle.Plan(mg;budget=Inf,profile))
        s
    catch
        close(s);rethrow()
    end
end

"""The gated MLP of one layer, `down(swiglu(gate(norm), up(norm)))`, returned
for the residual add. The SwiGLU is a graph value, so the pattern passes may fold
it into the down projection and the emitter may fuse its store with that
product's Q8 input; a packed down projection's result is a graph value too, so
the residual add may fold into its store."""
function declare_mlp!(g,m,branch,norm,gate,up,p;columns)
    f=Int(m.config.intermediate_size);wt(name)=m.weights[name]
    linear_batch!(g,gate,wt(p*"mlp.gate_proj.weight"),norm,columns;name=p*"gate")
    linear_batch!(g,up,wt(p*"mlp.up_proj.weight"),norm,columns;name=p*"up")
    act=DK.standardop!(g,"fused.swiglu",(Mantle.viewof(gate,(f,columns)),Mantle.viewof(up,(f,columns)));
        out=(Float32,(f,columns)),name=p*"swiglu")
    mlp_down!(g,branch,wt(p*"mlp.down_proj.weight"),act,columns;name=p*"down")
end
mlp_down!(g::DK.SemanticGraph,branch,A::DK.KQuantMatrix,act,columns;name)=
    DK.standardop!(g,"mm.default",(act,A);out=(Float32,(size(A,1),columns)),name)
function mlp_down!(g,branch,A,act,columns;name)
    linear_batch!(g,branch,A,act,columns;name)
    Mantle.viewof(branch,(A.m,columns))
end
function Base.close(s::Session)
    for exec in values(s.prefills)
        Mantle.free!(exec.plan);Mantle.free!(exec.tokens);Mantle.free!(exec.positions)
    end
    empty!(s.prefills)
    s.plan===nothing || Mantle.free!(s.plan)
    foreach(Mantle.free!,s.owned);empty!(s.owned);s.plan=nothing;nothing
end
function step!(s::Session,token::Integer)
    s.position<s.capacity || throw(ArgumentError("session cache capacity exceeded"))
    0<=token<s.model.config.vocab_size || throw(ArgumentError("token outside vocabulary"))
    s.token[]=Int32(token);s.posref[]=Int32(s.position)
    Mantle.run!(s.plan);s.position+=1;s.logits
end

"""Queue greedy decode steps with token feedback and position updates on the GPU.

Call after at least one prompt step. No token readback or host wait occurs here;
the caller synchronizes once when it needs the generated output.
"""
function decode!(s::Session,n::Integer)
    s.position>0 || throw(ArgumentError("ingest a prompt before decoding"))
    0<=n<=s.capacity-s.position || throw(ArgumentError("session cache capacity exceeded"))
    for _ in 1:n
        Mantle.run!(s.plan)
        s.position+=1
    end
    s
end

# Explicit inspection only; generation uses decode! and keeps best on the GPU.
function greedy(s::Session)
    KernelAbstractions.synchronize(s.model.device)
    Int(only(Array(s.best)))
end

include("prefill_qwen35.jl")

end # module
