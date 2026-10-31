# Experimental full hybrid MoE graph, declared as a DNNKernels semantic graph.
# The hand-fused kernels selected by `optimizations` are opaque operations with
# declared results; plain projections and residuals are standard IR operations.
module Qwen36MoEExperiment
include("qwen35.jl")
include("moe_qwen35.jl")
import Mantle
using JSON3
const Q=Qwen35Experiment
const M=Qwen35MoE
const DK=Q.DK
const BR=Q.BR
const Model=Q.Model
include("qwen36_optim.jl")
const DEFAULT_OPTIMIZATIONS=(:reuse_q8,:narrow_experts,:small_dense,:fuse_shared,
    :down_wg64,:fuse_router_gate,:fuse_boundary_norm,:fuse_boundary_q8,
    :parallel_q8,:dense_unroll32,:down_rows2,:no_alias,:paired_expert_rows,:capability_swiglu_q8)

declare_embedding!(g,out,weight,token,width)=Q.dispatch(g,Q.embedding!,
    (out,weight,token,Int32(width)),width;name="embedding")
declare_embedding_batch!(g,out,weight,tokens,width,n)=Q.dispatch(g,Q.embedding_batch!,
    (out,weight,tokens,Int32(width),Int32(width*n)),width*n;name="prompt.embedding")

# GGUF's converted recurrent weights use tiled V-head order. Keep that order
# throughout the recurrent branch; the converted output projection undoes it.
# The order is the checkpoint's, so a session defaults to the model's; a caller
# that converts weights itself (the tiled-layout test) names its own.
recurrent_layout(m::Model)=m.quantize===:gguf_q4ks ? :tiled : :native
function repeat_qk_tiled!(q,k,conv,heads::Int32,keyheads::Int32)
    i=Int32(Q.KI.get_global_id().x)-Int32(1)
    if i<128heads
        source=(i÷Int32(128)%keyheads)*Int32(128)+i%Int32(128)+Int32(1)
        @inbounds begin q[i+Int32(1)]=conv[source];k[i+Int32(1)]=conv[128keyheads+source];end
    end
    nothing
end
DK.kernelresults(::typeof(repeat_qk_tiled!))=(1,2)

struct Session
    inner::Q.Session
    routing::Dict{Int,Any}
    # What the decode graph was declared with; prompt chunks declare the same.
    head_layout::Symbol
    optimizations::Tuple
end
Base.getproperty(s::Session,k::Symbol)=k in (:inner,:routing,:head_layout,:optimizations) ? getfield(s,k) : getproperty(getfield(s,:inner),k)
Base.close(s::Session)=close(s.inner)
step!(s::Session,t::Integer)=Q.step!(s.inner,t)
decode!(s::Session,n::Integer)=Q.decode!(s.inner,n)

function validate_configuration(c)
    c.model_type=="qwen3_5_moe_text"||throw(ArgumentError("Qwen3.5/3.6 MoE text config required"))
    c.head_dim==256&&c.linear_key_head_dim==c.linear_value_head_dim==128&&c.linear_conv_kernel_dim==4||throw(ArgumentError("Attention dimensions"))
    1<=c.num_experts_per_tok<=c.num_experts<=256||throw(ArgumentError("Router capacity"))
    c.hidden_size%256==c.moe_intermediate_size%256==c.shared_expert_intermediate_size%256==0||throw(ArgumentError("Quantization dimensions"))
    length(c.layer_types)==c.num_hidden_layers&&all(t->t in ("linear_attention","full_attention"),c.layer_types)||throw(ArgumentError("Layer layout"))
    c.num_attention_heads%c.num_key_value_heads==0&&c.linear_num_value_heads%c.linear_num_key_heads==0||throw(ArgumentError("Head grouping"))
    c.rope_parameters.rope_type=="default"&&c.rope_parameters.partial_rotary_factor==0.25||throw(ArgumentError("Native RoPE only"))
    c.tie_word_embeddings===false||throw(ArgumentError("Independent output head required"))
    nothing
end

# The wider subgroup amortizes the fused producer's shuffle/register work.
# Keep the unfused producer on narrower hardware; no device/API identity is
# involved. The explicit fusion flag remains available for measured ablations.
fuse_swiglu_q8(caps,optimizations)=(:fuse_swiglu_q8 in optimizations||(:capability_swiglu_q8 in optimizations&&caps.subgroup>=64))&&
    caps.subgroup>=32&&caps.subgroup%32==0

"""The selected experts' gate/up, SwiGLU and down products for every token of
`x`, into `routed` [width, top-k, tokens]. Decode and prompt chunks share it."""
function declare_experts!(g,m,routed,x,il,selected,status;optimizations)
    f=Int(m.config.moe_intermediate_size);topk,tokens=size(selected);p="layers.$il.mlp."
    wt(n)=m.weights[p*n]
    gu=Mantle.Transient.Buffer(g,Float32,(2f,topk,tokens));act=Mantle.Transient.Buffer(g,Float32,(f,topk,tokens))
    routed_opt!(g,gu,wt("experts.gate_up_proj"),x,selected,status;name=p*"experts.gate_up",optimizations)
    expert_act=if fuse_swiglu_q8(Mantle.caps(g.dev),optimizations)&&supports_q8(wt("experts.down_proj"))
        prepare_swiglu_q8!(g,act,gu,gu,f,topk*tokens;packed=true,status,topk,per_expert=true,name=p*"experts.swiglu_q8")
    else
        Q.dispatch(g,M.swiglu_packed!,(act,gu,Int32(f),Int32(length(act))),length(act);name=p*"experts.swiglu")
        act
    end
    routed_opt!(g,routed,wt("experts.down_proj"),expert_act,selected,status;per_expert=true,name=p*"experts.down",optimizations)
end

"""Selected experts and shared branch declared into the caller's graph.

Uses the existing shared dense/Q4 arithmetic and native expert kernels. Router
IDs/status remain device resident; no per-layer/expert CPU feedback/submissions.
"""
function declare_ffn!(g,m,out,x,il,selected,routing,status;optimizations=())
    c=m.config;w=Int(c.hidden_size);f=Int(c.moe_intermediate_size);sf=Int(c.shared_expert_intermediate_size)
    topk=Int(c.num_experts_per_tok);ne=Int(c.num_experts);p="layers.$il.mlp."
    scratch(dims)=Mantle.Transient.Buffer(g,Float32,dims)
    x2=ndims(x)==2 ? x : Mantle.viewof(x,(w,1))
    out2=ndims(out)==2 ? out : Mantle.viewof(out,(w,1))
    fused_router=:fuse_router_gate in optimizations&&haskey(m.weights,p*"router_and_shared_gate")
    router_gate=fused_router ? scratch((ne+1,)) : nothing
    logits=fused_router ? Mantle.viewof(router_gate,(ne,1)) : scratch((ne,1))
    routed=scratch((w,topk,1))
    fused=:fuse_shared in optimizations&&haskey(m.weights,p*"shared_expert.gate_up_proj")
    shared_gu=fused ? scratch((2sf,)) : nothing
    sg=fused ? Mantle.viewof(shared_gu,(sf,)) : scratch((sf,))
    su=fused ? Mantle.viewof(shared_gu,(sf,);offset=sf) : scratch((sf,))
    shared=scratch((w,1));gate=fused_router ? Mantle.viewof(router_gate,(1,);offset=ne) : scratch((1,))
    wt(n)=m.weights[p*n]
    if fused_router
        linear_opt!(g,router_gate,wt("router_and_shared_gate"),x;name=p*"gate_logits_and_shared_scalar",optimizations)
    else
        linear_opt!(g,logits,wt("gate.weight"),x;name=p*"gate_logits",optimizations)
    end
    M.declare_router!(g,selected,routing,status,logits;name=p*"route")
    # Guard the one shared producer. On invalid tokens the final combine already
    # masks the entire FFN; no consumer converts invalid activations to integers.
    qx=maybe_prepare_q8!(g,x,(wt("experts.gate_up_proj"),wt("shared_expert.gate_proj.weight"),wt("shared_expert.up_proj.weight"));
        optimizations,status,name=p*"shared_input.q8")
    declare_experts!(g,m,routed,qx isa Q8Input ? qx : x2,il,selected,status;optimizations)
    fuse_q8=fuse_swiglu_q8(Mantle.caps(g.dev),optimizations)
    if fused
        linear_opt!(g,shared_gu,wt("shared_expert.gate_up_proj"),qx;name=p*"shared.gate_up",optimizations)
    else
        linear_opt!(g,sg,wt("shared_expert.gate_proj.weight"),qx;name=p*"shared.gate",optimizations)
        linear_opt!(g,su,wt("shared_expert.up_proj.weight"),qx;name=p*"shared.up",optimizations)
    end
    shared_act=if fuse_q8&&supports_q8(wt("shared_expert.down_proj.weight"))
        prepare_swiglu_q8!(g,scratch((sf,)),sg,su,sf,1;name=p*"shared.swiglu_q8")
    elseif fused
        # Windows of the concatenated projection: an opaque kernel reads them.
        sa=scratch((sf,))
        Q.dispatch(g,Q.swiglu!,(sa,sg,su,Int32(sf)),sf;name=p*"shared.swiglu")
        sa
    else
        # A graph value: the shared passes may fold it into the down projection.
        DK.standardop!(g,"fused.swiglu",(Mantle.viewof(sg,(sf,1)),Mantle.viewof(su,(sf,1)));
            out=(Float32,(sf,1)),name=p*"shared.swiglu")
    end
    linear_opt!(g,shared,wt("shared_expert.down_proj.weight"),shared_act;name=p*"shared.down",optimizations)
    fused_router||linear_opt!(g,gate,wt("shared_expert_gate.weight"),x;name=p*"shared.scalar_gate",optimizations)
    M.declare_combine!(g,out2,routed,shared,gate,routing,status;name=p*"combine")
end

function session(m::Model;capacity=4096,debug_layers=false,logit_hook=nothing,profile=false,
                 kv_type=Float32,attention=:grouped,head_layout=recurrent_layout(m),optimizations=DEFAULT_OPTIMIZATIONS)
    c=m.config;validate_configuration(c)
    0<capacity<=c.max_position_embeddings||throw(ArgumentError("Capacity"))
    kv_type in (Float32,Float16)||throw(ArgumentError("KV type"))
    attention in (:split,:grouped)||throw(ArgumentError("Attention implementation"))
    head_layout in (:native,:tiled)||throw(ArgumentError("Recurrent head layout"))
    all(x->x in (:reuse_q8,:narrow_experts,:small_dense,:fuse_shared,:fuse_router_gate,:down_wg64,:router_wg64,:fuse_boundary_norm,:fuse_boundary_q8,:dense_unroll16,:dense_unroll32,:parallel_q8,:down_rows2,:no_alias,:compact_schedule,:fuse_swiglu_q8,:capability_swiglu_q8,:capability_dense,:paired_expert_rows),optimizations)||throw(ArgumentError("Unknown optimization"))
    :fuse_boundary_q8 in optimizations&&!(:fuse_boundary_norm in optimizations&&:reuse_q8 in optimizations)&&throw(ArgumentError("Boundary Q8 requires boundary norm and Q8 reuse"))
    :fuse_boundary_q8 in optimizations&&c.hidden_size>8192&&throw(ArgumentError("Boundary Q8 supports widths up to8192"))
    :fuse_shared in optimizations&&prepare_shared_gateup!(m)
    :fuse_router_gate in optimizations&&prepare_router_gate!(m)
    attention==:grouped&&!(c.num_attention_heads÷c.num_key_value_heads in (1,2,4,8))&&throw(ArgumentError("Grouped attention ratio"))
    owned=Any[]
    persistent(T,dims)=begin a=Mantle.Buffer(m.device,T,dims);fill!(a,zero(T));push!(owned,a);a;end
    token=Mantle.GPURef(m.device,Int32(0));pos=Mantle.GPURef(m.device,Int32(0));push!(owned,token,pos)
    logits=persistent(Float32,(Int(c.vocab_size),));best=persistent(Int32,(1,));predictions=persistent(Int32,(capacity,))
    s=Q.Session(m,capacity,0,token,pos,logits,best,predictions,nothing,owned,Any[],Dict{Int,Any}(),Dict{Int,Any}(),Dict{Int,Any}(),logit_hook,attention)
    routes=Dict{Int,Any}()
    try
        g=DK.SemanticGraph(m.device;name="qwen36_decode");DK.semanticweights!(g,Q.neuralweights(m))
        scratch(n)=Mantle.Transient.Buffer(g,Float32,(Int(n),))
        w=Int(c.hidden_size);nh=Int(c.linear_num_value_heads);nkh=Int(c.linear_num_key_heads)
        kd=128nkh;vd=128nh;nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads);chunks=cld(capacity,64)
        x=scratch(w);norm=scratch(w);branch=scratch(w)
        boundary_q8=:fuse_boundary_q8 in optimizations ? Q8Input(norm,Mantle.Transient.Buffer(g,DK.UVec4,(w÷16,1)),Mantle.Transient.Buffer(g,Float32,(w÷16,1)),w,1) : nothing
        function boundary_dispatch(weight,add,name)
            if boundary_q8===nothing
                Q.dispatch(g,boundary_norm!,(norm,x,branch,weight,Int32(w),eps,Val(add),Val(Mantle.caps(m.device).subgroup)),256;name)
            else
                Q.dispatch(g,boundary_norm_q8!,(norm,x,branch,weight,boundary_q8.q,boundary_q8.ds,Int32(w),eps,Val(add),Val(Mantle.caps(m.device).subgroup),Val(w)),256;name)
            end
        end
        qkv=scratch(2kd+vd);z=scratch(vd);conv=scratch(2kd+vd)
        rq=scratch(vd);rk=scratch(vd);alpha=scratch(nh);beta=scratch(nh);gdn=scratch(vd)
        qfull=scratch(512nqh);kproj=scratch(256nkv);vproj=scratch(256nkv)
        aq=scratch(256nqh);qgate=scratch(256nqh);attn=scratch(256nqh)
        pm=scratch(nqh*chunks);ps=scratch(nqh*chunks);pa=scratch(nqh*chunks*256)
        wt(name)=m.weights[name];eps=Float32(c.rms_norm_eps);theta=Float32(c.rope_parameters.rope_theta)
        function snapshot(source,name)
            debug_layers||return nothing
            dest=persistent(Float32,(w,))
            Q.dispatch(g,BR.copy_kernel!,(dest,source,Int32(w)),w;name)
            dest
        end
        declare_embedding!(g,x,wt("embed_tokens.weight"),token,w)
        for il in 0:Int(c.num_hidden_layers)-1
            p="layers.$il."
            if :fuse_boundary_norm in optimizations
                il==0&&boundary_dispatch(wt(p*"input_layernorm.weight"),false,p*"input_norm")
            else
                Q.dispatch(g,BR.rmsnorm_batch_kernel!,(norm,x,wt(p*"input_layernorm.weight"),Int32(w),eps),256;name=p*"input_norm")
            end
            input_norm_snapshot=snapshot(norm,p*"debug_input_norm")
            if c.layer_types[il+1]=="linear_attention"
                a=p*"linear_attn."
                input=boundary_q8===nothing ? maybe_prepare_q8!(g,norm,Tuple(wt(a*n) for n in ("in_proj_qkv.weight","in_proj_z.weight","in_proj_a.weight","in_proj_b.weight"));
                    optimizations,name=a*"shared_input.q8") : boundary_q8
                for (out,name) in ((qkv,"in_proj_qkv.weight"),(z,"in_proj_z.weight"),(alpha,"in_proj_a.weight"),(beta,"in_proj_b.weight"))
                    linear_opt!(g,out,wt(a*name),input;name=a*name,optimizations)
                end
                hist=persistent(Float32,(3,2kd+vd));state=persistent(Float32,(128,128,nh));s.recurrent[il]=(hist,state)
                Q.dispatch(g,DK.depthwise_conv4_kernel!,(conv,hist,qkv,wt(a*"conv1d.weight"),Int32(2kd+vd)),2kd+vd;name=a*"conv")
                repeat_kernel=head_layout==:native ? Q.repeat_qk! : repeat_qk_tiled!
                Q.dispatch(g,repeat_kernel,(rq,rk,conv,Int32(nh),Int32(nkh)),vd;name=a*"repeat_qk")
                v=Mantle.viewof(conv,(vd,);offset=2kd)
                Q.dispatch(g,DK.gated_delta_net_kernel!,
                    (gdn,state,rq,rk,v,alpha,beta,wt(a*"dt_bias"),wt(a*"A_log"),z,wt(a*"norm.weight"),eps,Int32(nh),Int32(nh),Val(DK.GDN_LANES)),
                    nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=a*"gdn")
                Q.linear!(g,branch,wt(a*"out_proj.weight"),gdn;name=a*"out")
            else
                a=p*"self_attn."
                input=boundary_q8===nothing ? maybe_prepare_q8!(g,norm,Tuple(wt(a*n) for n in ("q_proj.weight","k_proj.weight","v_proj.weight"));
                    optimizations,name=a*"shared_input.q8") : boundary_q8
                linear_opt!(g,qfull,wt(a*"q_proj.weight"),input;name=a*"q",optimizations)
                linear_opt!(g,kproj,wt(a*"k_proj.weight"),input;name=a*"k",optimizations)
                linear_opt!(g,vproj,wt(a*"v_proj.weight"),input;name=a*"v",optimizations)
                Q.dispatch(g,BR.prepare_q_kernel!,(aq,qgate,qfull,wt(a*"q_norm.weight"),pos,theta,eps),nqh*256;name=a*"prepare_q")
                kc=persistent(kv_type,(256,nkv,capacity));vc=persistent(kv_type,(256,nkv,capacity));s.kv[il]=(kc,vc)
                Q.dispatch(g,Q.prepare_kv!,(kc,vc,kproj,vproj,wt(a*"k_norm.weight"),pos,Int32(nkv),theta,eps),nkv*256;name=a*"prepare_kv")
                Q.declare_attention!(g,attention,pm,ps,pa,aq,kc,vc,pos,nkv,nqh,chunks;name=a*"attention")
                Q.dispatch(g,BR.attention_combine_kernel!,(attn,pm,ps,pa,qgate,pos,Int32(chunks)),nqh*256;name=a*"combine")
                Q.linear!(g,branch,wt(a*"o_proj.weight"),attn;name=a*"out")
            end
            mixer_snapshot=snapshot(branch,p*"debug_mixer")
            Q.dispatch(g,BR.add_rmsnorm_batch_kernel!,
                (norm,x,branch,wt(p*"post_attention_layernorm.weight"),Int32(w),eps,Val(Mantle.caps(m.device).subgroup)),256;name=p*"residual_norm")
            selected=persistent(Int32,(Int(c.num_experts_per_tok),1));routing=persistent(Float32,size(selected));status=persistent(Int32,(1,))
            post_norm_snapshot=snapshot(norm,p*"debug_post_norm")
            declare_ffn!(g,m,branch,norm,il,selected,routing,status;optimizations)
            ffn_snapshot=snapshot(branch,p*"debug_ffn")
            routes[il]=(;selected,weights=routing,status,
                debug=(;input_norm=input_norm_snapshot,mixer=mixer_snapshot,post_norm=post_norm_snapshot,ffn=ffn_snapshot))
            if :fuse_boundary_norm in optimizations
                next_weight=il+1<Int(c.num_hidden_layers) ? wt("layers.$(il+1).input_layernorm.weight") : wt("norm.weight")
                boundary_dispatch(next_weight,true,p*"residual_next_norm")
            else
                DK.standardop!(g,"add.Tensor",(x,branch);out=x,name=p*"residual")
            end
            if debug_layers
                dest=persistent(Float32,(w,));push!(s.layer_outputs,dest)
                Q.dispatch(g,BR.copy_kernel!,(dest,x,Int32(w)),w;name=p*"debug_copy")
            end
        end
        :fuse_boundary_norm in optimizations||Q.dispatch(g,BR.rmsnorm_batch_kernel!,(norm,x,wt("norm.weight"),Int32(w),eps),256;name="final_norm")
        if boundary_q8===nothing
            Q.linear!(g,logits,wt("lm_head"),norm;name="lm_head")
        else
            linear_opt!(g,logits,wt("lm_head"),boundary_q8;name="lm_head",optimizations)
        end
        # Selection and token feedback follow the network in the same Mantle graph.
        neural=DK.emitsemantic!(g);mg=neural.mantle;s.fusion=neural.report
        logit_hook===nothing||logit_hook(mg,logits)
        BR.declare_argmax!(mg,best,logits;name="argmax")
        Q.dispatch(mg,Q.advance_decode!,(token,pos,predictions,best),1;group=1,name="advance_decode")
        policy=:compact_schedule in optimizations ? Mantle.Compact() : Mantle.Overlap()
        s.plan=Mantle.record!(Mantle.Plan(mg;budget=Inf,profile,alias=!(:no_alias in optimizations),policy))
        Session(s,routes,head_layout,Tuple(optimizations))
    catch
        close(s);rethrow()
    end
end

include("qwen36_prefill.jl")
end
