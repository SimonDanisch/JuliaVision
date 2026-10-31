# Included at the end of Qwen36MoEExperiment. A prompt chunk of `n` tokens as
# one semantic graph: projections and the MoE FFN span the chunk's columns, the
# recurrent and attention state advance column by column with the decode
# kernels, and only the chunk's last token produces logits.

"""The routed and shared experts over a prompt chunk's `n` columns, into `out`.

The router, expert kernels and combine are the decode graph's, over `n` tokens.
The decode layouts that concatenate the router with the shared scalar gate, or
the shared gate with its up projection, give each column both halves back to
back; the chunk reads the separate weights instead of strided windows.
"""
function declare_ffn_batch!(g,m,out,x,il,n;optimizations)
    c=m.config;w=Int(c.hidden_size);sf=Int(c.shared_expert_intermediate_size)
    topk=Int(c.num_experts_per_tok);ne=Int(c.num_experts);p="layers.$il.mlp."
    wt(name)=m.weights[p*name]
    scratch(T,dims)=Mantle.Transient.Buffer(g,T,dims)
    logits=scratch(Float32,(ne,n))
    Q.linear_batch!(g,logits,wt("gate.weight"),x,n;name=p*"gate_logits")
    selected=scratch(Int32,(topk,n));routing=scratch(Float32,(topk,n));status=scratch(Int32,(n,))
    M.declare_router!(g,selected,routing,status,logits;name=p*"route")
    routed=scratch(Float32,(w,topk,n))
    declare_experts!(g,m,routed,x,il,selected,status;optimizations)
    sg=scratch(Float32,(sf,n));su=scratch(Float32,(sf,n))
    Q.linear_batch!(g,sg,wt("shared_expert.gate_proj.weight"),x,n;name=p*"shared.gate")
    Q.linear_batch!(g,su,wt("shared_expert.up_proj.weight"),x,n;name=p*"shared.up")
    sa=DK.standardop!(g,"fused.swiglu",(sg,su);out=(Float32,(sf,n)),name=p*"shared.swiglu")
    shared=scratch(Float32,(w,n))
    Q.linear_batch!(g,shared,wt("shared_expert.down_proj.weight"),sa,n;name=p*"shared.down")
    gate=scratch(Float32,(1,n))
    Q.linear_batch!(g,gate,wt("shared_expert_gate.weight"),x,n;name=p*"shared.scalar_gate")
    M.declare_combine!(g,out,routed,shared,gate,routing,status;name=p*"combine")
end

function record_prefill(s::Session,n::Int)
    inner=s.inner;m=inner.model;c=m.config
    tokens=Mantle.Buffer(m.device,zeros(Int32,n))
    positions=nothing;plan=nothing
    try
        positions=Mantle.Buffer(m.device,zeros(Int32,n))
        g=DK.SemanticGraph(m.device;name="qwen36_prefill_$n")
        DK.semanticweights!(g,Q.neuralweights(m))
        scratch(rows)=Mantle.Transient.Buffer(g,Float32,(Int(rows),n))
        w=Int(c.hidden_size);nh=Int(c.linear_num_value_heads);nkh=Int(c.linear_num_key_heads)
        kd=128nkh;vd=128nh;channels=2kd+vd
        nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads)
        chunks=cld(inner.capacity,64);eps=Float32(c.rms_norm_eps)
        theta=Float32(c.rope_parameters.rope_theta);sg=Mantle.caps(m.device).subgroup
        column=Q.column
        x=scratch(w);norm=scratch(w);branch=scratch(w)
        qkv=scratch(channels);z=scratch(vd);conv=scratch(channels)
        rq=scratch(vd);rk=scratch(vd);alpha=scratch(nh);beta=scratch(nh);gdn=scratch(vd)
        qfull=scratch(512nqh);kproj=scratch(256nkv);vproj=scratch(256nkv)
        aq=scratch(256nqh);qgate=scratch(256nqh);attn=scratch(256nqh)
        pm=Mantle.Transient.Buffer(g,Float32,(nqh*chunks,))
        ps=Mantle.Transient.Buffer(g,Float32,(nqh*chunks,))
        pa=Mantle.Transient.Buffer(g,Float32,(nqh*chunks*256,))
        wt(name)=m.weights[name]
        repeat_kernel=s.head_layout===:native ? Q.repeat_qk! : repeat_qk_tiled!
        Q.dispatch(g,Q.prompt_positions!,(positions,inner.posref,Int32(n)),n;name="prompt.positions")
        declare_embedding_batch!(g,x,wt("embed_tokens.weight"),tokens,w,n)
        for il in 0:Int(c.num_hidden_layers)-1
            p="layers.$il."
            Q.dispatch(g,BR.rmsnorm_batch_kernel!,
                (norm,x,wt(p*"input_layernorm.weight"),Int32(w),eps),n*256;name=p*"input_norm")
            if c.layer_types[il+1]=="linear_attention"
                a=p*"linear_attn."
                Q.linear_batch!(g,qkv,wt(a*"in_proj_qkv.weight"),norm,n;name=a*"qkv")
                Q.linear_batch!(g,z,wt(a*"in_proj_z.weight"),norm,n;name=a*"z")
                Q.linear_batch!(g,alpha,wt(a*"in_proj_a.weight"),norm,n;name=a*"alpha")
                Q.linear_batch!(g,beta,wt(a*"in_proj_b.weight"),norm,n;name=a*"beta")
                hist,state=inner.recurrent[il]
                Q.dispatch(g,BR.depthwise_conv4_batch_kernel!,
                    (conv,hist,qkv,wt(a*"conv1d.weight"),Int32(channels),Int32(n)),channels;name=a*"conv")
                for j in 0:n-1
                    Q.dispatch(g,repeat_kernel,
                        (column(rq,vd,j),column(rk,vd,j),column(conv,channels,j),Int32(nh),Int32(nkh)),
                        vd;name=a*"repeat_qk.$j")
                    v=Mantle.viewof(conv,(vd,);offset=j*channels+2kd)
                    Q.dispatch(g,DK.gated_delta_net_kernel!,
                        (column(gdn,vd,j),state,column(rq,vd,j),column(rk,vd,j),v,
                         column(alpha,nh,j),column(beta,nh,j),wt(a*"dt_bias"),wt(a*"A_log"),
                         column(z,vd,j),wt(a*"norm.weight"),eps,Int32(nh),Int32(nh),Val(DK.GDN_LANES)),
                        nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=a*"gdn.$j")
                end
                Q.linear_batch!(g,branch,wt(a*"out_proj.weight"),gdn,n;name=a*"out")
            else
                a=p*"self_attn."
                Q.linear_batch!(g,qfull,wt(a*"q_proj.weight"),norm,n;name=a*"q")
                Q.linear_batch!(g,kproj,wt(a*"k_proj.weight"),norm,n;name=a*"k")
                Q.linear_batch!(g,vproj,wt(a*"v_proj.weight"),norm,n;name=a*"v")
                kc,vc=inner.kv[il]
                for j in 0:n-1
                    pos=column(positions,1,j)
                    Q.dispatch(g,BR.prepare_q_kernel!,
                        (column(aq,256nqh,j),column(qgate,256nqh,j),column(qfull,512nqh,j),
                         wt(a*"q_norm.weight"),pos,theta,eps),nqh*256;name=a*"prepare_q.$j")
                    Q.dispatch(g,Q.prepare_kv!,
                        (kc,vc,column(kproj,256nkv,j),column(vproj,256nkv,j),wt(a*"k_norm.weight"),
                         pos,Int32(nkv),theta,eps),nkv*256;name=a*"prepare_kv.$j")
                    Q.declare_attention!(g,inner.attention,pm,ps,pa,column(aq,256nqh,j),kc,vc,pos,
                        nkv,nqh,chunks;name=a*"attention.$j")
                    Q.dispatch(g,BR.attention_combine_kernel!,
                        (column(attn,256nqh,j),pm,ps,pa,column(qgate,256nqh,j),pos,Int32(chunks)),
                        nqh*256;name=a*"combine.$j")
                end
                Q.linear_batch!(g,branch,wt(a*"o_proj.weight"),attn,n;name=a*"out")
            end
            Q.dispatch(g,BR.add_rmsnorm_batch_kernel!,
                (norm,x,branch,wt(p*"post_attention_layernorm.weight"),Int32(w),eps,Val(sg)),
                n*256;name=p*"residual_norm")
            declare_ffn_batch!(g,m,branch,norm,il,n;optimizations=s.optimizations)
            DK.standardop!(g,"add.Tensor",(x,branch);out=x,name=p*"residual")
            if !isempty(inner.layer_outputs)
                Q.dispatch(g,BR.copy_kernel!,
                    (inner.layer_outputs[il+1],column(x,w,n-1),Int32(w)),w;name=p*"debug_copy")
            end
        end
        lastnorm=Mantle.Transient.Buffer(g,Float32,(w,))
        Q.dispatch(g,BR.rmsnorm_batch_kernel!,
            (lastnorm,column(x,w,n-1),wt("norm.weight"),Int32(w),eps),256;name="final_norm")
        Q.linear!(g,inner.logits,wt("lm_head"),lastnorm;name="lm_head")
        mg=DK.emitsemantic!(g).mantle
        inner.logit_hook===nothing || inner.logit_hook(mg,inner.logits)
        BR.declare_argmax!(mg,inner.best,inner.logits;name="argmax")
        Q.dispatch(mg,Q.advance_prompt!,(inner.token,inner.posref,inner.predictions,inner.best,Int32(n)),1;
            group=1,name="advance_prompt")
        plan=Mantle.record!(Mantle.Plan(mg;budget=Inf))
        (;tokens,positions,plan)
    catch
        plan===nothing || Mantle.free!(plan)
        positions===nothing || Mantle.free!(positions)
        Mantle.free!(tokens)
        rethrow()
    end
end

"""Ingest prompt chunks with batched projections and experts, sharing the decode
state. Only the final token's logits are produced per chunk.

The default chunk is the widest product the shared emitter still runs as Q8 GEMV
passes, which reproduce sequential decode bit for bit. Wider chunks take the
K-quant GEMM on fp16 activations: on the 8060S 32 columns ingest 8% faster and
64 columns 21%, at a KL of 0.006 against sequential decode on a 20-token prompt.
"""
function prefill!(s::Session,ids;chunk=DK.Q4K_GEMV_COLUMNS[])
    inner=s.inner
    chunk>0 || throw(ArgumentError("chunk must be positive"))
    length(ids)<=inner.capacity-inner.position || throw(ArgumentError("session cache capacity exceeded"))
    all(t->0<=t<inner.model.config.vocab_size,ids) || throw(ArgumentError("token outside vocabulary"))
    for start in 1:chunk:length(ids)
        n=min(chunk,length(ids)-start+1)
        if n==1
            step!(s,ids[start])
        else
            exec=get!(()->record_prefill(s,n),inner.prefills,n)
            exec.tokens[:]=Int32.(ids[start:start+n-1])
            inner.posref[]=Int32(inner.position)
            Mantle.run!(exec.plan)
            inner.position+=n
        end
    end
    inner.logits
end
