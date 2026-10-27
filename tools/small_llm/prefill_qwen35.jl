# Included inside Qwen35Experiment. Projections and MLPs span a prompt chunk;
# recurrent updates retain the validated decode arithmetic and shared state.
function embedding_batch!(out,weight,tokens,width::Int32,total::Int32)
    i=Int32(KI.get_global_id().x)-Int32(1)
    if i<total
        col=i÷width;row=i%width
        @inbounds out[i+Int32(1)]=weight[tokens[col+Int32(1)]*width+row+Int32(1)]
    end
    nothing
end

function prompt_positions!(positions,start,n::Int32)
    i=Int32(KI.get_global_id().x)
    i<=n && (@inbounds positions[i]=start[1]+i-Int32(1))
    nothing
end

function advance_prompt!(token,position,predictions,best,n::Int32)
    if KI.get_global_id().x==1
        @inbounds begin
            predictions[position[1]+n]=best[1]
            token[1]=best[1]
            position[1]+=n
        end
    end
    nothing
end

column(a,width,j)=Mantle.viewof(a,(width,);offset=j*width)

"""Whether a chunk of `n` columns runs the decode kernels column by column, which
reproduces sequential decode bit for bit: the products up to the shared
emitter's GEMV width do. Wider chunks have given that up in their products, and
take the chunk kernels for the recurrence and the attention as well."""
exactchunk(n)=n<=DK.Q4K_GEMV_COLUMNS[]

# A dense Float32 weight over a wide chunk: 64x64 output tiles, 16 of the
# reduction at a time staged in shared memory, 4x4 outputs a thread. Weight rows
# and activation columns are both contiguous along the reduction. The batch GEMV
# took a 64-lane reduction per output and re-read the weight for every column:
# 0.42 ms for the router's 256x2048 at 256 columns.
const DENSE_TILE=64
const DENSE_STEP=16
function dense_gemm_tn!(out,W,x,K::Int32,M::Int32,N::Int32)
    V4=NTuple{4,VecElement{Float32}}
    ws=KI.localmemory(Float32,Val((DENSE_STEP*DENSE_TILE,)),Val(1))     # (k, m) at k*64 + m
    xs=KI.localmemory(Float32,Val((DENSE_STEP*DENSE_TILE,)),Val(2))     # (k, n) at k*64 + n
    tid=Int32(KI.get_local_id().x)-Int32(1)
    wg=Int32(KI.get_group_id().x)-Int32(1)
    mt=cld(M,Int32(DENSE_TILE))
    m0=(wg%mt)*Int32(DENSE_TILE);n0=(wg÷mt)*Int32(DENSE_TILE)
    tm=(tid%Int32(16))*Int32(4);tn=(tid÷Int32(16))*Int32(4)
    # This thread stages four reduction values of row/column `lr` from `lc`.
    lr=tid÷Int32(4);lc=(tid%Int32(4))*Int32(4)
    wrow=min(m0+lr,M-Int32(1));xcol=min(n0+lr,N-Int32(1))
    w4=reinterpret(V4,W);x4=reinterpret(V4,x)
    Base.Cartesian.@nexprs 4 i->Base.Cartesian.@nexprs 4 j->(acc_i_j=0f0)
    k0=Int32(0)
    while k0<K
        wv=@inbounds w4[(wrow*K+k0+lc)÷Int32(4)+Int32(1)]
        xv=@inbounds x4[(xcol*K+k0+lc)÷Int32(4)+Int32(1)]
        Base.Cartesian.@nexprs 4 c->begin
            @inbounds ws[(lc+Int32(c-1))*Int32(DENSE_TILE)+lr+Int32(1)]=wv[c].value
            @inbounds xs[(lc+Int32(c-1))*Int32(DENSE_TILE)+lr+Int32(1)]=xv[c].value
        end
        KI.barrier()
        for kk in Int32(0):Int32(DENSE_STEP-1)
            Base.Cartesian.@nexprs 4 i->(a_i=@inbounds ws[kk*Int32(DENSE_TILE)+tm+Int32(i)])
            Base.Cartesian.@nexprs 4 j->(b_j=@inbounds xs[kk*Int32(DENSE_TILE)+tn+Int32(j)])
            Base.Cartesian.@nexprs 4 i->Base.Cartesian.@nexprs 4 j->(acc_i_j=muladd(a_i,b_j,acc_i_j))
        end
        KI.barrier()
        k0+=Int32(DENSE_STEP)
    end
    Base.Cartesian.@nexprs 4 i->Base.Cartesian.@nexprs 4 j->begin
        row=m0+tm+Int32(i-1);col=n0+tn+Int32(j-1)
        row<M&&col<N&&(@inbounds out[col*M+row+Int32(1)]=eltype(out)(acc_i_j))
    end
    nothing
end
DK.kernelresults(::typeof(dense_gemm_tn!))=(1,)

function linear_batch!(g,out,A::DenseWeight,x,n;name)
    if exactchunk(n)||A.k%DENSE_STEP!=0
        dispatch(g,BR.dense_gemv_batch_kernel!,
            (out,A.data,x,Int32(A.k),Int32(A.m),Int32(cld(A.m,4)),Val(Mantle.caps(g.dev).subgroup)),
            cld(A.m,4)*n*256;name)
    else
        dispatch(g,dense_gemm_tn!,(out,A.data,x,Int32(A.k),Int32(A.m),Int32(n)),
            cld(A.m,DENSE_TILE)*cld(n,DENSE_TILE)*256;group=256,name)
    end
    out
end

# A standard product over the chunk's columns. The shared emitter selects the
# kernel and its activation precision from the column count and capabilities.
linear_batch!(g::DK.SemanticGraph,out,A::DK.KQuantMatrix,x,n;name)=linear!(g,out,A,x;name)

for (kernel,writes) in ((embedding_batch!,(1,)),(prompt_positions!,(1,)),(advance_prompt!,(1,2,3)))
    @eval DK.kernelresults(::typeof($kernel))=$writes
end

function record_prefill(s::Session,n::Int)
    m=s.model;c=m.config
    tokens=Mantle.Buffer(m.device,zeros(Int32,n))
    positions=nothing;plan=nothing
    try
        positions=Mantle.Buffer(m.device,zeros(Int32,n))
        g=DK.SemanticGraph(m.device;name="qwen35_prefill_$n")
        DK.semanticweights!(g,neuralweights(m))
        scratch(rows)=Mantle.Transient.Buffer(g,Float32,(Int(rows),n))
        w=Int(c.hidden_size);f=Int(c.intermediate_size)
        nh=Int(c.linear_num_value_heads);nkh=Int(c.linear_num_key_heads)
        kd=128nkh;vd=128nh;channels=2kd+vd
        nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads)
        chunks=attention_slots(s.attention,s.capacity);eps=Float32(c.rms_norm_eps)
        theta=Float32(c.rope_parameters.rope_theta);sg=Mantle.caps(m.device).subgroup
        x=scratch(w);norm=scratch(w);branch=scratch(w)
        gate=scratch(f);up=scratch(f)
        qkv=scratch(channels);z=scratch(vd);conv=scratch(channels)
        rq=exactchunk(n) ? scratch(vd) : nothing;rk=exactchunk(n) ? scratch(vd) : nothing;alpha=scratch(nh);beta=scratch(nh);gdn=scratch(vd)
        qfull=scratch(512nqh);kproj=scratch(256nkv);vproj=scratch(256nkv)
        aq=scratch(256nqh);qgate=scratch(256nqh);attn=scratch(256nqh)
        # A chunk wider than the bit-exact products attends over the cache once;
        # narrower ones run the decode attention token by token, with one
        # query's split scratch at a time.
        batched=chunk_attention(m.device,n,nkv,nqh)
        pm=batched ? nothing : Mantle.Transient.Buffer(g,Float32,(nqh*chunks,))
        ps=batched ? nothing : Mantle.Transient.Buffer(g,Float32,(nqh*chunks,))
        pa=batched ? nothing : Mantle.Transient.Buffer(g,Float32,(nqh*chunks*256,))
        wt(name)=m.weights[name]
        dispatch(g,prompt_positions!,(positions,s.posref,Int32(n)),n;name="prompt.positions")
        dispatch(g,embedding_batch!,
            (x,wt("embed_tokens.weight"),tokens,Int32(w),Int32(w*n)),w*n;name="prompt.embedding")
        for il in 0:Int(c.num_hidden_layers)-1
            p="layers.$il."
            dispatch(g,BR.rmsnorm_batch_kernel!,
                (norm,x,wt(p*"input_layernorm.weight"),Int32(w),eps),n*256;name=p*"input_norm")
            if String(c.layer_types[il+1])=="linear_attention"
                a=p*"linear_attn."
                linear_batch!(g,qkv,wt(a*"in_proj_qkv.weight"),norm,n;name=a*"qkv")
                linear_batch!(g,z,wt(a*"in_proj_z.weight"),norm,n;name=a*"z")
                linear_batch!(g,alpha,wt(a*"in_proj_a.weight"),norm,n;name=a*"alpha")
                linear_batch!(g,beta,wt(a*"in_proj_b.weight"),norm,n;name=a*"beta")
                hist,state=s.recurrent[il]
                dispatch(g,BR.depthwise_conv4_batch_kernel!,
                    (conv,hist,qkv,wt(a*"conv1d.weight"),Int32(channels),Int32(n)),channels;name=a*"conv")
                if exactchunk(n)
                    for j in 0:n-1
                        dispatch(g,repeat_qk!,
                            (column(rq,vd,j),column(rk,vd,j),column(conv,channels,j),Int32(nh),Int32(nkh)),
                            vd;name=a*"repeat_qk.$j")
                        v=Mantle.viewof(conv,(vd,);offset=j*channels+2kd)
                        dispatch(g,DK.gated_delta_net_kernel!,
                            (column(gdn,vd,j),state,column(rq,vd,j),column(rk,vd,j),v,
                             column(alpha,nh,j),column(beta,nh,j),wt(a*"dt_bias"),wt(a*"A_log"),
                             column(z,vd,j),wt(a*"norm.weight"),eps,Int32(nh),Int32(nh),Val(DK.GDN_LANES)),
                            nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=a*"gdn.$j")
                    end
                else
                    dispatch(g,DK.gated_delta_net_chunk_kernel!,
                        (gdn,state,conv,alpha,beta,wt(a*"dt_bias"),wt(a*"A_log"),z,wt(a*"norm.weight"),
                         eps,Int32(nh),Int32(nkh),Int32(n),Val(DK.GDN_LANES),Val(false)),
                        nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=a*"gdn")
                end
                linear_batch!(g,branch,wt(a*"out_proj.weight"),gdn,n;name=a*"out")
            else
                a=p*"self_attn."
                linear_batch!(g,qfull,wt(a*"q_proj.weight"),norm,n;name=a*"q")
                linear_batch!(g,kproj,wt(a*"k_proj.weight"),norm,n;name=a*"k")
                linear_batch!(g,vproj,wt(a*"v_proj.weight"),norm,n;name=a*"v")
                kc,vc=s.kv[il]
                if batched
                    # The chunk's keys and values go into the cache first; then
                    # one causal pass reads it.
                    declare_prepare_chunk!(g,aq,qgate,qfull,wt(a*"q_norm.weight"),kc,vc,kproj,vproj,wt(a*"k_norm.weight"),
                        positions,nqh,nkv,theta,eps,n;name=a[1:end-1])
                    declare_prefill_attention!(g,attn,aq,qgate,kc,vc,s.posref,nkv,nqh,n;name=a*"attention")
                else
                    for j in 0:n-1
                        pos=column(positions,1,j)
                        dispatch(g,BR.prepare_q_kernel!,
                            (column(aq,256nqh,j),column(qgate,256nqh,j),column(qfull,512nqh,j),
                             wt(a*"q_norm.weight"),pos,theta,eps),nqh*256;name=a*"prepare_q.$j")
                        declare_prepare_kv!(g,kc,vc,column(kproj,256nkv,j),column(vproj,256nkv,j),wt(a*"k_norm.weight"),
                            pos,nkv,theta,eps;name=a*"prepare_kv.$j")
                        declare_attention!(g,s.attention,pm,ps,pa,column(aq,256nqh,j),kc,vc,pos,
                            nkv,nqh,chunks;name=a*"attention.$j")
                        declare_attention_combine!(g,s.attention,column(attn,256nqh,j),pm,ps,pa,column(qgate,256nqh,j),
                            pos,chunks,nqh;name=a*"combine.$j")
                    end
                end
                linear_batch!(g,branch,wt(a*"o_proj.weight"),attn,n;name=a*"out")
            end
            dispatch(g,BR.add_rmsnorm_batch_kernel!,
                (norm,x,branch,wt(p*"post_attention_layernorm.weight"),Int32(w),eps,Val(sg)),
                n*256;name=p*"residual_norm")
            mlp=declare_mlp!(g,m,branch,norm,gate,up,p;columns=n)
            DK.standardop!(g,"add.Tensor",(x,mlp);out=x,name=p*"residual")
            if !isempty(s.layer_outputs)
                dispatch(g,BR.copy_kernel!,
                    (s.layer_outputs[il+1],column(x,w,n-1),Int32(w)),w;name=p*"debug_copy")
            end
        end
        lastnorm=Mantle.Transient.Buffer(g,Float32,(w,))
        dispatch(g,BR.rmsnorm_batch_kernel!,
            (lastnorm,column(x,w,n-1),wt("norm.weight"),Int32(w),eps),256;name="final_norm")
        linear!(g,s.logits,wt("lm_head"),lastnorm;name="lm_head")
        mg=DK.emitsemantic!(g).mantle
        s.logit_hook===nothing || s.logit_hook(mg,s.logits)
        BR.declare_argmax!(mg,s.best,s.logits;name="argmax")
        dispatch(mg,advance_prompt!,(s.token,s.posref,s.predictions,s.best,Int32(n)),1;group=1,name="advance_prompt")
        # The default submission budget: a wide chunk deep in a long context runs
        # for seconds, and one submission that long is cancelled by the driver
        # (RADV: "guilty of a hard recovery"), losing the device.
        plan=Mantle.record!(Mantle.Plan(mg))
        (;tokens,positions,plan)
    catch
        plan===nothing || Mantle.free!(plan)
        positions===nothing || Mantle.free!(positions)
        Mantle.free!(tokens)
        rethrow()
    end
end

"""Ingest prompt chunks with batched projections and shared decode state.

Only the final token's logits are produced per chunk. This API does not provide
per-token prompt likelihoods. No generated token is read back to the host.
"""
function prefill!(s::Session,ids;chunk=32)
    chunk>0 || throw(ArgumentError("chunk must be positive"))
    length(ids)<=s.capacity-s.position || throw(ArgumentError("session cache capacity exceeded"))
    all(t->0<=t<s.model.config.vocab_size,ids) || throw(ArgumentError("token outside vocabulary"))
    for start in 1:chunk:length(ids)
        n=min(chunk,length(ids)-start+1)
        if n==1
            step!(s,ids[start])
        else
            exec=get!(()->record_prefill(s,n),s.prefills,n)
            exec.tokens[:]=Int32.(ids[start:start+n-1])
            s.posref[]=Int32(s.position)
            Mantle.run!(exec.plan)
            s.position+=n
        end
    end
    s.logits
end
