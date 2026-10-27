# Included at the end of Qwen36MoEExperiment, after speculative_qwen36.jl.
# Drafts from the checkpoint's multi-token prediction (MTP) layer, ported from
# vLLM's qwen3_5_mtp.py. For the token at position j+1 and the trunk's
# final-normed state at j, the layer norms the token's embedding and the state,
# projects the pair with `fc`, runs one gated-attention MoE decoder layer at
# position j over its own KV cache, norms, and reads the trunk's output head: a
# guess at the token at j+2. Its own normed state stands in for the trunk's to
# guess the tokens after that.
#
# The layer runs behind the trunk: the trunk's last state waits for the token
# after it. After a verify up to `k + 1` states wait (`pending`), one per
# consumed token. A round runs them, drafts `k` tokens and verifies them, all in
# one plan; a prompt chunk runs the one waiting state and its own columns but
# the last. Only the last column of a round's catch-up needs the whole layer:
# the other columns, and a prompt's, only write their keys and values.

"""Add the checkpoint's MTP layer from `dir` (safetensors holding the `mtp.*`
tensors, BF16) to `m`, quantized as the trunk is: projections and experts Q4_K,
the router and the shared expert's scalar gate Float32. The decoder layer goes
in as layer `num_hidden_layers`, the index vLLM gives it, so the trunk's layer
declarations read it. `fc` is split at its input's halves (embedding, state),
which Q4_K's 256-wide blocks keep identical to quantizing it whole."""
function loadmtp!(m::Model,dir)
    c=m.config;L=Int(c.num_hidden_layers);w=Int(c.hidden_size)
    get(c,:mtp_num_hidden_layers,0)==1 || throw(ArgumentError("the configuration has no single MTP layer"))
    get(c,:mtp_use_dedicated_embeddings,false)===false || throw(ArgumentError("dedicated MTP embeddings are not implemented"))
    haskey(m.weights,"mtp.fc_embedding") && throw(ArgumentError("the model's MTP layer is loaded already"))
    nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads);hd=Int(c.head_dim)
    f=Int(c.moe_intermediate_size);sf=Int(c.shared_expert_intermediate_size);ne=Int(c.num_experts)
    p="mtp.layers.0."
    # Julia's dimensions, the checkpoint's reversed.
    shapes=Dict("mtp.fc.weight"=>(2w,w),"mtp.pre_fc_norm_embedding.weight"=>(w,),"mtp.pre_fc_norm_hidden.weight"=>(w,),
        "mtp.norm.weight"=>(w,),p*"input_layernorm.weight"=>(w,),p*"post_attention_layernorm.weight"=>(w,),
        p*"self_attn.q_proj.weight"=>(w,2nqh*hd),p*"self_attn.k_proj.weight"=>(w,nkv*hd),p*"self_attn.v_proj.weight"=>(w,nkv*hd),
        p*"self_attn.o_proj.weight"=>(nqh*hd,w),p*"self_attn.q_norm.weight"=>(hd,),p*"self_attn.k_norm.weight"=>(hd,),
        p*"mlp.gate.weight"=>(w,ne),p*"mlp.shared_expert_gate.weight"=>(w,1),
        p*"mlp.shared_expert.gate_proj.weight"=>(w,sf),p*"mlp.shared_expert.up_proj.weight"=>(w,sf),
        p*"mlp.shared_expert.down_proj.weight"=>(sf,w),p*"mlp.experts.gate_up_proj"=>(w,2f,ne),p*"mlp.experts.down_proj"=>(f,w,ne))
    files=sort!(filter(endswith(".safetensors"),readdir(dir;join=true)))
    listed=[(name,file) for file in files for name in DK.safetensorsnames(file)]
    sort!(first.(listed))==sort!(collect(keys(shapes))) ||
        throw(ArgumentError("$dir does not hold exactly the MTP layer's $(length(shapes)) tensors"))
    added=Dict{String,Any}();owned=Any[]
    keep(b)=(push!(owned,b);b)
    function q4k(x)
        dense=Mantle.Buffer(m.device,permutedims(x))
        a=DK.quantizeq4k(m.device,dense)
        append!(owned,DK.kqarrays(a));Mantle.waitidle(m.device);Mantle.free!(dense)
        a
    end
    native(name)=startswith(name,p) ? "layers.$L."*name[length(p)+1:end] : name
    try
        for file in files
            tensors=DK.readsafetensors(file;mmap=true)
            for (name,raw) in tensors
                eltype(raw)===UInt16 || throw(ArgumentError("$name: BF16 expected"))
                size(raw)==shapes[name] || throw(DimensionMismatch("$name: $(size(raw)), expected $(shapes[name])"))
                x=reinterpret.(Float32,UInt32.(raw).<<16)
                key=native(name)
                if ndims(x)==1
                    # Every MTP norm is Gemma's, scaling by 1 + weight.
                    added[key]=keep(Mantle.Buffer(m.device,x.+1f0))
                elseif name=="mtp.fc.weight"
                    added["mtp.fc_embedding"]=q4k(x[1:w,:])
                    added["mtp.fc_hidden"]=q4k(x[w+1:2w,:])
                elseif ndims(x)==3
                    k,rows,experts=size(x)
                    added[key]=M.Q4Experts(q4k(reshape(x,k,rows*experts)),rows,experts)
                elseif endswith(name,"mlp.gate.weight")||endswith(name,"shared_expert_gate.weight")
                    k,rows=size(x)
                    added[key]=Q.DenseWeight(keep(Mantle.Buffer(m.device,x)),rows,k)
                else
                    added[key]=q4k(x)
                end
            end
            GC.gc(false)
        end
    catch
        foreach(Mantle.free!,reverse(owned));rethrow()
    end
    merge!(m.weights,added);append!(m.owned,owned)
    m
end

hasmtp(m::Model)=haskey(m.weights,"mtp.fc_embedding")

function shifted_positions!(positions,start,shift::Int32,n::Int32)
    i=Int32(Q.KI.get_global_id().x)
    i<=n && (@inbounds positions[i]=start[1]+shift+i-Int32(1))
    nothing
end
DK.kernelresults(::typeof(shifted_positions!))=(1,)

"""DECLARE the MTP layer over `n` columns: column j reads token `tokens[j+1]`
and the trunk's normed state `hidden[:, j+1]` at position `positions[j+1]`, and
writes its keys and values there. With `out` (w, 1) the last column runs the
whole layer, over the keys and values just written, and its normed state goes
into `out`."""
function declare_mtp!(g,s::Session,kv,tokens,hidden,positions,n;out=nothing,name)
    inner=s.inner;m=inner.model;c=m.config;L=Int(c.num_hidden_layers)
    w=Int(c.hidden_size);nqh=Int(c.num_attention_heads);nkv=Int(c.num_key_value_heads)
    eps=Float32(c.rms_norm_eps);theta=Float32(c.rope_parameters.rope_theta)
    p="layers.$L.";a=p*"self_attn.";wt(key)=m.weights[key];column=Q.column
    scratch(dims...)=Mantle.Transient.Buffer(g,Float32,Int.(dims))
    norm!(dst,src,weight,cols,label)=
        Q.dispatch(g,BR.rmsnorm_batch_kernel!,(dst,src,wt(weight),Int32(w),eps),cols*256;name="$name.$label")
    e=scratch(w,n);declare_embedding_batch!(g,e,wt("embed_tokens.weight"),tokens,w,n)
    en=scratch(w,n);norm!(en,e,"mtp.pre_fc_norm_embedding.weight",n,"embedding_norm")
    hn=scratch(w,n);norm!(hn,hidden,"mtp.pre_fc_norm_hidden.weight",n,"hidden_norm")
    x=scratch(w,n);xh=scratch(w,n)
    Q.linear_batch!(g,x,wt("mtp.fc_embedding"),en,n;name="$name.fc_embedding")
    Q.linear_batch!(g,xh,wt("mtp.fc_hidden"),hn,n;name="$name.fc_hidden")
    DK.standardop!(g,"add.Tensor",(x,xh);out=x,name="$name.fc")
    xn=scratch(w,n);norm!(xn,x,p*"input_layernorm.weight",n,"input_norm")
    kproj=scratch(256nkv,n);vproj=scratch(256nkv,n)
    Q.linear_batch!(g,kproj,wt(a*"k_proj.weight"),xn,n;name="$name.k")
    Q.linear_batch!(g,vproj,wt(a*"v_proj.weight"),xn,n;name="$name.v")
    kc,vc=kv
    Q.declare_prepare_kv_chunk!(g,kc,vc,kproj,vproj,wt(a*"k_norm.weight"),positions,nkv,theta,eps,n;name="$name.prepare_kv")
    out===nothing && return nothing
    # The last column, copied out whole: the standard products take whole tensors.
    j=n-1;pos=column(positions,1,j)
    xr=scratch(w,1);xl=scratch(w,1)
    Q.dispatch(g,BR.copy_kernel!,(xr,column(x,w,j),Int32(w)),w;name="$name.last")
    Q.dispatch(g,BR.copy_kernel!,(xl,column(xn,w,j),Int32(w)),w;name="$name.last_norm")
    qfull=scratch(512nqh);aq=scratch(256nqh);qgate=scratch(256nqh);attn=scratch(256nqh)
    Q.linear_batch!(g,qfull,wt(a*"q_proj.weight"),xl,1;name="$name.q")
    Q.dispatch(g,BR.prepare_q_kernel!,(aq,qgate,qfull,wt(a*"q_norm.weight"),pos,theta,eps),nqh*256;name="$name.prepare_q")
    chunks=Q.attention_slots(inner.attention,inner.capacity)
    pm=scratch(nqh*chunks);ps=scratch(nqh*chunks);pa=scratch(nqh*chunks*256)
    Q.declare_attention!(g,inner.attention,pm,ps,pa,aq,kc,vc,pos,nkv,nqh,chunks;name="$name.attention")
    Q.declare_attention_combine!(g,inner.attention,attn,pm,ps,pa,qgate,pos,chunks,nqh;name="$name.combine")
    branch=scratch(w,1)
    Q.linear_batch!(g,branch,wt(a*"o_proj.weight"),attn,1;name="$name.o")
    post=scratch(w,1)
    Q.dispatch(g,BR.add_rmsnorm_batch_kernel!,
        (post,xr,branch,wt(p*"post_attention_layernorm.weight"),Int32(w),eps,Val(Mantle.caps(m.device).subgroup)),
        256;name="$name.residual_norm")
    ffn=scratch(w,1)
    declare_ffn_batch!(g,m,ffn,post,L,1;optimizations=s.optimizations,name="$name.mlp.")
    DK.standardop!(g,"add.Tensor",(xr,ffn);out=xr,name="$name.residual")
    norm!(out,xr,"mtp.norm.weight",1,"norm")
end

"""Restrict the MTP layer's drafts to the tokens `ids` (0-based): their rows of
the output head, gathered as they are quantized, are all a draft reads, a
fraction of the trunk's 417 MB head (frequency-ranked drafting, FR-Spec, Zhao
et al. 2025). A token outside them is never drafted; the verify reads the whole
head, so the output is unchanged. A second call replaces the first's."""
function draftvocabulary!(m::Model,ids)
    vocab=Int(m.config.vocab_size)
    !isempty(ids)&&allunique(ids)&&all(t->0<=t<vocab,ids) || throw(ArgumentError("draft tokens must be distinct token ids"))
    head=gatherrows(m.device,m.weights["lm_head"],Int.(ids).+1)
    lookup=Mantle.Buffer(m.device,Int32.(ids))
    if haskey(m.weights,"mtp.draft_head")
        old=Any[DK.kqarrays(m.weights["mtp.draft_head"])...,m.weights["mtp.draft_ids"]]
        filter!(b->!any(o->o===b,old),m.owned);foreach(Mantle.free!,old)
    end
    append!(m.owned,DK.kqarrays(head));push!(m.owned,lookup)
    m.weights["mtp.draft_head"]=head;m.weights["mtp.draft_ids"]=lookup
    m
end

# Rows of a K-quant matrix: each array holds a row's blocks back to back.
function gatherrows(dev,A::DK.KQuantMatrix,rows)
    m=size(A,1)
    arrays=map(DK.kqarrays(A)) do x
        host=Array(x);g=reshape(host,length(host)÷m,m)[:,rows]
        Mantle.Buffer(dev,ndims(x)==1 ? vec(g) : g)
    end
    kqmatrix(A,arrays,length(rows))
end
kqmatrix(A::DK.Q6KMatrix,arrays,m)=DK.Q6KMatrix(arrays...,m,A.k)
kqmatrix(A::DK.Q4KMatrix,arrays,m)=DK.Q4KMatrix(arrays...,m,A.k)

function draft_token!(slot,ids,index)
    Q.KI.get_global_id().x==1 && (@inbounds slot[1]=ids[index[1]+Int32(1)])
    nothing
end
DK.kernelresults(::typeof(draft_token!))=(1,)

"""DECLARE the output head over the MTP state `state` (w, 1), the draft
vocabulary's rows if the model has one, and its argmax into `slot`, a one-token
view."""
function declare_draft!(g,s::Session,slot,state;name)
    m=s.inner.model;restricted=haskey(m.weights,"mtp.draft_head")
    head=m.weights[restricted ? "mtp.draft_head" : "lm_head"]
    logits=Mantle.Transient.Buffer(g,Float32,(size(head,1),1))
    Q.linear_batch!(g,logits,head,state,1;name="$name.head")
    best=restricted ? Mantle.Transient.Buffer(g,Int32,(1,)) : slot
    pv=Mantle.Transient.Buffer(g,Float32,(BR.ARGMAX_PARTS,));pidx=Mantle.Transient.Buffer(g,Int32,(BR.ARGMAX_PARTS,))
    Q.dispatch(g,BR.argmax_partial_kernel!,(pv,pidx,logits,Int32(length(logits))),BR.ARGMAX_PARTS*256;group=256,name="$name.argmax_parts")
    Q.dispatch(g,BR.argmax_final_kernel!,(best,pv,pidx),256;group=256,name="$name.argmax")
    restricted && Q.dispatch(g,draft_token!,(slot,m.weights["mtp.draft_ids"],best),1;group=1,name="$name.token")
end

"""Speculative decoding with drafts from the MTP layer (`loadmtp!`): rounds of
`k` drafts (by default; up to `kmax`, chosen per call), each verified in a chunk
of `k + 1` tokens with the n-gram `Speculator`'s verify and replay, the drafts
declared in the same plan. The drafts are the MTP head's argmax; a sampling
session samples the verify's columns and needs `undo`, as a `Speculator` does.
The layer has to see the whole context, so a speculator starts with an empty
session and its prompts go in through `prefill!(sp, ids)`. From position `deep`
on a verify takes the chunk attention; `deep = typemax(Int)` keeps decode's,
which makes the output plain decoding's, greedy or sampled."""
mutable struct MTPSpeculator
    session::Session
    buffers::VerifyBuffers    # `best` also holds the tokens after the waiting states
    kv::Tuple                 # the MTP layer's own keys and values
    hidden::Any               # Float32 (w, kmax+1): trunk states the MTP layer has yet to read
    pending::Int              # how many of them wait
    position::Int             # the session's position they were left at
    owned::Vector{Any}
    k::Int
    kmax::Int
    deep::Int
    rounds::Dict{Tuple{Int,Int,Bool},Any} # (drafts, pending, chunk attention) => recorded round
    flushes::Dict{Int,Any}                # pending => recorded flush
    prompts::Dict{Tuple{Int,Bool},Any}    # (width, first chunk) => recorded prompt chunk
    replays::Dict{Int,Any}
    undo::Any
    refused::Any              # GPURef{Int32}: refused columns of the last verify
    steps::Int
    drafted::Int
    accepted::Int
end

kvtype(c::Q.Q8KV)=Int8
kvtype(c::Q.Q4KV)=Q.Q4_0
kvtype(c::Mantle.Buffer)=eltype(c)

function MTPSpeculator(s::Session;k=2,kmax=k,deep=16384,undo=nothing)
    inner=s.inner;m=inner.model;c=m.config
    hasmtp(m) || throw(ArgumentError("the model has no MTP layer; `loadmtp!` it first"))
    (inner.logit_hook===nothing)==(undo===nothing) ||
        throw(ArgumentError("a sampling session needs `undo` for its draws, a greedy one has none"))
    1<=k<=kmax || throw(ArgumentError("1 <= k <= kmax drafts, not k = $k, kmax = $kmax"))
    kmax+1<=DK.Q4K_GEMV_COLUMNS[] || throw(ArgumentError("$kmax drafts make a verify chunk wider than the bit-exact width"))
    inner.position==0 || throw(ArgumentError("the MTP layer reads the whole context: give the speculator an empty session"))
    owned=Any[];buffers=nothing
    persistent(T,dims)=(a=Mantle.Buffer(m.device,T,dims);fill!(a,zero(T));push!(owned,a);a)
    try
        T=kvtype(first(values(inner.kv))[1]);nkv=Int(c.num_key_value_heads)
        kv=(Q.kvcache(persistent,T,nkv,inner.capacity),Q.kvcache(persistent,T,nkv,inner.capacity))
        hidden=persistent(Float32,(Int(c.hidden_size),kmax+1))
        refused=Mantle.GPURef(m.device,Int32(0));push!(owned,refused)
        buffers=VerifyBuffers(s,kmax+1)
        MTPSpeculator(s,buffers,kv,hidden,0,0,owned,k,kmax,deep,Dict{Tuple{Int,Int,Bool},Any}(),Dict{Int,Any}(),
            Dict{Tuple{Int,Bool},Any}(),Dict{Int,Any}(),undo,refused,0,0,0)
    catch
        buffers===nothing || close(buffers)
        foreach(Mantle.free!,owned);rethrow()
    end
end

function Base.close(sp::MTPSpeculator)
    for e in Iterators.flatten((values(sp.rounds),values(sp.prompts)))
        Mantle.free!(e.plan);Mantle.free!(e.tokens);Mantle.free!(e.positions)
    end
    foreach(Mantle.free!,values(sp.flushes));foreach(Mantle.free!,values(sp.replays))
    empty!(sp.rounds);empty!(sp.prompts);empty!(sp.flushes);empty!(sp.replays)
    close(sp.buffers);foreach(Mantle.free!,sp.owned);empty!(sp.owned)
    nothing
end

# What `record_prefill` declares for the MTP layer: a verify round's drafts and
# kept states, or a prompt chunk's MTP columns.
struct MTPRound
    speculator::MTPSpeculator
    pending::Int
    k::Int
end
struct MTPPrompt
    speculator::MTPSpeculator
    first::Bool               # the session's first chunk: no state waits before it
end

checkmtp(::Nothing,verify)=nothing
checkmtp(::MTPRound,verify)=verify===nothing && throw(ArgumentError("an MTP round is a verify chunk"))
checkmtp(::MTPPrompt,verify)=verify===nothing || throw(ArgumentError("an MTP prompt chunk is no verify"))

declare_mtp_drafts!(g,::Nothing,tokens)=nothing
declare_mtp_drafts!(g,::MTPPrompt,tokens)=nothing
"""The waiting states' MTP columns at the positions before the verify's, the
pending token into `tokens[1]` and `k` drafts after it."""
function declare_mtp_drafts!(g,r::MTPRound,tokens)
    sp=r.speculator;s=sp.session;inner=s.inner;p=r.pending
    w=Int(inner.model.config.hidden_size)
    slot(b,i)=Mantle.viewof(b,(1,);offset=i)
    positions=Mantle.Transient.Buffer(g,Int32,(p,))
    Q.dispatch(g,shifted_positions!,(positions,inner.posref,Int32(-p),Int32(p)),p;name="mtp.positions")
    state=Mantle.Transient.Buffer(g,Float32,(w,1))
    declare_mtp!(g,s,sp.kv,Mantle.viewof(sp.buffers.best,(p,)),Mantle.viewof(sp.hidden,(w,p)),positions,p;
        out=state,name="mtp.catchup")
    Q.dispatch(g,BR.copy_kernel!,(slot(tokens,0),slot(sp.buffers.best,p-1),Int32(1)),1;name="mtp.pending_token")
    declare_draft!(g,s,slot(tokens,1),state;name="mtp.draft_1")
    for i in 1:r.k-1
        at=Mantle.Transient.Buffer(g,Int32,(1,))
        Q.dispatch(g,shifted_positions!,(at,inner.posref,Int32(i-1),Int32(1)),1;name="mtp.position_$i")
        next=Mantle.Transient.Buffer(g,Float32,(w,1))
        declare_mtp!(g,s,sp.kv,slot(tokens,i),state,at,1;out=next,name="mtp.step_$i")
        declare_draft!(g,s,slot(tokens,i+1),next;name="mtp.draft_$(i+1)")
        state=next
    end
end

keep_states!(g,::Nothing,final,n)=nothing
keep_states!(g,::MTPPrompt,final,n)=nothing
keep_states!(g,r::MTPRound,final,n)=Q.dispatch(g,BR.copy_kernel!,
    (Mantle.viewof(r.speculator.hidden,(length(final),)),final,Int32(length(final))),length(final);name="mtp.keep_states")

declare_mtp_prompt!(g,::Nothing,x,lastnorm,tokens,positions,n)=nothing
declare_mtp_prompt!(g,::MTPRound,x,lastnorm,tokens,positions,n)=nothing
"""A prompt chunk's MTP columns: the waiting state with the chunk's first token
(unless the chunk is the session's first), then every column but the last with
the token after it. The last column's state waits."""
function declare_mtp_prompt!(g,r::MTPPrompt,x,lastnorm,tokens,positions,n)
    sp=r.speculator;s=sp.session;inner=s.inner;m=inner.model;c=m.config
    w=Int(c.hidden_size);eps=Float32(c.rms_norm_eps);column=Q.column
    cols=r.first ? n-1 : n
    if cols>0
        states=Mantle.Transient.Buffer(g,Float32,(w,cols))
        r.first || Q.dispatch(g,BR.copy_kernel!,(column(states,w,0),column(sp.hidden,w,0),Int32(w)),w;name="mtp.waiting_state")
        n>1 && Q.dispatch(g,BR.rmsnorm_batch_kernel!,
            (Mantle.viewof(states,(w*(n-1),);offset=r.first ? 0 : w),Mantle.viewof(x,(w*(n-1),)),m.weights["norm.weight"],Int32(w),eps),
            (n-1)*256;name="mtp.states")
        if r.first
            declare_mtp!(g,s,sp.kv,Mantle.viewof(tokens,(n-1,);offset=1),states,Mantle.viewof(positions,(n-1,)),n-1;name="mtp.prompt")
        else
            shifted=Mantle.Transient.Buffer(g,Int32,(n,))
            Q.dispatch(g,shifted_positions!,(shifted,inner.posref,Int32(-1),Int32(n)),n;name="mtp.positions")
            declare_mtp!(g,s,sp.kv,tokens,states,shifted,n;name="mtp.prompt")
        end
    end
    Q.dispatch(g,BR.copy_kernel!,(column(sp.hidden,w,0),lastnorm,Int32(w)),w;name="mtp.keep_state")
end

"""The MTP columns of all but the last of `p` waiting states, whose tokens the
verify consumed, and the last state moved first: what a prompt chunk expects."""
function record_flush(sp::MTPSpeculator,p::Int)
    s=sp.session;inner=s.inner;m=inner.model;w=Int(m.config.hidden_size)
    g=DK.SemanticGraph(m.device;name="mtp_flush_$p");DK.semanticweights!(g,Q.neuralweights(m))
    positions=Mantle.Transient.Buffer(g,Int32,(p-1,))
    Q.dispatch(g,shifted_positions!,(positions,inner.posref,Int32(-p),Int32(p-1)),p-1;name="mtp.positions")
    declare_mtp!(g,s,sp.kv,Mantle.viewof(sp.buffers.best,(p-1,)),Mantle.viewof(sp.hidden,(w,p-1)),positions,p-1;name="mtp.flush")
    Q.dispatch(g,BR.copy_kernel!,(Q.column(sp.hidden,w,0),Q.column(sp.hidden,w,p-1),Int32(w)),w;name="mtp.waiting_state")
    Mantle.record!(Mantle.Plan(DK.emitsemantic!(g).mantle;budget=Inf))
end

roundplan!(sp::MTPSpeculator,k,p,chunk)=get!(()->record_prefill(sp.session,k+1;verify=sp.buffers,chunk_attention=chunk,
    mtp=MTPRound(sp,p,k)),sp.rounds,(k,p,chunk))
flushplan!(sp::MTPSpeculator,p)=get!(()->record_flush(sp,p),sp.flushes,p)
promptplan!(sp::MTPSpeculator,n,first)=get!(()->record_prefill(sp.session,n;mtp=MTPPrompt(sp,first)),sp.prompts,(n,first))
replayplan!(sp::MTPSpeculator,a)=get!(()->record_replay(sp.session,sp.buffers,a;sp.undo,sp.refused),sp.replays,a)

"""Record every round, flush and replay a speculator can need with `k` drafts,
so none is recorded while it generates."""
function prepare!(sp::MTPSpeculator;k=sp.k)
    for p in 1:k+1,chunk in (false,true)
        (chunk ? sp.deep<sp.session.inner.capacity : sp.deep>0) && roundplan!(sp,k,p,chunk)
    end
    foreach(p->flushplan!(sp,p),2:k+1)
    foreach(a->replayplan!(sp,a),0:k-1)
    sp
end

checkposition(sp::MTPSpeculator)=sp.position==sp.session.inner.position ||
    throw(ArgumentError("the session moved without its MTP layer: it is at $(sp.session.inner.position), the layer at $(sp.position)"))

"""Ingest prompt tokens into the speculator's session, as `prefill!` does, with
the MTP layer's columns in every chunk."""
function prefill!(sp::MTPSpeculator,ids;chunk=DK.Q4K_GEMV_COLUMNS[])
    s=sp.session;inner=s.inner
    checkposition(sp)
    chunk>0 || throw(ArgumentError("chunk must be positive"))
    length(ids)<=inner.capacity-inner.position || throw(ArgumentError("session cache capacity exceeded"))
    all(t->0<=t<inner.model.config.vocab_size,ids) || throw(ArgumentError("token outside vocabulary"))
    isempty(ids) && return inner.logits
    if sp.pending>1
        inner.posref[]=Int32(inner.position)
        Mantle.run!(flushplan!(sp,sp.pending))
    end
    for start in 1:chunk:length(ids)
        n=min(chunk,length(ids)-start+1)
        exec=promptplan!(sp,n,inner.position==0)
        exec.tokens[:]=Int32.(ids[start:start+n-1])
        inner.posref[]=Int32(inner.position)
        Mantle.run!(exec.plan)
        inner.position+=n
    end
    sp.pending=1;sp.position=inner.position
    inner.logits
end

"""Generate up to `n` tokens, stopping after a token in `stops`, as
`speculative_decode!` does with n-gram drafts, `k` drafts a round: the first is
`first`, by default the session's `best` (right after `prefill!`), and the last
is left pending."""
function speculative_decode!(sp::MTPSpeculator,history,n;stops=(),first=nothing,k=sp.k)
    s=sp.session;inner=s.inner
    checkposition(sp)
    1<=k<=sp.kmax || throw(ArgumentError("1 <= k <= $(sp.kmax) drafts, not $k"))
    sp.pending>=1 || throw(ArgumentError("the speculator's session has consumed nothing"))
    n+k<=inner.capacity-inner.position || throw(ArgumentError("session cache capacity exceeded"))
    out=Int[]
    next=if first===nothing
        Mantle.waitidle(inner.model.device);Int(only(Array(inner.best)))
    else
        first
    end
    while true
        push!(out,next)
        (next in stops||length(out)>=n) && break
        P=inner.position;p=sp.pending
        exec=roundplan!(sp,k,p,P>=sp.deep)
        # The token after the last waiting state is the one the verify consumes first.
        sp.buffers.best[p:p]=Int32[next]
        inner.posref[]=Int32(P)
        Mantle.run!(exec.plan)
        predicted=Int.(Array(sp.buffers.best));proposal=Int.(Array(exec.tokens))[2:end]
        a=0;wanted=min(k,n-length(out)-1)
        while a<wanted&&predicted[a+1]==proposal[a+1]&&!(proposal[a+1] in stops);a+=1;end
        sp.steps+=1;sp.drafted+=wanted;sp.accepted+=a
        if a<k
            sp.refused[]=Int32(k-a)
            Mantle.run!(replayplan!(sp,a))
        end
        inner.position+=a+1;sp.position=inner.position;sp.pending=a+1
        append!(out,proposal[1:a])
        next=predicted[a+1]
    end
    inner.token[]=Int32(out[end]);inner.posref[]=Int32(inner.position)
    out
end

speculative_generate!(sp::MTPSpeculator,history,pending,n;kw...)=
    speculative_decode!(sp,history,n+1;first=pending,kw...)[2:end]
