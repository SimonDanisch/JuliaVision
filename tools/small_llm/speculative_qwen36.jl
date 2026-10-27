# Included at the end of Qwen36MoEExperiment. Greedy decoding with drafts from
# the context itself (prompt lookup): where the last tokens occurred before, the
# tokens that followed them are proposed, and one chunk of the network checks
# them all. Every accepted draft token saves a decode step; at depth the chunk
# reads the KV cache once for all of them.
#
# A verify chunk is at most the bit-exact width, so its products, recurrence
# and expert kernels are decode's own, column by column. Its attention is either
# decode's (then speculative output is greedy decode's, bit for bit) or the
# chunk attention on the matrix units, which reads the cache once and differs
# from decode by Float16 rounding. Decode's re-reads the cache for every draft
# token, which only costs at depth: on the shared 8060S, against 47 and 60 tok/s
# of plain decoding, decode's attention gave 87.5 tok/s at depth 0 and 45.7 at
# 64k, the chunk attention 61.1 and 83.1. A speculator switches at `deep`.
#
# The recurrent state cannot be cut back to the last accepted token. A verify
# copies it first and keeps each recurrent layer's inputs; when a draft token is
# refused, the state is copied back and the accepted tokens' recurrence replayed
# from those inputs with the same kernels. Attention needs nothing: positions
# past the accepted ones are written again before they are read.
#
# A sampling session samples every verify column with its own sampler hook, in
# column order, so column j draws what sequential decoding's j-th step would.
# A refusal also takes back the draws of the refused columns (`undo`), and with
# decode's attention sampled output is plain sampled decoding's, bit for bit.
#
# Every submission and every wait counts when the GPU is shared: a busy one made
# each submission cost ~200 ms whatever it held. So the copy is the verify
# plan's first pass and the copy back the replay's, and a stretch without a
# draft queues several decode steps before one wait.

"""Persistent buffers a verify chunk of up to `columns` tokens writes."""
struct VerifyBuffers
    best::Any                 # Int32 (columns,): each column's argmax
    logits::Any               # Float32 (vocab, columns)
    saved::Dict{Int,Any}      # recurrent layer => (qkv, alpha, beta), one column a token
    live::Vector{Any}         # the session's recurrent buffers, layer by layer
    snapshot::Vector{Any}     # their copy at the start of the last verify
end

function VerifyBuffers(s::Session,columns::Int)
    inner=s.inner;m=inner.model;c=m.config;owned=Any[]
    up(T,dims)=(b=Mantle.Buffer(m.device,zeros(T,dims));push!(owned,b);b)
    nh=Int(c.linear_num_value_heads);channels=256Int(c.linear_num_key_heads)+128nh
    try
        saved=Dict{Int,Any}(il=>(qkv=up(Float32,(channels,columns)),alpha=up(Float32,(nh,columns)),beta=up(Float32,(nh,columns)))
            for il in keys(inner.recurrent))
        live=collect(Any,Iterators.flatten(inner.recurrent[il] for il in sort(collect(keys(inner.recurrent)))))
        snapshot=Any[up(eltype(b),size(b)) for b in live]
        VerifyBuffers(up(Int32,(columns,)),up(Float32,(Int(c.vocab_size),columns)),saved,live,snapshot)
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end
Base.close(v::VerifyBuffers)=(Mantle.free!(v.best);Mantle.free!(v.logits);foreach(t->foreach(Mantle.free!,t),values(v.saved));
    foreach(Mantle.free!,v.snapshot);nothing)

"""DECLARE copies of `pairs` (destination, source) into a graph."""
function declare_copies!(g,pairs;name)
    for (k,(dst,src)) in enumerate(pairs)
        Q.dispatch(g,BR.copy_kernel!,(dst,src,Int32(length(src))),length(src);name="$name.$k")
    end
end

"""The recurrence of the first `a+1` tokens of the last verify chunk, from the
state the chunk started from (copied back first): each recurrent layer's
convolution over its kept projections, then the one-token Gated DeltaNet kernel
token by token, as the chunk ran them. Outputs are scratch; only the state and
history matter."""
function record_replay(s::Session,v::VerifyBuffers,a::Int;undo=nothing,refused=nothing)
    inner=s.inner;m=inner.model;c=m.config;dev=m.device;n=a+1
    nh=Int(c.linear_num_value_heads);nkh=Int(c.linear_num_key_heads);kd=128nkh;vd=128nh;channels=2kd+vd
    eps=Float32(c.rms_norm_eps);wt(name)=m.weights[name];column=Q.column
    repeat_kernel=s.head_layout===:native ? Q.repeat_qk! : repeat_qk_tiled!
    g=Mantle.Graph(dev)
    conv=Mantle.Transient.Buffer(g,Float32,(channels,n));rq=Mantle.Transient.Buffer(g,Float32,(vd,n))
    rk=Mantle.Transient.Buffer(g,Float32,(vd,n));gdn=Mantle.Transient.Buffer(g,Float32,(vd,n))
    # The gate `z` only shapes the output, which is scratch here.
    z=Mantle.Transient.Buffer(g,Float32,(vd,n))
    declare_copies!(g,zip(v.live,v.snapshot);name="replay.restore")
    for il in sort(collect(keys(inner.recurrent)))
        p="layers.$il.linear_attn.";hist,state=inner.recurrent[il];kept=v.saved[il]
        Q.dispatch(g,BR.depthwise_conv4_batch_kernel!,
            (conv,hist,Mantle.viewof(kept.qkv,(channels,n)),wt(p*"conv1d.weight"),Int32(channels),Int32(n)),channels;name=p*"replay_conv")
        for j in 0:a
            Q.dispatch(g,repeat_kernel,(column(rq,vd,j),column(rk,vd,j),column(conv,channels,j),Int32(nh),Int32(nkh)),vd;name=p*"replay_repeat.$j")
            Q.dispatch(g,DK.gated_delta_net_kernel!,
                (column(gdn,vd,j),state,column(rq,vd,j),column(rk,vd,j),Mantle.viewof(conv,(vd,);offset=j*channels+2kd),
                 Mantle.viewof(kept.alpha,(nh,);offset=j*nh),Mantle.viewof(kept.beta,(nh,);offset=j*nh),wt(p*"dt_bias"),wt(p*"A_log"),
                 column(z,vd,j),wt(p*"norm.weight"),eps,Int32(nh),Int32(nh),Val(DK.GDN_LANES)),
                nh*128*DK.GDN_LANES;group=128*DK.GDN_LANES,name=p*"replay_gdn.$j")
        end
    end
    # The draws of the refused columns, `a+1` onwards, `refused[]` of them.
    undo===nothing || undo(g,v.best,a+1,refused)
    Mantle.record!(Mantle.Plan(g;budget=Inf))
end

"""The latest earlier occurrence of every n-gram, `nmin <= n <= nmax`, of a
growing token history: a draft is what followed the longest one that matches the
history's end."""
mutable struct NgramIndex
    history::Vector{Int}
    tables::Vector{Dict{UInt64,Int}}    # n - nmin + 1 => hash of the n-gram => its start
    indexed::Int                        # n-grams ending at or before this position are in
    nmin::Int
    nmax::Int
end
NgramIndex(history;nmin=2,nmax=8)=NgramIndex(copy(history),[Dict{UInt64,Int}() for _ in nmin:nmax],0,nmin,nmax)

ngramhash(h,s,n)=hash(view(h,s:s+n-1))

function catchup!(ix::NgramIndex,upto)
    h=ix.history
    for e in ix.indexed+1:upto,(t,n) in enumerate(ix.nmin:ix.nmax)
        e>=n && (ix.tables[t][ngramhash(h,e-n+1,n)]=e-n+1)
    end
    ix.indexed=max(ix.indexed,upto)
end

"""Up to `kmax` tokens following the latest earlier occurrence of the longest
n-gram the history ends with, as many as the match is long beyond `nmin - 1`;
empty if none matches."""
function draft(ix::NgramIndex,kmax)
    h=ix.history;L=length(h)
    catchup!(ix,L-1)
    for n in min(ix.nmax,L-1):-1:ix.nmin
        s=get(ix.tables[n-ix.nmin+1],ngramhash(h,L-n+1,n),0)
        s>0 && view(h,s:s+n-1)==view(h,L-n+1:L) || continue
        k=min(kmax,n-ix.nmin+1)
        return h[s+n:min(s+n+k-1,L)]
    end
    Int[]
end

"""The index of `history`: the last call's if `history` continues what it
indexed (a chunked caller hands back what it consumed, plus what was generated
since), a new one otherwise (after a rewind). Rebuilding hashes every n-gram of
the history, ~0.1 s at 60k tokens, which a 16-token chunk cannot afford."""
function reindex(ix,history;nmin,nmax)
    ix isa NgramIndex&&ix.nmin==nmin&&ix.nmax==nmax || return NgramIndex(history;nmin,nmax)
    L=length(history);h=ix.history
    shared=min(L,length(h))
    # n-grams ending past the shared part may be indexed only if it is the history.
    ix.indexed<=shared && view(h,1:shared)==view(history,1:shared) || return NgramIndex(history;nmin,nmax)
    resize!(h,shared);append!(h,view(history,shared+1:L))
    ix
end

"""Speculative decoding for a session: verify chunks of up to `kmax + 1`
tokens, recorded per size and attention, and the replays that undo refused
drafts. From position `deep` on a verify takes the chunk attention; `deep =
typemax(Int)` keeps decode's everywhere, which makes the output decode's. A
session with a sampling `logit_hook` needs `undo(g, tokens, first, count)`, which
declares the reversal of the draws of `count[]` sampled tokens at
`tokens[first+1 ...]`."""
mutable struct Speculator
    session::Session
    buffers::VerifyBuffers
    verifies::Dict{Tuple{Int,Bool},Any}   # (width, chunk attention) => recorded chunk
    replays::Dict{Int,Any}
    undo::Any
    refused::Any                          # GPURef{Int32}: refused columns of the last verify
    kmax::Int
    deep::Int
    plainrun::Int             # decode steps queued before a wait where nothing is drafted
    index::Any                # the last call's n-gram index, or nothing
    steps::Int
    drafted::Int
    accepted::Int
end

function Speculator(s::Session;kmax=8,deep=16384,plainrun=4,undo=nothing)
    kmax+1<=DK.Q4K_GEMV_COLUMNS[] || throw(ArgumentError("a verify chunk of $(kmax+1) tokens is wider than the bit-exact width"))
    plainrun>=1 || throw(ArgumentError("plainrun must be positive"))
    (s.inner.logit_hook===nothing)==(undo===nothing) ||
        throw(ArgumentError("a sampling session needs `undo` for its draws, a greedy one has none"))
    Speculator(s,VerifyBuffers(s,kmax+1),Dict{Tuple{Int,Bool},Any}(),Dict{Int,Any}(),
        undo,Mantle.GPURef(s.inner.model.device,Int32(0)),kmax,deep,plainrun,nothing,0,0,0)
end
replayplan!(sp::Speculator,a)=get!(()->record_replay(sp.session,sp.buffers,a;sp.undo,sp.refused),sp.replays,a)

verifyplan!(sp::Speculator,w,chunk)=
    get!(()->record_prefill(sp.session,w;verify=sp.buffers,chunk_attention=chunk),sp.verifies,(w,chunk))

"""Record every verify chunk and replay a speculator can need, so none is
recorded while it generates."""
function prepare!(sp::Speculator)
    for w in 2:sp.kmax+1,chunk in (false,true)
        (chunk ? sp.deep<sp.session.inner.capacity : sp.deep>0) && verifyplan!(sp,w,chunk)
    end
    for a in 0:sp.kmax-1
        replayplan!(sp,a)
    end
    sp
end

function Base.close(sp::Speculator)
    for e in values(sp.verifies);Mantle.free!(e.plan);Mantle.free!(e.tokens);Mantle.free!(e.positions);end
    foreach(Mantle.free!,values(sp.replays))
    Mantle.free!(sp.refused);close(sp.buffers)
    nothing
end

"""Generate up to `n` tokens after `history` (the token ids the session has
consumed), stopping after a token in `stops`. The first token generated is
`first`, by default the session's `best`. Returns the tokens; the last one is
left pending, as decoding leaves it (the device's token and position too)."""
function speculative_decode!(sp::Speculator,history,n;stops=(),nmin=2,nmax=8,first=nothing)
    s=sp.session;inner=s.inner
    n<=inner.capacity-inner.position || throw(ArgumentError("session cache capacity exceeded"))
    ix=sp.index=reindex(sp.index,history;nmin,nmax);out=Int[]
    next=if first===nothing
        Mantle.waitidle(inner.model.device);Int(only(Array(inner.best)))
    else
        first
    end
    while true
        push!(out,next);push!(ix.history,next)
        (next in stops||length(out)>=n) && break
        proposal=draft(ix,min(sp.kmax,n-length(out)-1))
        if isempty(proposal)
            # `j` decode steps, one wait: `next` and the first j-1 of them are
            # consumed, the j-th is the new `next`.
            j=min(sp.plainrun,n-length(out));P=inner.position
            step!(s,next);j>1&&decode!(s,j-1)
            made=Int.(Array(inner.predictions)[P+1:P+j])
            for t in made[1:end-1]
                push!(out,t);push!(ix.history,t)
                if t in stops
                    # The stretch ran past the stop: the session stands after
                    # it, as plain decoding of the whole stretch would.
                    inner.token[]=Int32(made[end]);inner.posref[]=Int32(inner.position)
                    return out
                end
            end
            next=made[end]
            continue
        end
        w=1+length(proposal)
        exec=verifyplan!(sp,w,inner.position>=sp.deep)
        exec.tokens[:]=Int32.([next;proposal]);inner.posref[]=Int32(inner.position)
        Mantle.run!(exec.plan)
        predicted=Int.(Array(sp.buffers.best)[1:w])
        a=0
        while a<length(proposal)&&predicted[a+1]==proposal[a+1]&&!(proposal[a+1] in stops);a+=1;end
        sp.steps+=1;sp.drafted+=length(proposal);sp.accepted+=a
        if a<length(proposal)
            sp.refused[]=Int32(length(proposal)-a)
            Mantle.run!(replayplan!(sp,a))
        end
        inner.position+=a+1
        for t in proposal[1:a]
            push!(out,t);push!(ix.history,t)
        end
        # A refused draft token may still be a stop the model would have written:
        # that case is `predicted[a+1]`, which comes next.
        next=predicted[a+1]
    end
    inner.token[]=Int32(out[end]);inner.posref[]=Int32(inner.position)
    out
end

"""`generate!`'s contract with drafts: the session's pending token `pending` and
`n - 1` new ones consumed, `n` new tokens returned (the last left pending)."""
speculative_generate!(sp::Speculator,history,pending,n;kw...)=
    speculative_decode!(sp,history,n+1;first=pending,kw...)[2:end]
