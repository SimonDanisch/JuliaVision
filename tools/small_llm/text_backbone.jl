# The decoder a text harness drives. Each model family answers the same verbs for
# its own model and session types, so a harness never names a family. Loading
# this file creates no device.
isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
isdefined(@__MODULE__,:Qwen36GGUF) || include("qwen36_gguf.jl")
import Mantle

const Q35=SmallLLMChecks.Qwen35Experiment
const Q36=Qwen36MoEExperiment

# Qwen3.5 keeps the split attention its frozen studies ran with. Qwen3.6 takes
# its session's defaults (Float32 cache, grouped attention) up to 32k positions,
# which is what its receipts ran with; past that, where a Float32 cache is
# refused anyway (`maxcapacity`), flash-decoding over a q8_0 cache
# (`benchmark_long_context.jl` has the speeds and the divergence).
textsession(m::Q35.Model;capacity,logit_hook=nothing)=
    Q35.session(m;capacity,kv_type=Float32,attention=:split,logit_hook)
longsession(capacity)=capacity>FLOAT_KV_POSITIONS
textsession(m::Q36.Model;capacity,logit_hook=nothing,kv_type=longsession(capacity) ? Int8 : Float32,
            attention=longsession(capacity) ? :flash : :grouped)=
    Q36.session(m;capacity,logit_hook,kv_type,attention)

# Prompts are ingested in the widest chunk the shared emitter still runs as Q8
# GEMV passes, which reproduce sequential decode bit for bit. The studies ran
# with 32 when 32 columns were GEMV passes too; since wider products take the
# fp16 K-quant GEMM, 32 changed their answers (stable_groups: 150 -> 136 tokens)
# while 16 reproduces all 21 julia_tasks_results receipts (prompts up to 2473
# tokens) exactly.
const INGEST_CHUNK=Q35.DK.Q4K_GEMV_COLUMNS
# A long session's prompt goes in 512 tokens at a time: wide chunks give up bit
# identity with decode, and ingest 4-5x faster (benchmark_long_context.jl).
const LONG_INGEST_CHUNK=512
promptchunk(capacity)=longsession(capacity) ? LONG_INGEST_CHUNK : INGEST_CHUNK[]
ingest!(s::Q35.Session,ids;chunk=INGEST_CHUNK[])=Q35.prefill!(s,ids;chunk)
ingest!(s::Q36.Session,ids;chunk=INGEST_CHUNK[])=Q36.prefill!(s,ids;chunk)
feed!(s::Q35.Session,token)=Q35.step!(s,token)
feed!(s::Q36.Session,token)=Q36.step!(s,token)
generate!(s::Q35.Session,n)=Q35.decode!(s,n)
generate!(s::Q36.Session,n)=Q36.decode!(s,n)
# The host's count of consumed tokens, for a caller that restored the state of an
# earlier position; the next ingest or feed writes the device's from it.
setposition!(s::Q35.Session,p::Integer)=(s.position=p;s)
setposition!(s::Q36.Session,p::Integer)=(s.inner.position=p;s)

# Drafts from the context, checked a chunk at a time (Qwen3.6 only): with
# decode's attention in the verify, a drop-in for `generate!` whose tokens are
# the same. `undo` reverses a sampler's draws (`GPUSampling.declare_undo!`).
speculator(s::Q36.Session;undo=nothing,deep=typemax(Int))=Q36.Speculator(s;undo,deep)
speculator(s::Q35.Session;kw...)=throw(ArgumentError("speculative decoding is implemented for the Qwen3.6 session"))
speculate!(sp::Q36.Speculator,history,pending,n)=Q36.speculative_generate!(sp,history,pending,n)

# Drafts from the checkpoint's multi-token prediction layer (Qwen3.6 after
# `Q36.loadmtp!`), a drop-in for `generate!` as well. The layer runs over
# everything the session consumes, so a harness ingests, feeds, rewinds and keeps
# prefixes through the speculator: `consumer` is what it goes through.
mtpspeculator(s::Q36.Session;undo=nothing,k=2,deep=typemax(Int))=Q36.MTPSpeculator(s;undo,k,deep)
mtpspeculator(s::Q35.Session;kw...)=throw(ArgumentError("MTP drafts are implemented for the Qwen3.6 session"))
speculate!(sp::Q36.MTPSpeculator,history,pending,n)=Q36.speculative_generate!(sp,history,pending,n)
ingest!(sp::Q36.MTPSpeculator,ids;chunk=INGEST_CHUNK[])=Q36.prefill!(sp,ids;chunk)
feed!(sp::Q36.MTPSpeculator,token)=Q36.prefill!(sp,[token];chunk=1)
consumer(s,sp)=s
consumer(s,sp::Q36.MTPSpeculator)=sp

# The longest session a harness opens. Qwen3.5's 8192 is what its frozen studies
# ran under. Qwen3.6 keeps KV for 10 of its 40 layers: 16k positions are 0.7 GB
# in Float32, so a Float32 cache stops at 32k; the default past that (q8_0) and
# q4_0 hold the model's whole context.
const FLOAT_KV_POSITIONS=32768
maxcapacity(::Q35.Model;session=(;))=8192
maxcapacity(m::Q36.Model;session=(;))=get(session,:kv_type,nothing)===Float32 ? FLOAT_KV_POSITIONS : Int(m.config.max_position_embeddings)

"""What a result receipt records about the decoder that produced it."""
describe(m::Q35.Model)=(;family="Qwen3.5",quantization="Q4_K/Q8 activations; Q6_K head",cache="Float32 split")
describe(m::Q36.Model)=(;family="Qwen3.6 MoE",quantization="GGUF Q4_K_S (Q4_K/Q5_K/Q6_K) with Q8 activations",
    cache="Float32 grouped",optimizations=collect(string.(Q36.DEFAULT_OPTIMIZATIONS)))

# Paths from the workspace root, which every machine lays out the same way.
const WORKSPACE=normpath(joinpath(@__DIR__,"..","..","..",".."))
const QWEN36_GGUF=joinpath(WORKSPACE,"models","Qwen3.6-35B-A3B-Q4_K_S","Qwen3.6-35B-A3B.Q4_K_S.gguf")
# The checkpoint's multi-token prediction layer, from `download_qwen36_mtp.py`.
const QWEN36_MTP=joinpath(WORKSPACE,"models","Qwen3.6-35B-A3B-mtp")

isgguf(path)=isfile(path)&&endswith(path,".gguf")

"""A model directory (Qwen3.5 safetensors) or a GGUF checkpoint (Qwen3.6), with
the MTP layer from `QWEN36_MTP` if `mtp` (Qwen3.6, for `speculate=:mtp`)."""
function loadbackbone(path;device,verify_hash=true,mtp=false)
    isgguf(path) || (mtp && throw(ArgumentError("the MTP layer is Qwen3.6's"));return Q35.loadmodel(path;quantize=:q4k,device))
    model=Qwen36GGUF.loadmodel(path,joinpath(dirname(path),"config.json"),device;verify_hash)
    mtp && Q36.loadmtp!(model,QWEN36_MTP)
    model
end

"""The tokenizer for a model directory or a GGUF checkpoint.

Qwen3.6 ships the Qwen3.5 tokenizer. `shared` is any Qwen3.5 tokenizer
directory; its vocabulary, merges and special tokens are checked against the
checkpoint's own tables rather than trusting the pairing.
"""
function loadtokenizer(path;shared=joinpath(WORKSPACE,"gen","qwen35-4b"))
    isgguf(path) || return SmallLLMChecks.QwenText.Tokenizer(path)
    t=SmallLLMChecks.QwenText.Tokenizer(shared)
    meta=Q35.DK.readgguf(path).metadata
    # Past the text vocabulary the checkpoint lists 7 audio/TTS control tokens
    # and 243 padding entries (GGUF token types 3 and 5); a text harness that
    # decoded one would fail at the tokenizer's bounds rather than misread it.
    tokens=String.(meta["tokenizer.ggml.tokens"]);types=meta["tokenizer.ggml.token_type"];n=length(t.bpe.tokentostr)
    length(tokens)>=n&&tokens[1:n]==t.bpe.tokentostr&&all(in((3,5)),types[n+1:end])||
        error("Qwen3.6 vocabulary differs from $shared")
    merges=Dict(Tuple(String.(split(String(m),' ';limit=2)))=>i-1 for (i,m) in enumerate(meta["tokenizer.ggml.merges"]))
    merges==t.bpe.ranks||error("Qwen3.6 merges differ from $shared")
    (Int(meta["tokenizer.ggml.eos_token_id"]),Int(meta["tokenizer.ggml.bos_token_id"]))==(t.bpe.eos,t.bpe.bos)||
        error("Qwen3.6 special tokens differ from $shared")
    t
end

# ── Prompt prefixes kept on the device ───────────────────────────────────────

"""A session's state after a prompt prefix, on the device: every KV cache array's
rows of the prefix's positions and every recurrent buffer. Restored into a fresh
session of the same kind (cache format, attention) it replaces ingesting the
prefix, and what follows is bit for bit what a whole ingest gives: bit-exact
chunks run decode's arithmetic however they are cut, and wide chunks are cut at
the same places if the prefix ends on a chunk boundary (their attention splits
and products follow the chunk's width). A 128k-token context is then ingested
once and read many times."""
struct PrefixCache
    tokens::Vector{Int}
    kv::Vector{Any}
    recurrent::Vector{Any}
end

# The device arrays of a cache, each with positions as its last dimension.
kvarrays(c)=(c,)
kvarrays(c::Q35.QuantizedKV)=(c.q,c.scales)
kvarrays(c::Q36.Q.QuantizedKV)=(c.q,c.scales)
sessionkv(s)=collect(Any,Iterators.flatten(Iterators.flatten((kvarrays(kc),kvarrays(vc))) for (kc,vc) in (s.kv[il] for il in sort(collect(keys(s.kv))))))
sessionrecurrent(s)=collect(Any,Iterators.flatten(s.recurrent[il] for il in sort(collect(keys(s.recurrent)))))
# What consumed tokens left on the device, for a prefix or a rewind: a session's
# caches and recurrent state, and an MTP speculator's on top (its cache, the
# trunk states waiting for it and the tokens after them).
statesession(s)=s
statesession(sp::Q36.MTPSpeculator)=sp.session
statekv(s)=sessionkv(s)
statekv(sp::Q36.MTPSpeculator)=[sessionkv(sp.session);collect(Any,Iterators.flatten(map(kvarrays,sp.kv)))]
staterecurrent(s)=sessionrecurrent(s)
staterecurrent(sp::Q36.MTPSpeculator)=[sessionrecurrent(sp.session);Any[sp.hidden,sp.buffers.best]]
# An MTP prefix is kept right after an ingest, with the one state that waits then.
checkprefix(s)=nothing
checkprefix(sp::Q36.MTPSpeculator)=sp.pending==1&&sp.position==sp.session.position ||
    throw(ArgumentError("an MTP speculator's prefix is kept right after an ingest"))
restored!(s,p)=setposition!(s,p)
restored!(sp::Q36.MTPSpeculator,p)=(setposition!(sp.session,p);sp.pending=p>0 ? 1 : 0;sp.position=p;sp)
# The first `p` positions of a positions-last array, as a flat view.
prefixview(a,p)=Mantle.viewof(a,(p*(length(a)÷size(a,ndims(a))),))

function copyall!(device,pairs)
    g=Mantle.Graph(device)
    for (k,(dst,src)) in enumerate(pairs)
        Mantle.dispatch!(g,Q35.BR.copy_kernel!,(dst,src,Int32(length(src))),length(src);group=256,name="prefix.copy_$k")
    end
    plan=Mantle.record!(Mantle.Plan(g;budget=Inf))
    try
        Mantle.run!(plan);Mantle.waitidle(device)
    finally
        Mantle.free!(plan)
    end
end

"""The state of `x` (a session or an MTP speculator), which has consumed exactly
`tokens`, kept for `restore!`."""
function PrefixCache(x,tokens)
    s=statesession(x);p=length(tokens);device=s.model.device
    s.position==p || throw(ArgumentError("the session has consumed $(s.position) tokens, not the prefix's $p"))
    checkprefix(x)
    kv=[prefixview(a,p) for a in statekv(x)];rec=staterecurrent(x);owned=Any[]
    try
        kvcopy=Any[(b=Mantle.Buffer(device,eltype(v),(length(v),));push!(owned,b);b) for v in kv]
        reccopy=Any[(b=Mantle.Buffer(device,eltype(r),size(r));push!(owned,b);b) for r in rec]
        copyall!(device,[zip(kvcopy,kv)...,zip(reccopy,rec)...])
        PrefixCache(copy(tokens),kvcopy,reccopy)
    catch
        foreach(Mantle.free!,owned);rethrow()
    end
end
Base.close(c::PrefixCache)=(foreach(Mantle.free!,c.kv);foreach(Mantle.free!,c.recurrent);nothing)

"""Put a prefix's state into `x`, a session or an MTP speculator of the kind it
was kept from (fresh, or to be rewound to it)."""
function restore!(x,c::PrefixCache)
    s=statesession(x);p=length(c.tokens);kv=statekv(x);rec=staterecurrent(x)
    length(kv)==length(c.kv)&&all(i->eltype(kv[i])==eltype(c.kv[i])&&length(prefixview(kv[i],p))==length(c.kv[i]),eachindex(kv)) ||
        throw(ArgumentError("the session's caches are not the ones the prefix was kept from"))
    p<=s.capacity || throw(ArgumentError("the session holds $(s.capacity) positions, the prefix $p"))
    length(rec)==length(c.recurrent) || throw(ArgumentError("the prefix was kept from a different kind of session"))
    copyall!(s.model.device,[zip((prefixview(a,p) for a in kv),c.kv)...,zip(rec,c.recurrent)...])
    restored!(x,p)
end
