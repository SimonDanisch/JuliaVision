# The decoder a text harness drives. Each model family answers the same verbs for
# its own model and session types, so a harness never names a family. Loading
# this file creates no device.
isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
isdefined(@__MODULE__,:Qwen36MoEExperiment) || include("qwen36_moe_model.jl")
isdefined(@__MODULE__,:Qwen36GGUF) || include("qwen36_gguf.jl")
import Mantle

const Q35=SmallLLMChecks.Qwen35Experiment
const Q36=Qwen36MoEExperiment

# Qwen3.5 keeps the split attention its frozen studies ran with. Qwen3.6 has no
# earlier receipts to reproduce and takes its session's defaults.
textsession(m::Q35.Model;capacity,logit_hook=nothing)=
    Q35.session(m;capacity,kv_type=Float32,attention=:split,logit_hook)
textsession(m::Q36.Model;capacity,logit_hook=nothing)=Q36.session(m;capacity,logit_hook)

# Prompts are ingested in the widest chunk the shared emitter still runs as Q8
# GEMV passes, which reproduce sequential decode bit for bit. The studies ran
# with 32 when 32 columns were GEMV passes too; since wider products take the
# fp16 K-quant GEMM, 32 changed their answers (stable_groups: 150 -> 136 tokens)
# while 16 reproduces all 21 julia_tasks_results receipts (prompts up to 2473
# tokens) exactly.
const INGEST_CHUNK=Q35.DK.Q4K_GEMV_COLUMNS
ingest!(s::Q35.Session,ids)=Q35.prefill!(s,ids;chunk=INGEST_CHUNK[])
ingest!(s::Q36.Session,ids)=Q36.prefill!(s,ids;chunk=INGEST_CHUNK[])
feed!(s::Q35.Session,token)=Q35.step!(s,token)
feed!(s::Q36.Session,token)=Q36.step!(s,token)
generate!(s::Q35.Session,n)=Q35.decode!(s,n)
generate!(s::Q36.Session,n)=Q36.decode!(s,n)
# The host's count of consumed tokens, for a caller that restored the state of an
# earlier position; the next ingest or feed writes the device's from it.
setposition!(s::Q35.Session,p::Integer)=(s.position=p;s)
setposition!(s::Q36.Session,p::Integer)=(s.inner.position=p;s)

# The longest session a harness opens. Qwen3.5's 8192 is what its frozen studies
# ran under. Qwen3.6 keeps KV for 10 of its 40 layers: 16k positions are 0.7 GB.
maxcapacity(::Q35.Model)=8192
maxcapacity(::Q36.Model)=32768

"""What a result receipt records about the decoder that produced it."""
describe(m::Q35.Model)=(;family="Qwen3.5",quantization="Q4_K/Q8 activations; Q6_K head",cache="Float32 split")
describe(m::Q36.Model)=(;family="Qwen3.6 MoE",quantization="GGUF Q4_K_S (Q4_K/Q5_K/Q6_K) with Q8 activations",
    cache="Float32 grouped",optimizations=collect(string.(Q36.DEFAULT_OPTIMIZATIONS)))

# Paths from the workspace root, which every machine lays out the same way.
const WORKSPACE=normpath(joinpath(@__DIR__,"..","..","..",".."))
const QWEN36_GGUF=joinpath(WORKSPACE,"models","Qwen3.6-35B-A3B-Q4_K_S","Qwen3.6-35B-A3B.Q4_K_S.gguf")

isgguf(path)=isfile(path)&&endswith(path,".gguf")

"""A model directory (Qwen3.5 safetensors) or a GGUF checkpoint (Qwen3.6)."""
function loadbackbone(path;device,verify_hash=true)
    isgguf(path) || return Q35.loadmodel(path;quantize=:q4k,device)
    Qwen36GGUF.loadmodel(path,joinpath(dirname(path),"config.json"),device;verify_hash)
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
