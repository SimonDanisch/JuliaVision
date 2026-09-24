"""
    BonsaiRunner

Text-only inference for Prism's PTQ1 build of Ternary Bonsai 2 27B. Weights
remain packed ternary on the device. Decode and chunked prompt prefill are
recorded once and replayed with new token data.
"""
module BonsaiRunner

using DNNKernels
using KernelAbstractions
using LazyArtifacts
using Scratch
using SHA
import Mantle

export Bonsai2, BonsaiSession, BonsaiTokenizer, session, step!, prefill!, generate, release!
export encode, decode, chatprompt, loadweights!, checkpointpath

include("tokenizer.jl")
include("kernels.jl")

const WIDTH = 5120
const FFN = 17408
const NLAYERS = 64
const EPS = 1f-6

const CHECKPOINT_NAME = "Ternary-Bonsai-2-27B-PTQ1_0.gguf"
const CHECKPOINT_BYTES = 5_946_648_928
const CHECKPOINT_SHA256 =
    "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
const CHECKPOINT_PART_FILES = ntuple(
    i -> "Ternary-Bonsai-2-27B-PTQ1_0-part-$(i)of4", 4)
const CHECKPOINT_ARTIFACTS = ntuple(i -> "bonsai2-ptq1-p$i", 4)

# Literal artifact names are intentional: `@artifact_str` resolves against the
# Artifacts.toml beside this package at macro expansion. Each tree is lazy, so
# importing BonsaiRunner downloads nothing.
_checkpointpart(::Val{1}) = @artifact_str("bonsai2-ptq1-p1")
_checkpointpart(::Val{2}) = @artifact_str("bonsai2-ptq1-p2")
_checkpointpart(::Val{3}) = @artifact_str("bonsai2-ptq1-p3")
_checkpointpart(::Val{4}) = @artifact_str("bonsai2-ptq1-p4")

function _copyall!(out, src::AbstractString)
    open(src, "r") do input
        buf = Vector{UInt8}(undef, 16 * 1024 * 1024)
        while !eof(input)
            n = readbytes!(input, buf)
            write(out, view(buf, 1:n))
        end
    end
end

"""
    checkpointpath() -> String

Resolve the four lazy artifacts containing Bonsai's 5.95 GB PTQ1 checkpoint and
assemble their byte-exact concatenation once in Scratch.jl's package cache.

GitHub release assets are capped at 2 GiB, so the upstream GGUF cannot be one
release asset. Each part is independently checked by Julia's artifact system;
the assembled file is additionally checked against upstream's SHA-256 before it
is made visible. Subsequent calls reuse the size- and digest-versioned result.
"""
function checkpointpath()
    root = Scratch.get_scratch!(BonsaiRunner, "checkpoint-$CHECKPOINT_SHA256")
    path = joinpath(root, CHECKPOINT_NAME)
    isfile(path) && filesize(path) == CHECKPOINT_BYTES && return path

    tmp = path * ".tmp-$(getpid())-$(Threads.threadid())"
    try
        open(tmp, "w") do out
            for i in 1:4
                part = joinpath(_checkpointpart(Val(i)), CHECKPOINT_PART_FILES[i])
                isfile(part) || error("Bonsai checkpoint artifact part $i has no $(CHECKPOINT_PART_FILES[i])")
                _copyall!(out, part)
            end
        end
        filesize(tmp) == CHECKPOINT_BYTES || error(
            "assembled Bonsai checkpoint is $(filesize(tmp)) bytes, expected $CHECKPOINT_BYTES")
        digest = bytes2hex(open(SHA.sha256, tmp))
        digest == CHECKPOINT_SHA256 || error(
            "assembled Bonsai checkpoint SHA-256 is $digest, expected $CHECKPOINT_SHA256")
        mv(tmp, path; force=true)
    finally
        isfile(tmp) && rm(tmp)
    end
    return path
end

"""A loaded Bonsai checkpoint and the device selected for it."""
struct Bonsai2{B,C}
    backend::B
    # `Mantle.caps(device)`, which is the only thing this ever wanted. It held a
    # whole `DNNKernels.Ctx` — the eager execution context — to reach
    # `ctx.dev.subgroup`, and `Ctx.dev` IS a `DeviceCaps`. One indirection and
    # one API less.
    caps::C
    file::GGUFFile
    weights::Dict{String,Any}
    signs::Dict{Int,Any}
    folded::Set{String}
    inverse::Set{String}
    tokenizer::BonsaiTokenizer
    maxcontext::Int
    theta::Float32
    eps::Float32
end

# `Mantle.Buffer(dev, data)` IS the upload: it takes the region and fills it.
# One allocator for everything this runner owns, and one verb — `Mantle.free!` —
# that gives it back.
_upload(backend, host::AbstractArray) =
    Mantle.Buffer(Mantle.todevice(backend), Float32.(host))

_upload16(backend, host::AbstractArray) =
    Mantle.Buffer(Mantle.todevice(backend), Float16.(host))

"""
    Bonsai2(path; device=Mantle.device(), context=nothing, loadweights=true)

Open a PTQ1 Bonsai GGUF on `device`. `context=nothing` selects the checkpoint's
native context length (262144 for Ternary Bonsai 2); KV storage is allocated
lazily by [`session`](@ref), so selecting it does not reserve the full cache.
Metadata parsing is immediate; packed and dense tensors are uploaded when
`loadweights` is true. `loadweights=false` is useful for inspecting/tokenizing
a checkpoint without allocating model memory.
"""
function Bonsai2(path::AbstractString; device=Mantle.device(), context=nothing,
                 loadweights::Bool=true, progress::Bool=true)
    file = readgguf(path)
    md = file.metadata
    get(md, "general.architecture", "") == "qwen35" || error("checkpoint is not qwen35")
    Int(md["qwen35.block_count"]) == NLAYERS || error("BonsaiRunner expects 64 blocks")
    Int(md["qwen35.embedding_length"]) == WIDTH || error("BonsaiRunner expects width 5120")
    Int(md["qwen35.full_attention_interval"]) == 4 || error("unexpected full-attention interval")
    Int(md["prism.hadamard.version"]) == 1 || error("unsupported Hadamard metadata")
    Int(md["prism.hadamard.block_size"]) == 1024 || error("unsupported Hadamard block size")
    nativecontext = Int(md["qwen35.context_length"])
    context = context === nothing ? nativecontext : Int(context)
    context > 0 || throw(ArgumentError("context must be positive"))
    context <= nativecontext || throw(ArgumentError("context exceeds checkpoint limit"))

    backend = Mantle.backend(device)
    caps = Mantle.caps(Mantle.todevice(backend))
    weights = Dict{String,Any}()
    model = Bonsai2(backend, caps, file, weights, Dict{Int,Any}(),
        Set(String.(md["prism.hadamard.weight_names"])),
        Set(String.(get(md, "prism.hadamard.inverse_weight_names", String[]))),
        BonsaiTokenizer(file), Int(context), Float32(md["qwen35.rope.freq_base"]),
        Float32(md["qwen35.attention.layer_norm_rms_epsilon"]))
    _loadsigns!(model)
    loadweights && loadweights!(model; progress)
    model
end

"""Load Bonsai from the package's lazy, content-addressed checkpoint artifacts."""
Bonsai2(; kw...) = Bonsai2(checkpointpath(); kw...)

function _loadsigns!(model::Bonsai2)
    widths = Int.(model.file.metadata["prism.hadamard.sign_widths"])
    values = model.file.metadata["prism.hadamard.sign_values"]
    off = 0
    for width in widths
        off + width <= length(values) || error("truncated Hadamard sign metadata")
        host = Float32.(view(values, off+1:off+width))
        all(x -> x == -1f0 || x == 1f0, host) || error("Hadamard signs must be +/-1")
        model.signs[width] = _upload(model.backend, host)
        off += width
    end
    off == length(values) || error("extra Hadamard sign metadata")
    model
end

"""Upload every checkpoint tensor, retaining PTQ1 matrices in packed form."""
function loadweights!(model::Bonsai2; progress::Bool=true)
    isempty(model.weights) || return model
    names = sort!(collect(keys(model.file.tensors)))
    for (i, name) in enumerate(names)
        t = model.file[name]
        if t.typeid == DNNKernels.GGML_TYPE_PTQ1_0
            model.weights[name] = ptq1matrix(model.backend, model.file, t)
        elseif t.typeid in (DNNKernels.GGML_TYPE_F32, DNNKernels.GGML_TYPE_F16,
                            DNNKernels.GGML_TYPE_BF16)
            host = DNNKernels.gguffloat(model.file, t)
            model.weights[name] = _upload(model.backend, host)
        else
            error("unsupported GGML tensor type $(t.typeid) for $name")
        end
        progress && (i == 1 || i % 32 == 0 || i == length(names)) &&
            println("BonsaiRunner: loaded $i/$(length(names)) tensors")
    end
    for il in 0:NLAYERS-1
        (il + 1) % 4 == 0 && continue
        prefix = "blk.$il."
        ah = DNNKernels.gguffloat(model.file, model.file[prefix * "ssm_alpha.weight"])
        bh = DNNKernels.gguffloat(model.file, model.file[prefix * "ssm_beta.weight"])
        model.weights[prefix * "ssm_ab.weight.coop"] =
            _upload16(model.backend, permutedims(hcat(ah, bh)))
    end
    KernelAbstractions.synchronize(model.backend)
    model
end

mutable struct BonsaiSession{M}
    model::M
    position::Int
    recurrent::Dict{Int,Tuple{Any,Any}}
    kv::Dict{Int,Any}
    logits::Any                # session-owned; survives a plan rebuild
    scratch::Any               # GraphScratch over the current step graph
    tokenref::Any
    positionref::Any
    plan::Any
    kvpage::Int
    prefills::Dict{Int,Any}
end

struct PrefillExec
    tokens::Any
    position::Any
    scratch::Any
    plans::Vector{Any}
end

"""Quantized K/V storage for one full-attention layer."""
mutable struct Q8KVCache{A,S}
    k::A
    v::A
    kscale::S
    vscale::S
    capacity::Int
end

# Allocated and then filled, rather than `zeros` on the host and uploaded: the
# recurrent state is per layer and the host copy would be pure waste.
function _zeros(m::Bonsai2, ::Type{T}, dims...) where {T}
    b = Mantle.Buffer(Mantle.todevice(m.backend), T, Dims(dims))
    fill!(b, zero(T))
    b
end

function _q8cache(model::Bonsai2, capacity::Integer)
    dev = Mantle.todevice(model.backend)
    k = Mantle.Buffer(dev, Int8, (256, 4, capacity))
    v = Mantle.Buffer(dev, Int8, (256, 4, capacity))
    ks = Mantle.Buffer(dev, Float32, (4, capacity))
    vs = Mantle.Buffer(dev, Float32, (4, capacity))
    Q8KVCache(k, v, ks, vs, Int(capacity))
end

"""
    session(model; kv_page=4096)

Allocate recurrent state and the first page of a lazily growing Q8 KV cache.
Capacity doubles in page-sized increments as the sequence grows, up to the
model's declared context limit.
"""
function session(model::Bonsai2; kv_page::Integer=4096)
    isempty(model.weights) && error("weights are not loaded")
    kv_page > 0 || throw(ArgumentError("kv_page must be positive"))
    initial = min(Int(kv_page), model.maxcontext)
    recurrent = Dict{Int,Tuple{Any,Any}}()
    kv = Dict{Int,Any}()
    for il in 0:NLAYERS-1
        if (il + 1) % 4 == 0
            kv[il] = _q8cache(model, initial)
        else
            recurrent[il] = (_zeros(model, Float32, 3, 10240),
                              _zeros(model, Float32, 128, 128, 48))
        end
    end
    dev = Mantle.Device(model.backend)
    # Owned by the SESSION and not by the step graph: `_ensure_kv!` rebuilds the
    # plan when the cache grows, and a `:logits` declared into the graph would be
    # replaced on every rebuild with nothing freeing the old one.
    logits = _zeros(model, Float32, 248320)
    s = BonsaiSession(model, 0, recurrent, kv, logits, nothing,
                      Mantle.GPURef(dev, Int32(0)),
                      Mantle.GPURef(dev, Int32(0)), nothing, Int(kv_page),
                      Dict{Int,Any}())
    s.plan, s.scratch = _recordstep(s, dev)
    s
end

"""
    _stepscratch(g, logits) -> GraphScratch

One token's working set, declared into the step graph. Same story as
[`_batchscratch`](@ref): these were 25 permanent `_zeros` and the step plan
reported `naive = peak = 0` because nothing in it was declared.

`ntokens = 1`, so the widths are absolute. The Hadamard activations are Float32
here and Float16 in the batch path, which is why the precision set is per
scratch and not global.
"""
_stepscratch(g, logits) = GraphScratch(
    g, 1,
    Dict(:x=>WIDTH, :norm=>WIDTH, :branch=>WIDTH, :had5120=>WIDTH,
         :gate=>FFN, :up=>FFN, :ff=>FFN, :had17408=>FFN,
         :qkv=>10240, :z=>6144, :conv=>10240, :rq=>2048, :rk=>2048,
         :rv=>6144, :gdn=>6144, :had6144=>6144,
         :qfull=>12288, :kproj=>1024, :vproj=>1024, :aq=>6144,
         :qgate=>6144, :attn=>6144, :alpha=>48, :beta=>48),
    Dict{Symbol,Int}(), Set{Symbol}(),
    Dict{Symbol,Any}(:logits => logits), Dict{Symbol,Any}())

"""Current token capacity of a session's lazily allocated KV cache."""
kv_capacity(s::BonsaiSession) = first(values(s.kv)).capacity

"""Bytes occupied by all attention-layer Q8 K/V caches at `capacity`."""
kv_bytes(capacity::Integer) = (NLAYERS ÷ 4) * capacity *
    (2 * 256 * 4 * sizeof(Int8) + 2 * 4 * sizeof(Float32))

"""Declare the copy of `src` into the head of `dest` as a pass of `g`."""
function _declarecopy!(g, dest, src)
    n = length(src)
    Mantle.dispatch!(g, copy_kernel!, (dest, src, Int32(n)), n;
                     group = 256, name = "kv_copy")
    dest
end

function _ensure_kv!(s::BonsaiSession, needed::Integer)
    needed <= kv_capacity(s) && return s
    needed <= s.model.maxcontext || error("session context is full")
    oldcap = kv_capacity(s)
    target = min(s.model.maxcontext,
                 cld(max(Int(needed), 2oldcap), s.kvpage) * s.kvpage)
    dev = Mantle.Device(s.model.backend)
    # Wait before reallocating: the caches being replaced are named by the
    # recording still in flight.
    Mantle.waitidle(dev)
    # ONE graph for every layer's four copies — 64 passes and one submission,
    # which is the thing a graph is for. Four launches per layer was 64.
    g = Mantle.Graph(dev)
    replacement = Dict{Int,Any}()
    for (il, old) in s.kv
        new = _q8cache(s.model, target)
        _declarecopy!(g, new.k, old.k)
        _declarecopy!(g, new.v, old.v)
        _declarecopy!(g, new.kscale, old.kscale)
        _declarecopy!(g, new.vscale, old.vscale)
        replacement[il] = new
    end
    Mantle.runonce!(g)
    # The copy is submitted, not finished — and that is exactly why the old
    # caches can go back now: `free!` RETIRES a region, and the pool returns it
    # to the free list only once the device has passed the submission reading
    # it. Skipping this leaked a whole generation of the cache per doubling:
    # 130 MiB at the first growth, 260 at the second, and nothing ever gave them
    # back because Mantle frees on a verb and never from a finalizer.
    for old in values(s.kv)
        Mantle.free!(old.k); Mantle.free!(old.v)
        Mantle.free!(old.kscale); Mantle.free!(old.vscale)
    end
    s.kv = replacement
    oldplan = s.plan
    s.plan, s.scratch = _recordstep(s, Mantle.Device(s.model.backend))
    Mantle.free!(oldplan)
    for p in values(s.prefills)
        foreach(Mantle.free!, p.plans)
    end
    empty!(s.prefills)
    s
end

"""
    release!(session) -> nothing

Give back everything the session owns: its recorded step plan, the prefill plans
it built per chunk length, the Q8 KV caches, the recurrent state, the logits and
the two `GPURef`s.

**A dropped session frees nothing.** Mantle frees on an explicit verb and never
from a finalizer, so a caller that opens a session per conversation and lets it
go out of scope keeps every one of them: about 274 MiB of state per session on
the default page, plus a recording of 1527 passes. Measured while chasing what
looked like a slowdown: twelve sessions held 11 GB, and dropping the references
and running a full GC returned 281 MiB of it.

The model is untouched — `release!` is per session, and another session can be
opened on the same weights immediately.
"""
function release!(s::BonsaiSession)
    Mantle.free!(s.plan)
    for p in values(s.prefills)
        foreach(Mantle.free!, p.plans)
    end
    empty!(s.prefills)
    for c in values(s.kv)
        Mantle.free!(c.k); Mantle.free!(c.v)
        Mantle.free!(c.kscale); Mantle.free!(c.vscale)
    end
    empty!(s.kv)
    for (a, b) in values(s.recurrent)
        Mantle.free!(a); Mantle.free!(b)
    end
    empty!(s.recurrent)
    Mantle.free!(s.logits)
    Mantle.free!(s.tokenref)
    Mantle.free!(s.positionref)
    return nothing
end

@inline _w(s::BonsaiSession, name) = s.model.weights[name]

@inline function _dispatch!(g, kernel, args::Tuple, ndrange;
                            group::Int=256, name::AbstractString=string(nameof(kernel)))
    Mantle.dispatch!(g, kernel, args, ndrange; group, name)
end

function _hadamard!(s::BonsaiSession, g, x, signs; inverse::Bool=false)
    width = length(x)
    blocks = width ÷ 1024
    _dispatch!(g, DNNKernels.hadamard1024_kernel!,
        (x, signs, Int32(width), Int32(blocks), Val(true), Val(inverse)),
        blocks * 256; group=256, name=inverse ? "hadamard_inverse" : "hadamard")
    x
end

function _transform!(s::BonsaiSession, g, dest, x; permute::Bool=false)
    if permute
        _dispatch!(g, gdn_permute_kernel!, (dest, x), 6144;
                   group=256, name="gdn_permute")
    else
        _dispatch!(g, copy_kernel!, (dest, x, Int32(length(x))), length(x);
                   group=256, name="copy")
    end
    _hadamard!(s, g, dest, s.model.signs[length(dest)])
end

function _ptq!(s::BonsaiSession, g, out, name::String, transformed,
               workspace=nothing)
    name in s.model.folded || error("$name is PTQ1 but has no Hadamard declaration")
    A = _w(s, name)
    M, K = size(A)
    N = length(transformed) ÷ K
    coopmat = workspace !== nothing && N % DNNKernels.PTQ1_COOP_BN == 0 &&
              M % DNNKernels.PTQ1_COOP_BM == 0 &&
              K % DNNKernels.PTQ1_COOP_BK == 0 &&
              eltype(transformed) === Float16 &&
              g.dev isa Mantle.LavaDevice &&
              Mantle.coopmat_gemm_available(g.dev.ctx)
    if coopmat
        blocks = (M ÷ DNNKernels.PTQ1_COOP_BM) *
                 (N ÷ DNNKernels.PTQ1_COOP_BN)
        _dispatch!(g, DNNKernels.ptq1_coopmat_kernel!,
            (out, A.data, transformed, Val(M), Val(N), Val(K)),
            blocks * DNNKernels.PTQ1_COOP_WG;
            group=DNNKernels.PTQ1_COOP_WG, name)
    elseif N >= 8
        mtiles = cld(M, DNNKernels.PTQ1_MM_BM)
        ntiles = cld(N, DNNKernels.PTQ1_MM_BN)
        _dispatch!(g, DNNKernels.ptq1_mul_mm_kernel!,
            (out, A.data, transformed, Int32(M), Int32(K), Int32(N),
             Int32(mtiles)),
            mtiles * ntiles * DNNKernels.PTQ1_WG;
            group=DNNKernels.PTQ1_WG, name=name)
    else
        rows = cld(M, DNNKernels.PTQ1_ROWS_PER_WG)
        kernel = N == 1 ? DNNKernels.ptq1_mul_kernel! : DNNKernels.ptq1_mul4_kernel!
        columns = N == 1 ? N : cld(N, DNNKernels.PTQ1_COLS_PER_WG)
        _dispatch!(g, kernel,
            (out, A.data, transformed, A.data, Int32(M), Int32(K), Int32(N),
             Int32(rows), Val(false), Val(s.model.caps.subgroup)),
            rows * columns * DNNKernels.PTQ1_WG;
            group=DNNKernels.PTQ1_WG, name=name)
    end
    out
end

function _recurrent!(s::BonsaiSession, g, q, il::Int, xnorm)
    prefix = "blk.$il."
    _transform!(s, g, q[:had5120], xnorm)
    _ptq!(s, g, q[:qkv], prefix * "attn_qkv.weight", q[:had5120])
    _ptq!(s, g, q[:z], prefix * "attn_gate.weight", q[:had5120])
    _dispatch!(g, dense_gemv_kernel!,
        (q[:alpha], _w(s, prefix * "ssm_alpha.weight"), xnorm,
         Int32(length(xnorm)), Int32(48), Val(s.model.caps.subgroup)),
        cld(48, DENSE_ROWS_PER_WG) * 256;
        group=256, name=prefix * "ssm_alpha")
    _dispatch!(g, dense_gemv_kernel!,
        (q[:beta], _w(s, prefix * "ssm_beta.weight"), xnorm,
         Int32(length(xnorm)), Int32(48), Val(s.model.caps.subgroup)),
        cld(48, DENSE_ROWS_PER_WG) * 256;
        group=256, name=prefix * "ssm_beta")
    convstate, state = s.recurrent[il]
    _dispatch!(g, DNNKernels.depthwise_conv4_kernel!,
        (q[:conv], convstate, q[:qkv], _w(s, prefix * "ssm_conv1d.weight"),
         Int32(length(q[:qkv]))), length(q[:qkv]);
        group=256, name=prefix * "ssm_conv1d")
    _dispatch!(g, recurrent_split_kernel!,
        (q[:rq], q[:rk], q[:rv], q[:conv]), 6144;
        group=256, name=prefix * "recurrent_split")
    _dispatch!(g, DNNKernels.gated_delta_net_kernel!,
        (q[:gdn], state, q[:rq], q[:rk], q[:rv], q[:alpha], q[:beta],
         _w(s, prefix * "ssm_dt.bias"), _w(s, prefix * "ssm_a"), q[:z],
         _w(s, prefix * "ssm_norm.weight"), s.model.eps, Int32(48), Int32(16)),
        48 * 128; group=128, name=prefix * "gated_delta_net")
    _transform!(s, g, q[:had6144], q[:gdn]; permute=true)
    _ptq!(s, g, q[:branch], prefix * "ssm_out.weight", q[:had6144])
    q[:branch]
end

function _attention!(s::BonsaiSession, g, q, il::Int, xnorm)
    prefix = "blk.$il."
    _transform!(s, g, q[:had5120], xnorm)
    _ptq!(s, g, q[:qfull], prefix * "attn_q.weight", q[:had5120])
    _ptq!(s, g, q[:kproj], prefix * "attn_k.weight", q[:had5120])
    _ptq!(s, g, q[:vproj], prefix * "attn_v.weight", q[:had5120])
    _dispatch!(g, prepare_q_kernel!,
        (q[:aq], q[:qgate], q[:qfull], _w(s, prefix * "attn_q_norm.weight"),
         s.positionref, s.model.theta, s.model.eps), 24 * 256;
        group=256, name=prefix * "prepare_q")
    cache = s.kv[il]
    _dispatch!(g, prepare_kv_kernel!,
        (cache.k, cache.v, cache.kscale, cache.vscale,
         q[:kproj], q[:vproj], _w(s, prefix * "attn_k_norm.weight"),
         s.positionref, s.model.theta, s.model.eps), 4 * 256;
        group=256, name=prefix * "prepare_kv")
    _dispatch!(g, decode_attention_kernel!,
        (q[:attn], q[:aq], q[:qgate], cache.k, cache.v,
         cache.kscale, cache.vscale, s.positionref), 24 * 256;
        group=256, name=prefix * "decode_attention")
    _transform!(s, g, q[:had6144], q[:attn])
    _ptq!(s, g, q[:branch], prefix * "attn_output.weight", q[:had6144])
    q[:branch]
end

function _ffn!(s::BonsaiSession, g, q, il::Int, xnorm)
    prefix = "blk.$il."
    _transform!(s, g, q[:had5120], xnorm)
    _ptq!(s, g, q[:gate], prefix * "ffn_gate.weight", q[:had5120])
    _ptq!(s, g, q[:up], prefix * "ffn_up.weight", q[:had5120])
    _dispatch!(g, swiglu_kernel!,
        (q[:ff], q[:gate], q[:up], Int32(FFN)), FFN;
        group=256, name=prefix * "swiglu")
    _transform!(s, g, q[:had17408], q[:ff])
    _ptq!(s, g, q[:branch], prefix * "ffn_down.weight", q[:had17408])
    q[:branch]
end

"""
The prefill's working set, declared into the graph ON DEMAND.

Every entry used to be a `_zeros` — a permanent allocation, live for the whole
prefill, 305 MiB at 512 tokens across 29 buffers. That is Mantle's `Liveness`
and `Place` written by hand: the plans reported `naive = peak = 0` because there
was nothing declared in them to place.

Lazy rather than a table built up front, because which entries a prefill
actually touches depends on `ntokens` — a five-token prompt reaches a different
set from a 512-token chunk — and `Liveness` refuses a transient no pass uses.
That refusal is right: an interval derived from use has nothing to say about a
buffer with no uses.

It is also the thing the eager table was hiding. `_zeros` asks nothing and
allocates everything, so an entry this path never reads costs its full size on
every prefill and never complains; `:ff` is 34 MiB of exactly that, used by the
decode `_ffn!` and not by `_ffn_batch!`. Declared lazily it simply never
appears, which is better than deleting it from the table and risking a
`KeyError` on a path that was not exercised.

`:logits` stays a real buffer: `prefill!` hands it to the host after `run!`, so
it has to outlive the plan's arena.
"""
struct GraphScratch{G}
    g::G
    ntokens::Int                   # `widths` scale with this
    widths::Dict{Symbol,Int}
    fixed::Dict{Symbol,Int}        # …and these do not
    half::Set{Symbol}              # declared Float16 rather than Float32
    escapes::Dict{Symbol,Any}      # real buffers the host reads after `run!`
    buf::Dict{Symbol,Any}
end

function Base.getindex(q::GraphScratch, k::Symbol)
    get!(q.buf, k) do
        haskey(q.escapes, k) && return q.escapes[k]
        n = haskey(q.widths, k) ? q.widths[k] * q.ntokens :
            haskey(q.fixed, k) ? q.fixed[k] : throw(KeyError(k))
        # The Hadamard-transformed activations feed fp16 cooperative matrices.
        # Storing them in that precision removes a separate cast before every
        # projection and has the same rounding point as the old cast.
        Mantle.Transient.Buffer(q.g, k in q.half ? Float16 : Float32, (n,))
    end
end

function _batchscratch(g, logits, ntokens::Int)
    widths = Dict(
        :x=>WIDTH, :norm=>WIDTH, :branch=>WIDTH, :had5120=>WIDTH,
        :gate=>FFN, :up=>FFN, :ff=>FFN, :had17408=>FFN,
        :qkv=>10240, :z=>6144, :conv=>10240, :rq=>2048, :rk=>2048,
        :rv=>6144, :gdn=>6144, :had6144=>6144,
        :qfull=>12288, :kproj=>1024, :vproj=>1024, :aq=>6144,
        :qgate=>6144, :attn=>6144, :alpha=>48, :beta=>48, :alphabeta=>96,
        :densehalf=>WIDTH)
    GraphScratch(g, Int(ntokens), widths,
                 Dict(:lastnorm => WIDTH, :lasthad => WIDTH),
                 Set([:had5120, :had6144, :had17408, :densehalf]),
                 Dict{Symbol,Any}(:logits => logits),
                 Dict{Symbol,Any}())
end

function _hadamard_batch!(s::BonsaiSession, g, x, width::Int, ntokens::Int;
                          inverse::Bool=false)
    blocks = width ÷ 1024
    _dispatch!(g, DNNKernels.hadamard1024_kernel!,
        (x, s.model.signs[width], Int32(width), Int32(blocks),
         Val(true), Val(inverse)),
        blocks * ntokens * 256; group=256,
        name=inverse ? "hadamard_inverse_batch" : "hadamard_batch")
    x
end

function _transform_batch!(s::BonsaiSession, g, dest, x, width::Int, ntokens::Int;
                           permute::Bool=false, halfcopy=nothing)
    blocks = width ÷ 1024
    if permute
        width == 6144 || error("GDN permutation requires width 6144")
        _dispatch!(g, gdn_permute_hadamard1024_batch_kernel!,
            (dest, x, s.model.signs[width], Int32(width), Int32(blocks)),
            blocks * ntokens * 256; group=256,
            name="gdn_permute_hadamard_batch")
    else
        copydest = halfcopy === nothing ? dest : halfcopy
        _dispatch!(g, hadamard1024_out_batch_kernel!,
            (dest, copydest, x, s.model.signs[width], Int32(width),
             Int32(blocks), Val(halfcopy !== nothing)),
            blocks * ntokens * 256; group=256, name="hadamard_out_batch")
    end
    dest
end

function _dense_batch!(s::BonsaiSession, g, out, weight, x, K::Int, M::Int,
                       ntokens::Int, name::String)
    if ntokens >= 8
        mtiles = cld(M, DENSE_MM_BM)
        ntiles = cld(ntokens, DENSE_MM_BN)
        _dispatch!(g, dense_mul_mm_kernel!,
            (out, weight, x, Int32(K), Int32(M), Int32(ntokens), Int32(mtiles)),
            mtiles * ntiles * 256; group=256, name)
    else
        rows = cld(M, DENSE_ROWS_PER_WG)
        _dispatch!(g, dense_gemv_batch_kernel!,
            (out, weight, x, Int32(K), Int32(M), Int32(rows),
             Val(s.model.caps.subgroup)),
            rows * ntokens * 256; group=256, name)
    end
    out
end

function _dense_coop_batch!(g, out, weight, xhalf,
                            K::Int, M::Int, ntokens::Int, name::String)
    Mantle.coopmat_gemm_dispatch!(g, out, weight, xhalf, M, ntokens, K;
                                  name, blk_split=(2, 1))
    out
end

function _recurrent_batch!(s::BonsaiSession, g, q, il::Int, xnorm, ntokens::Int)
    prefix = "blk.$il."
    _transform_batch!(s, g, q[:had5120], xnorm, WIDTH, ntokens;
                      halfcopy=q[:densehalf])
    _ptq!(s, g, q[:qkv], prefix * "attn_qkv.weight", q[:had5120], q)
    _ptq!(s, g, q[:z], prefix * "attn_gate.weight", q[:had5120], q)
    densecoop = ntokens % 16 == 0 && g.dev isa Mantle.LavaDevice &&
                Mantle.coopmat_gemm_available(g.dev.ctx) &&
                haskey(s.model.weights, prefix * "ssm_ab.weight.coop")
    if densecoop
        _dense_coop_batch!(g, q[:alphabeta],
            _w(s, prefix * "ssm_ab.weight.coop"), q[:densehalf],
            WIDTH, 96, ntokens, prefix * "ssm_alpha_beta_batch")
        _dispatch!(g, dense_ab_split_kernel!,
            (q[:alpha], q[:beta], q[:alphabeta], Int32(ntokens)),
            48 * ntokens; group=256, name=prefix * "ssm_alpha_beta_split")
    else
        _dense_batch!(s, g, q[:alpha], _w(s, prefix * "ssm_alpha.weight"),
                      xnorm, WIDTH, 48, ntokens, prefix * "ssm_alpha_batch")
        _dense_batch!(s, g, q[:beta], _w(s, prefix * "ssm_beta.weight"),
                      xnorm, WIDTH, 48, ntokens, prefix * "ssm_beta_batch")
    end
    convstate, state = s.recurrent[il]
    _dispatch!(g, depthwise_conv4_batch_kernel!,
        (q[:conv], convstate, q[:qkv], _w(s, prefix * "ssm_conv1d.weight"),
         Int32(10240), Int32(ntokens)), 10240;
        group=256, name=prefix * "ssm_conv1d_batch")
    _dispatch!(g, recurrent_split_batch_kernel!,
        (q[:rq], q[:rk], q[:rv], q[:conv]), 6144 * ntokens;
        group=256, name=prefix * "recurrent_split_batch")
    _dispatch!(g, l2normalize_qk_batch_kernel!,
        (q[:rq], q[:rk], s.model.eps), ntokens * 16 * 128;
        group=128, name=prefix * "gdn_normalize_qk")
    subgroup = s.model.caps.subgroup
    nsubgroups = 256 ÷ subgroup
    groupsperhead = cld(128, nsubgroups)
    _dispatch!(g, gated_delta_state_batch_kernel!,
        (q[:gdn], state, q[:rq], q[:rk], q[:rv], q[:alpha], q[:beta],
         _w(s, prefix * "ssm_dt.bias"), _w(s, prefix * "ssm_a"),
         Int32(ntokens), Int32(groupsperhead), Val(subgroup)),
        48 * groupsperhead * 256; group=256,
        name=prefix * "gated_delta_state_batch")
    _dispatch!(g, gdn_norm_gate_batch_kernel!,
        (q[:gdn], q[:z], _w(s, prefix * "ssm_norm.weight"), s.model.eps),
        ntokens * 48 * 128; group=128, name=prefix * "gdn_norm_gate_batch")
    _transform_batch!(s, g, q[:had6144], q[:gdn], 6144, ntokens; permute=true)
    _ptq!(s, g, q[:branch], prefix * "ssm_out.weight", q[:had6144], q)
    q[:branch]
end

function _attention_batch!(s::BonsaiSession, g, q, il::Int, xnorm,
                           ntokens::Int, position)
    prefix = "blk.$il."
    _transform_batch!(s, g, q[:had5120], xnorm, WIDTH, ntokens)
    _ptq!(s, g, q[:qfull], prefix * "attn_q.weight", q[:had5120], q)
    _ptq!(s, g, q[:kproj], prefix * "attn_k.weight", q[:had5120], q)
    _ptq!(s, g, q[:vproj], prefix * "attn_v.weight", q[:had5120], q)
    _dispatch!(g, prepare_q_batch_kernel!,
        (q[:aq], q[:qgate], q[:qfull], _w(s, prefix * "attn_q_norm.weight"),
         position, s.model.theta, s.model.eps), ntokens * 24 * 256;
        group=256, name=prefix * "prepare_q_batch")
    cache = s.kv[il]
    _dispatch!(g, prepare_kv_batch_kernel!,
        (cache.k, cache.v, cache.kscale, cache.vscale, q[:kproj], q[:vproj],
         _w(s, prefix * "attn_k_norm.weight"), position,
         s.model.theta, s.model.eps), ntokens * 4 * 256;
        group=256, name=prefix * "prepare_kv_batch")
    _dispatch!(g, decode_attention_batch_fast_kernel!,
        (q[:attn], q[:aq], q[:qgate], cache.k, cache.v,
         cache.kscale, cache.vscale, position, Val(s.model.caps.subgroup)),
        ntokens * 24 * 256;
        group=256, name=prefix * "decode_attention_batch")
    _transform_batch!(s, g, q[:had6144], q[:attn], 6144, ntokens)
    _ptq!(s, g, q[:branch], prefix * "attn_output.weight", q[:had6144], q)
    q[:branch]
end

function _ffn_batch!(s::BonsaiSession, g, q, il::Int, xnorm, ntokens::Int)
    prefix = "blk.$il."
    _transform_batch!(s, g, q[:had5120], xnorm, WIDTH, ntokens)
    _ptq!(s, g, q[:gate], prefix * "ffn_gate.weight", q[:had5120], q)
    _ptq!(s, g, q[:up], prefix * "ffn_up.weight", q[:had5120], q)
    _dispatch!(g, swiglu_hadamard1024_batch_kernel!,
        (q[:had17408], q[:gate], q[:up], s.model.signs[FFN],
         Int32(FFN), Int32(FFN ÷ 1024)),
        (FFN ÷ 1024) * ntokens * 256;
        group=256, name=prefix * "swiglu_hadamard_batch")
    _ptq!(s, g, q[:branch], prefix * "ffn_down.weight", q[:had17408], q)
    q[:branch]
end

function _declareprefill_stage!(s::BonsaiSession, g, q, tokens, position,
                                ntokens::Int, layers;
                                initialize::Bool=false, finalize::Bool=false)
    if initialize
        embedding = _w(s, "token_embd.weight")
        _dispatch!(g, DNNKernels.ptq1_getrows_kernel!,
            (q[:x], embedding.data, tokens, Int32(embedding.m), Int32(embedding.k),
             Int32(ntokens)), embedding.k * ntokens;
            group=256, name="token_embedding_batch")
        _hadamard_batch!(s, g, q[:x], WIDTH, ntokens; inverse=true)
        _dispatch!(g, rmsnorm_batch_kernel!,
            (q[:norm], q[:x], _w(s, "blk.0.attn_norm.weight"),
             Int32(WIDTH), s.model.eps), ntokens * 256;
            group=256, name="blk.0.attn_norm_batch")
    end
    for il in layers
        prefix = "blk.$il."
        branch = (il + 1) % 4 == 0 ?
            _attention_batch!(s, g, q, il, q[:norm], ntokens, position) :
            _recurrent_batch!(s, g, q, il, q[:norm], ntokens)
        _dispatch!(g, add_rmsnorm_batch_kernel!,
            (q[:norm], q[:x], branch,
             _w(s, prefix * "post_attention_norm.weight"),
             Int32(WIDTH), s.model.eps, Val(s.model.caps.subgroup)), ntokens * 256;
            group=256, name=prefix * "attn_residual_norm_batch")
        branch = _ffn_batch!(s, g, q, il, q[:norm], ntokens)
        nextnorm = il == NLAYERS - 1 ? "output_norm.weight" :
                   "blk.$(il + 1).attn_norm.weight"
        _dispatch!(g, add_rmsnorm_batch_kernel!,
            (q[:norm], q[:x], branch, _w(s, nextnorm),
             Int32(WIDTH), s.model.eps, Val(s.model.caps.subgroup)), ntokens * 256;
            group=256, name=prefix * "ffn_residual_norm_batch")
    end
    if finalize
        _dispatch!(g, select_last_kernel!,
            (q[:lastnorm], q[:norm], Int32(WIDTH), Int32(ntokens)), WIDTH;
            group=256, name="select_last")
        _dispatch!(g, copy_kernel!,
            (q[:lasthad], q[:lastnorm], Int32(WIDTH)), WIDTH;
            group=256, name="copy_last")
        _hadamard!(s, g, q[:lasthad], s.model.signs[WIDTH])
        _ptq!(s, g, q[:logits], "output.weight", q[:lasthad])
    end
    q[:logits]
end

const PREFILL_CHUNK = 512

"""
How many submissions one prefill is split into.

A recording this long has to be broken up — a single submission of all 64
layers outruns what the driver allows and is cancelled, which presents as a
device loss with nothing naming the cause. That is a property of the
SUBMISSION, and `record!` takes it as `maxpasses`.

It used to be a property of the GRAPH: sixteen graphs of four layers each. That
works for the submission length and costs everything else, because Mantle
derives liveness, placement, aliasing and barriers per graph — so nothing
crossing a cut could be a transient and the whole working set had to be
permanent. Sixteen is kept here so the submissions stay the size they were.
"""
const PREFILL_SUBMISSIONS = 16

function _recordprefill(s::BonsaiSession, ntokens::Int; profile::Bool=false)
    dev = Mantle.Device(s.model.backend)
    tokens = Mantle.Buffer(dev, zeros(Int32, ntokens))
    position = Mantle.GPURef(dev, Int32(0))
    # ONE graph, all 64 layers, so the placer sees the whole working set.
    g = Mantle.Graph(dev)
    q = _batchscratch(g, s.logits, ntokens)
    _declareprefill_stage!(s, g, q, tokens, position, ntokens, 0:NLAYERS-1;
                           initialize=true, finalize=true)
    plan = Mantle.Plan(g; profile)
    Mantle.record!(plan; maxpasses = cld(length(plan.passes), PREFILL_SUBMISSIONS))
    PrefillExec(tokens, position, q, Any[plan])
end

function _declarestep!(s::BonsaiSession, g, q)
    embedding = _w(s, "token_embd.weight")
    _dispatch!(g, DNNKernels.ptq1_getrows_kernel!,
        (q[:x], embedding.data, s.tokenref, Int32(embedding.m), Int32(embedding.k), Int32(1)),
        embedding.k; group=256, name="token_embedding")
    _hadamard!(s, g, q[:x], s.model.signs[WIDTH]; inverse=true)
    for il in 0:NLAYERS-1
        prefix = "blk.$il."
        _dispatch!(g, rmsnorm_kernel!,
            (q[:norm], q[:x], _w(s, prefix * "attn_norm.weight"), Int32(WIDTH), s.model.eps),
            256; group=256, name=prefix * "attn_norm")
        branch = (il + 1) % 4 == 0 ? _attention!(s, g, q, il, q[:norm]) :
                                     _recurrent!(s, g, q, il, q[:norm])
        _dispatch!(g, add_kernel!, (q[:x], q[:x], branch, Int32(WIDTH)), WIDTH;
                   group=256, name=prefix * "attn_residual")
        _dispatch!(g, rmsnorm_kernel!,
            (q[:norm], q[:x], _w(s, prefix * "post_attention_norm.weight"),
             Int32(WIDTH), s.model.eps),
            256; group=256, name=prefix * "post_attention_norm")
        branch = _ffn!(s, g, q, il, q[:norm])
        _dispatch!(g, add_kernel!, (q[:x], q[:x], branch, Int32(WIDTH)), WIDTH;
                   group=256, name=prefix * "ffn_residual")
    end
    _dispatch!(g, rmsnorm_kernel!,
        (q[:norm], q[:x], _w(s, "output_norm.weight"), Int32(WIDTH), s.model.eps),
        256; group=256, name="output_norm")
    _transform!(s, g, q[:had5120], q[:norm])
    _ptq!(s, g, q[:logits], "output.weight", q[:had5120])
    q[:logits]
end

function _recordstep(s::BonsaiSession, dev)
    g = Mantle.Graph(dev)
    q = _stepscratch(g, s.logits)
    _declarestep!(s, g, q)
    (Mantle.record!(Mantle.Plan(g)), q)
end

"""
    step!(session, token) -> device logits

Advance the recorded 64-layer decoder by one zero-based token id. The returned
248320-element logits remain on the device.
"""
function step!(s::BonsaiSession, token::Integer)
    s.position < s.model.maxcontext || error("session context is full")
    0 <= token < length(s.model.tokenizer.tokentostr) || throw(BoundsError(s.model.tokenizer.tokentostr, token + 1))
    _ensure_kv!(s, s.position + 1)
    s.tokenref[] = Int32(token)
    s.positionref[] = Int32(s.position)
    Mantle.run!(s.plan)
    s.position += 1
    s.logits
end

"""
    prefill!(session, ids; chunk=512) -> device logits

Advance `session` through a prompt with recorded chunk graphs. A graph is built
once for each encountered chunk length and reused by later calls. Recurrent
state is advanced in sequence inside each chunk; projections and attention are
batched across its tokens. The returned logits belong to the final input token.
"""
function prefill!(s::BonsaiSession, ids::AbstractVector{<:Integer};
                  chunk::Integer=PREFILL_CHUNK)
    isempty(ids) && throw(ArgumentError("prompt must contain at least one token"))
    chunk > 0 || throw(ArgumentError("chunk must be positive"))
    chunk <= PREFILL_CHUNK || throw(ArgumentError(
        "chunk must be at most $PREFILL_CHUNK"))
    s.position + length(ids) <= s.model.maxcontext || error("session context is full")
    vocab = length(s.model.tokenizer.tokentostr)
    all(id -> 0 <= id < vocab, ids) || throw(BoundsError(s.model.tokenizer.tokentostr))
    logits = nothing
    first = 1
    while first <= length(ids)
        n = min(Int(chunk), length(ids) - first + 1)
        _ensure_kv!(s, s.position + n)
        exec = get!(s.prefills, n) do
            _recordprefill(s, n)
        end
        exec.tokens[:] = Int32.(view(ids, first:first+n-1))
        exec.position[] = Int32(s.position)
        foreach(Mantle.run!, exec.plans)
        s.position += n
        logits = exec.scratch[:logits]
        first += n
    end
    logits
end

function _greedy(s::BonsaiSession, logits)
    KernelAbstractions.synchronize(s.model.backend)
    argmax(Array(logits)) - 1
end

"""Greedy token generation with chunked prompt prefill."""
function generate(s::BonsaiSession, ids::AbstractVector{<:Integer}; max_tokens::Integer=32,
                  stop=(s.model.tokenizer.eos,), prefill_chunk::Integer=PREFILL_CHUNK)
    isempty(ids) && throw(ArgumentError("prompt must contain at least one token"))
    logits = prefill!(s, ids; chunk=prefill_chunk)
    out = Int[]
    for i in 1:max_tokens
        id = _greedy(s, logits)
        push!(out, id)
        (id in stop || i == max_tokens) && break
        logits = step!(s, id)
    end
    out
end

function generate(s::BonsaiSession, prompt::AbstractString; max_tokens::Integer=32, kwargs...)
    ids = encode(s.model.tokenizer, prompt)
    generated = generate(s, ids; max_tokens, kwargs...)
    decode(s.model.tokenizer, generated)
end

end # module
