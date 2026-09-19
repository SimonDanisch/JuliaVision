"""
    BonsaiRunner

Text-only inference for Prism's PTQ1 build of Ternary Bonsai 2 27B. Weights
remain packed ternary on the device. The initial runtime is batch-one decode;
prompt ingestion deliberately uses the same token step until a chunked prefill
kernel is added.
"""
module BonsaiRunner

using DNNKernels
using KernelAbstractions
import Mantle

export Bonsai2, BonsaiSession, BonsaiTokenizer, session, step!, generate
export encode, decode, chatprompt, loadweights!

include("tokenizer.jl")
include("kernels.jl")

const WIDTH = 5120
const FFN = 17408
const NLAYERS = 64
const EPS = 1f-6

"""A loaded Bonsai checkpoint and the device selected for it."""
struct Bonsai2{B,C}
    backend::B
    ctx::C
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

function _upload(backend, host::AbstractArray)
    out = KernelAbstractions.allocate(backend, Float32, size(host)...)
    copyto!(out, Float32.(host))
    out
end

"""
    Bonsai2(path; device=Mantle.device(), context=8192, loadweights=true)

Open a PTQ1 Bonsai GGUF on `device`. Metadata parsing is immediate; packed and
dense tensors are uploaded when `loadweights` is true. `loadweights=false` is
useful for inspecting/tokenizing a checkpoint without allocating model memory.
"""
function Bonsai2(path::AbstractString; device=Mantle.device(), context::Integer=8192,
                 loadweights::Bool=true, progress::Bool=true)
    file = readgguf(path)
    md = file.metadata
    get(md, "general.architecture", "") == "qwen35" || error("checkpoint is not qwen35")
    Int(md["qwen35.block_count"]) == NLAYERS || error("BonsaiRunner expects 64 blocks")
    Int(md["qwen35.embedding_length"]) == WIDTH || error("BonsaiRunner expects width 5120")
    Int(md["qwen35.full_attention_interval"]) == 4 || error("unexpected full-attention interval")
    Int(md["prism.hadamard.version"]) == 1 || error("unsupported Hadamard metadata")
    Int(md["prism.hadamard.block_size"]) == 1024 || error("unsupported Hadamard block size")
    context > 0 || throw(ArgumentError("context must be positive"))
    context <= Int(md["qwen35.context_length"]) || throw(ArgumentError("context exceeds checkpoint limit"))

    backend = Mantle.backend(device)
    ctx = DNNKernels.Ctx(backend)
    weights = Dict{String,Any}()
    model = Bonsai2(backend, ctx, file, weights, Dict{Int,Any}(),
        Set(String.(md["prism.hadamard.weight_names"])),
        Set(String.(get(md, "prism.hadamard.inverse_weight_names", String[]))),
        BonsaiTokenizer(file), Int(context), Float32(md["qwen35.rope.freq_base"]),
        Float32(md["qwen35.attention.layer_norm_rms_epsilon"]))
    _loadsigns!(model)
    loadweights && loadweights!(model; progress)
    model
end

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
            model.weights[name] = _upload(model.backend, DNNKernels.gguffloat(model.file, t))
        else
            error("unsupported GGML tensor type $(t.typeid) for $name")
        end
        progress && (i == 1 || i % 32 == 0 || i == length(names)) &&
            println("BonsaiRunner: loaded $i/$(length(names)) tensors")
    end
    KernelAbstractions.synchronize(model.backend)
    model
end

mutable struct BonsaiSession{M}
    model::M
    position::Int
    recurrent::Dict{Int,Tuple{Any,Any}}
    kv::Dict{Int,Tuple{Any,Any}}
    scratch::Dict{Symbol,Any}
    tokenref::Any
    positionref::Any
    plan::Any
end

_zeros(m::Bonsai2, ::Type{T}, dims...) where {T} = KernelAbstractions.zeros(m.backend, T, dims...)

"""Allocate mutable recurrent and KV state for one batch-one sequence."""
function session(model::Bonsai2)
    isempty(model.weights) && error("weights are not loaded")
    recurrent = Dict{Int,Tuple{Any,Any}}()
    kv = Dict{Int,Tuple{Any,Any}}()
    for il in 0:NLAYERS-1
        if (il + 1) % 4 == 0
            kv[il] = (_zeros(model, Float16, 256, 4, model.maxcontext),
                      _zeros(model, Float16, 256, 4, model.maxcontext))
        else
            recurrent[il] = (_zeros(model, Float32, 3, 10240),
                              _zeros(model, Float32, 128, 128, 48))
        end
    end
    sizes = Dict(
        :x=>WIDTH, :norm=>WIDTH, :branch=>WIDTH, :had5120=>WIDTH,
        :gate=>FFN, :up=>FFN, :ff=>FFN, :had17408=>FFN,
        :qkv=>10240, :z=>6144, :conv=>10240, :rq=>2048, :rk=>2048,
        :rv=>6144, :gdn=>6144, :had6144=>6144,
        :qfull=>12288, :kproj=>1024, :vproj=>1024, :aq=>6144,
        :qgate=>6144, :attn=>6144, :alpha=>48, :beta=>48,
        :logits=>248320)
    scratch = Dict{Symbol,Any}(k => _zeros(model, Float32, n) for (k,n) in sizes)
    dev = Mantle.Device(model.backend)
    s = BonsaiSession(model, 0, recurrent, kv, scratch,
                      Mantle.GPURef(dev, Int32(0)),
                      Mantle.GPURef(dev, Int32(0)), nothing)
    s.plan = _recordstep(s, dev)
    s
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

function _ptq!(s::BonsaiSession, g, out, name::String, transformed)
    name in s.model.folded || error("$name is PTQ1 but has no Hadamard declaration")
    A = _w(s, name)
    M, K = size(A)
    N = length(transformed) ÷ K
    rows = cld(M, DNNKernels.PTQ1_ROWS_PER_WG)
    _dispatch!(g, DNNKernels.ptq1_mul_kernel!,
        (out, A.data, transformed, A.data, Int32(M), Int32(K), Int32(N),
         Int32(rows), Val(false), Val(s.model.ctx.dev.subgroup)),
        rows * N * DNNKernels.PTQ1_WG;
        group=DNNKernels.PTQ1_WG, name=name)
    out
end

function _recurrent!(s::BonsaiSession, g, il::Int, xnorm)
    q = s.scratch
    prefix = "blk.$il."
    _transform!(s, g, q[:had5120], xnorm)
    _ptq!(s, g, q[:qkv], prefix * "attn_qkv.weight", q[:had5120])
    _ptq!(s, g, q[:z], prefix * "attn_gate.weight", q[:had5120])
    _dispatch!(g, dense_gemv_kernel!,
        (q[:alpha], _w(s, prefix * "ssm_alpha.weight"), xnorm,
         Int32(length(xnorm)), Int32(48), Val(s.model.ctx.dev.subgroup)),
        cld(48, DENSE_ROWS_PER_WG) * 256;
        group=256, name=prefix * "ssm_alpha")
    _dispatch!(g, dense_gemv_kernel!,
        (q[:beta], _w(s, prefix * "ssm_beta.weight"), xnorm,
         Int32(length(xnorm)), Int32(48), Val(s.model.ctx.dev.subgroup)),
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

function _attention!(s::BonsaiSession, g, il::Int, xnorm)
    q = s.scratch
    prefix = "blk.$il."
    _transform!(s, g, q[:had5120], xnorm)
    _ptq!(s, g, q[:qfull], prefix * "attn_q.weight", q[:had5120])
    _ptq!(s, g, q[:kproj], prefix * "attn_k.weight", q[:had5120])
    _ptq!(s, g, q[:vproj], prefix * "attn_v.weight", q[:had5120])
    _dispatch!(g, prepare_q_kernel!,
        (q[:aq], q[:qgate], q[:qfull], _w(s, prefix * "attn_q_norm.weight"),
         s.positionref, s.model.theta, s.model.eps), 24 * 256;
        group=256, name=prefix * "prepare_q")
    kc, vc = s.kv[il]
    _dispatch!(g, prepare_kv_kernel!,
        (kc, vc, q[:kproj], q[:vproj], _w(s, prefix * "attn_k_norm.weight"),
         s.positionref, s.model.theta, s.model.eps), 4 * 256;
        group=256, name=prefix * "prepare_kv")
    _dispatch!(g, decode_attention_kernel!,
        (q[:attn], q[:aq], q[:qgate], kc, vc, s.positionref), 24 * 256;
        group=256, name=prefix * "decode_attention")
    _transform!(s, g, q[:had6144], q[:attn])
    _ptq!(s, g, q[:branch], prefix * "attn_output.weight", q[:had6144])
    q[:branch]
end

function _ffn!(s::BonsaiSession, g, il::Int, xnorm)
    q = s.scratch; prefix = "blk.$il."
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

function _declarestep!(s::BonsaiSession, g)
    q = s.scratch
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
        branch = (il + 1) % 4 == 0 ? _attention!(s, g, il, q[:norm]) :
                                     _recurrent!(s, g, il, q[:norm])
        _dispatch!(g, add_kernel!, (q[:x], q[:x], branch, Int32(WIDTH)), WIDTH;
                   group=256, name=prefix * "attn_residual")
        _dispatch!(g, rmsnorm_kernel!,
            (q[:norm], q[:x], _w(s, prefix * "post_attention_norm.weight"),
             Int32(WIDTH), s.model.eps),
            256; group=256, name=prefix * "post_attention_norm")
        branch = _ffn!(s, g, il, q[:norm])
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
    _declarestep!(s, g)
    Mantle.record!(Mantle.Plan(g))
end

"""
    step!(session, token) -> device logits

Advance the recorded 64-layer decoder by one zero-based token id. The returned
248320-element logits remain on the device.
"""
function step!(s::BonsaiSession, token::Integer)
    s.position < s.model.maxcontext || error("session context is full")
    0 <= token < length(s.model.tokenizer.tokentostr) || throw(BoundsError(s.model.tokenizer.tokentostr, token + 1))
    s.tokenref[] = Int32(token)
    s.positionref[] = Int32(s.position)
    Mantle.run!(s.plan)
    s.position += 1
    s.scratch[:logits]
end

function _greedy(s::BonsaiSession, logits)
    KernelAbstractions.synchronize(s.model.backend)
    argmax(Array(logits)) - 1
end

"""Greedy token generation. Prompt prefill currently advances one token at a time."""
function generate(s::BonsaiSession, ids::AbstractVector{<:Integer}; max_tokens::Integer=32,
                  stop=(s.model.tokenizer.eos,))
    isempty(ids) && throw(ArgumentError("prompt must contain at least one token"))
    logits = nothing
    for id in ids
        logits = step!(s, id)
    end
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
