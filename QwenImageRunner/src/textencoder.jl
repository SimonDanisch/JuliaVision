"""
Qwen-Image 2.1's conditioner: the Qwen3-VL-8B language model, run over the
pipeline's own prompt template, read at the last decoder layer.

Three things this file owns, none of which are the transformer's business:

**The template and the drop index.** The checkpoint was trained on a literal
chat string, not on `apply_chat_template`'s rendering of one, and the leading
system turn is dropped from the hidden states afterwards. Both come from
`QwenImage21Pipeline._get_qwen_prompt_embeds`.

**Right padding.** The exported graph is static at `prompt_tokens`, and the
encoder is causal: position `i` reads nothing after it, so tokens appended past
the prompt cannot change the rows we keep. One 64-token graph therefore serves
every prompt up to that length, bit for bit. (The *denoiser* has no such freedom
— its text stream is part of a joint attention — so its context length is bound
to this prompt's embedding count at export.)

**The embedding table stays on the host.** It is 151936x4096 INT8 with a
per-row scale, a fifth of the checkpoint, and a prompt reads forty rows of it.
Those rows are gathered and un-rotated here; the graph takes embeddings.
"""

const QWEN_SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."

"""The system turn, which is also exactly what is dropped from the output."""
qwen_system_prefix() = "<|im_start|>system\n$(QWEN_SYSTEM_PROMPT)<|im_end|>\n"

"""The text-to-image prompt string the 2.1 checkpoint was trained on."""
qwen_prompt_template(prompt::AbstractString) =
    qwen_system_prefix() * "<|im_start|>user\n$(prompt)<|im_end|>\n<|im_start|>assistant\n"

"""
    convrot!(x, group_size) -> x

The radix-4 regular-Hadamard transform Comfy's ConvRot quantization uses, over
`group_size`-wide blocks of the first axis, in place.

It is its own inverse: each stage applies a symmetric orthogonal 4x4 matrix, so
the composition squares to the identity. That is what makes an *embedding* row
recoverable — the table is stored rotated, exactly like a linear layer's weight,
but nothing rotates the activation on the way into a lookup.

The same arithmetic as `DNNKernels.convrot_stage_kernel!`, on the host, because
forty rows of a table that never reaches the device is not a dispatch.
"""
function convrot!(x::AbstractMatrix{Float32}, group_size::Integer=256)
    K = size(x, 1)
    K % group_size == 0 || throw(DimensionMismatch(
        "ConvRot group size $group_size does not divide $K"))
    stages = round(Int, log(4, group_size))
    4^stages == group_size || throw(ArgumentError("ConvRot group size must be a power of four"))
    src = x
    dst = similar(x)
    for stage in 0:stages-1
        stride = 4^stage
        period = 4stride
        @inbounds for col in axes(src, 2), k in 0:K-1
            base = (k ÷ period) * period + (k % stride) + 1
            a = src[base, col]
            b = src[base + stride, col]
            c = src[base + 2stride, col]
            d = src[base + 3stride, col]
            row = (k ÷ stride) % 4
            v = row == 0 ? a + b + c - d :
                row == 1 ? a + b - c + d :
                row == 2 ? a - b + c + d : -a + b + c + d
            dst[k + 1, col] = v * 0.5f0
        end
        src, dst = dst, src
    end
    src === x ? x : copyto!(x, src)
end

"""
    token_embeddings(ids; compact) -> Matrix{Float16}

The `(hidden, tokens)` input embeddings for `ids` (0-based token ids), read from
the compact checkpoint's INT8 table and rotated back out of the ConvRot basis.
"""
function token_embeddings(ids::AbstractVector{<:Integer};
        compact::AbstractDict=compact_encoder(),
        group_size::Integer=256)
    q = compact["model.embed_tokens.weight"]            # (hidden, vocab), Int8
    scale = compact["model.embed_tokens.weight_scale"]  # (1, vocab), Float32
    hidden, vocab = size(q)
    out = Matrix{Float32}(undef, hidden, length(ids))
    for (j, id) in enumerate(ids)
        0 <= id < vocab || throw(ArgumentError("token id $id is outside the vocabulary"))
        s = Float32(scale[1, id + 1])
        @inbounds for i in 1:hidden
            out[i, j] = Float32(q[i, id + 1]) * s
        end
    end
    Float16.(convrot!(out, group_size))
end

"""
    compact_text_encoder_weights(graph; compact, constants_dir)

Map Comfy-Org's W4A8 Qwen3-VL encoder into the exported graph. The quantized
matrices stay packed until DNNKernels decodes them onto the backend; the norms
are BF16 in the file and Float16 in the graph; the rotary tables come from the
export's own constants.

`compact` is the checkpoint as a dictionary rather than a directory to read it
from, because it arrives as five artifacts that `compact_encoder` merges.
"""
function compact_text_encoder_weights(graph;
        compact::AbstractDict=compact_encoder(),
        constants_dir::AbstractString=assetdir())
    constants_path = joinpath(constants_dir, "text_encoder_constants.safetensors")
    isfile(constants_path) || throw(ArgumentError(
        "text encoder graph constants not found at $constants_path"))
    constants = readsafetensors(constants_path; mmap=false)
    out = Dict{String,Any}()

    for buffer in values(graph.buffers)
        buffer.kind === :weight || continue
        key = buffer.key
        (isempty(key) || haskey(out, key)) && continue
        if haskey(constants, key)
            value = constants[key]
            out[key] = value isa BitArray ? _densebool(value) : value
            continue
        end
        haskey(compact, key) || throw(KeyError(key))
        value = compact[key]
        rel_key = replace(key, r"\.weight$" => ".weight_s_rel")
        channel_key = replace(key, r"\.weight$" => ".weight_s_channel")
        book_key = replace(key, r"\.weight$" => ".weight_codebook")
        if eltype(value) === Int8 && haskey(compact, rel_key)
            out[key] = DNNKernels.W4A8ConvRotHostMatrix(
                value, compact[rel_key], compact[channel_key], compact[book_key], 16, 256)
        elseif eltype(value) === UInt16
            out[key] = _bf16array(value)
        else
            out[key] = value
        end
    end
    out
end

"""The encoder graph's name, which is also its key in the `Model`."""
const ENCODERGRAPH = "qwenimage21_text_encoder"

"""A prepared Qwen3-VL text encoder: its `Model`, tokenizer and lengths."""
struct QwenTextEncoder{B,M,C}
    backend::B
    model::M
    tokenizer::QwenTokenizer
    tokens::Int
    drop::Int
    # The checkpoint is kept, not the path it came from: `encode_prompt` reads
    # the embedding table out of it once per prompt, and re-opening five
    # memory-mapped shards for two tensors is work for nothing. Holding it
    # resident costs nothing either — `readsafetensors` maps, so this is five
    # mappings and no pages.
    compact::C
end

"""
    qwenimagetextencoder(; backend, dir=assetdir(), compact, processor_dir)

Load and prepare the exported Qwen3-VL conditioner.

The encoder decodes to ~6.9 GB of INT8 on the device and the denoiser is
another 7.26 GB, which do not fit at once on an 8060S. Encode first, drop this
object, then build the denoiser: the prompt is encoded once per image while the
denoiser runs every step.
"""
function qwenimagetextencoder(; backend=Mantle.defaultbackend(),
                              dir::AbstractString=assetdir(),
                              compact::AbstractDict=compact_encoder(),
                              processor_dir::AbstractString=processordir(),
                              maxpasses::Integer=64)
    graph = qwenimagegraph(:text_encoder; dir)
    weights = compact_text_encoder_weights(graph; compact, constants_dir=dir)
    model = Model(Dict(ENCODERGRAPH => graph), weights; backend,
                  record_maxpasses = Dict(ENCODERGRAPH => Int(maxpasses)))
    prepared = model.graphs[ENCODERGRAPH]
    # `planahead!` and not a lazy first call: this is built once and the prompt
    # is encoded once per image, so the compiling belongs here rather than
    # inside the first encode. It builds the plan `call` would build, under the
    # key `call` looks up. The graph is not symbolic — `dims = (;)` — so there
    # is exactly one.
    DNNKernels.planahead!(model, ENCODERGRAPH)
    tokenizer = QwenTokenizer(processor_dir)
    tokens = Int(prepared.buffers[only(prepared.inputs)].shape[2])
    drop = length(encode(tokenizer, qwen_system_prefix()))
    QwenTextEncoder(model.backend, model, tokenizer, tokens, drop, compact)
end

"""
    encode_prompt(encoder, prompt) -> (hidden, tokens)

The prompt embeddings Qwen-Image 2.1's denoiser reads: the last decoder layer's
output over the prompt's own tokens, with the system turn dropped. The returned
device array has shape `(4096, prompt_tokens, 1)`, and its token count is what
the denoiser must have been exported with.
"""
function encode_prompt(encoder::QwenTextEncoder, prompt::AbstractString;
                       pad_token::Integer=151643)
    ids = encode(encoder.tokenizer, qwen_prompt_template(prompt))
    length(ids) <= encoder.tokens || throw(ArgumentError(
        "prompt is $(length(ids)) tokens and the exported encoder holds " *
        "$(encoder.tokens); re-export with " *
        "`--component text_encoder --prompt-tokens $(length(ids))`"))
    padded = vcat(ids, fill(Int(pad_token), encoder.tokens - length(ids)))
    embeddings = token_embeddings(padded; compact=encoder.compact)
    input = DNNKernels.toback(encoder.backend, reshape(embeddings, size(embeddings, 1),
                                                       size(embeddings, 2), 1))
    hidden = first(DNNKernels.call(encoder.model, ENCODERGRAPH, input; dims = (;)))
    # Causal attention: everything past `ids` is padding that no kept row read.
    kept = (encoder.drop + 1):length(ids)
    host = Array(hidden)
    reshape(host[:, kept], size(host, 1), length(kept), 1)
end
