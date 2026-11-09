"""
Qwen3-Embedding 0.6B: a text to one unit vector, so that texts meaning the same
thing land close together. What semantic search ranks by. 1024 dimensions.

A Qwen3 decoder read at its final norm, at the last token: the text is followed
by `<|endoftext|>` and that token's hidden state, L2-normalized, is the
embedding. Queries carry an instruction ([`embed_query`](@ref)), documents do
not ([`embed`](@ref)).

**Buckets.** The graph's length is its `t` symbol, 2 to 512, and a plan is
compiled per length, so a text is right-padded to the next of [`BUCKETS`](@ref).
Attention is causal: no token reads anything to its right, so the padding cannot
change the kept row, which is the one at the text's own last token.

**The embedding table stays on the host**, as in Qwen-Image's text encoder: it is
151669x1024 and a text reads a few hundred rows of it.

The tokenizer is `QwenImageRunner`'s: Qwen3 and Qwen3-VL share the byte-level
BPE and its special tokens; only the files differ, and the export writes them in
the layout that tokenizer reads.

Upstream: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
License: **Apache-2.0**

See `tools/export_qwen3embedding.py` for the export that feeds it.
"""
module Qwen3EmbeddingRunner

import DNNKernels, Mantle
using DNNKernels: Model, loadgraph, readsafetensors, toback
using QwenImageRunner: QwenTokenizer, encode
using LinearAlgebra: normalize!
import JSON3
using LazyArtifacts

export qwen3embedding, embed, embed_query, embed_long, Qwen3Embedding

const GRAPH = "qwen3embedding"

"""The `<|endoftext|>` token that ends every text; its hidden state is the embedding."""
const ENDOFTEXT = 151643

"""The lengths a text is padded to; one compiled plan each."""
const BUCKETS = (32, 64, 128, 256, 512)

"""
    assetdir() -> String

The graph, weights, embedding table and tokenizer in the `qwen3embedding`
artifact, downloaded on first use and cached across environments on this machine.
Re-exported assets take effect after `tools/make_artifacts.jl qwen3embedding`
rebinds `Artifacts.toml`.
"""
assetdir() = artifact"qwen3embedding"

"""The separately downloaded fp32 reference embeddings and token ids used by the tests."""
refsdir() = artifact"qwen3embedding-refs"

"""Whether an export is installed."""
ready(dir::AbstractString = assetdir()) =
    all(f -> isfile(joinpath(dir, f)), ("qwen3embedding.json", "weights.safetensors",
                                        "embed_tokens.safetensors", "export.json", "vocab.json",
                                        "merges.txt", "added_tokens.json"))

"""A loaded model: the graph on the device, the host-side token table and the tokenizer."""
struct Qwen3Embedding{B,M,E}
    backend::B
    model::M
    table::E                 # (1024, vocab) Float16, memory-mapped
    tokenizer::QwenTokenizer
    maxtokens::Int
    task::String             # the instruction queries are embedded with
end

"""
    qwen3embedding(; backend = Mantle.defaultbackend(), dir = assetdir()) -> Qwen3Embedding

Load the exported model.
"""
function qwen3embedding(; backend = Mantle.defaultbackend(), dir::AbstractString = assetdir())
    ready(dir) || throw(ArgumentError("Qwen3-Embedding artifact at $dir is incomplete; reinstall the qwen3embedding artifact"))
    graph = loadgraph(joinpath(dir, "qwen3embedding.json"))
    model = Model(Dict(GRAPH => graph), readsafetensors(joinpath(dir, "weights.safetensors")); backend)
    table = readsafetensors(joinpath(dir, "embed_tokens.safetensors"); mmap = true)["embed_tokens"]
    meta = JSON3.read(read(joinpath(dir, "export.json"), String))
    Int(meta.endoftext) == ENDOFTEXT || throw(ArgumentError("export ends texts with $(meta.endoftext), not $ENDOFTEXT"))
    return Qwen3Embedding(model.backend, model, table, QwenTokenizer(dir), Int(meta.max_tokens), String(meta.task))
end

"""The token ids of `text` as the model reads it: truncated to fit, then `<|endoftext|>`."""
function tokens(m::Qwen3Embedding, text::AbstractString)
    ids = encode(m.tokenizer, text)
    return vcat(ids[1:min(end, m.maxtokens - 1)], ENDOFTEXT)
end

"""The bucket an `n`-token input runs at: the first of `BUCKETS` it fits, else `maxtokens`."""
function bucket(n::Integer, maxtokens::Integer)
    for b in BUCKETS
        n <= b <= maxtokens && return b
    end
    return maxtokens
end

"""
    embed(m, text) -> Vector{Float32}

The unit-length embedding of a document.
"""
function embed(m::Qwen3Embedding, text::AbstractString)
    return embed_ids(m, tokens(m, text))
end

function embed_ids(m::Qwen3Embedding, ids::AbstractVector{<:Integer})
    t = bucket(length(ids), m.maxtokens)
    padded = vcat(ids, fill(ENDOFTEXT, t - length(ids)))
    x = m.table[:, padded .+ 1]
    hidden = first(DNNKernels.call(m.model, GRAPH, toback(m.backend, reshape(x, size(x, 1), t, 1)); dims = (; t)))
    # Features for one token are contiguous. Download just that token, rather
    # than every hidden state (up to 512 times as much data).
    width = size(hidden, 1)
    row = Vector{eltype(hidden)}(undef, width)
    copyto!(row, 1, hidden, (length(ids) - 1) * width + 1, width)
    return normalize!(Float32.(row))
end

"""
    embed_long(m, text; overlap = 64) -> Vector{Float32}

Embed every token of a long document using overlapping model-sized windows.
Pool normalized window vectors, weighted by newly covered tokens, then normalize.
Short documents use exactly the ordinary embedding path. This is chunk pooling,
not a longer-context forward pass.
"""
function embed_long(m::Qwen3Embedding, text::AbstractString; overlap::Integer = 64)
    capacity = m.maxtokens - 1
    0 <= overlap < capacity || throw(ArgumentError("overlap must be between zero and $(capacity - 1)"))
    ids = encode(m.tokenizer, text)
    length(ids) <= capacity && return embed_ids(m, vcat(ids, ENDOFTEXT))
    pooled = zeros(Float32, size(m.table, 1))
    start, covered = 1, 0
    while start <= length(ids)
        stop = min(start + capacity - 1, length(ids))
        vector = embed_ids(m, vcat(ids[start:stop], ENDOFTEXT))
        pooled .+= (stop - covered) .* vector
        covered = stop
        stop == length(ids) && break
        start += capacity - overlap
    end
    return normalize!(pooled)
end

"""
    embed_query(m, query) -> Vector{Float32}

The unit-length embedding of a search query, under the export's retrieval instruction.
"""
embed_query(m::Qwen3Embedding, query::AbstractString) = embed(m, "Instruct: $(m.task)\nQuery:$(query)")

end
