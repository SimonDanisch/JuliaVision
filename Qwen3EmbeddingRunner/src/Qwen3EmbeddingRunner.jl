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

export qwen3embedding, embed, embed_query, Qwen3Embedding

const GRAPH = "qwen3embedding"

"""The `<|endoftext|>` token that ends every text; its hidden state is the embedding."""
const ENDOFTEXT = 151643

"""The lengths a text is padded to; one compiled plan each."""
const BUCKETS = (32, 64, 128, 256, 512)

"""
    assetdir() -> String

The export, at the workspace's `gen/graphs/qwen3embedding` (the `gen` symlink
next to this repository). Not an artifact yet: `uv run tools/export_qwen3embedding.py`
writes it, and `tools/make_artifacts.jl qwen3embedding` would bind it.
"""
assetdir() = normpath(joinpath(realpath(@__DIR__), "..", "..", "gen", "graphs", "qwen3embedding"))

"""Whether an export is installed."""
ready(dir::AbstractString = assetdir()) =
    all(f -> isfile(joinpath(dir, f)), ("qwen3embedding.json", "weights.safetensors",
                                        "embed_tokens.safetensors", "export.json", "vocab.json"))

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
    ready(dir) || throw(ArgumentError(
        "Qwen3-Embedding export not found at $dir. Run `uv run tools/models.py fetch qwen3embedding` " *
        "and `uv run tools/export_qwen3embedding.py`."))
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
    ids = tokens(m, text)
    t = bucket(length(ids), m.maxtokens)
    padded = vcat(ids, fill(ENDOFTEXT, t - length(ids)))
    x = m.table[:, padded .+ 1]
    hidden = first(DNNKernels.call(m.model, GRAPH, toback(m.backend, reshape(x, size(x, 1), t, 1)); dims = (; t)))
    row = Array(hidden)[:, length(ids), 1]
    return normalize!(Float32.(row))
end

"""
    embed_query(m, query) -> Vector{Float32}

The unit-length embedding of a search query, under the export's retrieval instruction.
"""
embed_query(m::Qwen3Embedding, query::AbstractString) = embed(m, "Instruct: $(m.task)\nQuery:$(query)")

end
