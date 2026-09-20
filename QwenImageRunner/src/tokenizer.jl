"""
Qwen's byte-level BPE, read from the processor directory the checkpoint ships:
`vocab.json`, `merges.txt` and `added_tokens.json`.

The same scheme as GPT-2 — NFC, a split regex, the 256-byte printable map, then
merges by rank — with Qwen's own vocabulary and its 26 special tokens. Those are
matched before anything else: `<|im_start|>` is one token and must never be
split into the pieces its characters would make.

This is the text half of `Qwen3VLProcessor`. The image half is not here: the
condition-image path of Qwen-Image 2.1 needs the vision tower as well, and
neither is ported yet.
"""

# `Base.Unicode`, not the `Unicode` stdlib: the same `normalize`, without a
# dependency entry for one call.
const normalize = Base.Unicode.normalize

"""Byte-level BPE tables for the Qwen tokenizer that ships with a checkpoint."""
struct QwenTokenizer
    tokentostr::Dict{Int,String}
    strtotoken::Dict{String,Int}
    ranks::Dict{Tuple{String,String},Int}
    special::Vector{Pair{String,Int}}
    byteencoder::Vector{Char}
    bytedecoder::Dict{Char,UInt8}
end

# GPT-2's printable byte map: the 188 bytes that are already printable keep
# their character, the rest move into the 256..288 range, so a token is always a
# valid string and whitespace is visible in the vocabulary.
function _bytemap()
    bs = vcat(collect(UInt8('!'):UInt8('~')), collect(UInt8('¡'):UInt8('¬')),
              collect(UInt8('®'):UInt8('ÿ')))
    cs = Int.(bs)
    n = 0
    for b in 0:255
        if !(UInt8(b) in bs)
            push!(bs, UInt8(b)); push!(cs, 256 + n); n += 1
        end
    end
    enc = Vector{Char}(undef, 256)
    dec = Dict{Char,UInt8}()
    for (b, c) in zip(bs, cs)
        enc[Int(b) + 1] = Char(c)
        dec[Char(c)] = b
    end
    enc, dec
end

# Qwen's pre-tokenizer, copied from `tokenizer.json`'s `pre_tokenizer.Split`
# pattern. PCRE reads it as written; the one Rust spelling Julia does not share
# is `(?i:...)`, which it does.
const QWEN_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

"""
    QwenTokenizer(processordir) -> QwenTokenizer

Load the tokenizer from a `processor/` directory — the layout of the Qwen-Image
2.1 repository, not a GGUF's embedded copy.
"""
function QwenTokenizer(dir::AbstractString)
    vocabpath = joinpath(dir, "vocab.json")
    mergepath = joinpath(dir, "merges.txt")
    addedpath = joinpath(dir, "added_tokens.json")
    for path in (vocabpath, mergepath, addedpath)
        isfile(path) || throw(ArgumentError(
            "QwenImageRunner: no tokenizer at $path. It is the `processor/` " *
            "directory of https://huggingface.co/Qwen/Qwen-Image-2.1"))
    end
    vocab = JSON3.read(read(vocabpath, String))
    strtotoken = Dict{String,Int}(String(k) => Int(v) for (k, v) in pairs(vocab))
    ranks = Dict{Tuple{String,String},Int}()
    rank = 0
    for line in eachline(mergepath)
        (isempty(line) || startswith(line, "#")) && continue
        parts = split(line, ' '; limit=2)
        length(parts) == 2 || throw(ArgumentError("invalid BPE merge $(repr(line))"))
        ranks[(String(parts[1]), String(parts[2]))] = rank
        rank += 1
    end
    added = JSON3.read(read(addedpath, String))
    special = Pair{String,Int}[String(k) => Int(v) for (k, v) in pairs(added)]
    for (token, id) in special
        strtotoken[token] = id
    end
    # Longest first: `<|im_start|>` and `<|im_end|>` share a prefix, and a scan
    # that takes the shorter one leaves the rest of the tag as ordinary text.
    sort!(special; by=p -> (-ncodeunits(first(p)), first(p)))
    enc, dec = _bytemap()
    QwenTokenizer(Dict(v => k for (k, v) in strtotoken), strtotoken, ranks,
                  special, enc, dec)
end

function _encodeordinary!(out, tk::QwenTokenizer, text::AbstractString)
    for match in eachmatch(QWEN_SPLIT, text)
        mapped = String([tk.byteencoder[Int(b) + 1] for b in codeunits(match.match)])
        syms = [string(c) for c in mapped]
        n = length(syms)
        n == 0 && continue
        # A doubly linked list over the symbols plus a queue of candidate pairs,
        # so a merge costs its two new neighbours rather than a rescan.
        prev = collect(0:n-1)
        next = vcat(collect(2:n), 0)
        queue = Tuple{Int,Int,String,String}[]
        function offer(l, r)
            (l == 0 || r == 0 || isempty(syms[l]) || isempty(syms[r])) && return
            rank = get(tk.ranks, (syms[l], syms[r]), nothing)
            rank === nothing || push!(queue, (rank, l, syms[l], syms[r]))
        end
        for i in 1:n-1
            offer(i, i + 1)
        end
        while !isempty(queue)
            best = 1
            for i in 2:length(queue)
                (queue[i][1], queue[i][2]) < (queue[best][1], queue[best][2]) && (best = i)
            end
            _, l, lt, rt = queue[best]
            deleteat!(queue, best)
            r = next[l]
            # The entry is stale if either side has since been merged away.
            (r == 0 || syms[l] != lt || syms[r] != rt) && continue
            syms[l] = lt * rt; syms[r] = ""
            next[l] = next[r]
            next[r] != 0 && (prev[next[r]] = l)
            offer(prev[l], l); offer(l, next[l])
        end
        i = 1
        while i != 0
            s = syms[i]
            if !isempty(s)
                id = get(tk.strtotoken, s, nothing)
                id === nothing ? throw(ArgumentError(
                    "QwenImageRunner: $(repr(s)) is not in the tokenizer vocabulary")) :
                    push!(out, id)
            end
            i = next[i]
        end
    end
    out
end

"""
    encode(tokenizer, text) -> Vector{Int}

Token ids for `text`, special tags included where they appear literally.
"""
function encode(tk::QwenTokenizer, text::AbstractString)
    out = Int[]
    isempty(text) && return out
    text = normalize(text, :NFC)
    pos = firstindex(text)
    while pos <= lastindex(text)
        best = nothing
        bestid = 0
        for (special, id) in tk.special
            found = findnext(special, text, pos)
            found === nothing && continue
            if best === nothing || first(found) < first(best) ||
               (first(found) == first(best) && last(found) > last(best))
                best, bestid = found, id
            end
        end
        if best === nothing
            _encodeordinary!(out, tk, SubString(text, pos))
            break
        end
        pos < first(best) &&
            _encodeordinary!(out, tk, SubString(text, pos, prevind(text, first(best))))
        push!(out, bestid)
        pos = nextind(text, last(best))
    end
    out
end

"""Text for `ids`, with the byte map undone. For inspecting a tokenization."""
function decode(tk::QwenTokenizer, ids::AbstractVector{<:Integer})
    joined = IOBuffer()
    for id in ids
        s = get(tk.tokentostr, Int(id), nothing)
        s === nothing || print(joined, s)
    end
    bytes = UInt8[]
    for c in String(take!(joined))
        b = get(tk.bytedecoder, c, nothing)
        b === nothing ? append!(bytes, codeunits(string(c))) : push!(bytes, b)
    end
    String(bytes)
end
