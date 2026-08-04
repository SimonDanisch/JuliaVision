# Whisper's tokenizer: GPT-2 byte-level BPE, plus the special tokens that make a
# Whisper prompt.
#
# ## Which reference, and why not the obvious one
#
# whisper.cpp's `tokenize()` (`dev/whisper.cpp/src/whisper.cpp`) is **greedy
# longest-match**, not BPE: it walks the word taking the longest prefix present
# in the vocabulary. That is smaller and faster and it is *not the same
# function* — greedy takes `["Ġunbelie", "vable"]` where BPE's merge order takes
# something else, and any such disagreement changes the prompt the model
# conditions on. Our acceptance criterion is matching HuggingFace exactly, so the
# encoder here follows **llama.cpp's `llm_tokenizer_bpe`**
# (`dev/llama.cpp/src/llama-vocab.cpp`, MIT): regex pre-split, one symbol per
# UTF-8 character in a doubly-linked list, a priority queue over adjacent pairs
# keyed by merge rank, merge and re-offer the two new neighbours.
#
# Decoding needs neither: id -> vocabulary string -> bytes through the inverse of
# GPT-2's byte map. whisper.cpp and llama.cpp do the same thing there.
#
# `tokenizer.json` is the *data* — 50257 vocabulary entries and 50000 ranked
# merges — not the algorithm. It is HuggingFace's serialisation of exactly the
# table `tiktoken` and llama.cpp consume.
#
# ## Cost
#
# Encoding is O(n log n) in the characters of a word and runs on prompts of a few
# hundred tokens, so it is microseconds either way; the priority queue is here
# because it is *correct*, not because the naive scan would be slow. Decoding is
# a table lookup per token.

"""
    Tokenizer

Whisper's vocabulary: the BPE tables, the byte map, and the special token ids.

`ranks` is keyed by the *pair* rather than by a merged string, because that is
what the merge loop asks: given two adjacent symbols, how early does this
vocabulary want them joined. Missing means never.
"""
struct Tokenizer
    tokentostr::Vector{String}          # id -> vocabulary string (byte-mapped)
    strtotoken::Dict{String,Int}        # and back
    ranks::Dict{Tuple{String,String},Int}
    bytedecoder::Dict{Char,UInt8}       # GPT-2 byte map, inverted
    byteencoder::Vector{Char}           # 0x00..0xff -> the char that stands for it
    special::Dict{String,Int}           # "<|en|>" and friends
    eot::Int
    sot::Int
    prev::Int
    notimestamps::Int
    timestampbegin::Int
    transcribe::Int
    translate::Int
    langs::Dict{String,Int}             # "en" -> 50259
end

"""
    bytemap() -> (encoder, decoder)

GPT-2's byte-to-unicode table.

BPE operates on characters, but the input is bytes, and a byte string is not
valid UTF-8 in general. GPT-2's answer is a bijection from all 256 bytes to
printable code points: the 188 bytes that are already printable map to
themselves, and the other 68 are shifted to U+0100 and up. So every byte string
becomes a valid, printable Julia `String` that BPE can chew on, reversibly.

This is the reason a vocabulary entry looks like `"Ġthe"` — `Ġ` is U+0120, the
stand-in for byte 0x20, a space.
"""
function bytemap()
    bs = vcat(collect(UInt8('!'):UInt8('~')),
              collect(UInt8('¡'):UInt8('¬')),
              collect(UInt8('®'):UInt8('ÿ')))
    cs = Int.(bs)
    n = 0
    for b in 0:255
        if !(UInt8(b) in bs)
            push!(bs, UInt8(b))
            push!(cs, 256 + n)
            n += 1
        end
    end
    enc = Vector{Char}(undef, 256)
    dec = Dict{Char,UInt8}()
    for (b, c) in zip(bs, cs)
        enc[Int(b) + 1] = Char(c)
        dec[Char(c)] = b
    end
    return enc, dec
end

"""
    tokenizer(; dir = decoderdir()) -> Tokenizer

Read `tokenizer.json`.

Special tokens come from the file's own `added_tokens` rather than from
hard-coded ids. large-v3 shifted every one of them by one against large-v2 (the
extra Cantonese language token), and whisper.cpp carries exactly that patch as
`vocab.token_sot++`. Reading them by name means the next such shift is not a
silent off-by-one in every prompt.
"""
function tokenizer(; dir::AbstractString = decoderdir())
    path = joinpath(dir, "tokenizer.json")
    isfile(path) || throw(ArgumentError(
        "no tokenizer.json at $path — it ships in the whisper-decoder artifact; " *
        "re-bind with `julia --project=. tools/make_artifacts.jl whisper-decoder`"))
    t = JSON3.read(read(path, String))
    model = t["model"]

    nvocab = length(model["vocab"]) + length(t["added_tokens"])
    tokentostr = fill("", nvocab)
    strtotoken = Dict{String,Int}()
    for (k, v) in pairs(model["vocab"])
        s = String(k)
        tokentostr[Int(v) + 1] = s          # ids are 0-based, this vector is not
        strtotoken[s] = Int(v)
    end

    ranks = Dict{Tuple{String,String},Int}()
    for (i, m) in enumerate(model["merges"])
        # Serialised either as "a b" or as ["a", "b"] depending on the version of
        # `tokenizers` that wrote the file. Both appear in the wild.
        a, b = m isa AbstractString ? split(String(m), ' '; limit = 2) : (String(m[1]), String(m[2]))
        ranks[(String(a), String(b))] = i - 1
    end

    special = Dict{String,Int}()
    for a in t["added_tokens"]
        s = String(a["content"]); id = Int(a["id"])
        special[s] = id
        id + 1 <= nvocab && (tokentostr[id + 1] = s)
    end

    langs = Dict{String,Int}()
    for (s, id) in special
        m = match(r"^<\|([a-z]{2,3})\|>$", s)
        m === nothing || (langs[m.captures[1]] = id)
    end

    enc, dec = bytemap()
    notimestamps = special["<|notimestamps|>"]
    return Tokenizer(tokentostr, strtotoken, ranks, dec, enc, special,
                     special["<|endoftext|>"], special["<|startoftranscript|>"],
                     special["<|startofprev|>"], notimestamps,
                     notimestamps + 1,          # <|0.00|> follows <|notimestamps|>
                     special["<|transcribe|>"], special["<|translate|>"], langs)
end

"""
    istimestamp(tk, id) / timestampseconds(tk, id) / timestamptoken(tk, seconds)

Timestamp tokens are a contiguous block above `<|notimestamps|>`, one every
**0.02 s** from 0.00 to 30.00 — 1501 of them. The step is the encoder's frame
rate: 1500 output positions for a 30 s window.
"""
istimestamp(tk::Tokenizer, id::Integer) = id >= tk.timestampbegin
timestampseconds(tk::Tokenizer, id::Integer) = (id - tk.timestampbegin) * 0.02
timestamptoken(tk::Tokenizer, s::Real) = tk.timestampbegin + round(Int, s / 0.02)

"""
    isspecial(tk, id)

Whether `id` is a control token rather than text. Everything from `<|endoftext|>`
up is special: the vocabulary proper is the ids below it.
"""
isspecial(tk::Tokenizer, id::Integer) = id >= tk.eot

"""
    decode(tk, ids; skipspecial = true) -> String

Ids to text.

Concatenate the vocabulary strings, then map each character back to its byte and
interpret the result as UTF-8 — the concatenation has to happen *before* the byte
mapping, because a single multi-byte character is routinely split across two
tokens and decoding them separately would produce two invalid fragments.

Invalid UTF-8 can still survive that when a sequence is truncated mid-character
(a window boundary does it), so the final step is lenient rather than throwing.
"""
function decode(tk::Tokenizer, ids::AbstractVector{<:Integer}; skipspecial::Bool = true)
    io = IOBuffer()
    for id in ids
        skipspecial && isspecial(tk, id) && continue
        (0 <= id < length(tk.tokentostr)) || continue
        print(io, tk.tokentostr[id + 1])
    end
    s = String(take!(io))
    bytes = UInt8[]
    for c in s
        b = get(tk.bytedecoder, c, nothing)
        b === nothing ? append!(bytes, codeunits(string(c))) : push!(bytes, b)
    end
    return String(bytes)
end

"""The GPT-2 pre-tokenizer split. Contractions, then letter / digit / symbol runs."""
const BPE_SPLIT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""
    encode(tk, text) -> Vector{Int}

Text to ids, byte-level BPE.

Ported from llama.cpp's `llm_tokenizer_bpe::tokenize`. Per pre-split word: one
symbol per character in a doubly-linked list, every adjacent pair that has a
merge rank pushed onto a priority queue, then repeatedly take the *lowest-ranked*
pair, join it, and offer the two pairs that just became adjacent.

Two guards from the reference, both load-bearing:

  * a popped pair whose symbols have since been merged into something else is
    stale and is skipped — the queue is not kept consistent, entries are
    invalidated lazily. llama.cpp checks `left_token + right_token != bigram.text`;
    the same check is `sym.n == 0` plus the recorded text here.
  * a symbol that is still not in the vocabulary after all merges is emitted
    byte by byte, which cannot fail because all 256 single-byte strings are in
    the vocabulary by construction.

Needed for the previous-window context prompt, not for producing a transcript.
"""
function encode(tk::Tokenizer, text::AbstractString)
    out = Int[]
    for m in eachmatch(BPE_SPLIT, text)
        word = m.match
        isempty(word) && continue
        # bytes -> the printable stand-ins BPE actually operates on
        mapped = String([tk.byteencoder[Int(b) + 1] for b in codeunits(word)])
        chars = collect(mapped)
        n = length(chars)
        if n == 0
            continue
        end
        syms = [string(c) for c in chars]     # "" marks a symbol merged away
        prev = collect(0:(n - 1))             # 1-based indices, 0 = none
        next = vcat(collect(2:n), 0)
        # (rank, left index, the two texts at the time of offering). Equal ranks
        # resolve left to right, which is the reference's comparator.
        #
        # A linear scan for the minimum, not a heap: llama.cpp uses a
        # `priority_queue` because it tokenizes whole documents, whereas this
        # loop runs per pre-split *word* — a handful of characters — where the
        # scan is shorter than the heap's bookkeeping. The complexity is the same
        # O(n^2) either way for n that small.
        queue = Tuple{Int,Int,String,String}[]
        function offer(l, r)
            (l == 0 || r == 0) && return
            (isempty(syms[l]) || isempty(syms[r])) && return
            rk = get(tk.ranks, (syms[l], syms[r]), nothing)
            rk === nothing && return
            push!(queue, (rk, l, syms[l], syms[r]))
        end
        for i in 1:(n - 1)
            offer(i, i + 1)
        end
        while !isempty(queue)
            best = 1
            for q in 2:length(queue)
                if queue[q][1] < queue[best][1] ||
                   (queue[q][1] == queue[best][1] && queue[q][2] < queue[best][2])
                    best = q
                end
            end
            rk, l, lt, rt = queue[best]
            deleteat!(queue, best)
            r = next[l]
            # stale: either side already merged elsewhere, or the text moved on
            (r == 0 || isempty(syms[l]) || isempty(syms[r])) && continue
            (syms[l] == lt && syms[r] == rt) || continue
            syms[l] = lt * rt
            syms[r] = ""
            next[l] = next[r]
            next[r] != 0 && (prev[next[r]] = l)
            offer(prev[l], l)
            offer(l, next[l])
        end
        i = 1
        while i != 0
            s = syms[i]
            if !isempty(s)
                id = get(tk.strtotoken, s, nothing)
                if id === nothing
                    for c in s                       # every single char is in the vocab
                        cid = get(tk.strtotoken, string(c), nothing)
                        cid === nothing || push!(out, cid)
                    end
                else
                    push!(out, id)
                end
            end
            i = next[i]
        end
    end
    return out
end

"""
    prompt(tk; language = "en", task = :transcribe, timestamps = true,
           context = Int[]) -> Vector{Int}

The forced prefix a window is decoded with.

    [<|startofprev|> context...] <|startoftranscript|> <|lang|> <|task|> [<|notimestamps|>]

`context` is the previous window's tokens. Whisper conditions on them so that
spelling, casing and speaker style carry across a boundary, and openai/whisper
caps the prompt at half the 448-token context — the tail is kept, since the
nearest words matter most.
"""
function prompt(tk::Tokenizer; language::AbstractString = "en",
                task::Symbol = :transcribe, timestamps::Bool = true,
                context::AbstractVector{<:Integer} = Int[], maxcontext::Int = 223)
    ids = Int[]
    if !isempty(context)
        c = length(context) > maxcontext ? context[(end - maxcontext + 1):end] : context
        push!(ids, tk.prev)
        append!(ids, c)
    end
    push!(ids, tk.sot)
    haskey(tk.langs, language) || throw(ArgumentError("unknown language $language"))
    push!(ids, tk.langs[language])
    push!(ids, task === :translate ? tk.translate : tk.transcribe)
    timestamps || push!(ids, tk.notimestamps)
    return ids
end
