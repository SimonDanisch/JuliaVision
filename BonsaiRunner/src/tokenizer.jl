"""Byte-level GPT-2 BPE tables stored directly in the Bonsai GGUF."""
struct BonsaiTokenizer
    tokentostr::Vector{String}
    strtotoken::Dict{String,Int}
    ranks::Dict{Tuple{String,String},Int}
    types::Vector{Int}
    byteencoder::Vector{Char}
    bytedecoder::Dict{Char,UInt8}
    special::Vector{Pair{String,Int}}
    bos::Int
    eos::Int
end

function _bytemap()
    bs = vcat(collect(UInt8('!'):UInt8('~')),
              collect(UInt8('¡'):UInt8('¬')),
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

function BonsaiTokenizer(file::GGUFFile)
    md = file.metadata
    String(md["tokenizer.ggml.model"]) == "gpt2" || error("BonsaiRunner supports the GGUF GPT-2 tokenizer")
    tokens = String.(md["tokenizer.ggml.tokens"])
    lookup = Dict(s => i - 1 for (i, s) in enumerate(tokens))
    ranks = Dict{Tuple{String,String},Int}()
    for (i, merge) in enumerate(md["tokenizer.ggml.merges"])
        parts = split(String(merge), ' '; limit=2)
        length(parts) == 2 || error("invalid BPE merge $(repr(merge))")
        ranks[(parts[1], parts[2])] = i - 1
    end
    enc, dec = _bytemap()
    types = Int.(md["tokenizer.ggml.token_type"])
    special = sort!([tokens[i] => i-1 for i in eachindex(tokens) if types[i] != 1];
                    by=p -> (-ncodeunits(first(p)), first(p)))
    BonsaiTokenizer(tokens, lookup, ranks, types, enc, dec, special,
        Int(md["tokenizer.ggml.bos_token_id"]), Int(md["tokenizer.ggml.eos_token_id"]))
end

# The exact qwen35 pre-tokenizer from the reference fork. Its combining-mark
# branch is the one difference from Qwen2's otherwise similar expression.
const QWEN35_SPLIT = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

function _encodeordinary!(out, tk::BonsaiTokenizer, text::AbstractString)
    for match in eachmatch(QWEN35_SPLIT, text)
        mapped = String([tk.byteencoder[Int(b) + 1] for b in codeunits(match.match)])
        syms = [string(c) for c in mapped]
        n = length(syms)
        n == 0 && continue
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
                if id === nothing
                    for c in s
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
    out
end

function encode(tk::BonsaiTokenizer, text::AbstractString)
    out = Int[]
    isempty(text) && return out
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
        if pos < first(best)
            _encodeordinary!(out, tk, SubString(text, pos, prevind(text, first(best))))
        end
        push!(out, bestid)
        pos = nextind(text, last(best))
    end
    out
end

function decode(tk::BonsaiTokenizer, ids::AbstractVector{<:Integer}; skipspecial::Bool=true)
    joined = IOBuffer()
    for id in ids
        0 <= id < length(tk.tokentostr) || continue
        skipspecial && tk.types[id + 1] != 1 && continue
        print(joined, tk.tokentostr[id + 1])
    end
    bytes = UInt8[]
    for c in String(take!(joined))
        b = get(tk.bytedecoder, c, nothing)
        b === nothing ? append!(bytes, codeunits(string(c))) : push!(bytes, b)
    end
    String(bytes)
end

"""A minimal text-only rendering of the checkpoint's Qwen chat template."""
function chatprompt(messages; add_generation_prompt::Bool=true, thinking::Bool=true)
    io = IOBuffer()
    for message in messages
        role = message isa Pair ? first(message) : getproperty(message, :role)
        content = message isa Pair ? last(message) : getproperty(message, :content)
        print(io, "<|im_start|>", role, '\n', content, "<|im_end|>\n")
    end
    if add_generation_prompt
        print(io, "<|im_start|>assistant\n<think>\n")
        thinking || print(io, "\n</think>\n\n")
    end
    String(take!(io))
end
