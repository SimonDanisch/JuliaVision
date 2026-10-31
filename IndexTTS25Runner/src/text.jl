# Text to token ids, as `IndexTTS2.infer_generator` does it.
#
# The vocabulary is a tiktoken BPE (`multilingual_zh_ja_yue_char_del.tiktoken`)
# with Whisper's special tokens appended in `indextts/utils/tokenizer.py`'s order.
# Encoding is tiktoken's: the special tokens split the text, the GPT-2 pattern
# splits the rest, and each piece is merged by `_byte_pair_merge` (tiktoken's
# `src/lib.rs`) — lowest-ranked adjacent pair first, leftmost on ties.
#
# Around it, upstream's steps in order: the punctuation table, (normalisation),
# case, pronunciation annotations `<word|reading>`, upper-cased special-token
# names, then segments of at most `max_tokens` tokens split at punctuation.
# German has its own frontend (`german.jl`), which the German checkpoint was
# trained with and which replaces upstream's normalisation.

# ── the vocabulary ───────────────────────────────────────────────────────────

"""
`indextts.utils.tokenizer.LANGUAGES`, in order: Whisper's 99 and seven more. The
first 99 are special tokens (`get_encoding(num_languages = 99)`); all 106 index
the GPT's language embedding (`LANGUAGE_DICT`), `common` for anything else.
"""
const LANGUAGES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect",
    "zh/en", "en/zh", "common"
]
const SPECIAL_LANGUAGES = 99
const AUDIO_EVENTS = ["ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter",
                      "Applause", "/Applause"]
const EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]
const TTS_TOKENS = vcat(["TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H"],
                        ["TTS/SP" * lpad(i, 2, '0') for i in 1:13])

"""The special tokens of `get_encoding`, in the order they are numbered."""
function specialtokens()
    s = ["<|endoftext|>", "<|startoftranscript|>"]
    append!(s, "<|$l|>" for l in LANGUAGES[1:SPECIAL_LANGUAGES])
    append!(s, "<|$e|>" for e in AUDIO_EVENTS)
    append!(s, "<|$e|>" for e in EMOTIONS)
    append!(s, ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
                "<|nospeech|>", "<|notimestamps|>"])
    append!(s, "<|SPECIAL_TOKEN_$i|>" for i in 1:30)
    append!(s, "<|$t|>" for t in TTS_TOKENS)
    # `f"<|{i * 0.02:.2f}|>"`: i * 0.02 is within an ulp of 2i hundredths.
    append!(s, "<|$(2i ÷ 100).$(lpad(2i % 100, 2, '0'))|>" for i in 0:1500)
    return s
end

"""
    Tokenizer

The BPE ranks and the special tokens.
"""
struct Tokenizer
    ranks::Dict{Vector{UInt8},Int}
    special::Dict{String,Int}
    specialrx::Regex
end

"""`'s|'t|...`, the GPT-2 pre-tokenizer tiktoken is constructed with here."""
const BPE_SPLIT = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""Escape a literal for a regex."""
rxescape(s::AbstractString) = replace(s, r"[\\^$.|?*+()\[\]{}/-]" => s"\\\0")

function Tokenizer(path::AbstractString)
    ranks = Dict{Vector{UInt8},Int}()
    for line in eachline(path)
        isempty(line) && continue
        tok, rank = split(line)
        ranks[base64decode(tok)] = parse(Int, rank)
    end
    n = length(ranks)
    special = Dict{String,Int}()
    for (i, t) in enumerate(specialtokens())
        special[t] = n + i - 1
    end
    # Longest first so a prefix of another special never wins the alternation.
    names = sort!(collect(keys(special)); by = length, rev = true)
    Tokenizer(ranks, special, Regex(join(rxescape.(names), "|")))
end

"""Decode RFC 4648 base64 (the `.tiktoken` format), padding included."""
function base64decode(s::AbstractString)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    val = Dict(c => i - 1 for (i, c) in enumerate(alphabet))
    out = UInt8[]
    acc, bits = UInt32(0), 0
    for c in s
        c == '=' && break
        acc = (acc << 6) | UInt32(val[c])
        bits += 6
        if bits >= 8
            bits -= 8
            push!(out, UInt8((acc >> bits) & 0xff))
        end
    end
    return out
end

"""
    bytepairmerge(ranks, piece) -> Vector{Int}

tiktoken's `_byte_pair_merge` and `byte_pair_encode`: part boundaries carry the
rank of the pair they would merge with the next part; the lowest rank merges,
leftmost first, and its neighbours' ranks are recomputed.
"""
function bytepairmerge(ranks::Dict{Vector{UInt8},Int}, piece::Vector{UInt8})
    n = length(piece)
    n == 1 && return [ranks[piece]]
    big = typemax(Int)
    rank(a, b) = get(ranks, piece[a:(b - 1)], big)     # piece[a, b) 1-based
    starts = collect(1:(n + 1))                          # part starts, then the end
    rk = [i < n ? rank(i, i + 2) : big for i in 1:(n + 1)]
    rankat(i) = i + 3 <= length(starts) ? rank(starts[i], starts[i + 3]) : big
    while true
        m, i = big, 0
        for j in 1:(length(starts) - 1)
            if rk[j] < m
                m, i = rk[j], j
            end
        end
        m == big && break
        i > 1 && (rk[i - 1] = rankat(i - 1))
        rk[i] = rankat(i)
        deleteat!(starts, i + 1)
        deleteat!(rk, i + 1)
    end
    return [ranks[piece[starts[j]:(starts[j + 1] - 1)]] for j in 1:(length(starts) - 1)]
end

"""Ordinary text (no specials) to ids: pre-split, then each piece merged."""
function encodeordinary!(ids::Vector{Int}, tk::Tokenizer, text::AbstractString; pattern::Regex = BPE_SPLIT)
    for m in eachmatch(pattern, text)
        piece = Vector{UInt8}(codeunits(m.match))
        r = get(tk.ranks, piece, nothing)
        r === nothing ? append!(ids, bytepairmerge(tk.ranks, piece)) : push!(ids, r)
    end
    return ids
end

"""
    encode(tk, text) -> Vector{Int}

`encoding.encode(text, allowed_special = "all")`.
"""
function encode(tk::Tokenizer, text::AbstractString; pattern::Regex = BPE_SPLIT)
    ids = Int[]
    pos = 1
    for m in eachmatch(tk.specialrx, text)
        m.offset > pos && encodeordinary!(ids, tk, SubString(text, pos, prevind(text, m.offset)); pattern)
        push!(ids, tk.special[m.match])
        pos = m.offset + ncodeunits(m.match)
    end
    pos <= ncodeunits(text) && encodeordinary!(ids, tk, SubString(text, pos); pattern)
    return ids
end

ntokens(tk::Tokenizer, text::AbstractString) = length(encode(tk, text))

"""`lang_to_token`: the language embedding row, `common` for anything unknown."""
function langtoken(lang::AbstractString)
    i = findfirst(==(lowercase(lang)), LANGUAGES)
    return i === nothing ? length(LANGUAGES) - 1 : i - 1
end

# ── upstream's text steps ────────────────────────────────────────────────────

"""`TextNormalizer.char_rep_map`, in its order: the alternation tries them in turn."""
const CHAR_REPLACEMENTS = [
    "：" => ",", "；" => ",", ";" => ",", "，" => ",", "。" => ".", "！" => "!", "？" => "?",
    "\n" => " ", "·" => "-", "、" => ",", "..." => "…", ",,," => "…", "，，，" => "…",
    "……" => "…", "“" => "'", "”" => "'", "\"" => "'", "‘" => "'", "’" => "'",
    "（" => "'", "）" => "'", "(" => "'", ")" => "'", "《" => "'", "》" => "'",
    "【" => "'", "】" => "'", "[" => "'", "]" => "'", "—" => "-", "～" => "-", "~" => "-",
    "「" => "'", "」" => "'", ":" => ",",
]
const CLEAN_RX = Regex(join((rxescape(first(p)) for p in CHAR_REPLACEMENTS), "|"))
const CLEAN_MAP = Dict(CHAR_REPLACEMENTS)

"""`clean_pattern.sub(lambda x: char_rep_map[x.group()], text)`."""
cleantext(text::AbstractString) = replace(text, CLEAN_RX => m -> CLEAN_MAP[m])

const ANNOTATION_RX = r"<([^|>\n]+)\|([^>\n]+)>"

iskana(s::AbstractString) = occursin(r"^[\u3040-\u309F]+$", s) || occursin(r"^[\u30A0-\u30FF]+$", s)

"""
`apply_pronunciation_annotations`: `<word|reading>` becomes the upper-cased
reading wrapped in `<|SPECIAL_TOKEN_2|>` (Chinese word) or `_1|>`; a kana
reading goes in bare, between spaces.
"""
function annotate(text::AbstractString)
    out = IOBuffer()
    pos = 1
    for m in eachmatch(ANNOTATION_RX, text)
        print(out, SubString(text, pos, prevind(text, m.offset)))
        word, reading = m.captures[1], uppercase(m.captures[2])
        if iskana(reading)
            print(out, ' ', reading, ' ')
        else
            tok = occursin(r"[\u4e00-\u9fff]", word) ? "SPECIAL_TOKEN_2" : "SPECIAL_TOKEN_1"
            print(out, "<|", tok, "|>", reading, "<|", tok, "|>")
        end
        pos = m.offset + ncodeunits(m.match)
    end
    print(out, SubString(text, pos))
    return String(take!(out))
end

"""Special-token names upper-cased: `<\\|([^|]+)\\|>`."""
upperspecials(text::AbstractString) = replace(text, r"<\|([^|]+)\|>" => m -> "<|" * uppercase(m[3:(end - 2)]) * "|>")

const PROTECTED_RX = r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>"
const SEGMENT_BREAK_RX = r"(?<=[，。！？、；：,\.!\?;:\n])"

"""`_split_atomic_pieces`: annotated readings stay whole."""
function atomicpieces(text::AbstractString)
    pieces = Tuple{String,Bool}[]
    pos = 1
    for m in eachmatch(PROTECTED_RX, text)
        m.offset > pos && push!(pieces, (String(SubString(text, pos, prevind(text, m.offset))), false))
        push!(pieces, (String(m.match), true))
        pos = m.offset + ncodeunits(m.match)
    end
    pos <= ncodeunits(text) && push!(pieces, (String(SubString(text, pos)), false))
    return pieces
end

"""
    segments(tk, text, maxtokens, prefix) -> Vector{String}

`split_text_by_tokens`: the whole text if it fits the budget (`min(maxtokens,
600)` less the prefix), else pieces cut after punctuation, a piece still too
long cut character by character, and the pieces packed greedily.
"""
function segments(tk::Tokenizer, text::AbstractString, maxtokens::Int, prefix::AbstractString;
                  capacity::Int = 602)
    budget = max(1, min(maxtokens, capacity - 2) - ntokens(tk, prefix))
    ntokens(tk, text) <= budget && return [String(text)]
    chunks = String[]
    for (piece, atomic) in atomicpieces(text)
        if atomic
            push!(chunks, piece)
            continue
        end
        for part in split(piece, SEGMENT_BREAK_RX)
            isempty(part) && continue
            if ntokens(tk, part) <= budget
                push!(chunks, part)
                continue
            end
            current = ""
            for ch in part
                if !isempty(current) && ntokens(tk, current * ch) > budget
                    push!(chunks, current)
                    current = string(ch)
                else
                    current *= ch
                end
            end
            isempty(current) || push!(chunks, current)
        end
    end
    segs, current = String[], ""
    for chunk in chunks
        if !isempty(current) && ntokens(tk, current * chunk) > budget
            push!(segs, current)
            current = chunk
        else
            current *= chunk
        end
    end
    isempty(current) || push!(segs, current)
    return isempty(segs) ? [String(text)] : segs
end

"""
    TextInput

What the GPT reads for one utterance: each segment's ids (with upstream's
trailing 1) and the language row.
"""
struct TextInput
    segments::Vector{String}
    ids::Vector{Vector{Int}}
    lang::Int
end

"""
    prepare(tk, text, lang; maxtokens = 120) -> TextInput

`infer_generator`'s text path with `text_normalization = false`, which is what
the German checkpoint runs with (after [`preparegerman`](@ref)) and what any text
already written out in words needs. Case follows upstream: lower for ja, zh,
zhen, en; upper for es; unchanged otherwise.
"""
function prepare(tk::Tokenizer, text::AbstractString, lang::AbstractString; maxtokens::Int = 120)
    l = lowercase(lang)
    prefix = "<|$l|> "
    t = cleantext(text)
    if l in ("ja", "zh", "zhen", "en")
        t = lowercase(t)
    elseif l == "es"
        t = uppercase(t)
    end
    t = upperspecials(annotate(t))
    segs = segments(tk, t, maxtokens, prefix)
    ids = [vcat(encode(tk, prefix * s), 1) for s in segs]
    return TextInput(segs, ids, langtoken(l))
end
