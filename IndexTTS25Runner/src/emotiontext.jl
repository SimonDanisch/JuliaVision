# The bundled Qwen3-0.6B model, its own tokenizer/template, and QwenEmotion's
# output conversion. The template is the checkpoint's System:/Human: format.

const EMOTION_SPLIT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

struct EmotionTokenizer
    bpe::Tokenizer
    tokens::Dict{Int,Vector{UInt8}}
end

function EmotionTokenizer(dir::AbstractString)
    printable = vcat(collect(33:126), collect(161:172), collect(174:255))
    chars = Char.(printable)
    extra = 0
    for byte in 0:255
        byte in printable && continue
        push!(printable, byte); push!(chars, Char(256 + extra)); extra += 1
    end
    decodebyte = Dict(c => UInt8(b) for (b, c) in zip(printable, chars))
    vocab = JSON3.read(read(joinpath(dir, "vocab.json"), String))
    ranks = Dict{Vector{UInt8},Int}()
    tokens = Dict{Int,Vector{UInt8}}()
    for (token, id) in pairs(vocab)
        bytes = UInt8[decodebyte[c] for c in String(token)]
        ranks[bytes] = Int(id); tokens[Int(id)] = bytes
    end
    special = Dict{String,Int}(String(k) => Int(v) for (k, v) in
                              pairs(JSON3.read(read(joinpath(dir, "added_tokens.json"), String))))
    names = sort!(collect(keys(special)); by = length, rev = true)
    return EmotionTokenizer(Tokenizer(ranks, special, Regex(join(rxescape.(names), "|"))), tokens)
end

emotionprompt(text::AbstractString) = "System: 文本情感分类<|endoftext|>\nHuman: $text<|endoftext|>\nAssistant:"
emotionids(tk::EmotionTokenizer, text::AbstractString) =
    encode(tk.bpe, Base.Unicode.normalize(emotionprompt(text), :NFC); pattern = EMOTION_SPLIT)

function emotiondecode(tk::EmotionTokenizer, ids)
    bytes = UInt8[]
    for id in ids
        haskey(tk.tokens, id) && append!(bytes, tk.tokens[id])
    end
    return String(bytes)
end

struct EmotionText{M}
    model::M
    tokenizer::EmotionTokenizer
    capacity::Int
    eos::Int
end

function loademotiontext(dir::AbstractString; device)
    graphs = Dict(name => loadgraph(joinpath(dir, "$name.json"))
                  for name in ("emotion_prefill", "emotion_decode"))
    model = Model(graphs, readpartweights(dir); device)
    cfg = JSON3.read(read(joinpath(dir, "config.json"), String))
    return EmotionText(model, EmotionTokenizer(dir), Int(cfg.capacity), Int(cfg.eos))
end

function emotionmask(n::Int, capacity::Int, start::Int)
    mask = fill(-floatmax(Float16), capacity, n, 1, 1)
    for i in 1:n
        mask[1:(start + i), i, 1, 1] .= Float16(0)
    end
    return mask
end

function emotiontokens(model::EmotionText, text::AbstractString; max_new_tokens::Int = 32768)
    ids = emotionids(model.tokenizer, text)
    n, capacity, dev = length(ids), model.capacity, model.model.device
    2 <= n < capacity || throw(ArgumentError("emotion prompt needs $n tokens, capacity is $capacity"))
    shape = (128, capacity, 8, 1, 28)
    kc = fill!(Mantle.Buffer(dev, Float16, shape), Float16(0))
    vc = fill!(Mantle.Buffer(dev, Float16, shape), Float16(0))
    function forward(name, tokens, positions, K, V, mask; dims = (;))
        inputs = (Mantle.Buffer(dev, reshape(Int64.(tokens), :, 1)),
                  Mantle.Buffer(dev, Int64.(positions)), Mantle.Buffer(dev, mask))
        try
            logits, keys, values = call(model.model, name, inputs[1], K, V, inputs[2], inputs[3]; dims)
            return argmax(vec(Array(logits))) - 1, keys, values
        finally
            foreach(Mantle.free!, inputs)
        end
    end
    try
        token, K, V = forward("emotion_prefill", ids, 0:(n - 1), kc, vc,
                              emotionmask(n, capacity, 0); dims = (; n))
        result = Int[]
        while token != model.eos && length(result) < max_new_tokens
            push!(result, token)
            n < capacity || throw(ArgumentError("emotion generation exceeded $capacity cached tokens"))
            token, K, V = forward("emotion_decode", [token], [n], K, V, emotionmask(1, capacity, n))
            n += 1
        end
        return result
    finally
        Mantle.free!(kc); Mantle.free!(vc)
    end
end

const EMOTION_CN = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]

function emotionweights(response::AbstractString, text::AbstractString)
    content = try
        value = JSON3.read(response)
        value isa AbstractDict || value isa JSON3.Object ? Dict{String,Any}(String(k) => v for (k,v) in pairs(value)) : value
    catch
        Dict{String,Any}(String(m[1]) => parse(Float64, m[2]) for m in
                        eachmatch(r"([^\s\":.,]+?)\"?\s*:\s*([\d.]+)", response))
    end
    label(value) = value isa AbstractString ? findfirst(i -> strip(value) == EMOTION_CN[i] ||
                   lowercase(strip(value)) == String(EMOTIONS8[i]), eachindex(EMOTION_CN)) : nothing
    normalized = content isa AbstractDict ? copy(content) : Dict{String,Any}()
    # Upstream swaps raw fields before normalizing labels.
    if any(word -> occursin(word, lowercase(text)), ("低落", "melancholy", "melancholic", "depression", "depressed", "gloomy"))
        normalized["悲伤"], normalized["低落"] = get(normalized, "低落", 0.0), get(normalized, "悲伤", 0.0)
    end
    detected = content isa AbstractString ? label(content) : nothing
    if detected === nothing
        for key in ("emotion", "emotion_label", "label", "情感", "情绪")
            detected = label(get(normalized, key, nothing))
            detected === nothing || break
        end
    end
    if detected !== nothing && all(k -> !haskey(normalized, k), EMOTION_CN)
        normalized[EMOTION_CN[detected]] = 1.0
    end
    for key in EMOTION_CN
        detected = label(get(normalized, key, nothing))
        detected === nothing && continue
        normalized[key] = EMOTION_CN[detected] == key ? 1.0 : 0.0
        EMOTION_CN[detected] == key || (normalized[EMOTION_CN[detected]] = 1.0)
    end
    result = [clamp(Float64(get(normalized, key, 0.0)), 0.0, 1.2) for key in EMOTION_CN]
    all(<=(0), result) && (result[end] = 1.0)
    return result
end

function emotiontext(model::EmotionText, text::AbstractString)
    ids = emotiontokens(model, text)
    lastthink = findlast(==(151668), ids)
    lastthink === nothing || (ids = ids[(lastthink + 1):end])
    return emotionweights(emotiondecode(model.tokenizer, ids), text)
end
