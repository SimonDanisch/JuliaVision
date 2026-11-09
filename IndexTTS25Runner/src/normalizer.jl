# WeText's optimized tropical transducers, exported as CSR arrays. Applying a
# fixed input leaves only epsilon edges and edges matching its next UTF-8 byte.
# Queue selection follows OpenFst's AutoQueue/SingleShortestPath, including its
# equal-cost choices: https://www.openfst.org/ . Tables come from WeTextProcessing
# (Apache-2.0, Zhendong Peng and Xingchen Song); its TokenParser ordering is below.

struct TextFST
    start::Int
    offsets::Vector{Int64}
    finals::Vector{Float32}
    ilabels::Vector{Int32}
    olabels::Vector{Int32}
    weights::Vector{Float32}
    targets::Vector{Int32}
end

function TextFST(path::AbstractString)
    w = readsafetensors(path)
    TextFST(Int(only(w["start"])) + 1, vec(w["offsets"]), vec(w["finals"]),
            vec(w["ilabels"]), vec(w["olabels"]), vec(w["weights"]), vec(w["targets"]))
end

function transduce(fst::TextFST, text::AbstractString)
    isempty(text) && return ""
    input = codeunits(text)
    states = [(0, fst.start)]
    ids = Dict(first(states) => 1)
    arcs = Vector{Tuple{Int,Int32,Float32}}[]
    finals = Float32[]
    function intern(position, state)
        get!(ids, (position, state)) do
            push!(states, (position, state))
            length(states)
        end
    end
    head = 1
    while head <= length(states)
        position, state = states[head]
        outgoing = Tuple{Int,Int32,Float32}[]
        for arc in (fst.offsets[state] + 1):fst.offsets[state + 1]
            label = fst.ilabels[arc]
            if label == 0 || (position < length(input) && label == input[position + 1])
                target = intern(position + (label != 0), Int(fst.targets[arc]) + 1)
                push!(outgoing, (target, fst.olabels[arc], fst.weights[arc]))
            end
        end
        push!(arcs, outgoing)
        push!(finals, position == length(input) ? fst.finals[state] : Inf32)
        head += 1
    end
    # Composition removes states with no successful continuation before
    # shortestpath chooses its queue. Do the same: dead weighted branches must
    # not turn an otherwise unweighted lattice into a weighted one.
    incoming = [Int[] for _ in states]
    for (i, outgoing) in enumerate(arcs), (target, _, _) in outgoing
        push!(incoming[target], i)
    end
    live = isfinite.(finals)
    queue = findall(live)
    head = 1
    while head <= length(queue)
        for source in incoming[queue[head]]
            live[source] && continue
            live[source] = true; push!(queue, source)
        end
        head += 1
    end
    live[1] || throw(ArgumentError("normalizer has no path for $(repr(text))"))
    unweighted = all(!live[i] || !isfinite(finals[i]) || finals[i] == 0 for i in eachindex(states)) &&
        all(!live[i] || all(!live[t] || w == 0 for (t, _, w) in arcs[i]) for i in eachindex(states))
    # OpenFst AutoQueue selects LIFO for unweighted idempotent lattices and a
    # DFS topological queue for acyclic weighted lattices. Equal-cost paths keep
    # their first predecessor; queue order therefore affects decimal readings.
    order = Int[]
    if !unweighted
        color = zeros(UInt8, length(states))
        stack = [(1, 1)]
        color[1] = 1
        while !isempty(stack)
            state, edge = stack[end]
            if edge > length(arcs[state])
                push!(order, state); color[state] = 2; pop!(stack)
                continue
            end
            stack[end] = (state, edge + 1)
            target = arcs[state][edge][1]
            live[target] || continue
            color[target] == 1 && throw(ArgumentError("normalizer epsilon cycle"))
            color[target] == 2 && continue
            color[target] = 1; push!(stack, (target, 1))
        end
        reverse!(order)
    end
    scores = fill(Inf32, length(states)); scores[1] = 0
    parents = zeros(Int, length(states))
    labels = zeros(Int32, length(states))
    queue = unweighted ? [1] : order
    enqueued = falses(length(states)); enqueued[1] = true
    winner, cost = 0, Inf32
    head = 1
    while unweighted ? !isempty(queue) : head <= length(queue)
        state = unweighted ? pop!(queue) : queue[head]
        head += 1; enqueued[state] = false
        value = scores[state] + finals[state]
        if value < cost
            winner, cost = state, value
        end
        for (target, label, weight) in arcs[state]
            live[target] || continue
            score = scores[state] + weight
            score < scores[target] || continue
            scores[target] = score; parents[target] = state; labels[target] = label
            if unweighted && !enqueued[target]
                push!(queue, target); enqueued[target] = true
            end
        end
    end
    bytes = UInt8[]
    while winner != 0
        labels[winner] == 0 || push!(bytes, UInt8(labels[winner]))
        winner = parents[winner]
    end
    return String(reverse!(bytes))
end

const TN_ORDERS = Dict(
    "date" => ["year", "month", "day"], "fraction" => ["denominator", "numerator"],
    "measure" => ["denominator", "numerator", "value"], "money" => ["value", "currency"],
    "time" => ["noon", "hour", "minute", "second"])
const EN_TN_ORDERS = Dict("date" => ["preserve_order", "text", "day", "month", "year"],
                         "money" => ["integer_part", "fractional_part", "quantity", "currency_maj"])

"""WeText TokenParser.reorder, retaining quoted escapes for the verbalizer."""
function reordertags(text::AbstractString, lang::Symbol)
    orders = lang === :en ? EN_TN_ORDERS : TN_ORDERS
    tokens = String[]
    for token in eachmatch(r"([a-zA-Z_]+)\s*\{((?:[^\"}]|\"(?:\\.|[^\"\\])*\")*)\}", text)
        members = Dict{String,String}()
        order = String[]
        for field in eachmatch(r"([a-zA-Z_]+):\s*\"((?:\\.|[^\"\\])*)\"", token[2])
            key = String(field[1])
            push!(order, key); members[key] = String(field[2])
        end
        if get(members, "preserve_order", "") != "true"
            order = get(orders, String(token[1]), order)
        end
        fields = [" $key: \"$(members[key])\"" for key in order if haskey(members, key)]
        push!(tokens, token[1] * " {" * join(fields) * " }")
    end
    return join(tokens, ' ')
end

struct TextNormalizerTables
    taggers::Dict{Symbol,TextFST}
    verbalizers::Dict{Symbol,TextFST}
end

TextNormalizerTables(dir::AbstractString) = TextNormalizerTables(
    Dict(lang => TextFST(joinpath(dir, "$(lang)_tagger.safetensors")) for lang in (:en, :zh)),
    Dict(lang => TextFST(joinpath(dir, "$(lang)_verbalizer.safetensors")) for lang in (:en, :zh)))

function normalizetext(tables::TextNormalizerTables, text::AbstractString, lang::Symbol)
    tagged = transduce(tables.taggers[lang], text)
    return transduce(tables.verbalizers[lang], reordertags(tagged, lang))
end

const PINYIN_RX = r"(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])"i
const CHINESE_NAME_RX = r"[\u4e00-\u9fff]+(?:[-·—][\u4e00-\u9fff]+){1,2}"
const TECH_TERM_RX = r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+"
const CONTRACTION_RX = r"(what|where|who|which|how|t?here|it|s?he|that|this)'s"i

function alphabeticindex(i::Int)
    out = ""
    while true
        out = string(Char(Int('a') + i % 26)) * out
        i = i ÷ 26 - 1
        i < 0 && return out
    end
end

function protectmatches(text::AbstractString, pattern::Regex, prefix::String, suffix::String = "")
    names = unique(String(m.match) for m in eachmatch(pattern, text))
    pairs = [prefix * alphabeticindex(i - 1) * suffix => name for (i, name) in enumerate(names)]
    for (placeholder, name) in pairs
        text = replace(text, name => placeholder)
    end
    return String(text), pairs
end

function normalizetext(tables::TextNormalizerTables, text::AbstractString, lang::AbstractString)
    lowercase(lang) in ("en", "zh", "zhen") || return String(text)
    text = cleantext(text)
    text, annotations = protectmatches(text, ANNOTATION_RX, "PRONPLACEHOLDER", "PRONPLACEHOLDER")
    chinese = occursin(r"[\u4e00-\u9fff]", text) || !occursin(r"[a-zA-Z]", text) ||
              occursin(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$", text) || occursin(PINYIN_RX, text)
    text = replace(text, CONTRACTION_RX => s"\1 is")
    terms = unique(String(m.match) for m in eachmatch(TECH_TERM_RX, text))
    for term in sort!(terms; by = length, rev = true)
        text = replace(text, term => replace(term, "-" => "<H>"))
    end
    if chinese
        text = rstrip(text)
        text, pinyin = protectmatches(text, PINYIN_RX, "<pinyin_", ">")
        text, names = protectmatches(text, CHINESE_NAME_RX, "<n_", ">")
        text = normalizetext(tables, text, :zh)
        for (placeholder, name) in names
            text = replace(text, placeholder => name)
        end
        for (placeholder, pronunciation) in pinyin
            if first(pronunciation) in "jqxJQX"
                pronunciation = uppercase(replace(pronunciation, r"([jqx])[uü](n|e|an)*(\d)"i => s"\1v\2\3"))
            end
            text = replace(text, placeholder => pronunciation)
        end
    else
        text = normalizetext(tables, text, :en)
    end
    isempty(terms) || (text = replace(text, r"\s*<H>\s*" => "-"))
    chinese && (text = replace(text, "\$" => "."))
    text = cleantext(text)
    for (placeholder, annotation) in annotations
        text = replace(text, placeholder => annotation)
    end
    return String(text)
end
