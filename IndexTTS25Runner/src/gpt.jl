# The GPT: prefix, then semantic codes until the stop token, as upstream's
# `inference_speech` drives transformers' generation (index-tts vendors 4.52's
# `_sample` and `_beam_search` with `BeamSearchScorer`).
#
# Each step's raw logits come back from the device; the processors, the sampling
# and the beam bookkeeping run here, in upstream's order:
#
#   greedy / sampling: repetition penalty on the raw logits, then (sampling)
#     temperature, top-k, top-p, softmax, one draw;
#   beams: log-softmax, repetition penalty, (sampling) temperature, top-k and
#     top-p keeping at least 2, plus each beam's score, then over all beams
#     `2 num_beams` candidates — drawn without replacement when sampling, the
#     best otherwise — sorted, and `BeamSearchScorer.process`; length penalty 0.
#
# A draw without replacement is ATen's exponential race, `topk(p / q)` with
# `q ~ Exp(1)`; the noise comes from a counter hash (`noise`) so a seed names a
# decode on any device and `dump_indextts25_refs.py` can draw the same.

const START_MEL = 8192
const STOP_MEL = 8193
const VOCAB_MEL = 8194

"""Cache rows: a prefix of 3 + 122 rows and 1500 mel tokens fit."""
const CACHE_ROWS = 2176

"""
    Sampling(; do_sample, num_beams, temperature, top_k, top_p, penalty, seed, max_tokens)

Generation settings; the defaults are `infer_v2_5.py`'s (beam search with
sampling over 3 beams). The German checkpoint's card uses `num_beams = 1,
temperature = 0.7`. `max_tokens` is upstream's `max_mel_tokens`.
"""
Base.@kwdef struct Sampling
    do_sample::Bool = true
    num_beams::Int = 3
    temperature::Float32 = 0.8f0
    top_k::Int = 30
    top_p::Float32 = 0.8f0
    penalty::Float32 = 10f0
    seed::Int = 1234
    max_tokens::Int = 1500
end

"""Chris Wellons' `lowbias32`."""
function lowbias32(x::UInt32)
    x ⊻= x >> 16
    x *= 0x7feb352d
    x ⊻= x >> 15
    x *= 0x846ca68b
    x ⊻= x >> 16
    return x
end

"""Exp(1) noise of draw `call`, element `i` (zero-based over the probabilities)."""
function noise(seed::Integer, call::Integer, i::Integer)
    key = lowbias32(UInt32(call) ⊻ lowbias32(UInt32(seed % 2^32)))
    h = lowbias32(UInt32(i) ⊻ key)
    return -log(Float32((h >> 8) + 0x00000001) * 2f0^-24)
end

"""ATen divides by a host scalar as a multiply by its float reciprocal."""
scalardiv(x::Float32, s::Float32) = x * (1f0 / s)

function penalize!(x::AbstractVector{Float32}, seen, penalty::Float32)
    for t in seen
        s = x[t + 1]
        x[t + 1] = s < 0 ? s * penalty : scalardiv(s, penalty)
    end
    return x
end

function logsoftmax!(x::AbstractVector{Float32})
    m = maximum(x)
    l = log(sum(v -> exp(v - m), x))
    x .= x .- m .- l
end

softmax(x) = (e = exp.(x .- maximum(x)); e ./ sum(e))

"""Temperature, top-k, top-p, in transformers' order, on one row."""
function warp!(x::AbstractVector{Float32}, s::Sampling, minkeep::Int)
    s.temperature != 1f0 && (x .= scalardiv.(x, s.temperature))
    if s.top_k > 0
        k = min(max(s.top_k, minkeep), length(x))
        kth = partialsort(x, k; rev = true)
        x[x .< kth] .= -Inf32
    end
    if s.top_p < 1f0
        order = sortperm(x)
        p = softmax(x[order])
        remove = cumsum(p) .<= Float32(1 - Float64(s.top_p))
        remove[(end - minkeep + 1):end] .= false
        x[order[remove]] .= -Inf32
    end
    return x
end

"""`multinomial(p, n)` without replacement as ATen draws it, from [`noise`](@ref)."""
function race(p::AbstractVector{Float32}, n::Int, seed::Int, call::Int)
    q = [p[i] / noise(seed, call, i - 1) for i in eachindex(p)]
    return n == 1 ? [argmax(q)] : partialsortperm(q, 1:n; rev = true)
end

"""Finished hypotheses of a beam search: `BeamHypotheses` with length penalty 0."""
mutable struct Hypotheses
    beams::Vector{Tuple{Float32,Vector{Int}}}
    worst::Float32
    n::Int
end
Hypotheses(n) = Hypotheses(Tuple{Float32,Vector{Int}}[], 1f9, n)

function add!(h::Hypotheses, score::Float32, tokens::Vector{Int})
    if length(h.beams) < h.n || score > h.worst
        push!(h.beams, (score, tokens))
        if length(h.beams) > h.n
            sort!(h.beams; by = first)
            popfirst!(h.beams)
            h.worst = h.beams[1][1]
        else
            h.worst = min(score, h.worst)
        end
    end
    return h
end

"""`BeamHypotheses.is_done` without early stopping, length penalty 0."""
isdone(h::Hypotheses, best::Float32) = length(h.beams) >= h.n && h.worst >= best

"""Prefix row text: 0 and 1 dropped, then `[0, text..., 1]` (`prepare_gpt_inputs`)."""
prefixids(ids::AbstractVector{<:Integer}) = Int64[0; filter(t -> t != 0 && t != 1, ids); 1]


# Prefill, cache seeding and its logits readback are one recorded graph. All
# changing inputs are queued stores, so replay does not allocate or submit input
# uploads separately. These plans have the same model-owned lifetime as call's.
function prefilldecode(gpt::GPT, dec::Decoder, text, lang, style, emovec)
    dev = gpt.prefill.device
    data = (reshape(style, :, 1), reshape(emovec, :, 1), reshape(text, :, 1), Int64[lang])
    key = (:generate_prefill, length(text), dec.beams)
    mp = get!(gpt.prefill.scratch, key) do
        g = Mantle.Graph(dev)
        aten = gpt.prefill.graphs["prefill"]
        inputs = map(a -> Mantle.Buffer(dev, eltype(a), size(a)), data)
        resources = emitgraph!(g, gpt.prefill, "prefill"; dims=(; n=length(text)),
                               inputs=Dict(zip(aten.inputs, inputs)))
        logits, K, V = map(id -> resources[id], Tuple(aten.outputs))
        P = seedcache!(g, dec, K, V)
        Mantle.Readback(g, logits)
        plan = Mantle.record!(Mantle.Plan(g; budget=Inf))
        DNNKernels.RecordedPlan(plan, inputs, (; logits, prefix=P), Any[inputs...], true)
    end
    for (dst, src) in zip(mp.inputs, data)
        dst[:] = vec(src)
    end
    future = Mantle.Readback(mp.plan.graph, mp.outputs.logits)
    Mantle.run!(mp.plan)
    return fetch(future), mp.outputs.prefix
end

"""
    generate(gpt, ids, lang, style, emovec, s::Sampling) -> Vector{Int}

The codes of one segment, the stop token excluded: prefill over the prefix, then
one [`Decoder`](@ref) step per token on `num_beams` rows. `ids` are the segment's
token ids as `encode` gives them (upstream's trailing 1 included or not; prefix
building drops it).
"""
function generate(gpt::GPT, ids::AbstractVector{<:Integer}, lang::Integer,
                  style::AbstractVector{Float32}, emovec::AbstractVector{Float32}, s::Sampling)
    dev = gpt.prefill.device
    text = prefixids(ids)
    B = s.num_beams
    dec = decoder(gpt, B)
    logits, P = prefilldecode(gpt, dec, text, lang, style, emovec)
    rows = [copy(vec(logits)) for _ in 1:B]
    hist = [Int[] for _ in 1:B]
    seen(b) = union(Set([1, START_MEL]), hist[b])
    score = Float32[b == 1 ? 0f0 : -1f9 for b in 1:B]
    hyps = Hypotheses(B)
    minkeep = B > 1 ? 2 : 1
    draws = 0
    j = 0
    local out
    while true
        if B == 1
            x = penalize!(rows[1], seen(1), s.penalty)
            if s.do_sample
                warp!(x, s, minkeep)
                tok = race(softmax(x), 1, s.seed, draws)[1] - 1
                draws += 1
            else
                tok = argmax(x) - 1
            end
            if tok == STOP_MEL || j + 1 >= s.max_tokens
                tok == STOP_MEL || push!(hist[1], tok)
                out = hist[1]
                break
            end
            push!(hist[1], tok)
            nexttok, beamidx = [tok], [0]
        else
            scores = Float32[]
            for b in 1:B
                x = logsoftmax!(rows[b])
                penalize!(x, seen(b), s.penalty)
                s.do_sample && warp!(x, s, minkeep)
                append!(scores, x .+ score[b])
            end
            if s.do_sample
                cand = race(softmax(scores), 2B, s.seed, draws)
                draws += 1
                cs = scores[cand]
                o = sortperm(cs; rev = true)
                cand, cs = cand[o], cs[o]
            else
                cand = partialsortperm(scores, 1:(2B); rev = true)
                cs = scores[cand]
            end
            nb, nt, ni = Float32[], Int[], Int[]
            for (rank, (c, sc)) in enumerate(zip(cand, cs))
                beam, tok = divrem(c - 1, VOCAB_MEL)
                if tok == STOP_MEL
                    rank <= B && add!(hyps, sc, copy(hist[beam + 1]))
                else
                    push!(nb, sc); push!(nt, tok); push!(ni, beam)
                end
                length(nb) == B && break
            end
            done = isdone(hyps, maximum(cs))
            hist = [vcat(hist[ni[b] + 1], nt[b]) for b in 1:B]
            score = nb
            if done || j + 1 >= s.max_tokens
                done || foreach(b -> add!(hyps, score[b], copy(hist[b])), 1:B)
                out = last(sort(hyps.beams; by = first))[2]
                break
            end
            nexttok, beamidx = nt, ni
        end
        j += 1
        slot = P - 1 + j
        slot < dec.rows || break
        lh = step!(dec, nexttok, beamidx, j + 1, slot; cpu=true)
        rows = [lh[:, b] for b in 1:B]
    end
    return out
end
