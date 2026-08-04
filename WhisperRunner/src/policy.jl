# The decoding policy: which token a step is allowed to pick, and whether the
# result is worth keeping.
#
# Ported from whisper.cpp's `whisper_process_logits` and the fallback loop in
# `whisper_full_with_state` (`dev/whisper.cpp/src/whisper.cpp`, MIT), which are
# themselves a port of openai/whisper's `decoding.py` — every rule below carries
# the reference's own line reference in its comment.
#
# **This is not decoration on top of a working decoder, it is most of what makes
# a transcript readable.** Without it the model emits `<|notimestamps|>` mid
# sequence, opens a segment it never closes, repeats a phrase until the token cap
# and reports silence as a hallucinated sentence. `greedy` in `decode.jl` is
# still there and still exact against a pure-argmax reference; this is the layer
# that the *reference itself* is not.

"""
    DecodeOptions

The policy constants. Defaults are whisper.cpp's own (`whisper_full_default_params`),
which are openai/whisper's.

`temperatures` is the fallback ladder: decode at 0, and only if the result fails
one of the quality tests decode again hotter. That ordering matters — a greedy
pass is deterministic and usually right, and sampling is a recovery mechanism for
the cases where it collapses into a loop.

**`nospeechthreshold` is measured to be inert on large-v3-turbo.** The gate is
`p(<|nospeech|>) > 0.6 AND avg logprob < -1.0`, and the first half never fires
here: at the position openai/whisper reads it — the distribution predicting the
token after `<|startoftranscript|>` — this checkpoint puts **0.000000** on
`<|nospeech|>` (id 50363), on silence and on speech alike. Measured on both
windows of `jfk.wav`, where after the SOT the mass is 0.53 on `<|transcribe|>`
and 0.44 on `<|en|>`. Distillation appears to have trained the token out.

It is kept, at the reference's value, because it costs nothing and a non-distilled
checkpoint will use it. What actually rejects a bad window here is
`logprobthreshold`, and the consequence is worth knowing: a short tail of applause
decodes to a confident-looking hallucination (`jfk.wav`'s last 0.6 s give
`"Yeah!"` at avg -1.19), the temperature ladder retries and fails, and the text is
emitted anyway. whisper.cpp does the same thing on the same input — line 7576
fails the decoder when `avg < thold && no_speech < thold`, and still keeps the
best result. HuggingFace avoids it only by never decoding a second window for
audio under 30 s.
"""
Base.@kwdef struct DecodeOptions
    temperatures::Vector{Float32} = Float32[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    compressionthreshold::Float64 = 2.4    # `entropy_thold`; a repetition detector
    logprobthreshold::Float64 = -1.0
    nospeechthreshold::Float64 = 0.6       # inert on turbo; see above
    timestamps::Bool = true
    maxinitialtimestamp::Float64 = 1.0
    suppressblank::Bool = true
    maxtokens::Int = 224
end

"""
    suppresslists(dir) -> (suppress, beginsuppress)

The two suppression lists from `generation_config.json`.

`suppress_tokens` is a fixed list of ~90 ids — punctuation-only tokens, musical
note glyphs, and other things Whisper was trained never to emit — banned at every
step. `begin_suppress_tokens` is `[220, 50257]`: a leading space and
end-of-text, banned only as the *first* generated token, so a window cannot
immediately give up or start with whitespace.

Read from the file rather than hard-coded: they are model-specific, and
`.en`-only checkpoints ship a different list.
"""
function suppresslists(dir::AbstractString)
    p = joinpath(dir, "generation_config.json")
    isfile(p) || return (Int[], Int[])
    g = JSON3.read(read(p, String))
    sup = haskey(g, :suppress_tokens) ? Int.(collect(g[:suppress_tokens])) : Int[]
    beg = haskey(g, :begin_suppress_tokens) ? Int.(collect(g[:begin_suppress_tokens])) : Int[]
    return (sup, beg)
end

"""
    applypolicy!(logits, tk, opts, generated, suppress, beginsuppress) -> logits

Every rule whisper.cpp applies between the model and the argmax, in its order.

`logits` is the host copy for one step, 0-based token `i` at index `i + 1`.
`generated` is what this window has produced so far, *excluding* the prompt —
which is what "initial" means below.

The rules, in the reference's order:

 1. **suppress blank** (`decoding.py` L388) — at the first generated token only,
    ban `<|endoftext|>` and a leading space.
 2. **suppress the fixed list** — `suppress_tokens`, always.
 3. **ban `<|notimestamps|>`** (L410) — it is a *prompt* token; the model must
    never generate it. Whisper.cpp does this unconditionally and so do we.
 4. **ban `<|startoftranscript|>` and `<|startofprev|>`** — likewise prompt-only.
 5. **timestamp pairing** (L414-L424) — this is the rule that makes segments
    well-formed. If the last token was a timestamp: when the one before it was
    *also* a timestamp the pair just closed, so the next token must be text;
    otherwise a segment is open and the next token must be a timestamp. Getting
    this backwards produces a transcript whose segment boundaries drift.
 6. **max initial timestamp** (L426) — the first timestamp cannot be later than
    1.0 s, so a window cannot open by skipping a second of audio.
 7. **monotonic timestamps** — no timestamp earlier than the one already emitted.
 8. **timestamp mass beats text mass** (L431) — if the *total* probability over
    all timestamps exceeds the single best text token, force a timestamp. Total,
    not maximum: the mass is spread over 1501 neighbouring times, and comparing
    maxima would essentially never fire.
"""
function applypolicy!(logits::Vector{Float32}, tk::Tokenizer, opts::DecodeOptions,
                      generated::AbstractVector{<:Integer},
                      suppress::AbstractVector{<:Integer},
                      beginsuppress::AbstractVector{<:Integer})
    n = length(logits)
    tb = tk.timestampbegin
    ninf = -Inf32
    at(id) = id + 1                              # 0-based token -> 1-based index

    if opts.suppressblank && isempty(generated)
        for id in beginsuppress
            0 <= id < n && (logits[at(id)] = ninf)
        end
    end
    for id in suppress
        0 <= id < n && (logits[at(id)] = ninf)
    end
    logits[at(tk.notimestamps)] = ninf
    logits[at(tk.sot)] = ninf
    logits[at(tk.prev)] = ninf

    if !opts.timestamps
        for i in at(tb):n
            logits[i] = ninf
        end
    else
        lastts = !isempty(generated) && generated[end] >= tb
        # `length < 2` counts as "penultimate was a timestamp" exactly as the
        # reference does: at the very start a lone timestamp closes nothing.
        penults = length(generated) < 2 || generated[end - 1] >= tb
        if lastts
            if penults
                for i in at(tb):n; logits[i] = ninf; end          # pair closed -> text
            else
                for i in 1:at(tk.eot) - 1; logits[i] = ninf; end  # open -> must close
            end
        end
        if isempty(generated) && opts.maxinitialtimestamp > 0
            lastallowed = tb + round(Int, opts.maxinitialtimestamp / 0.02)
            for i in at(lastallowed + 1):n
                logits[i] = ninf
            end
        end
        # monotonic: never go back before the newest timestamp already emitted
        newest = 0
        for t in generated
            t >= tb && (newest = max(newest, Int(t)))
        end
        if newest > 0
            for i in at(tb):at(newest) - 1
                logits[i] = ninf
            end
        end
        # timestamp mass vs the best text token, in log space
        mx = maximum(logits)
        isfinite(mx) || return logits
        tsmax = ninf
        for i in at(tb):n
            logits[i] > tsmax && (tsmax = logits[i])
        end
        if isfinite(tsmax)
            acc = 0.0
            for i in at(tb):n
                isfinite(logits[i]) && (acc += exp(Float64(logits[i] - tsmax)))
            end
            tslogsum = log(acc) + Float64(tsmax)
            txtmax = ninf
            for i in 1:at(tb) - 1
                logits[i] > txtmax && (txtmax = logits[i])
            end
            if tslogsum > Float64(txtmax)
                for i in 1:at(tb) - 1
                    logits[i] = ninf
                end
            end
        end
    end
    return logits
end

"""
    logsoftmax!(x) -> x

In place, numerically stable. `-Inf` entries stay `-Inf` rather than becoming
`NaN`, which is what a naive `x .- logsumexp(x)` would do to every suppressed
token and what would then poison the sequence's average log-probability.
"""
function logsoftmax!(x::Vector{Float32})
    mx = maximum(x)
    isfinite(mx) || return x
    acc = 0.0
    for v in x
        isfinite(v) && (acc += exp(Float64(v - mx)))
    end
    lse = Float32(log(acc)) + mx
    @inbounds for i in eachindex(x)
        x[i] = isfinite(x[i]) ? x[i] - lse : -Inf32
    end
    return x
end

"""
    sample(logprobs, temperature, rng) -> id

Argmax at temperature 0, otherwise a draw from the softmax at that temperature.

0-based id out, because that is what the model's vocabulary is indexed by
everywhere else in this file.
"""
function sample(logprobs::Vector{Float32}, temperature::Real, rng)
    if temperature <= 0
        return argmax(logprobs) - 1
    end
    mx = maximum(logprobs)
    acc = 0.0
    for v in logprobs
        isfinite(v) && (acc += exp((Float64(v) - mx) / temperature))
    end
    r = rand(rng) * acc
    c = 0.0
    @inbounds for i in eachindex(logprobs)
        isfinite(logprobs[i]) || continue
        c += exp((Float64(logprobs[i]) - mx) / temperature)
        c >= r && return i - 1
    end
    return argmax(logprobs) - 1
end

"""
    tokenentropy(ids; window = 32) -> Float64

Shannon entropy of the id counts over the last `window` tokens.

**The repetition detector**, and whisper.cpp's own (`entropy_thold`, default
2.4): a window is rejected when the entropy falls *below* the threshold, having
generated more than `window` tokens.

The failure it catches is the model looping — "and then and then and then" —
which has a perfectly good average log-probability, because each repeated token
is confidently predicted, so the log-probability test cannot see it at all. A
loop has few distinct ids and therefore low entropy.

openai/whisper uses a gzip compression ratio for the same job, thresholded from
the other side. This is the whisper.cpp formulation, chosen because it needs no
compression library, and because it is computed on *ids* rather than on decoded
text, so it does not depend on the tokenizer being right.
"""
function tokenentropy(ids::AbstractVector{<:Integer}; window::Int = 32)
    isempty(ids) && return 0.0
    lo = max(1, length(ids) - window + 1)
    counts = Dict{Int,Int}()
    n = 0
    for i in lo:length(ids)
        counts[Int(ids[i])] = get(counts, Int(ids[i]), 0) + 1
        n += 1
    end
    e = 0.0
    for (_, c) in counts
        p = c / n
        e -= p * log(p)
    end
    return e
end
