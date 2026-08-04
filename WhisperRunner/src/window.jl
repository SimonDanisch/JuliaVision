# Audio in, transcript out: the 30 s window loop.
#
# Ported from whisper.cpp's `whisper_full_with_state`
# (`dev/whisper.cpp/src/whisper.cpp`, MIT), itself openai/whisper's
# `transcribe.py`. The model only ever sees 30 s; everything about transcribing
# an hour is here.
#
# ## Why the window does not simply advance by 30 s
#
# It advances to the **last completed segment's end timestamp**. Whisper's own
# output tells us where it stopped understanding, and a fixed 30 s hop would cut
# through the middle of a word every time — the model then sees half a word at
# the end of one window and half at the start of the next, and hallucinates
# across both. If a window produces no usable timestamp at all the hop falls back
# to the full 30 s, which is the only way to guarantee progress.

"""
    Segment

One timestamped run of text, in seconds from the start of the audio.
"""
struct Segment
    start::Float64
    stop::Float64
    text::String
    tokens::Vector{Int}
end

const SAMPLERATE = 16000
const WINDOWSECONDS = 30.0
const WINDOWSAMPLES = SAMPLERATE * 30            # 480000
const NFRAMES = 3000                             # mel frames per window
const NMELS = 128

# whisper.cpp's two tail rules (`whisper_full_with_state`), both of which exist
# because the *last* window of a file is mostly zero padding and the model
# invents speech to fill it.
#
#   * stop with 100 ms or less left (`seek + delta_min >= seek_end`). There is
#     nothing there.
#   * with 5 s or less left, decode WITHOUT the previous window's context. The
#     reference's own comment: it "tends to confuse the decoder and often make it
#     repeat or hallucinate stuff". Measured here on `jfk.wav`, whose last 0.6 s
#     are applause: with the context prompt the tail window produces `Yes!`.
const MINTAIL = SAMPLERATE ÷ 10                  # 100 ms
const NOCONTEXTTAIL = SAMPLERATE * 5             # 5 s

"""
    melwindow(w, audio, offset) -> (3000, 128, 1)

The log-mel for one 30 s window starting at sample `offset`, zero-padded when the
audio runs out.

Padding rather than a shorter window: every extent in the encoder graph is baked,
so a partial final window has to be filled. Whisper is trained with exactly this
padding, so the model expects it.
"""
function melwindow(w::Whisper, audio::AbstractVector{<:Real}, offset::Int)
    chunk = zeros(Float32, WINDOWSAMPLES)
    lo = offset + 1
    hi = min(length(audio), offset + WINDOWSAMPLES)
    lo <= hi && copyto!(chunk, 1, Float32.(@view audio[lo:hi]), 1, hi - lo + 1)
    a = KA.allocate(w.backend, Float32, WINDOWSAMPLES)
    copyto!(a, chunk)
    ctx = Ctx(w.backend)
    mel = logmelspectrogram(ctx, a, w.melfilters)     # (nmels, frames)
    host = Array(mel)
    m = KA.allocate(w.backend, Float32, NFRAMES, NMELS, 1)
    # (nmels, frames) -> the encoder's (frames, nmels, 1)
    copyto!(m, reshape(permutedims(host[:, 1:NFRAMES]), NFRAMES, NMELS, 1))
    return m
end

"""
    detectlanguage(w, tk) -> (code, probability)

One decoder step from `<|startoftranscript|>`, argmax restricted to the language
tokens.

Requires [`fillcross!`](@ref) to have run for this window. Exactly what
whisper.cpp's `whisper_lang_auto_detect` does, and it costs a single token.
"""
function detectlanguage(w::Whisper, tk::Tokenizer)
    reset!(w.cache)
    logits = Array(vec(decodestep!(w, tk.sot)))
    best, bestp = "en", -Inf32
    for (code, id) in tk.langs
        0 <= id < length(logits) || continue
        v = logits[id + 1]
        v > bestp && ((best, bestp) = (code, v))
    end
    ls = logsoftmax!(copy(logits))
    return best, exp(Float64(ls[tk.langs[best] + 1]))
end

"""
    decodewindow(w, tk, opts, prompttokens, temperature, rng)
        -> (tokens, avglogprob, nospeechprob)

Decode one window at one temperature, with the policy applied at every step.

`nospeechprob` is read from the **first** step only — the probability the model
assigns to `<|nospeech|>` when it has seen nothing but the prompt, which is
whisper.cpp's definition and the reason it is captured before any filtering:
suppression would zero the very token being measured.
"""
function decodewindow(w::Whisper, tk::Tokenizer, opts::DecodeOptions,
                      prompttokens::AbstractVector{<:Integer}, temperature::Real,
                      rng, suppress, beginsuppress)
    reset!(w.cache)
    logits = nothing
    for t in prompttokens
        logits = decodestep!(w, t)
    end
    generated = Int[]
    sumlogprob = 0.0
    nospeech = 0.0
    nosp = get(tk.special, "<|nospeech|>", get(tk.special, "<|nocaptions|>", nothing))
    first = true
    while length(generated) < opts.maxtokens
        l = Array(vec(logits))
        if first && nosp !== nothing
            ls = logsoftmax!(copy(l))
            nospeech = exp(Float64(ls[nosp + 1]))
            first = false
        end
        applypolicy!(l, tk, opts, generated, suppress, beginsuppress)
        logsoftmax!(l)
        id = sample(l, temperature, rng)
        id == tk.eot && break
        sumlogprob += Float64(l[id + 1])
        push!(generated, id)
        w.cache.position >= size(w.cache.self_k, 2) && break
        logits = decodestep!(w, id)
    end
    avg = isempty(generated) ? -Inf : sumlogprob / length(generated)
    return generated, avg, nospeech
end

"""
    segments(tk, tokens, offset) -> (Vector{Segment}, advance_seconds)

Split a window's tokens on timestamp pairs.

`advance_seconds` is where the next window starts: the end of the last *closed*
segment. A window whose final segment never closed contributes its text but not
its boundary — that text will be produced again by the next window, which is the
behaviour that keeps a sentence intact across the seam.
"""
function segments(tk::Tokenizer, tokens::AbstractVector{<:Integer}, offset::Float64)
    out = Segment[]
    advance = 0.0
    i = 1
    n = length(tokens)
    while i <= n
        istimestamp(tk, tokens[i]) || (i += 1; continue)
        tstart = timestampseconds(tk, tokens[i])
        j = i + 1
        while j <= n && !istimestamp(tk, tokens[j])
            j += 1
        end
        if j > n                       # segment left open at the window edge
            break
        end
        tstop = timestampseconds(tk, tokens[j])
        body = collect(tokens[(i + 1):(j - 1)])
        text = strip(decode(tk, body))
        isempty(text) || push!(out, Segment(offset + tstart, offset + tstop, text, body))
        advance = max(advance, tstop)
        i = j                          # a closing stamp opens the next segment
    end
    return out, advance
end

"""
    transcribechunk(w, tk, audio; language, task, options, context, rng) -> (text, tokens, avglogprob)

**One** window, no loop, no timestamps by default — the streaming entry point.

`transcribe` is the wrong shape for a live microphone: it runs the 30 s window
loop with the temperature ladder and the timestamp-driven advance, all of which
assume the audio is finished and on disk. A stream instead wants "here is the
last few seconds, what does it say", called again a moment later on overlapping
audio.

`audio` is padded to 30 s by [`melwindow`](@ref); anything up to that length
works, and shorter is not cheaper — the encoder's extents are baked, so a 3 s
chunk costs exactly what a 30 s one does. That is what sets the streaming step
size, not the decoder.

Timestamps default off, matching whisper.cpp's own sliding-window mode
(`params.no_timestamps = !use_vad`): a partial window's timestamps are relative
to a boundary that is about to move, so they mislead more than they inform.
"""
function transcribechunk(w::Whisper, tk::Tokenizer, audio::AbstractVector{<:Real};
                         language::AbstractString = "en", task::Symbol = :transcribe,
                         options::DecodeOptions = DecodeOptions(timestamps = false),
                         context::AbstractVector{<:Integer} = Int[],
                         rng = Random.default_rng())
    m = melwindow(w, audio, 0)
    hid, = call(w.encoder, "whisper", m; dims = (;))
    fillcross!(w, hid)
    pr = prompt(tk; language, task, timestamps = options.timestamps, context)
    toks, avg, _ = decodewindow(w, tk, options, pr, 0.0f0, rng,
                                w.suppress, w.beginsuppress)
    return strip(decode(tk, toks)), toks, avg
end

"""
    transcribe(w, audio; language, task, options, rng) -> (text, segments)

The whole thing: samples at 16 kHz in, text and timestamped segments out.

The loop, per window:

 1. mel, encoder, cross-attention K/V — the expensive part, once per window.
 2. decode at temperature 0. If the result fails a quality test, decode again at
    the next temperature. The ladder is `0, 0.2, … 1.0` and the tests are
    whisper.cpp's: **entropy below 2.4** (the model is looping) or **average
    log-probability below -1.0** (it is not confident). Both are computed on the
    result, so a fallback costs a full re-decode and is why temperature 0 is
    tried first.
 3. If `<|nospeech|>` beats 0.6 *and* the log-probability is still poor, the
    window is silence: emit nothing and skip a full 30 s. Both conditions,
    because a confident transcript over a low-energy passage is speech.
 4. Split on timestamp pairs, advance to the last closed segment.
 5. Carry this window's tokens into the next window's prompt.

`language = nothing` detects it from the first window, once, and then holds it
fixed — Whisper will otherwise switch language mid-file on an ambiguous window.
"""
function transcribe(w::Whisper, audio::AbstractVector{<:Real};
                    language::Union{Nothing,AbstractString} = nothing,
                    task::Symbol = :transcribe,
                    options::DecodeOptions = DecodeOptions(),
                    tk::Tokenizer = tokenizer(),
                    rng = Random.default_rng(),
                    verbose::Bool = false)
    allsegs = Segment[]
    context = Int[]
    lang = language
    offset = 0                                    # in samples
    while offset < length(audio)
        # Nothing but padding left.
        length(audio) - offset <= MINTAIL && break
        m = melwindow(w, audio, offset)
        hid, = call(w.encoder, "whisper", m; dims = (;))
        fillcross!(w, hid)
        toffset = offset / SAMPLERATE

        if lang === nothing
            lang, p = detectlanguage(w, tk)
            verbose && @info "detected language" language=lang probability=p
        end

        # A short tail decodes without the previous window's words. See
        # NOCONTEXTTAIL — this is the rule that removes the `Yes!` jfk.wav's
        # applause otherwise produces.
        tailonly = offset > 0 && length(audio) - offset <= NOCONTEXTTAIL
        ctxnow = tailonly ? Int[] : context

        toks, avg, nospeech = Int[], -Inf, 0.0
        for (k, temp) in enumerate(options.temperatures)
            pr = prompt(tk; language = lang, task, timestamps = options.timestamps,
                        context = ctxnow)
            toks, avg, nospeech = decodewindow(w, tk, options, pr, temp, rng,
                                               w.suppress, w.beginsuppress)
            ent = tokenentropy(toks)
            ok = !(length(toks) > 32 && ent < options.compressionthreshold) &&
                 !(avg < options.logprobthreshold)
            verbose && @info "window" t=round(toffset; digits=2) temperature=temp ntokens=length(toks) avglogprob=round(avg; digits=3) entropy=round(ent; digits=3) accepted=ok
            ok && break
            k == length(options.temperatures) && break
        end

        if nospeech > options.nospeechthreshold && avg < options.logprobthreshold
            verbose && @info "silence" t=round(toffset; digits=2) nospeech=round(nospeech; digits=3)
            offset += WINDOWSAMPLES
            continue
        end

        segs, advance = segments(tk, toks, toffset)
        append!(allsegs, segs)
        # Context is the *text* of this window, without timestamps: the prompt is
        # meant to carry wording, and feeding timestamps back makes the model
        # believe the new window starts where the old one did.
        context = isempty(segs) ? Int[] : reduce(vcat, s.tokens for s in segs)
        offset += advance > 0 ? max(1, round(Int, advance * SAMPLERATE)) : WINDOWSAMPLES
    end
    text = join((s.text for s in allsegs), " ")
    return text, allsegs
end

"""
    transcribe(w, path::AbstractString; kw...)

Same, reading the audio with `VideoIO`/`FFMPEG` and resampling to 16 kHz mono —
the format Whisper's front end is defined on. Anything ffmpeg can open works.
"""
function transcribe(w::Whisper, path::AbstractString; kw...)
    isfile(path) || throw(ArgumentError("no such audio file: $path"))
    return transcribe(w, readaudio(path); kw...)
end

"""
    readaudio(path; rate = 16000) -> Vector{Float32}

Decode any container to mono float samples at `rate`, through ffmpeg.

Raw `f32le` on stdout rather than a temporary wav: no header to parse, no file to
clean up, and the sample format is stated rather than discovered.
"""
function readaudio(path::AbstractString; rate::Int = SAMPLERATE)
    out = IOBuffer()
    cmd = `$(FFMPEG_jll.ffmpeg()) -v quiet -i $path -f f32le -acodec pcm_f32le -ac 1 -ar $rate -`
    run(pipeline(cmd; stdout = out))
    return reinterpret(Float32, take!(out)) |> collect
end
