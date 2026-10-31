# The reference clips: the speaker clip conditions the GPT (CAMPPlus style), the
# flow matching (prompt features and mel) and, unless a second clip, a vector or
# text says otherwise, the emotion; an emotion clip only goes through w2v-BERT and
# the emotion encoder.

"""
    Voice

One speaker clip, conditioned once and reused for every utterance in its voice:

- `cond`: w2v-BERT's normalised layer-17 features, `(1024, t, 1)` (`spk_cond_emb`);
- `style`: the CAMPPlus embedding, 192 floats;
- `prompt`: the length regulator over `cond`, stretched to the mel's frames,
  `(512, frames, 1)` (`prompt_condition`);
- `mel`: the clip's 22.05 kHz log mel, `(frames, 80, 1)` (`ref_mel`);
- `base`: the emotion encoder over `cond`, 1280 floats (`get_emovec(spk_cond_emb)`);
- `own`: the emotion encoder over the same file loaded at 16 kHz directly, which
  is what upstream uses as the emotion clip when none is given. The two loads
  differ (22.05 kHz then a sinc to 16 kHz, against libsoxr straight to 16 kHz),
  and so do the two vectors, by about 2% of their range.
"""
mutable struct Voice{B,P,M,L}
    cond::B
    style::Vector{Float32}
    prompt::P
    mel::M
    base::Vector{Float32}
    own::Vector{Float32}
    lease::L
end

# A compiled voice graph owns its output buffers. A live Voice leases one slot,
# so another clip cannot overwrite it. Weak ownership also returns a dropped
# voice without needing GPU work in a finalizer.
Voice(cond, style, prompt, mel, base, own; lease=nothing) =
    Voice(cond, style, prompt, mel, base, own, lease)

frames(v::Voice) = size(v.mel, 1)

function Base.close(v::Voice)
    if v.lease === nothing
        foreach(Mantle.free!, (v.cond, v.prompt, v.mel))
    elseif v.lease[].value === v
        v.lease[] = WeakRef(nothing)
    end
    return nothing
end

"""
    nearestindex(n, m) -> Vector{Int64}

ATen's `nearest_idx` for `upsample_nearest1d` from `n` frames to `m`: identity,
`i >> 1` for exactly twice, else `min(floorf(i * (float)n / m), n - 1)`, all in
float32 as the kernel computes it. Zero-based, as the regulator graph indexes.
"""
function nearestindex(n::Int, m::Int)
    m == n && return collect(Int64, 0:(m - 1))
    m == 2n && return Int64[i >> 1 for i in 0:(m - 1)]
    scale = Float32(n) / Float32(m)
    return Int64[min(floor(Int64, Float32(i) * scale), n - 1) for i in 0:(m - 1)]
end

"""
    emotionfeatures!(g, tts, x16, n16) -> 1280 emotion vector transient

Declare SeamlessM4T features of a 16 kHz signal, w2v-BERT and the emotion encoder.
"""
function emotionfeatures!(g, tts, x16, n16::Int, out)
    N = tts.networks
    feats, mask, t = seamlessfeatures!(g, tts.frontend, x16, n16)
    cond = emit!(g, N.w2v, (; t), feats, mask)
    return emit!(g, N.emo, (; t), cond; out)
end

"""
    voice(tts, path) -> Voice
    voice(tts, audio::Audio) -> Voice
    voice(tts, a22, a16) -> Voice

The speaker clip's conditioning as one graph: resampling, both Kaldi front ends,
the mel, w2v-BERT, CAMPPlus, the emotion encoder and the length regulator, and
the same file's own emotion from its direct 16 kHz load. A path or a decoded
`Audio` at its own rate is loaded as upstream loads it (`loadaudio`: libsoxr to
22.05 and to 16 kHz, 15 s at most each).
"""
voice(tts, path::AbstractString) = voice(tts, readaudio(path))
voice(tts, a::Audio) = voice(tts, cut(soxr(a, 22050)), cut(soxr(a, 16000)))

function voice(tts, a22::Audio, a16::Audio)
    a22.rate == 22050 && a16.rate == 16000 ||
        throw(ArgumentError("voice: expected 22050 and 16000 Hz audio, got $(a22.rate) and $(a16.rate)"))
    mp = voiceplan(tts, length(a22), length(a16))
    inputs, outputs = mp.inputs, mp.outputs
    inputs.x22[:] = a22.samples
    inputs.y16[:] = a16.samples
    futures = map(out -> Mantle.Readback(mp.plan.graph, out), outputs.cpu)
    Mantle.run!(mp.plan)
    s, b, o = map(f -> vec(fetch(f)), futures)
    v = Voice(outputs.cond, s, outputs.prompt, outputs.mel, b, o; lease=inputs.lease)
    inputs.lease[] = WeakRef(v)
    return v
end


# One recorded graph per shape and concurrent Voice. Reuse commands and storage,
# never conditioning values: each call stages both fresh decoded input signals.
function voiceplan(tts, n22::Int, direct16::Int)
    cache = tts.networks.w2v.scratch
    for slot in Iterators.countfrom(1)
        key = (:voice, n22, direct16, slot)
        mp = get(cache, key, nothing)
        if mp === nothing
            mp = buildvoiceplan(tts, n22, direct16)
            cache[key] = mp
            return mp
        elseif mp.inputs.lease[].value === nothing
            return mp
        end
    end
end

function buildvoiceplan(tts, n22::Int, direct16::Int)
    dev, fe, N = tts.device, tts.frontend, tts.networks
    nm = s2melframecount(n22)
    g = Mantle.Graph(dev)
    x22 = Mantle.Buffer(dev, Float32, n22)
    x16, n16 = resample16k!(g, fe, x22, n22)
    feats, mask, t = seamlessfeatures!(g, fe, x16, n16)
    fbank, nf = kaldifbank!(g, fe, x16, n16)
    mel = Mantle.Buffer(dev, Float32, (nm, 80, 1))
    s2melmel!(g, fe, x22, n22, mel)
    cond = outputbuffer(N.w2v, (; t))
    emit!(g, N.w2v, (; t), feats, mask; out = cond)
    style = outputbuffer(N.campplus, (; t = nf))
    emit!(g, N.campplus, (; t = nf), fbank; out = style)
    base = outputbuffer(N.emo, (; t))
    emit!(g, N.emo, (; t), cond; out = base)
    index = Mantle.Buffer(dev, nearestindex(t, nm))
    prompt = outputbuffer(N.regulator, (; t, y = nm))
    emit!(g, N.regulator, (; t, y = nm), cond, index; out = prompt)
    y16 = Mantle.Buffer(dev, Float32, direct16)
    own = outputbuffer(N.emo, (; t))
    emotionfeatures!(g, tts, y16, direct16, own)
    outputs = (; cond, prompt, mel, cpu=(style, base, own))
    foreach(out -> Mantle.Readback(g, out), outputs.cpu)
    plan = Mantle.record!(Mantle.Plan(g; budget=Inf))
    inputs = (; x22, y16, lease=Ref(WeakRef(nothing)))
    owned = Any[x22, y16, index, style, base, own, cond, prompt, mel]
    return DNNKernels.RecordedPlan(plan, inputs, outputs, owned, true)
end

"""
    emotionclip(tts, path) -> Vector{Float32}
    emotionclip(tts, audio::Audio) -> Vector{Float32}

An emotion clip's `get_emovec`, from the file loaded at 16 kHz directly, as
upstream does (`_load_and_cut_audio(..., sr = 16000)`): SeamlessM4T features,
w2v-BERT, the emotion encoder.
"""
emotionclip(tts, path::AbstractString) = emotionclip(tts, readaudio(path))

function emotionclip(tts, a::Audio)
    a16 = a.rate == 16000 ? a : cut(soxr(a, 16000))
    key = (:emotionclip, length(a16))
    mp = get!(tts.networks.emo.scratch, key) do
        g = Mantle.Graph(tts.device)
        x16 = Mantle.Buffer(tts.device, Float32, length(a16))
        out = emotionfeatures!(g, tts, x16, length(a16), nothing)
        Mantle.Readback(g, out)
        plan = Mantle.record!(Mantle.Plan(g; budget=Inf))
        DNNKernels.RecordedPlan(plan, (x16,), (out,), Any[x16], true)
    end
    only(mp.inputs)[:] = a16.samples
    future = Mantle.Readback(mp.plan.graph, only(mp.outputs))
    Mantle.run!(mp.plan)
    return vec(fetch(future))
end

"""Upstream's eight emotions, in the order its vectors and matrices use."""
const EMOTIONS8 = (:happy, :angry, :sad, :afraid, :disgusted, :melancholic, :surprised, :calm)

"""Rows of `feat1.pt` / `feat2.pt` per emotion (`emo_num`)."""
const EMOTION_ROWS = (3, 17, 2, 8, 4, 5, 10, 24)

"""
    emotionmix(tts, v, weights, own) -> Vector{Float32}

`infer_generator`'s emotion-vector path without the `emo_alpha` scaling: for
each emotion, the row of its block of `spk_matrix` most similar (cosine) to the
speaker's style, then `Σ w_i emo_matrix[row_i] + (1 - Σ w) · own`.
"""
function emotionmix(tts, v::Voice, weights::AbstractVector{<:Real}, own::Vector{Float32})
    length(weights) == 8 || throw(ArgumentError("an emotion vector has 8 weights ($(join(EMOTIONS8, ", ")))"))
    w = Float32.(weights)
    mix = zeros(Float32, size(tts.emomatrix, 1))
    base = 0
    for (i, n) in enumerate(EMOTION_ROWS)
        block = (base + 1):(base + n)
        sims = [cosine(v.style, view(tts.spkmatrix, :, r)) for r in block]
        row = block[argmax(sims)]
        mix .+= w[i] .* view(tts.emomatrix, :, row)
        base += n
    end
    return mix .+ (1f0 - sum(w)) .* own
end

"""`F.cosine_similarity` of two vectors, eps 1e-8."""
cosine(a, b) = sum(a .* b) / sqrt(max(sum(abs2, a) * sum(abs2, b), 1f-16))

"""`merge_emovec`: `base + alpha (emo - base)`, float32."""
mergeemotion(base::Vector{Float32}, emo::AbstractVector{Float32}, alpha::Real) = base .+ Float32(alpha) .* (emo .- base)

"""
    emovector(tts, v; emotion, alpha, weights) -> Vector{Float32}

The GPT's emotion input (`emovec`), by upstream's precedence:

- `weights` (8 floats; or text through [`emotiontext`](@ref)) wins over a clip.
  They are scaled by `clamp(alpha, 0, 1)` and truncated to four decimals when
  that is not 1, then mixed by [`emotionmix`](@ref) with the speaker's own
  emotion.
- else an `emotion` clip (path, `Audio`, or its [`emotionclip`](@ref) vector),
  blended in by `alpha`;
- else the speaker clip's own emotion, `alpha` forced to 1.
"""
function emovector(tts, v::Voice; emotion = nothing, alpha::Real = 1.0, weights = nothing)
    own = mergeemotion(v.base, v.own, 1)
    if weights !== nothing
        s = clamp(Float64(alpha), 0.0, 1.0)
        w = s == 1.0 ? Float64.(weights) : [trunc(Int, x * s * 10000) / 10000 for x in weights]
        return emotionmix(tts, v, w, own)
    end
    emotion === nothing && return own
    e = emotion isa AbstractVector{Float32} ? emotion : emotionclip(tts, emotion)
    return mergeemotion(v.base, e, alpha)
end
