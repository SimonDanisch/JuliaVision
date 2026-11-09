"""
IndexTTS-2.5 — zero-shot text to speech with the voice from one clip and the
emotion from another.

Bilibili's IndexTTS-2.5 (`IndexTeam/IndexTTS-2.5`), with the community German
checkpoint (`sharrnah/index-tts-2.5-german`) as a second GPT. Text goes through a
GPT-2 that writes semantic codes at 25 per second, conditioned on a CAMPPlus
speaker embedding and an emotion vector; a flow-matching DiT turns the codes into
a mel spectrogram in the reference speaker's voice, and BigVGAN v2 into 22.05 kHz
audio.

Upstream: https://github.com/index-tts/index-tts (d9e41aa)
License: bilibili Model Use License Agreement (code and weights); the German
checkpoint is a derivative under the same terms.
"""
module IndexTTS25Runner

using Mantle, Lava, DNNKernels, KernelAbstractions
using LazyArtifacts, JSON3, FFMPEG_jll, SoXResampler_jll
using Artifacts: artifact_hash, artifact_exists, artifact_path
using LazyArtifacts: ensure_artifact_installed
import Random

export IndexTTS25, Voice, Sampling, indextts25, speak, voice, emotionclip, emotiontext, preparegerman, SAMPLERATE

"""The output rate, BigVGAN v2's 22.05 kHz."""
const SAMPLERATE = 22050

include("audio.jl")
include("features.jl")
include("text.jl")
include("german.jl")
include("model.jl")
include("normalizer.jl")
include("voice.jl")
include("decode.jl")
include("gpt.jl")
include("s2mel.jl")
include("emotiontext.jl")

"""
    IndexTTS25

The loaded model: the networks, the GPT (base or German), the front-end tables,
the tokenizer and the speaker/emotion matrices of the emotion-vector path.
"""
struct IndexTTS25{D,M,G,F}
    device::D
    networks::Networks{M}
    gpt::G
    german::Bool
    frontend::F
    tokenizer::Tokenizer
    spkmatrix::Matrix{Float32}     # (192, 73)
    emomatrix::Matrix{Float32}     # (1280, 73)
    extras::Dict{Symbol,Any}       # lazily loaded normalizer and emotion-text model
end

"""
    loadindextts(graphs, tokenizer; device, german, precision) -> IndexTTS25

Build the model from an export directory (`graphs/<part>/...`, `aux/matrices`)
and a `.tiktoken` vocabulary. `precision` is `:fast` (fp16 DiT, codec and GPT
weights) or `:exact` (fp32 everywhere; see [`PRECISIONS`](@ref)).
"""
function loadindextts(graphs::AbstractString, tokenizer::AbstractString;
                      device = Mantle.Device(), german::Bool = false, precision::Symbol = :fast)
    nets = Networks(graphs; device, precision)
    gpt = loadgpt(graphs, german ? "gpt_de" : "gpt", precision === :exact ? Float32 : Float16; device)
    mats = readsafetensors(joinpath(graphs, "aux", "matrices.safetensors"))
    return IndexTTS25(device, nets, gpt, german, Frontend(device), Tokenizer(tokenizer),
                      Matrix{Float32}(mats["spk_matrix"]), Matrix{Float32}(mats["emo_matrix"]),
                      Dict{Symbol,Any}(:graphs => String(graphs)))
end

"""Resolve a lazily downloaded model component from this package's bindings."""
function assetdir(name::AbstractString = "indextts25")
    toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    hash = artifact_hash(name, toml)
    hash === nothing && error("IndexTTS25Runner: artifact $name is not bound; run tools/pack_indextts25.jl")
    ensure_artifact_installed(name, toml)
    return artifact_path(hash)
end

"""
    indextts25(; device, german=false, precision=:fast, dir=assetdir())

Load the packaged runner. Weight components are downloaded on demand; the base
and German models share the clip encoders, codec, flow model and vocoder.
"""
function indextts25(; dir::AbstractString = assetdir(), kw...)
    loadindextts(joinpath(dir, "graphs"), joinpath(dir, "tokenizer.tiktoken"); kw...)
end

function emotiontext(tts::IndexTTS25, text::AbstractString)
    model = get!(tts.extras, :emotion_text) do
        loademotiontext(joinpath(tts.extras[:graphs], "emotion_text"); device = tts.device)
    end
    return emotiontext(model, text)
end

function normalizetext(tts::IndexTTS25, text::AbstractString, lang::AbstractString)
    lowercase(lang) in ("en", "zh", "zhen") || return String(text)
    tables = get!(tts.extras, :normalizer) do
        TextNormalizerTables(joinpath(tts.extras[:graphs], "normalizer"))
    end
    return normalizetext(tables, text, lang)
end

Base.show(io::IO, t::IndexTTS25) = print(io, "IndexTTS25(", t.german ? "German" : "base", ")")

"""
    speak(tts, text; voice, emotion, alpha, weights, lang, duration, sampling,
          maxtokens, interval, noise) -> Vector{Int16}

`infer`: 22.05 kHz 16-bit PCM of `text` in `voice`'s voice (a [`Voice`](@ref) or
a path), segment by segment, joined with `interval` ms of silence and faded out
over the last 20 ms.

- `lang`: the language prefix and embedding row; "de" for the German model.
  English/Chinese use the upstream WeText tables; `normalize=false` disables
  normalization. German uses the checkpoint's [`preparegerman`](@ref).
- `emotion`: a clip (path, [`Audio`](@ref) at 16 kHz) or its vector, blended by
  `alpha`; `weights`: eight emotion weights instead (see [`emovector`](@ref)).
- `duration`: upstream's `duration_factor`, scaling the frames per code.
- `sampling`: see [`Sampling`](@ref).
- `noise`: per segment, the flow solve's start (to compare against upstream).
- `emotion_text`: a description such as "angry"; `use_emotion_text=true` uses
  the speech text itself when no separate description is supplied. The Qwen
  emotion classifier is loaded on first use and takes precedence over `weights`.
"""
function speak(tts::IndexTTS25, text::AbstractString; voice, emotion = nothing, alpha::Real = 1.0,
               weights = nothing, lang::AbstractString = tts.german ? "de" : "en", duration::Real = 1.0,
               sampling::Sampling = tts.german ? Sampling(num_beams = 1, temperature = 0.7f0) : Sampling(),
               maxtokens::Int = 120, interval::Real = 200, noise = nothing,
               emotion_text::Union{Nothing,AbstractString} = nothing, use_emotion_text::Bool = false,
               normalize::Bool = true)
    v = voice isa Voice ? voice : IndexTTS25Runner.voice(tts, voice)
    try
        if use_emotion_text || emotion_text !== nothing
            weights = emotiontext(tts, emotion_text === nothing ? text : emotion_text)
        end
        emovec = emovector(tts, v; emotion, alpha, weights)
        t = lowercase(lang) == "de" ? preparegerman(text) : normalize ? normalizetext(tts, text, lang) : text
        input = prepare(tts.tokenizer, t, lang; maxtokens)
        out = Float32[]
        for (k, ids) in enumerate(input.ids)
            codes = generate(tts.gpt, ids, input.lang, v.style, emovec, sampling)
            pcm = synthesize(tts, v, codes; duration, noise = noise === nothing ? nothing : noise[k],
                             seed = sampling.seed + k - 1)
            k > 1 && append!(out, zeros(Float32, silence(interval)))
            append!(out, pcm)
        end
        fadeout!(out)
        return trunc.(Int16, out)
    finally
        voice isa Voice || close(v)
    end
end

function Base.close(tts::IndexTTS25)
    close(tts.gpt)
    for name in fieldnames(typeof(tts.networks))
        close(getfield(tts.networks, name))
    end
    for name in fieldnames(typeof(tts.frontend))
        resource = getfield(tts.frontend, name)
        resource isa Mantle.Resource && Mantle.free!(resource)
    end
    model = get(tts.extras, :emotion_text, nothing)
    model === nothing || close(model.model)
    empty!(tts.extras)
    return nothing
end

end # module
