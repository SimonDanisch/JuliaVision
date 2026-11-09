"""Compare saved Julia PCM and upstream PCM using Whisper and CAMPPlus.

Results are saved after each clip. Pass generated=... with a validation/
subdirectory containing the saved waveforms. Variants default to upstream
and exact; fast reads <case>_fast.wav. Text is normalized before WER scoring.
"""
module IndexTTS25AudioCheck
using IndexTTS25Runner, WhisperRunner, Mantle, DNNKernels, JSON3
const R = IndexTTS25Runner
const W = WhisperRunner
function worderror(expected, actual)
    words(s) = split(strip(replace(lowercase(s), r"[^\p{L}\p{N}]+" => " ")))
    a, b = words(expected), words(actual)
    previous = collect(0:length(b))
    for i in eachindex(a)
        current = [i; zeros(Int, length(b))]
        for j in eachindex(b)
            current[j + 1] = min(previous[j + 1] + 1, current[j] + 1,
                                 previous[j] + (a[i] != b[j]))
        end
        previous = current
    end
    return previous[end] / max(length(a), 1)
end


function embedding(model, frontend, audio::R.Audio)
    dev = model.device
    g = Mantle.Graph(dev)
    input = Mantle.Buffer(dev, audio.samples)
    out = nothing
    try
        bank, t = R.kaldifbank!(g, frontend, input, length(audio))
        out = R.outputbuffer(model, (; t))
        R.emit!(g, model, (; t), bank; out)
        Mantle.runonce!(g)
        return vec(Array(out))
    finally
        Mantle.free!(input)
        out === nothing || Mantle.free!(out)
    end
end

function run(; select="7900", cases=("en_greedy", "en_emotext", "de_greedy", "de_emoclip", "de_sample"),
               variants=("upstream", "exact"), generated=expanduser("~/gen/indextts25"))
    fixtures = R.assetdir("indextts25-refs")
    output = joinpath(generated, "validation")
    mkpath(output)
    path = joinpath(output, "quality-comparison.jsonl")
    done = isfile(path) ? Set((String(r.name), String(r.variant)) for r in JSON3.read.(readlines(path))) : Set{Tuple{String,String}}()
    dev = Mantle.LavaDevice(; select)
    model = R.loadpart(joinpath(R.assetdir(), "graphs"), "campplus"; device=dev)
    frontend = R.Frontend(dev)
    recognizer = W.whisper(; backend=dev)
    styles = Dict{String,Vector{Float32}}()
    tables = nothing
    try
        for name in cases
            meta = JSON3.read(read(joinpath(fixtures, "refs", "$name.json"), String))
            lang = String(meta.lang)
            style = get!(styles, String(meta.spk)) do
                a = R.soxr(R.readaudio(joinpath(fixtures, "examples", String(meta.spk))), 16000)
                embedding(model, frontend, a)
            end
            if lang == "de"
                normalize = R.preparegerman
            else
                tables === nothing && (tables = R.TextNormalizerTables(joinpath(R.assetdir(), "graphs", "normalizer")))
                normalize = s -> R.normalizetext(tables, s, lang)
            end
            expected = normalize(String(meta.text))
            for variant in variants
                (name, variant) in done && continue
                if variant == "upstream"
                    ref = DNNKernels.readsafetensors(joinpath(fixtures, "refs", "$name.safetensors"))
                    audio = R.Audio(Float32.(vec(ref["out/wav_int16"])) ./ 32768f0, R.SAMPLERATE)
                else
                    file = variant == "exact" ? "$name.wav" : "$(name)_$(variant).wav"
                    isfile(joinpath(output, file)) || error("Missing audio: $(joinpath(output, file))")
                    audio = R.readaudio(joinpath(output, file))
                end
                audio = R.soxr(audio, 16000)
                println("Checking $name / $variant"); flush(stdout)
                transcript, _ = W.transcribe(recognizer, audio.samples; language=lang)
                generatedstyle = embedding(model, frontend, audio)
                row = (; name, variant, transcript, expected, normalized_transcript=normalize(transcript),
                         word_error=worderror(expected, normalize(transcript)),
                         speaker_cosine=R.cosine(style, generatedstyle), device=dev.device_name)
                open(path, "a") do io
                    println(io, JSON3.write(row))
                end
                println(JSON3.write(row)); flush(stdout)
            end
        end
    finally
        close(model)
        close(recognizer.encoder); close(recognizer.decoder)
        for name in fieldnames(typeof(frontend))
            r = getfield(frontend, name)
            r isa Mantle.Resource && Mantle.free!(r)
        end
    end
    return path
end
end

check_indextts25_audio(; kw...) = IndexTTS25AudioCheck.run(; kw...)
