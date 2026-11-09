const ASSETS = R.assetdir()
const FIXTURES = R.assetdir("indextts25-refs")
const SELECT_DEVICE = get(ENV, "INDEXTTS25_TEST_DEVICE", "")
const OUTPUT = get(ENV, "INDEXTTS25_TEST_OUTPUT", "")

function checkpoint(name, phase)
    @info "IndexTTS25 progress" name phase
    flush(stdout); flush(stderr)
    isempty(OUTPUT) && return
    mkpath(OUTPUT)
    open(joinpath(OUTPUT, "progress.jsonl"), "a") do io
        println(io, JSON3.write((; name, phase, time = time())))
    end
end

function savecase(name, pcm, metrics)
    isempty(OUTPUT) && return
    mkpath(OUTPUT)
    # Mono little-endian PCM for the independent intelligibility checks.
    path = joinpath(OUTPUT, "$name.wav")
    open(path * ".partial", "w") do io
        write(io, "RIFF", htol(UInt32(36 + 2length(pcm))), "WAVEfmt ", htol(UInt32(16)),
              htol(UInt16(1)), htol(UInt16(1)), htol(UInt32(R.SAMPLERATE)),
              htol(UInt32(2R.SAMPLERATE)), htol(UInt16(2)), htol(UInt16(16)),
              "data", htol(UInt32(2length(pcm))), htol.(pcm))
    end
    mv(path * ".partial", path; force = true)
    open(joinpath(OUTPUT, "parity.jsonl"), "a") do io
        println(io, JSON3.write(metrics))
    end
end

@testset "text front ends against upstream" begin
    tk = R.Tokenizer(joinpath(ASSETS, "tokenizer.tiktoken"))
    tables = R.TextNormalizerTables(joinpath(ASSETS, "graphs", "normalizer"))
    for file in ("reference.json", "battery.json")
        rows = JSON3.read(read(joinpath(FIXTURES, "normalizer", file), String))
        for row in rows
            if haskey(row, :lang)
                lang = Symbol(row.lang)
                @test R.transduce(tables.taggers[lang], String(row.text)) == row.tagged
                @test R.normalizetext(tables, String(row.text), lang) == row.normalized
            else
                @test R.normalizetext(tables, String(row.text), "en") == row.normalized
            end
        end
    end
    for file in readdir(joinpath(FIXTURES, "refs"))
        endswith(file, ".json") || continue
        meta = JSON3.read(read(joinpath(FIXTURES, "refs", file), String))
        text = meta.lang == "de" ? R.preparegerman(String(meta.text)) :
               R.normalizetext(tables, String(meta.text), String(meta.lang))
        prepared = R.prepare(tk, text, String(meta.lang);
                             maxtokens = Int(get(meta, :max_text_tokens_per_segment, 120)))
        @test prepared.segments == String.(meta.segments)
        reference = readsafetensors(joinpath(FIXTURES, "refs", replace(file, ".json" => ".safetensors")))
        for (i, ids) in enumerate(prepared.ids)
            key = i == 1 ? "gpt/text_tokens" : "gpt/text_tokens/$(i - 1)"
            @test ids == vec(reference[key])
        end
    end
    for row in JSON3.read(read(joinpath(@__DIR__, "fixtures", "german_frontend.json"), String))
        @test only(R.prepare(tk, String(row.prepared), "de").ids) == row.ids
    end
    emodir = joinpath(ASSETS, "graphs", "emotion_text")
    tk = R.EmotionTokenizer(emodir)
    for row in JSON3.read(read(joinpath(FIXTURES, "emotion_text", "reference.json"), String))
        @test R.emotionids(tk, String(row.text)) == row.ids
    end
end

function sampling(meta)
    gen = get(meta, :gen, Dict())
    R.Sampling(; do_sample = meta.decode in ("sample", "beamsample"),
                num_beams = meta.decode in ("beam", "beamsample") ? 3 : 1,
                temperature = Float32(get(gen, :temperature, 0.8)),
                top_p = Float32(get(gen, :top_p, 0.8)), top_k = Int(get(gen, :top_k, 30)),
                penalty = Float32(get(gen, :repetition_penalty, 10.0)), seed = Int(get(meta, :seed, 1234)))
end

function casecheck(tts, name)
    meta = JSON3.read(read(joinpath(FIXTURES, "refs", "$name.json"), String))
    reference = readsafetensors(joinpath(FIXTURES, "refs", "$name.safetensors"))
    noise = [reference[i == 1 ? "cfm/z" : "cfm/z/$(i - 1)"] for i in 1:length(meta.segments)]
    path = joinpath(FIXTURES, "examples", String(meta.spk))
    emotion = haskey(meta, :emo) ? joinpath(FIXTURES, "examples", String(meta.emo)) : nothing
    weights = haskey(meta, :emo_vector) ? Float64.(meta.emo_vector) : nothing
    checkpoint(name, "voice")
    v = R.voice(tts, path)
    try
        checkpoint(name, "speak")
        seconds = @elapsed pcm = R.speak(tts, String(meta.text); voice = v, emotion, weights,
                      alpha = get(meta, :emo_alpha, 1.0), lang = String(meta.lang),
                      emotion_text = get(meta, :emo_text, nothing),
                      use_emotion_text = get(meta, :use_emo_text, false),
                      sampling = sampling(meta), duration = get(meta, :duration_factor, 1.0),
                      maxtokens = Int(get(meta, :max_text_tokens_per_segment, 120)), noise)
        want = vec(reference["out/wav_int16"])
        @test length(pcm) == length(want)
        if length(pcm) == length(want)
            diff = abs.(Int.(pcm) .- Int.(want))
            @test sum(diff) / length(diff) < 5
            @test maximum(diff) < 512
            @info "IndexTTS25 PCM agreement" name samples=length(pcm) mean=sum(diff)/length(diff) peak=maximum(diff)
            savecase(name, pcm, (; name, samples = length(pcm), mean = sum(diff) / length(diff),
                                  peak = maximum(diff), seconds, device = tts.device.device_name))
        end
        checkpoint(name, "complete")
        return pcm
    finally
        close(v)
    end
end

if get(ENV, "INDEXTTS25_TEST_GPU", "true") == "true"
    include("packing.jl")
    @testset "complete inference against fixed-noise upstream runs" begin
        dev = isempty(SELECT_DEVICE) ? Mantle.Device() : Mantle.LavaDevice(select = SELECT_DEVICE)
        testpacking(dev)
        for german in (false, true)
            cases = german ? ("de_greedy", "de_sample", "de_emoclip") :
                    ("en_greedy", "en_emoclip", "en_emovec", "en_emotext", "en_duration",
                     "en_beam", "en_beamsample", "en_long", "zh_greedy")
            selected = split(get(ENV, "INDEXTTS25_TEST_CASES", join(cases, ',')), ',')
            cases = filter(in(selected), cases)
            qwen = !german && get(ENV, "INDEXTTS25_TEST_QWEN", "true") == "true"
            isempty(cases) && !qwen && continue
            checkpoint(german ? "german" : "base", "load")
            tts = R.indextts25(; device = dev, german, precision = :exact)
            try
                if qwen
                    @testset "Qwen emotion generation" begin
                        model = R.loademotiontext(joinpath(ASSETS, "graphs", "emotion_text"); device = dev)
                        try
                            for row in JSON3.read(read(joinpath(FIXTURES, "emotion_text", "reference.json"), String))
                                ids = R.emotiontokens(model, String(row.text))
                                # The exporter records the EOS; the runner returns content only.
                                @test vcat(ids, model.eos) == row.generated
                                @test R.emotiontext(model, String(row.text)) == R.emotionweights(String(row.response), String(row.text))
                            end
                        finally
                            close(model.model)
                        end
                    end
                end
                for name in cases
                    @testset "$name" begin
                        casecheck(tts, name)
                    end
                end
            finally
                close(tts)
            end
        end
    end
end
