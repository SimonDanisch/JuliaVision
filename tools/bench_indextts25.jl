"""
Clock-gated IndexTTS-2.5 stage and complete-call measurements.

    include("tools/bench_indextts25.jl")
    benchmark_indextts25(; select="7900", precision=:exact)
    benchmark_indextts25(; select="4000", precision=:fast)

Voice, s2mel, vocoder and complete calls build fresh graphs, including planning.
GPT uses its reusable prefill and device decode plans, as normal inference does.
Every result, including rejected clock samples, is saved before the next stage.
"""
module IndexTTS25Bench

using IndexTTS25Runner, Mantle, DNNKernels, JSON3, Statistics
const R = IndexTTS25Runner
include(joinpath(@__DIR__, "measure.jl"))

function execute(g, out)
    Mantle.runonce!(g)
    Array(out)
end

function stage_mel(tts, v, codes, noise)
    g = Mantle.Graph(tts.device)
    out = Mantle.Buffer(tts.device, Float32, (R.melframes(length(codes), 1), 80, 1))
    _, owned = R.synthesismel!(g, tts, v, codes; noise, out)
    try
        execute(g, out)
    finally
        foreach(Mantle.free!, owned)
        Mantle.free!(out)
    end
end

function stage_vocoder(tts, mel)
    g = Mantle.Graph(tts.device)
    out = Mantle.Buffer(tts.device, Float32, (256size(mel, 1), 1, 1))
    try
        R.emit!(g, tts.networks.bigvgan, (; t = size(mel, 1)), mel; out)
        execute(g, out)
    finally
        Mantle.free!(out)
    end
end

function run(; select::AbstractString = "7900", precision::Symbol = :exact,
              generated::AbstractString = expanduser("~/gen/indextts25"), samples::Int = 7,
              stages = ("voice", "gpt", "gpt_beam", "s2mel", "bigvgan", "synthesis", "cached_voice", "complete"))
    dev = Mantle.LavaDevice(; select)
    source = occursin("NVIDIA", dev.device_name) ? NvidiaSmi() : telemetry()
    fixtures = R.assetdir("indextts25-refs")
    ref = DNNKernels.readsafetensors(joinpath(fixtures, "refs", "en_greedy.safetensors"))
    meta = JSON3.read(read(joinpath(fixtures, "refs", "en_greedy.json"), String))
    text = String(meta.text)
    clip = joinpath(fixtures, "examples", String(meta.spk))
    codes = Int.(vec(ref["gpt/codes"]))[1:end-1]
    noise = ref["cfm/z"]
    path = joinpath(generated, "bench_$(select)_$(precision).jsonl")
    mkpath(generated)
    tts = R.indextts25(; device = dev, precision)
    v = nothing
    mel = nothing
    try
        v = R.voice(tts, clip)
        input = R.prepare(tts.tokenizer, R.normalizetext(tts, text, "en"), "en")
        emotion = R.emovector(tts, v)
        mel = Mantle.Buffer(dev, ref["bigvgan/in"])
        greedy = R.Sampling(do_sample = false, num_beams = 1)
        beam = R.Sampling(do_sample = true, num_beams = 3)
        speaknoise = precision === :exact ? [noise] : nothing
        # Check real output before accepting timings on a new device.
        validation = if precision === :exact
            pcm = R.speak(tts, text; voice = v, sampling = greedy, noise = speaknoise)
            want = vec(ref["out/wav_int16"])
            @assert length(pcm) == length(want)
            difference = abs.(Int.(pcm) .- Int.(want))
            metrics = (; mean = mean(difference), peak = maximum(difference), samples = length(pcm))
            println("Benchmark PCM validation: ", metrics); flush(stdout)
            open(joinpath(generated, "validation_$(select)_$(precision).json"), "w") do io
                write(io, JSON3.write((; device = dev.device_name, precision, metrics...)))
            end
            write(joinpath(generated, "validation_$(select)_$(precision).pcm"), htol.(pcm))
            @assert metrics.mean < 5 && metrics.peak < 512 "PCM mismatch: $metrics"
            metrics
        else
            pcm = R.speak(tts, text; voice = v, sampling = greedy)
            @assert !isempty(pcm) && any(!iszero, pcm) "empty or silent fast output"
            metrics = (; samples = length(pcm), reference_samples = length(ref["out/wav_int16"]))
            open(joinpath(generated, "validation_$(select)_$(precision).json"), "w") do io
                write(io, JSON3.write((; device = dev.device_name, precision, metrics...)))
            end
            write(joinpath(generated, "validation_$(select)_$(precision).pcm"), htol.(pcm))
            metrics
        end
        output_samples = Dict{String,Int}()
        function decode(name, options)
            result = R.generate(tts.gpt, only(input.ids), input.lang, v.style, emotion, options)
            output_samples[name] = 256R.melframes(length(result), 1)
            return result
        end
        function capture(f, name)
            result = f()
            output_samples[name] = length(result)
            return result
        end
        calls = Dict(
            "voice" => () -> close(R.voice(tts, clip)),
            "gpt" => () -> decode("gpt", greedy),
            "gpt_beam" => () -> decode("gpt_beam", beam),
            "s2mel" => () -> stage_mel(tts, v, codes, noise),
            "bigvgan" => () -> stage_vocoder(tts, mel),
            "synthesis" => () -> R.synthesize(tts, v, codes; noise),
            "cached_voice" => () -> capture(() -> R.speak(tts, text; voice = v, sampling = greedy, noise = speaknoise), "cached_voice"),
            "complete" => () -> capture(() -> R.speak(tts, text; voice = clip, sampling = greedy, noise = speaknoise), "complete"))
        for name in stages
            println("Benchmarking $select / $precision / $name"); flush(stdout)
            started_unix = time()
            result = bench(calls[name]; samples, source, label = name, sync = () -> Mantle.waitidle(dev))
            finished_unix = time()
            println(result); flush(stdout)
            duration = get(output_samples, name, length(ref["out/wav_int16"])) / R.SAMPLERATE
            row = (; stage = name, device = dev.device_name, precision, audio_seconds = duration,
                     median_ms = 1000result.median, rtf = result.median / duration,
                     spread = result.spread, kept = result.kept, rejected = result.rejected,
                     clock_fraction = result.clock, samples = result.samples, validation,
                     started_unix, finished_unix)
            open(path, "a") do io
                println(io, JSON3.write(row))
            end
        end
    finally
        mel === nothing || Mantle.free!(mel)
        v === nothing || close(v)
        close(tts)
    end
    return path
end

end

benchmark_indextts25(; kw...) = IndexTTS25Bench.run(; kw...)
