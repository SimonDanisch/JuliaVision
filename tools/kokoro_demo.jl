# Kokoro-82M — text to speech on the GPU, through Lava.
#
# A VS Code eval-block script: send each `begin`/`end` block with Shift+Enter and
# change the plain variables between runs. Nothing is `const`, so `voice = ...`
# and re-running a block is the whole workflow.
#
# The model runs on Lava (Vulkan/SPIR-V). The only thing on the host is turning
# text into phonemes, which is a lexicon lookup — see `KokoroRunner`'s `g2p.jl`.

using KokoroRunner, FFMPEG_jll, Printf
using KokoroRunner: Kokoro, speak, phonemize, pronounce!, voices, SAMPLERATE


begin
    k = Kokoro()
    # The lexicon is 90k English words and knows no proper nouns, so it does not
    # know the model's own name — `phonemize` warns and DROPS an unknown word
    # rather than guessing, and this is how you answer that.
    pronounce!(k, "Kokoro" => "kəkˈɔɹO")
    @time speak(k; phonemes = "hˈɛlO")          # warm the shapes
    println("ready — ", length(voices(k)), " voices")
end


## ─────────────────────────────────────────────────────────────── helpers ─────
begin
    """
        wav(path, audio) -> path

    Mono 16-bit PCM at 24 kHz. Through `FFMPEG_jll` rather than a WAV package
    because ffmpeg is already a dependency of this repo and writes every other
    container the editor needs; a second audio-file library for one header would
    be the odd thing.
    """
    function wav(path, audio::Vector{Float32}; rate = SAMPLERATE)
        open(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error
              -f f32le -ar $rate -ac 1 -i pipe:0 $path`, "w") do io
            write(io, audio)
        end
        path
    end

    "Play through whatever PipeWire/PulseAudio is offering."
    play(audio::Vector{Float32}; rate = SAMPLERATE) =
        open(`paplay --raw --format=float32le --rate=$rate --channels=1`, "w") do io
            write(io, audio)
        end

    "Say it, report what it cost, play it."
    function say(k, text; voice = "af_heart", speed = 1.0)
        ps = phonemize(k, text)
        t = @elapsed audio = speak(k; phonemes = ps, voice, speed)
        secs = length(audio) / SAMPLERATE
        @printf("  %-12s %5.2f s audio in %5.2f s  (%.1fx realtime)\n",
                voice, secs, t, secs / t)
        play(audio)
        audio
    end
end


## ────────────────────────────────────────────────────────────── say it ───────
begin
    text = "Hello! This is Kokoro, running entirely on the graphics processor through Lava."
    voice = "af_heart"

    println(phonemize(k, text))
    audio = say(k, text; voice)
end


## ─────────────────────────────────────────────── the same line, every voice ──
#
# The voice is a 256-number style vector, not a separate model — all 54 share one
# set of weights, so switching costs a lookup.
begin
    text = "The quick brown fox jumps over the lazy dog."
    for voice in ["af_heart", "af_bella", "am_michael", "am_puck", "bf_emma", "bm_george"]
        say(k, text; voice)
        sleep(0.3)
    end
end


## ──────────────────────────────────────────────────────────────── speed ──────
#
# `speed` divides the predicted durations, so the model re-renders at the new
# pace rather than the audio being resampled — the pitch does not move.
begin
    text = "This sentence is spoken at three different speeds."
    for speed in (0.8, 1.0, 1.3)
        @printf("speed %.1f\n", speed)
        say(k, text; speed)
        sleep(0.3)
    end
end


## ──────────────────────────────────────────────── one export, any length ─────
#
# The graph is symbolic in the token count. These are 4 to 200-odd tokens and all
# go through the same two graphs — there is no bucket and no re-export.
begin
    for text in ["Yes.",
                 "Just a short line.",
                 "A somewhat longer sentence, with a comma in the middle of it.",
                 "And a much longer one: speech synthesis converts written language " *
                 "into spoken words, and the whole pipeline — the text encoder, the " *
                 "prosody predictor, and the vocoder — runs in one process on one " *
                 "graphics card, at twenty four kilohertz."]
        ps = phonemize(k, text)
        @printf("%3d phonemes | ", length(ps))
        say(k, text)
    end
end


## ───────────────────────────────────────────────────────── write a file ──────
begin
    text = "This line was written to a file."
    voice = "am_michael"
    path = joinpath(tempdir(), "kokoro_demo.wav")

    audio = speak(k, text; voice)
    wav(path, audio)
    @printf("%s  —  %.2f s, %d kB\n", path, length(audio) / SAMPLERATE,
            filesize(path) ÷ 1024)
end
