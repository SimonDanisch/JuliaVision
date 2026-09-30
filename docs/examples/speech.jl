# The KokoroRunner and WhisperRunner README examples: Kokoro reads two lines in
# two voices, the audio is written as MP3, and Whisper transcribes the files.
isdefined(@__MODULE__, :ExampleCommon) || include(joinpath(@__DIR__, "common.jl"))
using .ExampleCommon
using KokoroRunner, FFMPEG_jll
import WhisperRunner

"Mono MP3 of 24 kHz Float32 samples, through the ffmpeg the runners already depend on."
function writemp3(path::AbstractString, audio::Vector{Float32}; rate = KokoroRunner.SAMPLERATE)
    open(`$(FFMPEG_jll.ffmpeg()) -y -loglevel error -f f32le -ar $rate -ac 1 -i pipe:0
          -codec:a libmp3lame -q:a 4 $path`, "w") do io
        write(io, audio)
    end
    return path
end

const SPEECH_LINES = [
    ("af_heart", "The fox in these pictures was never photographed. " *
                 "A diffusion model drew it, and every other model on this page took it from there."),
    ("bm_george", "Each network runs on the graphics card through Vulkan, " *
                  "from one Julia process, with no Python anywhere."),
]

function speech_examples()
    k = Kokoro()
    # The lexicon knows no proper nouns and drops what it does not know, with a
    # warning; this is how a word is taught to it.
    pronounce!(k, "Vulkan" => "vˈʌlkən")
    spoken = map(enumerate(SPEECH_LINES)) do (i, (voice, text))
        audio = speak(k, text; voice)                           # the first call per length compiles
        t = @elapsed audio = speak(k, text; voice)
        path = writemp3(media("kokoro", "line$i.mp3"), audio)
        (; voice, text, path, seconds = length(audio) / KokoroRunner.SAMPLERATE, took = t)
    end
    k = nothing
    GC.gc(true)

    w = WhisperRunner.whisper()
    heard = map(spoken) do s
        WhisperRunner.transcribe(w, s.path; language = "en")
        t = @elapsed text, segs = WhisperRunner.transcribe(w, s.path; language = "en")
        (; s.path, text, took = t)
    end
    return (; spoken, heard)
end

abspath(PROGRAM_FILE) == (@__FILE__) && speech_examples()
