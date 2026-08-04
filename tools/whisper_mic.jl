using WhisperRunner, Printf
using WhisperRunner: whisper, tokenizer, transcribechunk, transcribe, DecodeOptions


begin
    tk = tokenizer()
    w = whisper()
    @time transcribechunk(w, tk, zeros(Float32, 16000))
    println("ready")
end


## ──────────────────────────────────────────────────────────── helpers ────────
begin
    "The capture sources PipeWire/PulseAudio is offering."
    function listdevices()
        for l in eachline(`pactl list short sources`)
            f = split(l, '\t')
            length(f) >= 2 && println("  ", f[2])
        end
    end

    function micstream(dev; rate = 16000)
        cmd = `parecord --raw --format=float32le --rate=$rate --channels=1
               --latency-msec=100 --device=$dev`
        io = Base.PipeEndpoint()
        p = run(pipeline(cmd; stdout = io); wait = false)
        return p, io
    end

    "Drain the pipe forever into `buf`. Own task — see the header."
    function reader!(io, buf::Vector{Float32}, lk::ReentrantLock)
        @async begin
            chunk = Vector{UInt8}(undef, 4 * 1600)          # 100 ms
            while isopen(io)
                n = readbytes!(io, chunk)
                n == 0 && break
                s = reinterpret(Float32, @view chunk[1:n])
                lock(() -> append!(buf, s), lk)
            end
        end
    end

    "Capture `secs` seconds, showing the level as it goes."
    function record(dev, secs; rate = 16000)
        p, io = micstream(dev; rate)
        buf = Float32[]; lk = ReentrantLock()
        reader!(io, buf, lk)
        t0 = time()
        while time() - t0 < secs
            sleep(0.1)
            n, pk = lock(lk) do
                length(buf), isempty(buf) ? 0.0f0 : maximum(abs, @view buf[max(1, end - 4800):end])
            end
            @printf("\r  %4.1f s  level %s%s", n / rate,
                    "#"^clamp(round(Int, pk * 60), 0, 30), " "^30)
            flush(stdout)
        end
        kill(p)
        println()
        return lock(() -> copy(buf), lk)
    end
end

## ────────────────────────────────────────────── which microphone is there ────
listdevices()

## ──────────────────────────────────── record a few seconds, then transcribe ──
# The simplest check: speak while the level meter moves, then read the text.
begin
    rate = 16000
    device = "alsa_input.usb-Samson_Technologies_Samson_Meteorite_Mic-00.analog-stereo"
    language = "de"
    step = 3.0                 # seconds of new audio between passes
    window = 10.0              # seconds re-transcribed each pass
    keep = 0.2                 # seconds carried across a commit
    seconds = 5.0              # for the record-once block

    @printf("recording %.1f s — speak now...\n", seconds)
    audio = record(device, seconds; rate)
    @printf("captured %.2f s, peak %.3f\n", length(audio) / rate, maximum(abs, audio))
    text, segs = transcribe(w, audio; tk, language)
    println(isempty(text) ? "  [no speech]" : "  " * text)
    for s in segs
        @printf("    [%6.2f -> %6.2f] %s\n", s.start, s.stop, s.text)
    end
end
