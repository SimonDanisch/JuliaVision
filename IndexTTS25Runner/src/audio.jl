# Reference audio, as upstream loads it.
#
# `IndexTTS2._load_and_cut_audio` is `librosa.load(path, sr)`: the file decoded to
# float32 (libsndfile), channels averaged in float32, resampled with libsoxr at its
# HQ recipe (`res_type = "soxr_hq"`), then cut to 15 s. The speaker clip is loaded
# at 22050 Hz and taken to 16 kHz by torchaudio's sinc resampler; the emotion clip
# is loaded at 16 kHz directly. Each step here is that step: ffmpeg decodes (its
# integer-to-float conversion is libsndfile's, x / 2^(bits-1)), the downmix is the
# same float32 mean, and the resampler is libsoxr itself.

"""Seconds of reference audio upstream keeps (`_load_and_cut_audio(..., 15)`)."""
const MAXSECONDS = 15

"""
    Audio

Mono float32 samples and their rate.
"""
struct Audio
    samples::Vector{Float32}
    rate::Int
end

Base.length(a::Audio) = length(a.samples)
duration(a::Audio) = length(a.samples) / a.rate

"""
    probeaudio(path) -> (rate, channels)

The first audio stream's sample rate and channel count, from ffprobe.
"""
function probeaudio(path::AbstractString)
    out = read(`$(FFMPEG_jll.ffprobe()) -v error -select_streams a:0 -show_entries stream=sample_rate,channels -of csv=p=0 $path`, String)
    rate, channels = parse.(Int, split(strip(out), ','))
    return rate, channels
end

"""
    readaudio(path) -> Audio

Decode any file ffmpeg opens at its own rate, and average its channels the way
librosa's `to_mono` does: summed in float32, divided by the count.
"""
function readaudio(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such audio file: $path"))
    rate, channels = probeaudio(path)
    io = IOBuffer()
    run(pipeline(`$(FFMPEG_jll.ffmpeg()) -v quiet -i $path -map 0:a:0 -f f32le -acodec pcm_f32le -`; stdout = io))
    raw = reinterpret(Float32, take!(io))
    n = length(raw) ÷ channels
    frames = reshape(view(raw, 1:(n * channels)), channels, n)
    mono = Vector{Float32}(undef, n)
    for i in 1:n
        s = 0f0
        for c in 1:channels
            s += frames[c, i]
        end
        mono[i] = s / Float32(channels)
    end
    return Audio(mono, rate)
end

# ── libsoxr ──────────────────────────────────────────────────────────────────

# `soxr.h`, 0.1.3: the three specs `soxr_oneshot` takes, by pointer.
struct SoxrQualitySpec
    precision::Cdouble
    phase_response::Cdouble
    passband_end::Cdouble
    stopband_begin::Cdouble
    e::Ptr{Cvoid}
    flags::Culong
end

struct SoxrIOSpec
    itype::Cint
    otype::Cint
    scale::Cdouble
    e::Ptr{Cvoid}
    flags::Culong
end

struct SoxrRuntimeSpec
    log2_min_dft_size::Cuint
    log2_large_dft_size::Cuint
    coef_size_kbytes::Cuint
    num_threads::Cuint
    e::Ptr{Cvoid}
    flags::Culong
end

"""`SOXR_HQ`, the recipe python-soxr's `"HQ"`/`"soxr_hq"` names (20-bit)."""
const SOXR_HQ = Culong(4)
const SOXR_FLOAT32_I = Cint(0)

"""
    soxr(x, inrate, outrate) -> Vector{Float32}

libsoxr's HQ resampler over a whole signal, `soxr_oneshot` with float32 in and
out and one thread: what python-soxr runs for `librosa.load`'s `"soxr_hq"`. Its
streaming entry point gives the same samples; the output length is
`ceil(n * outrate / inrate)`, as it is there.
"""
function soxr(x::AbstractVector{Float32}, inrate::Integer, outrate::Integer)
    inrate == outrate && return Vector{Float32}(x)
    q = Ref(ccall((:soxr_quality_spec, SoXResampler_jll.libsoxr), SoxrQualitySpec, (Culong, Culong), SOXR_HQ, 0))
    io = Ref(ccall((:soxr_io_spec, SoXResampler_jll.libsoxr), SoxrIOSpec, (Cint, Cint), SOXR_FLOAT32_I, SOXR_FLOAT32_I))
    rt = Ref(ccall((:soxr_runtime_spec, SoXResampler_jll.libsoxr), SoxrRuntimeSpec, (Cuint,), 1))
    src = Vector{Float32}(x)
    olen = ceil(Int, length(src) * outrate / inrate)
    out = Vector{Float32}(undef, olen)
    idone, odone = Ref{Csize_t}(0), Ref{Csize_t}(0)
    err = ccall((:soxr_oneshot, SoXResampler_jll.libsoxr), Ptr{Cchar},
                (Cdouble, Cdouble, Cuint, Ptr{Float32}, Csize_t, Ref{Csize_t}, Ptr{Float32}, Csize_t,
                 Ref{Csize_t}, Ref{SoxrIOSpec}, Ref{SoxrQualitySpec}, Ref{SoxrRuntimeSpec}),
                inrate, outrate, 1, src, length(src), idone, out, olen, odone, io, q, rt)
    err == C_NULL || error("soxr_oneshot: $(unsafe_string(err))")
    return resize!(out, Int(odone[]))
end

"""`soxr` on an `Audio`."""
soxr(a::Audio, rate::Integer) = Audio(soxr(a.samples, a.rate, rate), rate)

"""The first `seconds` of `a`: `audio[:, :int(seconds * sr)]`."""
function cut(a::Audio; seconds::Real = MAXSECONDS)
    keep = trunc(Int, seconds * a.rate)
    return length(a) > keep ? Audio(a.samples[1:keep], a.rate) : a
end

"""
    loadaudio(path, rate; seconds = MAXSECONDS) -> Audio

`IndexTTS2._load_and_cut_audio`: decoded, averaged to mono, resampled to `rate`
with libsoxr HQ, and cut to `seconds`.
"""
loadaudio(path::AbstractString, rate::Integer; seconds::Real = MAXSECONDS) =
    cut(soxr(readaudio(path), rate); seconds)
