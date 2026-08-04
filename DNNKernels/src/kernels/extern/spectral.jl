"""
Spectral front ends: the STFT and the mel spectrogram every audio model here
starts with.

The transform itself lives in Lava (`Lava.fft!`, `Lava.rfft`, `Lava.stft`,
ported from VkFFT) for the same reason `mul!` does — it is an array primitive,
not a neural-network one, and the graphics side wants it too. What is here is the
model-facing layer: the entry points take a `Ctx`, so they compose with the rest
of the kernel library, and the mel filterbank, which is a property of the model
rather than of the device.

## The sizes are not powers of two

Worth stating because it drove the whole design of the Lava side:

    Whisper mel      n_fft = 400  = 2^4 * 5^2
    DeepFilterNet3   n_fft = 960  = 2^6 * 3 * 5
    Demucs htdemucs  n_fft = 4096 = 2^12

Only the last is a power of two. `Lava.fftany!` dispatches to a mixed-radix plan
for the other two — `400 -> (8,5,5,2)`, `960 -> (8,8,5,3)`.
"""

"""
    stft(ctx, x, nfft, hop, window; center = true) -> AbstractArray

Short-time Fourier transform, `(nfft ÷ 2 + 1, frames)` complex.

Matches `torch.stft(..., center, return_complex=true)`. The `ctx` argument is
this library's convention — every kernel entry point takes one — and carries the
backend the transform runs on.
"""
function stft(ctx::Ctx, x, nfft::Int, hop::Int, window; center::Bool = true)
    return Lava.stft(x, nfft, hop, window; center)
end

"""
    hznmel(f) / melnhz(m)

The **Slaney** mel scale, which is what `torchaudio` and HuggingFace's
`WhisperFeatureExtractor` use by default — linear below 1 kHz, logarithmic above.

Not the HTK formula `2595 log10(1 + f/700)`. They differ by a few percent in bin
placement, which is enough to move every mel coefficient and therefore every
number the encoder sees. If a model turns out to want HTK the fix is a keyword
here, not a different filterbank.
"""
hznmel(f) = f < 1000 ? 3f0 * f / 200f0 :
            15f0 + log(f / 1000) * (27f0 / log(6.4f0))
melnhz(m) = m < 15 ? 200f0 * m / 3f0 :
            1000f0 * exp(log(6.4f0) * (m - 15f0) / 27f0)

"""
    melfilters(nmels, nfreq, sr; fmin = 0, fmax = sr / 2, norm = :slaney) -> Matrix{Float32}

The triangular mel filterbank, `(nmels, nfreq)`, on the host.

Built here rather than shipped in an artifact because it is a dozen lines and a
closed form, and an artifact would make the mel scale a binary blob nobody can
check against the model that expects it.

`norm = :slaney` scales each filter by `2 / (f[i+2] - f[i])` so that filters have
unit *area* rather than unit peak — again what HuggingFace does by default, and
skipping it makes the low mel bins several times too loud.
"""
function melfilters(nmels::Int, nfreq::Int, sr::Real;
                    fmin::Real = 0, fmax::Real = sr / 2, norm::Symbol = :slaney)
    fftfreqs = Float32[(sr / 2) * k / (nfreq - 1) for k in 0:(nfreq - 1)]
    mmin, mmax = hznmel(Float32(fmin)), hznmel(Float32(fmax))
    mpts = range(mmin, mmax; length = nmels + 2)
    fpts = Float32[melnhz(m) for m in mpts]
    W = zeros(Float32, nmels, nfreq)
    for i in 1:nmels
        lo, ctr, hi = fpts[i], fpts[i + 1], fpts[i + 2]
        for k in 1:nfreq
            f = fftfreqs[k]
            v = f < ctr ? (f - lo) / max(ctr - lo, eps(Float32)) :
                          (hi - f) / max(hi - ctr, eps(Float32))
            W[i, k] = max(0.0f0, v)
        end
        if norm === :slaney
            W[i, :] .*= 2.0f0 / max(hi - lo, eps(Float32))
        end
    end
    return W
end

"""
    logmelspectrogram(ctx, audio, filters; nfft, hop, window, droplast = true) -> AbstractArray

Whisper's front end: STFT, magnitude squared, mel projection, log10, then the
two clamps that make it a bounded input.

`droplast` drops the final STFT frame, which `torch.stft(center=true)` emits and
Whisper's own feature extractor discards — keeping it shifts every subsequent
frame and the encoder's positional embedding no longer lines up.

The dynamic-range clamp (`max(log_spec, max(log_spec) - 8)`) is a **global**
maximum over the whole spectrogram, not per frame. Doing it per frame is a
tempting simplification that changes quiet passages, so it is a full reduction.

`filters` may be a host matrix — [`melfilters`](@ref) builds one, since it is a
closed-form table and there is nothing to compute on a device. It is uploaded
here rather than at the call site: handing a `Matrix` to a device `mul!` reaches
`densify`, which broadcasts a host `Vector` inside a kernel and fails with
"passing non-bitstype argument … `Memory{Float32}` is not isbits" — an error that
names the broadcast machinery and not the array that should not have been there.
"""
function logmelspectrogram(ctx::Ctx, audio, filters;
                           nfft::Int = 400, hop::Int = 160, window = nothing,
                           droplast::Bool = true)
    backend = ctx.backend
    w = window === nothing ? Lava.hannwindow(backend, nfft) : window
    S = Lava.stft(audio, nfft, hop, w; center = true)
    nb, nt = size(S)
    keep = droplast ? nt - 1 : nt
    mag = abs2.(view(S, :, 1:keep))                 # (nfreq, frames), real
    F = toback(backend, filters)
    mel = F * mag                                   # (nmels, frames)
    ls = log10.(max.(mel, 1.0f-10))
    ls = max.(ls, maximum(ls) - 8.0f0)
    return (ls .+ 4.0f0) ./ 4.0f0
end
