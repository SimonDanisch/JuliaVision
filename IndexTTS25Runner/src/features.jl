# The spectral front ends of the reference audio, on the device.
#
# Three, because upstream feeds three models and each brought its own:
#
#   * w2v-BERT's input is `SeamlessM4TFeatureExtractor`: Kaldi-style frames of the
#     16 kHz signal scaled to 16-bit range, a 512-point power spectrum, 80 Kaldi-mel
#     filters triangular in mel space, `log`, then each mel bin normalised to zero
#     mean and unit variance over the clip (`ddof = 1`) and frames paired into 160.
#   * CAMPPlus reads `torchaudio.compliance.kaldi.fbank` of the same signal,
#     unscaled, minus its mean over time.
#   * The flow-matching prompt is s2mel's `mel_spectrogram` of the 22.05 kHz
#     signal: reflect-padded 1024-point Hann STFT, magnitude, librosa's Slaney mel,
#     `log(max(·, 1e-5))`.
#
# The 16 kHz signal itself comes from the 22.05 kHz one through
# `torchaudio.transforms.Resample(22050, 16000)`, a windowed-sinc polyphase filter.
#
# The filter banks and the resampling kernel are closed forms built on the host,
# each in the precision the library that defines it computes in, so that what
# reaches the device is the same table to within a rounding.

import KernelInterface as KI

# ── host tables ──────────────────────────────────────────────────────────────

"""The Povey window, `(0.5 - 0.5 cos(2πn/(N-1)))^0.85`, symmetric."""
poveywindow(n::Int) = Float32[(0.5 - 0.5 * cospi(2k / (n - 1)))^0.85 for k in 0:(n - 1)]

"""`torch.hann_window(n)`: periodic."""
hannwindow(n::Int) = Float32[0.5 - 0.5 * cospi(2k / n) for k in 0:(n - 1)]

"""`np.linspace(a, b, n)`: `a + i * (b - a) / (n - 1)`, the last point exactly `b`."""
linspace(a::Float64, b::Float64, n::Int) = [i == n - 1 ? b : a + i * ((b - a) / (n - 1)) for i in 0:(n - 1)]

"""
    seamlessmel() -> Matrix{Float32}

`mel_filter_bank(257, 80, 20, 8000, 16000, norm = None, mel_scale = "kaldi",
triangularize_in_mel_space = True)` (transformers `audio_utils.py`), in float64
as numpy builds it, transposed to `(80, 257)`.
"""
function seamlessmel(; nbins::Int = 257, nmels::Int = 80, fmin = 20.0, fmax = 8000.0, sr = 16000.0)
    hz2mel(f) = 1127.0 * log(1.0 + f / 700.0)
    melf = linspace(hz2mel(fmin), hz2mel(fmax), nmels + 2)
    width = sr / ((nbins - 1) * 2)
    fftf = [hz2mel(width * k) for k in 0:(nbins - 1)]
    W = zeros(Float32, nmels, nbins)
    for i in 1:nmels, k in 1:nbins
        down = -(melf[i] - fftf[k]) / (melf[i + 1] - melf[i])
        up = (melf[i + 2] - fftf[k]) / (melf[i + 2] - melf[i + 1])
        W[i, k] = max(0.0, min(down, up))
    end
    return W
end

"""
    kaldimel() -> Matrix{Float32}

`torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000, 20, 0, ...)`, padded
with a zero Nyquist column to `(80, 257)` as `fbank` does. torchaudio builds it in
float32 from float64 scalars, and so does this.
"""
function kaldimel(; nmels::Int = 80, padded::Int = 512, sr = 16000.0, low = 20.0)
    mel(f) = 1127.0 * log(1.0 + f / 700.0)
    high = 0.5 * sr
    width = Float32(sr / padded)
    lo, hi = mel(low), mel(high)
    delta = Float32((hi - lo) / (nmels + 1))
    lo32 = Float32(lo)
    nfft = padded ÷ 2
    m = Float32[1127f0 * log(1f0 + (width * Float32(k)) / 700f0) for k in 0:(nfft - 1)]
    W = zeros(Float32, nmels, nfft + 1)
    for b in 0:(nmels - 1)
        left = lo32 + Float32(b) * delta
        centre = lo32 + (Float32(b) + 1f0) * delta
        right = lo32 + (Float32(b) + 2f0) * delta
        for k in 1:nfft
            up = (m[k] - left) / (centre - left)
            down = (right - m[k]) / (right - centre)
            W[b + 1, k] = max(0f0, min(up, down))
        end
    end
    return W
end

"""
    slaneymel(sr, nfft, nmels; fmin = 0, fmax = sr / 2) -> Matrix{Float32}

`librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)`: Slaney mel scale, Slaney
area normalisation, built in float64, `(nmels, nfft ÷ 2 + 1)`.
"""
function slaneymel(sr::Real, nfft::Int, nmels::Int; fmin::Real = 0.0, fmax::Real = sr / 2)
    fsp = 200.0 / 3
    minloghz = 1000.0
    minlogmel = minloghz / fsp
    logstep = log(6.4) / 27.0
    hz2mel(f) = f >= minloghz ? minlogmel + log(f / minloghz) / logstep : f / fsp
    mel2hz(m) = m >= minlogmel ? minloghz * exp(logstep * (m - minlogmel)) : fsp * m
    nf = nfft ÷ 2 + 1
    fftf = [k * (Float64(sr) / nfft) for k in 0:(nf - 1)]
    melf = mel2hz.(linspace(hz2mel(Float64(fmin)), hz2mel(Float64(fmax)), nmels + 2))
    W = zeros(Float64, nmels, nf)
    for i in 1:nmels, k in 1:nf
        lower = -(melf[i] - fftf[k]) / (melf[i + 1] - melf[i])
        upper = (melf[i + 2] - fftf[k]) / (melf[i + 2] - melf[i + 1])
        W[i, k] = max(0.0, min(lower, upper))
    end
    for i in 1:nmels
        W[i, :] .*= 2.0 / (melf[i + 2] - melf[i])
    end
    return Float32.(W)
end

"""
    sinckernel(orig, new; width = 6, rolloff = 0.99) -> (Matrix{Float32}, pad)

`torchaudio.functional._get_sinc_resample_kernel` with `sinc_interp_hann`, for
rates already divided by their gcd: built in float64 and stored as float32, one
column of `2 pad + orig` taps per output phase, `(2 pad + orig, new)`.
"""
function sinckernel(orig::Int, new::Int; lowpass::Int = 6, rolloff::Float64 = 0.99)
    base = min(orig, new) * rolloff
    pad = ceil(Int, lowpass * orig / base)
    K = zeros(Float32, 2pad + orig, new)
    for p in 0:(new - 1), (j, s) in enumerate(-pad:(pad + orig - 1))
        # `arange(0, -new, -1) / new` is an integer tensor divided in float32.
        t = Float64(Float32(-p) / Float32(new)) + s / orig
        t *= base
        t = clamp(t, -lowpass, lowpass)
        window = cos(t * π / lowpass / 2)^2
        t *= π
        k = t == 0 ? 1.0 : sin(t) / t
        K[j, p + 1] = Float32(k * (window * (base / orig)))
    end
    return K, pad
end

# ── kernels ──────────────────────────────────────────────────────────────────

"""Polyphase sinc resampling: output `i` is phase `i % new` of block `i ÷ new`."""
function sincresample!(out, x, K, orig::Int32, new::Int32, pad::Int32, n::Int32, nout::Int32)
    i = Int32(KI.get_global_id().x)
    i <= nout || return nothing
    f, p = divrem(i - Int32(1), new)
    base = f * orig - pad
    acc = 0f0
    @inbounds for j in Int32(1):Int32(size(K, 1))
        s = base + j - Int32(1)
        if s >= Int32(0) && s < n
            acc += K[j, p + Int32(1)] * x[s + Int32(1)]
        end
    end
    @inbounds out[i] = acc
    return nothing
end

"""Each Kaldi frame's mean, for `remove_dc_offset`: one thread a frame."""
function framemeans!(means, x, scale::Float32, len::Int32, hop::Int32, nframes::Int32)
    f = Int32(KI.get_global_id().x)
    f <= nframes || return nothing
    base = (f - Int32(1)) * hop
    s = 0f0
    @inbounds for n in Int32(1):len
        s += x[base + n] * scale
    end
    @inbounds means[f] = s / Float32(len)
    return nothing
end

"""
Kaldi frames, `(padded, nframes)`: `scale x` less its frame mean, pre-emphasised
(the first sample against itself, as torchaudio's replicate pad), windowed, and
zero past `len`.
"""
function kaldiframes!(frames, x, means, window, scale::Float32, len::Int32, hop::Int32,
                      padded::Int32, nframes::Int32)
    g = Int32(KI.get_global_id().x) - Int32(1)
    g < padded * nframes || return nothing
    n, f = g % padded, g ÷ padded
    @inbounds if n >= len
        frames[g + Int32(1)] = 0f0
    else
        m = means[f + Int32(1)]
        base = f * hop
        v = x[base + n + Int32(1)] * scale - m
        prev = n == Int32(0) ? v : x[base + n] * scale - m
        frames[g + Int32(1)] = (v - 0.97f0 * prev) * window[n + Int32(1)]
    end
    return nothing
end

"""`log(max(floor, mel · |X|²))`, `(nmels, nframes)`."""
function powermel!(out, spec, mel, floor::Float32, nbins::Int32, nmels::Int32, nframes::Int32)
    g = Int32(KI.get_global_id().x) - Int32(1)
    g < nmels * nframes || return nothing
    m, t = g % nmels + Int32(1), g ÷ nmels + Int32(1)
    acc = 0f0
    @inbounds for k in Int32(1):nbins
        z = spec[k, t]
        acc += mel[m, k] * (real(z) * real(z) + imag(z) * imag(z))
    end
    @inbounds out[g + Int32(1)] = log(max(acc, floor))
    return nothing
end

"""
Per-row mean and unbiased variance over the columns of `x` `(rows, n)`: one
workgroup a row, a strided share per item, then a tree over workgroup memory —
the shape of a pairwise sum rather than a running one.
"""
function rowstats!(mean, var, x, n::Int32, ::Val{G}) where {G}
    l = Int32(KI.get_local_id().x)
    r = Int32(KI.get_group_id().x)
    acc = KI.localmemory(Float32, Val((G,)), Val(1))
    s = 0f0
    i = l
    @inbounds while i <= n
        s += x[r, i]
        i += Int32(G)
    end
    @inbounds acc[l] = s
    KI.barrier()
    half = Int32(G ÷ 2)
    while half >= Int32(1)
        @inbounds l <= half && (acc[l] += acc[l + half])
        KI.barrier()
        half ÷= Int32(2)
    end
    @inbounds mu = acc[1] / Float32(n)
    KI.barrier()
    s = 0f0
    i = l
    @inbounds while i <= n
        d = x[r, i] - mu
        s += d * d
        i += Int32(G)
    end
    @inbounds acc[l] = s
    KI.barrier()
    half = Int32(G ÷ 2)
    while half >= Int32(1)
        @inbounds l <= half && (acc[l] += acc[l + half])
        KI.barrier()
        half ÷= Int32(2)
    end
    if l == Int32(1)
        @inbounds mean[r] = mu
        @inbounds var[r] = acc[1] / Float32(n - Int32(1))
    end
    return nothing
end

"""
SeamlessM4T's last two steps: each mel bin normalised, then frames paired into
`(2 nmels, ceil(nframes / 2))`, an odd count padded with a zero frame.
"""
function pairnormalised!(out, x, mean, var, nmels::Int32, nframes::Int32, npairs::Int32)
    g = Int32(KI.get_global_id().x) - Int32(1)
    g < Int32(2) * nmels * npairs || return nothing
    c, k = g % (Int32(2) * nmels), g ÷ (Int32(2) * nmels)
    frame = Int32(2) * k + c ÷ nmels + Int32(1)
    b = c % nmels + Int32(1)
    @inbounds out[g + Int32(1)] = frame <= nframes ?
        (x[b, frame] - mean[b]) / sqrt(var[b] + 1f-7) : 0f0
    return nothing
end

"""`x` less its row mean: torchaudio's fbank minus `feat.mean(dim=0)`."""
function centrerows!(out, x, mean, nrows::Int32, n::Int32)
    g = Int32(KI.get_global_id().x)
    g <= nrows * n || return nothing
    r = (g - Int32(1)) % nrows + Int32(1)
    @inbounds out[g] = x[g] - mean[r]
    return nothing
end

"""s2mel's frames: reflect-padded by `pad` on both sides, Hann-windowed, `center = False`."""
function reflectframes!(frames, x, window, nfft::Int32, hop::Int32, pad::Int32, n::Int32, nframes::Int32)
    g = Int32(KI.get_global_id().x) - Int32(1)
    g < nfft * nframes || return nothing
    i, t = g % nfft, g ÷ nfft
    p = t * hop + i - pad
    p < Int32(0) && (p = -p)
    p >= n && (p = Int32(2) * n - Int32(2) - p)
    @inbounds frames[g + Int32(1)] = x[p + Int32(1)] * window[i + Int32(1)]
    return nothing
end

"""
s2mel's `log(clamp(mel · sqrt(|X|² + 1e-9), 1e-5))`, written `(nframes, nmels)`:
the `(1, 80, t)` torch tensor in the reversed layout the graphs read.
"""
function magmel!(out, spec, mel, nbins::Int32, nmels::Int32, nframes::Int32)
    g = Int32(KI.get_global_id().x) - Int32(1)
    g < nmels * nframes || return nothing
    t, m = g % nframes + Int32(1), g ÷ nframes + Int32(1)
    acc = 0f0
    @inbounds for k in Int32(1):nbins
        z = spec[k, t]
        acc += mel[m, k] * sqrt(real(z) * real(z) + imag(z) * imag(z) + 1f-9)
    end
    @inbounds out[g + Int32(1)] = log(max(acc, 1f-5))
    return nothing
end

# ── the front end ────────────────────────────────────────────────────────────

"""
    Frontend(dev)

The tables of all three front ends and the resampler, on `dev`.
"""
struct Frontend{V,M}
    povey::V
    hann::V
    seamless::M
    kaldi::M
    s2mel::M
    sinc::M
    sincpad::Int
end

function Frontend(dev)
    up(x) = Mantle.Buffer(dev, x)
    K, pad = sinckernel(441, 320)
    Frontend(up(poveywindow(400)), up(hannwindow(1024)), up(seamlessmel()), up(kaldimel()),
             up(slaneymel(22050, 1024, 80)), up(K), pad)
end

kaldiframecount(n::Int) = n < 400 ? 0 : 1 + (n - 400) ÷ 160
s2melframecount(n::Int) = (n + 2 * 384 - 1024) ÷ 256 + 1
resampledlength(n22::Int) = ceil(Int, 320 * n22 / 441)

"""
    resample16k!(g, fe, x22, n22) -> 16 kHz transient

Declare `torchaudio.transforms.Resample(22050, 16000)`.
"""
function resample16k!(g, fe::Frontend, x22, n22::Int)
    n16 = resampledlength(n22)
    x16 = Mantle.Transient.Buffer(g, Float32, n16)
    Mantle.dispatch!(g, sincresample!, (x16, x22, fe.sinc, Int32(441), Int32(320), Int32(fe.sincpad),
                                        Int32(n22), Int32(n16)), n16; name = "resample 16k")
    return x16, n16
end

"""
    kaldimel!(g, fe, x16, n; scale, mel, floor, name) -> (80, frames) transient

Declare Kaldi framing, the 512-point power spectrum and the log mel energies.
"""
function kaldimel!(g, fe::Frontend, x16, n::Int; scale::Float32, mel, floor::Float32, name)
    nf = kaldiframecount(n)
    means = Mantle.Transient.Buffer(g, Float32, nf)
    frames = Mantle.Transient.Buffer(g, Float32, (512, nf))
    packed = Mantle.Transient.Buffer(g, ComplexF32, (256, nf))
    spectrum = Mantle.Transient.Buffer(g, ComplexF32, (256, nf))
    spec = Mantle.Transient.Buffer(g, ComplexF32, (257, nf))
    out = Mantle.Transient.Buffer(g, Float32, (80, nf))
    Mantle.dispatch!(g, framemeans!, (means, x16, scale, Int32(400), Int32(160), Int32(nf)), nf;
                     name = "$name/means")
    Mantle.dispatch!(g, kaldiframes!, (frames, x16, means, fe.povey, scale, Int32(400), Int32(160),
                                       Int32(512), Int32(nf)), 512nf; name = "$name/frames")
    Mantle.rfft_dispatch!(g, spec, frames, packed, spectrum; name = "$name/rfft")
    Mantle.dispatch!(g, powermel!, (out, spec, mel, floor, Int32(257), Int32(80), Int32(nf)), 80nf;
                     name = "$name/mel")
    return out, nf
end

"""
    seamlessfeatures!(g, fe, x16, n16; group) -> ((160, pairs) transient, mask, pairs)

Declare `SeamlessM4TFeatureExtractor`: 16-bit range, `mel_floor = 1.192092955078125e-07`,
each bin normalised over the clip, frames paired. An odd frame count is padded
with a zero frame, and `mask`, the extractor's attention mask as float
`(pairs, 1)`, then excludes the last pair, as w2v-BERT reads it.
"""
function seamlessfeatures!(g, fe::Frontend, x16, n16::Int; group::Int = 256)
    mel, nf = kaldimel!(g, fe, x16, n16; scale = 32768f0, mel = fe.seamless,
                        floor = 1.192092955078125f-7, name = "seamless")
    nf >= 2 || throw(ArgumentError("seamless features: $(n16 / 16000) s of audio is too short"))
    npairs = cld(nf, 2)
    mean = Mantle.Transient.Buffer(g, Float32, 80)
    var = Mantle.Transient.Buffer(g, Float32, 80)
    out = Mantle.Transient.Buffer(g, Float32, (160, npairs, 1))
    Mantle.dispatch!(g, rowstats!, (mean, var, mel, Int32(nf), Val(group)), 80group; group,
                     name = "seamless/stats")
    Mantle.dispatch!(g, pairnormalised!, (out, mel, mean, var, Int32(80), Int32(nf), Int32(npairs)),
                     160npairs; name = "seamless/pairs")
    mask = Mantle.Buffer(g.dev, Float32[i < npairs || iseven(nf) ? 1f0 : 0f0 for i in 1:npairs, _ in 1:1])
    return out, mask, npairs
end

"""
    kaldifbank!(g, fe, x16, n16; group) -> ((80, frames) transient, frames)

Declare `torchaudio.compliance.kaldi.fbank(x16, num_mel_bins = 80, dither = 0)`
less its mean over time, the CAMPPlus input.
"""
function kaldifbank!(g, fe::Frontend, x16, n16::Int; group::Int = 256)
    mel, nf = kaldimel!(g, fe, x16, n16; scale = 1f0, mel = fe.kaldi, floor = eps(Float32), name = "fbank")
    mean = Mantle.Transient.Buffer(g, Float32, 80)
    var = Mantle.Transient.Buffer(g, Float32, 80)
    out = Mantle.Transient.Buffer(g, Float32, (80, nf, 1))
    Mantle.dispatch!(g, rowstats!, (mean, var, mel, Int32(nf), Val(group)), 80group; group,
                     name = "fbank/stats")
    Mantle.dispatch!(g, centrerows!, (out, mel, mean, Int32(80), Int32(nf)), 80nf; name = "fbank/centre")
    return out, nf
end

"""
    s2melmel!(g, fe, x22, n22, out)

Declare s2mel's `mel_spectrogram` of the 22.05 kHz signal into `out`, `(frames, 80, 1)`.
"""
function s2melmel!(g, fe::Frontend, x22, n22::Int, out)
    nm = s2melframecount(n22)
    frames = Mantle.Transient.Buffer(g, Float32, (1024, nm))
    packed = Mantle.Transient.Buffer(g, ComplexF32, (512, nm))
    spectrum = Mantle.Transient.Buffer(g, ComplexF32, (512, nm))
    spec = Mantle.Transient.Buffer(g, ComplexF32, (513, nm))
    Mantle.dispatch!(g, reflectframes!, (frames, x22, fe.hann, Int32(1024), Int32(256), Int32(384),
                                         Int32(n22), Int32(nm)), 1024nm; name = "s2mel/frames")
    Mantle.rfft_dispatch!(g, spec, frames, packed, spectrum; name = "s2mel/rfft")
    Mantle.dispatch!(g, magmel!, (out, spec, fe.s2mel, Int32(513), Int32(80), Int32(nm)), 80nm;
                     name = "s2mel/mel")
    return out
end
