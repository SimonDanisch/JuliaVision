"""
Kokoro-82M — text -> speech, on Lava.

82M parameters, 54 fixed voices, 24 kHz. StyleTTS2-derived: an ALBERT text
encoder, a prosody predictor, and an iSTFTNet vocoder, with six **bidirectional
LSTMs** threaded through them. Those LSTMs are what made this the awkward one —
`run_decompositions()` unrolls a recurrent layer into one copy of the cell per
timestep, so the graphs came out at 7298 ops for a single short utterance and
grew with the sentence.

Two things fixed that, and both live below the model:

  * `DNNKernels` implements `aten::lstm` as **one op** with the timestep loop
    inside a single kernel — 532 nodes per timestep became 1.
  * the export is **length-generic**: `torch.export` cannot keep the sequence
    symbolic through an LSTM, so the axis is recovered afterwards by exporting at
    three lengths and fitting the integers that move. One export runs any text.

## The model is cut in two, because it chooses its own output length

`KModel.forward_with_tokens` predicts a per-token duration, rounds it, and builds
an alignment whose width is `sum(pred_dur)` — the length of the audio, decided by
the model from its own logits. No exporter can bake that, so:

  * [`Kokoro`](@ref) runs **`kokorotext`** — `(ids, ref_s) -> (d, t_en, duration)`
  * [`align`](@ref) rounds the durations and expands them, on the host
  * then **`kokorovoc`** — `(en, asr, ref_s) -> audio`

The alignment the reference builds is a one-hot `ntokens x nframes` matrix it
multiplies by; here it is the index vector that matrix would have selected, and
the "matmul" is a gather.

Upstream: https://github.com/hexgrad/kokoro — Apache-2.0.
"""
module KokoroRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using JSON3
using DNNKernels: loadgraph, execute!, readsafetensors, toback, Model, call

export Kokoro, speak, phonemize, pronounce!, voices, SAMPLERATE

const KA = KernelAbstractions

"""
    SAMPLERATE

24000. Kokoro's iSTFTNet decoder emits **600 samples per prosody frame** — the
graph's output shape is literally `600*f` — so the frame rate is 40 Hz and the
sample rate follows from it. Writing a wav at any other rate plays the right
audio at the wrong pitch, which sounds like a bad voice rather than a bug.
"""
const SAMPLERATE = 24000

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

The exported graphs, weights, voice packs, phoneme vocabulary and G2P lexicon.
Downloaded on first use and cached across every environment on this machine.
"""
assetdir() = @artifact_str("kokoro")

"""
    ready() -> Bool

Whether a usable export is installed. The workload and the tests both branch on
this, because neither may fail on a machine that has not run the exporter.
"""
ready() =
    all(isfile(joinpath(assetdir(), f)) for f in
        ("kokorotext.json", "kokorovoc.json", "weights.safetensors",
         "vocab.json", "voices.safetensors"))

"""
    refsdir() -> String

The **parity fixtures**: PyTorch's audio for a set of phrases, and misaki's
phonemisation of a sentence corpus. A second, small artifact rather than part of
[`assetdir`](@ref) — a caller who wants to synthesise speech should not download
the test material, which is the same split `sam2-large-refs` makes beside
`sam2-large`.

An artifact and not a path into `gen/` for the reason `DNNKernels/src/assets.jl`
gives: a test whose fixtures come from a local export tree passes for whoever ran
the exporter and is unreachable for everyone else, so the gate that catches a
kernel which is fast and subtly wrong would run on exactly one machine.

Rebind with `julia --project=. tools/make_artifacts.jl kokoro-refs`.
"""
refsdir() = @artifact_str("kokoro-refs")

include("g2p.jl")

"""
    Kokoro(; backend = Mantle.LavaBackend(), dir = assetdir()) -> Kokoro

Load the model. Holds device weights and a scratch slab, so build it once and
keep it — the constructor is the expensive call, [`speak`](@ref) is not.

`voicepack` is `(256, 511)` per voice and is indexed **by phoneme count**: Kokoro
conditions prosody on how long the utterance is. Picking the wrong column is
silent — it gives fluent speech in a subtly wrong cadence, not an error — so
[`refstyle`](@ref) is the only thing that indexes it.
"""
struct Kokoro{B,M,V}
    backend::B
    model::M
    vocab::Dict{Char,Int}
    voicepack::Dict{String,V}
    lexicon::Lexicon
end

function Kokoro(; backend = Mantle.LavaBackend())
    dir = assetdir()
    ready() || throw(ArgumentError(
        "Kokoro assets incomplete in $dir. Generate them with " *
        "`uv run tools/export_kokoro.py` and `uv run tools/export_kokoro_assets.py`, " *
        "then bind with `julia --project=. tools/make_artifacts.jl`."))
    model = Model(Dict(n => loadgraph(joinpath(dir, "$n.json"))
                       for n in ("kokorotext", "kokorovoc")),
                  readsafetensors(joinpath(dir, "weights.safetensors")); backend)
    raw = JSON3.read(read(joinpath(dir, "vocab.json"), String))
    # The keys are single characters; the values are the ids the embedding
    # indexes with directly (0-based — `embedding.default` adds the 1).
    vocab = Dict{Char,Int}(only(String(k)) => Int(v) for (k, v) in pairs(raw))
    packs = readsafetensors(joinpath(dir, "voices.safetensors"))
    Kokoro(backend, model, vocab, packs, Lexicon(joinpath(dir, "lexicon.json")))
end

"""
    voices(k) -> Vector{String}

The 54 voice names, sorted. `af_*`/`am_*` are American, `bf_*`/`bm_*` British;
the rest are the other languages Kokoro ships, which need a G2P this package does
not have (see [`phonemize`](@ref)).
"""
voices(k::Kokoro) = sort!(collect(keys(k.voicepack)))

"""
    phonemize(k, text) -> String

Text -> misaki phonemes, using the lexicon this model was loaded with. See
`g2p.jl` for the algorithm and for what a missing part-of-speech tagger costs.
"""
phonemize(k::Kokoro, text::AbstractString) = phonemize(k.lexicon, text)

"""
    pronounce!(k, word => phonemes) -> k

Teach this model's G2P a word. See the `Lexicon` method for the why.
"""
pronounce!(k::Kokoro, p::Pair) = (pronounce!(k.lexicon, p); k)

"""
    refstyle(k, voice, nphonemes) -> (256, 1) on the device

The style vector, which is **two vectors concatenated and they are not
interchangeable**: `ref_s[129:256]` conditions the prosody predictor and
`ref_s[1:128]` the decoder. Passing one for both is silent — measured, it gives
speech-shaped audio at rel rms 1.3 from the reference, i.e. a different voice
rather than a broken one.

Indexed at `nphonemes`, matching `KPipeline`'s `pack[len(ps)-1]` one-for-one
(0-based there, 1-based here). Clamped at 511, the pack's height.
"""
function refstyle(k::Kokoro, voice::AbstractString, nphonemes::Integer)
    pack = get(k.voicepack, voice) do
        throw(ArgumentError("no voice $voice; have $(join(voices(k), ", "))"))
    end
    i = clamp(nphonemes, 1, size(pack, 2))
    toback(k.backend, reshape(pack[:, i], :, 1))
end

"""
    tokenize(k, phonemes) -> Vector{Int32}

Phoneme characters -> embedding ids, wrapped in the boundary token `0` at both
ends. Characters outside the vocabulary are **dropped**, which is what
`KModel.forward` does — a phoneme the model was never trained on has no id, and
substituting one would put a different sound in its place.
"""
function tokenize(k::Kokoro, phonemes::AbstractString)
    ids = Int32[0]
    for c in phonemes
        v = get(k.vocab, c, nothing)
        v === nothing || push!(ids, Int32(v))
    end
    push!(ids, Int32(0))
    ids
end

"""
    align(duration, speed) -> Vector{Int32}

Per-token durations -> the per-frame token index, on the host.

`pred_dur[i]` frames of token `i`, concatenated: this is exactly the column
`pred_aln_trg` would have selected, without materialising the matrix. `speed`
divides the durations, so `speed > 1` is faster speech.

The clamp at 1 is the reference's and it matters: a token whose duration rounds
to 0 would vanish from the alignment entirely.
"""
function align(duration::AbstractVector, speed::Real)
    idx = Int32[]
    for (i, d) in enumerate(duration)
        n = max(1, round(Int, d / speed))
        append!(idx, fill(Int32(i), n))
    end
    idx
end

"""
    gather_align_kernel!(en, asr, d, t_en, idx)

Both halves of the frame-level gather in one launch.

`idx` is the per-frame token index [`align`](@ref) produces, so this is
`en[j, c] = d[c, idx[j]]` and `asr[j, c] = t_en[idx[j], c]` — the duration
model's expansion of `t` tokens into `f` frames.

One kernel rather than two because `asr`'s 512 channels are a prefix of `en`'s
640: the wider launch covers both and the narrower write is one predicate, which
is cheaper than a second dispatch on a call that is already host-bound.

`j` varies fastest so the writes are coalesced — both destinations have the frame
axis first. The reads are not, and do not need to be: `idx` is a
`repeat_interleave`, so consecutive lanes mostly share one `i`.

**No `cpu = false`**, unlike the other kernels here, so this is the only copy of
the gather for every backend. It replaced a host loop that main had just fixed a
different way — a `gatherframes!` function barrier, to stop `dh[c, i, 1]` boxing
a scalar per element because `speak` does not type what `call` returns. That
diagnosis was right and the barrier worked; keeping both would have left two
implementations of one gather to drift apart, which is the thing that must not
happen.
"""
@kernel function gather_align_kernel!(en, asr, @Const(d), @Const(t_en),
                                      @Const(idx), F::Int32, C2::Int32)
    g = @index(Global, Linear) - 1
    j = g % Int(F) + 1
    c = g ÷ Int(F) + 1
    @inbounds i = Int(idx[j])
    @inbounds en[j, c, 1] = d[c, i, 1]
    @inbounds if c <= Int(C2)
        asr[j, c, 1] = t_en[i, c, 1]
    end
end

"""
    gather_align!(backend, en, asr, d, t_en, idx) -> nothing

Launch [`gather_align_kernel!`](@ref) over `size(en, 1) * size(en, 2)` elements.
"""
function gather_align!(backend, en, asr, d, t_en, idx)
    f, c1 = size(en, 1), size(en, 2)
    gather_align_kernel!(backend, 256)(en, asr, d, t_en, idx,
                                       Int32(f), Int32(size(asr, 2));
                                       ndrange = f * c1)
    nothing
end

"""
    speak(k, text; voice, speed, trim) -> Vector{Float32}

Text -> 24 kHz mono audio. The whole path: [`phonemize`](@ref), tokenize, the
text graph, [`align`](@ref), the vocoder graph.

`speed` scales the predicted durations. `trim` drops the leading and trailing
silence the boundary tokens produce; the reference's `KPipeline` does the same
thing by other means.

`noise = false` zeroes the vocoder's excitation noise. **That is a measurement
setting, not a quality one** — the noise is a real part of Kokoro's
harmonic-plus-noise source, and without it the voice is thinner. It exists
because the model is otherwise stochastic and cannot be compared to a reference
at all; see `DNNKernels.NoiseSource`.

Pass `phonemes` instead of `text` to skip the G2P entirely.
"""
function speak(k::Kokoro, text::AbstractString; voice::AbstractString = "af_heart",
               speed::Real = 1.0, trim::Bool = true, noise::Bool = true)
    speak(k; phonemes = phonemize(k, text), voice, speed, trim, noise)
end

function speak(k::Kokoro; phonemes::AbstractString, voice::AbstractString = "af_heart",
               speed::Real = 1.0, trim::Bool = true, noise::Bool = true)
    ids = tokenize(k, phonemes)
    t = length(ids)
    ref_s = refstyle(k, voice, length(phonemes))

    # (t, 1): the graphs carry torch's shapes, and a `LavaArray`'s dimensions are
    # the reverse of them.
    d, t_en, duration = call(k.model, "kokorotext",
                             toback(k.backend, reshape(ids, :, 1)), ref_s;
                             dims = (t = t,))

    idx = align(vec(Array(duration)), speed)
    f = length(idx)

    # `en` is `d` transposed and gathered; `asr` is `t_en` gathered, **on the
    # device**. Only `duration` — `t` floats — has to reach the host, because the
    # model decides its own output length and `f` is a launch dimension.
    #
    # This was a host loop, defended on the grounds that "the gather is `f` int
    # reads and the graphs want contiguous inputs, so a device kernel here would
    # buy nothing and cost an upload either way". Right about the bytes (0.83 MB
    # round trip) and wrong about everything else, measured 2026-08-12:
    #
    #   * `call` returns `Any`, so `dh`/`teh` were untyped and the inner loops
    #     **boxed every element** — 166 660 of the utterance's 324 967 host
    #     allocations, from two lines;
    #   * three extra device syncs, on a call where the GPU is already idle 60% of
    #     the time against SAM 2's 3%.
    #
    # The gather itself is `f * 1152` reads and the device does it in one launch.
    en = KA.allocate(k.backend, Float32, f, 640, 1)
    asr = KA.allocate(k.backend, Float32, f, 512, 1)
    gather_align!(k.backend, en, asr, d, t_en, toback(k.backend, idx))

    audio, = call(k.model, "kokorovoc", en, asr, ref_s; dims = (f = f,),
                  noise = noise ? DNNKernels.RandomNoise() : DNNKernels.ZeroNoise())
    out = Array(audio)
    trim ? trimsilence(out) : out
end

"""
    trimsilence(x; thresh = 1e-3) -> view

Drop the leading and trailing near-silence. The boundary token at each end is a
real token with a real predicted duration, so the model renders it — as a short
pause. Keeping it makes every clip start late and end long.
"""
function trimsilence(x::AbstractVector{Float32}; thresh::Real = 1e-3)
    a = findfirst(v -> abs(v) > thresh, x)
    b = findlast(v -> abs(v) > thresh, x)
    (a === nothing || b === nothing) && return x
    x[a:b]
end

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Mantle.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# The workload drives `speak` rather than the graphs directly, and that is the
# whole point — SAM2Runner learned the expensive way that a workload running a
# different path than the caller leaves the caller compiling on first use. The
# measurement that matters is `Lava.frozen_stats().misses == 0` on a fresh
# process, through this entry point.
@setup_workload begin
    if ready()
        try
            k = Kokoro()
            @compile_workload KERNELS_VERSION begin
                speak(k; phonemes = "hˈEllO", voice = "af_heart")
            end
        catch err
            @warn "KokoroRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "KokoroRunner: assets not installed — nothing precompiled"
    end
end

end # module
