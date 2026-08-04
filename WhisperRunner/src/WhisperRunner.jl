"""
Whisper large-v3-turbo — speech -> text.

Transcription, and the transcript-driven editing that follows from it. 809M parameters, ~1.6 GB in fp16.

The reason this one is first is not the feature. It is the only model in the set that decodes autoregressively, so it forces two things the runtime does not have: an FFT for the log-mel front end, and a KV cache. The FFT is shared with every other audio model here; the KV cache puts the engine in a bandwidth-bound batch-1 GEMV regime that none of the GEMM tiling work in perf-plan.md applies to.

Both halves are ported. The encoder is one 617-op graph per 30 s window; the
decoder is a 96-op *step* plus a host loop, because the loop is not a static
graph and a step is (`tools/export_whisper_decoder.py` has the decomposition).

    w = whisper()                        # both graphs, weights resident, KV cache
    text, segs = transcribe(w, "talk.mp4")   # or a Vector of 16 kHz samples

    m = whispermodel()          # or the encoder alone, at half the download
    h = encode(m, mel)          # mel :: (3000, 128, 1) log-mel, host or device

`transcribe` is the whole pipeline: ffmpeg -> log-mel on the GPU -> encoder ->
the 30 s window loop with language detection, temperature fallback, the
no-speech test, timestamp rules and context carry-over -> BPE -> text and
`Segment`s. `window.jl` and `policy.jl` are ports of whisper.cpp's
`whisper_full_with_state` and `whisper_process_logits`; `tokenizer.jl` follows
llama.cpp's `llm_tokenizer_bpe`, not whisper.cpp's greedy longest-match, because
matching HuggingFace exactly is the acceptance criterion.

Measured against the PyTorch fp32 reference through the rewritten graph and a
planned slab: encoder **rel rms 6.30e-5, cosine 0.999999998**, no NaN under a
slab poisoned with `0xff` first (GUARDRAILS 3). `tools/verify_whisper.jl gpu
fp32` re-runs it; `tools/verify_whisper_decoder.jl` does the decoder, node by
node and then as a token sequence against `generate`.

## fp32, and why not fp16

The artifact carries the **fp32** export, at 2.37 GiB of weights. There is an
fp16 export beside it in `gen/graphs/whisper-fp16` (1.19 GiB, and it would halve
the download) and it is **not shipped, because it does not currently match**:
rel rms 0.137 against PyTorch's *own* fp16 reference, where the gate is 0.03.
Even the first two blocks fail 2 of 49 ops, worst at `addmm_3` — a GEMM, which is
the shape of the problem `verify_whisper.jl`'s docstring already names ("in fp16
the two paths differ by two orders of magnitude in accuracy", meaning whether a
matmul reaches the cooperative-matrix path).

For scale, PyTorch fp16 vs PyTorch fp32 on this encoder is rel rms 0.0254 — so
fp16 costs something real here even done right, and ours costs 5x that. Fixing it
is a kernel-accuracy task, not a packaging one.

Upstream: https://github.com/openai/whisper
License: **MIT**

The log-mel front end runs on the GPU through `Lava.stft` (0.712 ms), with
`melfilters` building the Slaney filterbank.

`greedy` remains beside the policy-driven path on purpose: it is a bare argmax
loop, it is what the token-for-token parity test in
`tools/verify_whisper_decoder.jl` checks, and keeping it means a policy bug can
never be mistaken for a decoder bug.

See `models-to-port.md` for the state of this one, `tools/export_whisper.py` and
`tools/export_whisper_decoder.py` for the exports that feed it.
"""
module WhisperRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using LazyArtifacts, JSON3, Random, FFMPEG_jll
using DNNKernels: loadgraph, execute!, readsafetensors, toback, Model, call,
                  Ctx, logmelspectrogram, melfilters

export whispergraph, whisperweights
export whispermodel, encode, WhisperEncoder
export Whisper, whisper, transcribe, greedy, decodestep!, KVCache
export Tokenizer, tokenizer, encode_text, decode_tokens
export DecodeOptions, Segment, detectlanguage, readaudio, transcribechunk

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Where the encoder's graph and weights live: its artifact, downloaded on first use
and cached across every environment on this machine. **1.33 GiB** — the largest
of the set, because these are fp32 weights and the fp16 export does not yet match
(see the module docstring).

**Changing these assets means re-binding the artifact**, not editing a directory.
Re-export, then `julia --project=. tools/make_artifacts.jl whisper` — that hashes
the new content and rewrites `../Artifacts.toml`, so this call resolves to it
immediately. Uploading is only needed to publish it to anyone else.
"""
assetdir() = @artifact_str("whisper")

"""
    whispergraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function whispergraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "whisper.json")
    isfile(p) || throw(ArgumentError(
        "Whisper large-v3-turbo graph not found at $p. Generate it with " *
        "`uv run tools/export_whisper.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl`."))
    return loadgraph(p)
end

"""
    whisperweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function whisperweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Whisper large-v3-turbo weights not found at $p"))
    return readsafetensors(p)
end

"""
    decoderdir() -> String

Where the DECODER's graph and weights live — a separate artifact from the
encoder's.

Separate on purpose. The encoder is 1.33 GiB and the decoder another 909 MiB, and
a caller who wants embeddings (or the editor, which wants the encoder for
alignment) should not download the autoregressive half to get them. They are also
different exports with different lifetimes: re-exporting one does not invalidate
the other.

Throws until the artifact is bound. Produce it with
`uv run tools/export_whisper_decoder.py`, then
`julia --project=. tools/make_artifacts.jl whisper-decoder`.
"""
decoderdir() = @artifact_str("whisper-decoder")

"""
    decoderready(; dir = decoderdir()) -> Bool

Whether the decoder half is installed. Separate from [`ready`](@ref) because the
encoder alone is useful and the decoder alone is not.
"""
decoderready(; dir::AbstractString = decoderdir()) =
    all(isfile(joinpath(dir, f)) for f in
        ("whisperdec.json", "whispercross.json", "weights.safetensors",
         "tokenizer.json", "generation_config.json"))

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "whisper.json")) && isfile(joinpath(dir, "weights.safetensors"))

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Lava.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ------------------------------------------------------------------- the model

"""
    WhisperEncoder

A loaded encoder: the rewritten graph, its weights on the device, and the backend
they belong to. Build one with [`whispermodel`](@ref) and hand it to
[`encode`](@ref).

Distinct from [`Whisper`](@ref), which carries both halves and a KV cache. The
encoder alone is the useful component for alignment and embeddings, and it is
1.33 GiB against the pair's 2.2 GiB.

Holding the `Model` rather than rebuilding it per call is the whole point — the
2.37 GiB of weights are uploaded once, and `Model`'s rewrite passes (above all
`hoistpermutes`, which is what decides whether a matmul reaches the
cooperative-matrix path) run once.
"""
struct WhisperEncoder{B,M}
    backend::B
    model::M
end

"""
    whispermodel(; backend = LavaBackend(), dir = assetdir()) -> WhisperEncoder

Load the encoder. Downloads the artifact on first use.

Not called at module scope and not cached in a global: a `Model` holds device
buffers, and a module global holding one is baked into the package image with a
`VkContext` that is dead by the time anyone loads it.
"""
function whispermodel(; backend = LavaBackend(), dir::AbstractString = assetdir())
    ready(; dir) || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_whisper.py`"))
    WhisperEncoder(backend, Model(dir, joinpath(dir, "weights.safetensors");
                                  names = ["whisper"], backend))
end

"""
    encode(w::WhisperEncoder, mel) -> AbstractArray

Run the encoder over one 30 s window. `mel` is the log-mel spectrogram in the
export's own layout, `(3000, 128, 1)` — 3000 frames of 128 mel bins — either on
the host or already on the device.

Returns the `(1280, 1500, 1)` hidden state, on the device. The caller decides
whether to download it; a decoder would not.

**There is no mel front end here yet.** Whisper's own is an STFT, and
`DNNKernels` has no FFT, so producing `mel` is the caller's problem for now (or
the reference dump's — `refs["whisper/in0"]` is exactly one).
"""
function encode(w::WhisperEncoder, mel)
    out, = call(w.model, "whisper", toback(w.backend, mel); dims = (;))
    return out
end

# Order matters: `tokenizer.jl` defines `Tokenizer`, which `policy.jl` dispatches
# on, and `decode.jl`'s `whisper` calls `policy.jl`'s `suppresslists` at load
# time — so the two data files come before the two that use them.
include("tokenizer.jl")
include("policy.jl")
include("decode.jl")
include("window.jl")

# `encode`/`decode` are already taken here — `encode(::WhisperEncoder, mel)` is
# the audio encoder — so the tokenizer's pair is exported under names that say
# which direction they go. Inside the package they keep the short names, which is
# what every reference calls them.
const encode_text = encode
const decode_tokens = decode

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
            # `whispermodel` INSIDE `@compile_workload`, not in front of it —
            # `Model`'s last pass folds constant subgraphs by running them on the
            # device, and building it outside leaves those dispatches unfrozen.
            # RIFERunner measured exactly that: `frozen_stats().misses == 9` on a
            # fresh process, every time, no matter the frame size.
            #
            # One real window, not a token shape: the encoder's extents are baked
            # (3000 frames in, 1500 out), so there is only one shape to freeze and
            # a smaller one would freeze the wrong kernels.
            @compile_workload KERNELS_VERSION begin
                w = whispermodel(; backend)
                mel = KA.allocate(backend, Float32, 3000, 128, 1)
                fill!(mel, 0.0f0)
                encode(w, mel)
                KA.synchronize(backend)
            end
        catch err
            @warn "WhisperRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "WhisperRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
