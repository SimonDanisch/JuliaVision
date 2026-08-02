"""
Whisper large-v3-turbo — speech -> text.

Transcription, and the transcript-driven editing that follows from it. 809M parameters, ~1.6 GB in fp16.

The reason this one is first is not the feature. It is the only model in the set that decodes autoregressively, so it forces two things the runtime does not have: an FFT for the log-mel front end, and a KV cache. The FFT is shared with every other audio model here; the KV cache puts the engine in a bandwidth-bound batch-1 GEMV regime that none of the GEMM tiling work in perf-plan.md applies to.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`whispergraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/openai/whisper
License: **MIT**

Ops `DNNKernels` does not have yet:
  * stft / fft (host-side mel is acceptable to start)
  * KV-cache attention (incremental, not a new ATen op but a new execution mode)

See `models-to-port.md` for the state of this one, and `tools/export_whisper.py`
for the export that feeds it.
"""
module WhisperRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, assetpath

export whispergraph, whisperweights, assetdir

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Where the exported graph and weights live.

No `Artifacts.toml` yet, deliberately: a lazy artifact needs the sha256 of a
tarball that has been uploaded to a release, and there is nothing to upload
until the export runs. Until then `assetpath` falls through to the generated
directory, and the error message names the place it looked. Adding the artifact
is what turns a working port into an installable one.
"""
assetdir() = assetpath(; generated = joinpath("gen", "graphs", "whisper"),
                       env = "JULIA_WHISPER_ASSETS", from = @__DIR__)

"""
    whispergraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function whispergraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "whisper.json")
    isfile(p) || throw(ArgumentError(
        "Whisper large-v3-turbo graph not found at $p. Generate it with " *
        "`uv run tools/export_whisper.py`, or set JULIA_WHISPER_ASSETS."))
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

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# TODO(port): drive the real call here once the graph runs. The measurement that
# matters is `Lava.frozen_stats().misses == 0` on a *fresh* process — a workload
# that runs a different path than the editor does leaves the editor compiling on
# first use, which is the entire cost this package exists to remove. SAM2Runner
# learned that the expensive way: its `runsam2` workload still left 45 s on the
# first click because the editor goes through a closure `runsam2` never touches.
@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
            graph = whispergraph()
            weights = whisperweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: log-mel spectrogram (1, 128, 3000) + decoder token ids
                nothing
            end
        catch err
            @warn "WhisperRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "WhisperRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
