"""
Demucs v4 (htdemucs) — stem separation.

Splits a mix into vocals / drums / bass / other. In an edit that means pulling dialogue off a music bed, or replacing the bed and keeping the dialogue. ~300 MB.

A different feature from DeepFilterNet3, not a better one: separation, not denoising. It will not clean a noisy recording and DFN3 will not remove a song. Both are cheap and both want the same FFT.

Hybrid Transformer Demucs runs two branches — waveform and spectrogram — and sums them, so the graph carries a real FFT *inside* it rather than only in a front end. That makes it the model that decides whether the FFT is a proper device kernel or a host convenience.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`demucsgraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/adefossez/demucs
License: **MIT**

Ops `DNNKernels` does not have yet:
  * stft / istft on device
  * lstm (the encoder's bottleneck)

See `models-to-port.md` for the state of this one, and `tools/export_demucs.py`
for the export that feeds it.
"""
module DemucsRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, execute!, readsafetensors

export demucsgraph, demucsweights

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

The **upstream Demucs v4 (htdemucs) checkpoint**, as an artifact — downloaded on first use and
cached across every environment on this machine.

**This is not an export.** Demucs v4 (htdemucs) is not ported: nothing has been traced into a
`DNNKernels` graph, so [`demucsgraph`](@ref) still throws and [`ready`](@ref) is
still `false`. What the artifact buys is that the *port* can start on any machine
without re-fetching by hand.

What is actually blocking the port:
Hybrid Transformer Demucs runs a waveform and a spectrogram branch and sums
them, so the graph carries a real FFT *inside* it rather than only in a front
end — it is the model that decides whether the FFT is a device kernel or a host
convenience.

When the export lands, pack `gen/graphs/demucs` under the artifact name
`demucs` (the `MODELS` table in `tools/make_artifacts.jl`), point this at
`@artifact_str("demucs")`, and the checkpoint artifact becomes developer-only.
"""
assetdir() = @artifact_str("demucs-ckpt")

"""
    demucsgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function demucsgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "demucs.json")
    isfile(p) || throw(ArgumentError(
        "Demucs v4 (htdemucs) graph not found at $p. Generate it with " *
        "`uv run tools/export_demucs.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl`."))
    return loadgraph(p)
end

"""
    demucsweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function demucsweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Demucs v4 (htdemucs) weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
# `false`, and not a placeholder: `assetdir()` resolves — it carries the upstream
# checkpoint — but a checkpoint is not a graph. `ready()` answers "can this package
# run the model", which stays false until `demucs.json` exists.
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "demucs.json")) && isfile(joinpath(dir, "weights.safetensors"))

"""
    checkpoints(; dir = assetdir()) -> Vector{String}

The upstream checkpoint files the artifact carries, absolute. What
`tools/export_demucs.py` will read when the port starts.
"""
checkpoints(; dir::AbstractString = assetdir()) =
    [joinpath(dir, f) for f in sort(readdir(dir)) if isfile(joinpath(dir, f))]

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
            graph = demucsgraph()
            weights = demucsweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: waveform (1, 2, 343980) at 44.1 kHz
                nothing
            end
        catch err
            @warn "DemucsRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "DemucsRunner: not ported yet — nothing precompiled"
    end
end

end # module
