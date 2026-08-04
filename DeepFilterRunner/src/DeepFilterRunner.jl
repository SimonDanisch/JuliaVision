"""
DeepFilterNet3 — voice denoising.

Room tone, hum and hiss off a dialogue track, in real time. Roughly 2 MB of weights — the smallest model in the set by two orders of magnitude.

Nearly free once Whisper's FFT exists: it works in the complex STFT domain, and `view_as_complex`/`view_as_real` are already ops the runtime has.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`deepfilternetgraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/Rikorose/DeepFilterNet
License: **MIT or Apache-2.0 (dual)**

Ops `DNNKernels` does not have yet:
  * ERB filterbank (a small matmul, not a new op)

See `models-to-port.md` for the state of this one, and `tools/export_deepfilternet.py`
for the export that feeds it.
"""
module DeepFilterRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, execute!, readsafetensors

export deepfilternetgraph, deepfilternetweights

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

The **upstream DeepFilterNet3 checkpoint**, as an artifact — downloaded on first use and
cached across every environment on this machine.

**This is not an export.** DeepFilterNet3 is not ported: nothing has been traced into a
`DNNKernels` graph, so [`deepfilternetgraph`](@ref) still throws and [`ready`](@ref) is
still `false`. What the artifact buys is that the *port* can start on any machine
without re-fetching by hand.

What is actually blocking the port:
it works in the complex STFT domain, so it needs the same FFT Whisper's mel
front end does — nearly free once that exists.

When the export lands, pack `gen/graphs/deepfilternet` under the artifact name
`deepfilternet` (the `MODELS` table in `tools/make_artifacts.jl`), point this at
`@artifact_str("deepfilternet")`, and the checkpoint artifact becomes developer-only.
"""
assetdir() = @artifact_str("deepfilternet-ckpt")

"""
    deepfilternetgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function deepfilternetgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "deepfilternet.json")
    isfile(p) || throw(ArgumentError(
        "DeepFilterNet3 graph not found at $p. Generate it with " *
        "`uv run tools/export_deepfilternet.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl`."))
    return loadgraph(p)
end

"""
    deepfilternetweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function deepfilternetweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("DeepFilterNet3 weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
# `false`, and not a placeholder: `assetdir()` resolves — it carries the upstream
# checkpoint — but a checkpoint is not a graph. `ready()` answers "can this package
# run the model", which stays false until `deepfilternet.json` exists.
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "deepfilternet.json")) && isfile(joinpath(dir, "weights.safetensors"))

"""
    checkpoints(; dir = assetdir()) -> Vector{String}

The upstream checkpoint files the artifact carries, absolute. What
`tools/export_deepfilternet.py` will read when the port starts.
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
            graph = deepfilternetgraph()
            weights = deepfilternetweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: complex STFT frames (1, T, F, 2)
                nothing
            end
        catch err
            @warn "DeepFilterRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "DeepFilterRunner: not ported yet — nothing precompiled"
    end
end

end # module
