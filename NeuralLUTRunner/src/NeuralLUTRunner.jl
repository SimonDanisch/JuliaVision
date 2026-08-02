"""
Image-Adaptive 3D LUT — style / mood grading.

A look, applied as a grade. Under 600K parameters: a small CNN predicts blend weights over a handful of basis 3D LUTs, and the result is a LUT.

That output shape is the whole argument for it. Per-frame diffusion style transfer flickers and cannot be edited; a predicted LUT is a first-class grading object — it drops into the inspector, the user can push it around, and it keyframes with the machinery already built. Temporal stability comes from smoothing the predicted weights across frames rather than from the network.

Almost nothing new for the runtime, which is why it is second in the order: it is the fastest visible result in the set. The LUT apply is trilinear interpolation and belongs in GPUFiltering, not in the graph.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`neurallutgraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/HuiZeng/Image-Adaptive-3DLUT
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * trilinear 3D LUT apply (a GPUFiltering kernel, not an ATen op)

See `models-to-port.md` for the state of this one, and `tools/export_neurallut.py`
for the export that feeds it.
"""
module NeuralLUTRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, assetpath

export neurallutgraph, neurallutweights, assetdir

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
assetdir() = assetpath(; generated = joinpath("gen", "graphs", "neurallut"),
                       env = "JULIA_NEURALLUT_ASSETS", from = @__DIR__)

"""
    neurallutgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function neurallutgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "neurallut.json")
    isfile(p) || throw(ArgumentError(
        "Image-Adaptive 3D LUT graph not found at $p. Generate it with " *
        "`uv run tools/export_neurallut.py`, or set JULIA_NEURALLUT_ASSETS."))
    return loadgraph(p)
end

"""
    neurallutweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function neurallutweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Image-Adaptive 3D LUT weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "neurallut.json")) && isfile(joinpath(dir, "weights.safetensors"))

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
            graph = neurallutgraph()
            weights = neurallutweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: image (1, 3, H, W) in 0..1
                nothing
            end
        catch err
            @warn "NeuralLUTRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "NeuralLUTRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
