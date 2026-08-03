"""
BasicVSR++ — video upscaling.

Temporally consistent upscaling. 7.3M parameters, but the footprint is activations rather than weights — it is recurrent over a clip, so VRAM scales with sequence length.

The furthest along: `tools/export_basicvsrpp.py` already produces the graph into `gen/graphs/basicvsrpp-fp32`, and the runner package is what is missing.

Engine-wise the interesting part is flow-guided deformable alignment — DCNv2 is an irregular per-pixel gather with no clean coopmat mapping, and it is the hardest kernel in this set.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`basicvsrppgraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/open-mmlab/mmagic
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * deformable_conv2d (DCNv2)

See `models-to-port.md` for the state of this one, and `tools/export_basicvsrpp.py`
for the export that feeds it.
"""
module BasicVSRRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors

export basicvsrppgraph, basicvsrppweights

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Throws. BasicVSR++ is **not ported yet**, so there is no artifact to read
from and nothing on disk that a user of this package would have.

Porting it means, in order: export it with `uv run tools/export_basicvsrpp.py`, bind the
result with `julia --project=. tools/make_artifacts.jl basicvsrpp`, and replace
this definition with `@artifact_str("basicvsrpp")`. Assets come from the artifact
and from nowhere else — see `DNNKernels/src/assets.jl`.
"""
assetdir() = error(
    "BasicVSRRunner: BasicVSR++ is not ported yet, so no artifact is bound. " *
    "Export it with `uv run tools/export_basicvsrpp.py`, bind it with " *
    "`julia --project=. tools/make_artifacts.jl basicvsrpp`, then set " *
    "`assetdir() = @artifact_str(\"basicvsrpp\")`.")

"""
    basicvsrppgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function basicvsrppgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "basicvsrpp.json")
    isfile(p) || throw(ArgumentError(
        "BasicVSR++ graph not found at $p. Generate it with " *
        "`uv run tools/export_basicvsrpp.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl`."))
    return loadgraph(p)
end

"""
    basicvsrppweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function basicvsrppweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("BasicVSR++ weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready() = false        # not ported: see `assetdir`

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
            graph = basicvsrppgraph()
            weights = basicvsrppweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: clip (1, T, 3, H, W)
                nothing
            end
        catch err
            @warn "BasicVSRRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "BasicVSRRunner: not ported yet — nothing precompiled"
    end
end

end # module
