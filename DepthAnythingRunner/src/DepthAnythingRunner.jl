"""
Depth Anything V2 Small — monocular depth.

Depth per frame, which buys fake shallow depth of field, depth-keyed grading, parallax push-ins and sky masks. 25M parameters.

**Small specifically.** The Base and Large checkpoints are CC-BY-NC-4.0; only Small is Apache-2.0. Large is better and can be added later as its own non-commercial package, the way MatAnyone already is.

A ViT with scaled-dot-product attention and nothing else unusual, so the runtime needs nothing new. Pure editor value, no engine work — which is why it sits where it does in the order.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`depthanythinggraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://github.com/DepthAnything/Depth-Anything-V2
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * none — the runtime already covers it

See `models-to-port.md` for the state of this one, and `tools/export_depthanything.py`
for the export that feeds it.
"""
module DepthAnythingRunner

using Lava, DNNKernels, KernelAbstractions
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, assetpath

export depthanythinggraph, depthanythingweights, assetdir

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
assetdir() = assetpath(; generated = joinpath("gen", "graphs", "depthanything"),
                       env = "JULIA_DEPTHANYTHING_ASSETS", from = @__DIR__)

"""
    depthanythinggraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function depthanythinggraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "depthanything.json")
    isfile(p) || throw(ArgumentError(
        "Depth Anything V2 Small graph not found at $p. Generate it with " *
        "`uv run tools/export_depthanything.py`, or set JULIA_DEPTHANYTHING_ASSETS."))
    return loadgraph(p)
end

"""
    depthanythingweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function depthanythingweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Depth Anything V2 Small weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "depthanything.json")) && isfile(joinpath(dir, "weights.safetensors"))

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
            graph = depthanythinggraph()
            weights = depthanythingweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: image (1, 3, 518, 518)
                nothing
            end
        catch err
            @warn "DepthAnythingRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "DepthAnythingRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
