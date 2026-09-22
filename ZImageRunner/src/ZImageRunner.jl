"""
Z-Image-Turbo (6B) — text to image.

8-step text-to-image, 6B parameters. Published fp32 (24.6 GB), so ~12 GB converted to bf16 — fits, but only just, and it is generation rather than editing so it buys title cards and background plates rather than an edit operation. Second to FLUX.2-klein on both counts.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`zimagegraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * group_norm
  * VAE decoder

See `models-to-port.md` for the state of this one, and `tools/export_zimage.py`
for the export that feeds it.
"""
module ZImageRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors

export zimagegraph, zimageweights, assetdir

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Throws. Z-Image-Turbo (6B) is **not ported yet**, so there is no artifact to read from and
nothing on disk that a user of this package would have.

Porting it means, in order: export it with `uv run tools/export_zimage.py`, bind
the result with `julia --project=. tools/make_artifacts.jl zimage`, and
replace this definition with `@artifact_str("zimage")`. Assets come from the
artifact and from nowhere else — see `DNNKernels/src/assets.jl`.
"""
assetdir() = error(
    "ZImageRunner: Z-Image-Turbo (6B) is not ported yet, so no artifact is bound. " *
    "Export it with `uv run tools/export_zimage.py`, bind it with " *
    "`julia --project=. tools/make_artifacts.jl zimage`, then set " *
    "`assetdir() = @artifact_str(\"zimage\")`.")

"""
    zimagegraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function zimagegraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "zimage.json")
    isfile(p) || throw(ArgumentError(
        "Z-Image-Turbo (6B) graph not found at $p. Generate it with " *
        "`uv run tools/export_zimage.py`, or set JULIA_ZIMAGE_ASSETS."))
    return loadgraph(p)
end

"""
    zimageweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function zimageweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Z-Image-Turbo (6B) weights not found at $p"))
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
    Mantle.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# TODO(port): drive the real call here once the graph runs. The measurement that
# matters is `Mantle.no_pipeline_compilation` reporting **0 refusals** on a *fresh*
# process — a workload
# that runs a different path than the editor does leaves the editor compiling on
# first use, which is the entire cost this package exists to remove. SAM2Runner
# learned that the expensive way: its `runsam2` workload still left 45 s on the
# first click because the editor goes through a closure `runsam2` never touches.
#
# NOT `frozen_stats().misses == 0`, which reads stronger than it is: it cannot
# distinguish the frozen cache working from the driver's own shader cache having
# served everything, and its miss report identifies modules by the *sampling*
# hash, so two modules differing in one byte count as one (`STATUS.md`,
# cross-project). `no_pipeline_compilation` empties `PIPELINE_CACHE` first and
# refuses anything needing a compile. Pair it with a negative control whose
# kernel body is novel per RUN — a `Val{K}` with `K` from `RandomDevice` — or a
# green means nothing; verified firing here at refused = 1.
@setup_workload begin
    if ready()
        try
            backend = Mantle.defaultbackend()
            graph = zimagegraph()
            weights = zimageweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: prompt -> image
                nothing
            end
        catch err
            @warn "ZImageRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "ZImageRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
