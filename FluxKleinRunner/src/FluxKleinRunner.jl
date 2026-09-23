"""
FLUX.2-klein-4B — generative fill / outpaint / instruction edit.

Generative fill behind a SAM 2 mask, outpainting to reframe without cropping, replacing a sign, relighting a subject. 4B parameters, 4 steps.

**The only edit-capable diffusion model that fits this card at full precision.** 7.8 GB of transformer in bf16, against 40.9 GB for Qwen-Image-Edit-2511, 34.2 for HiDream-E1-1 and 23.8 for FLUX.1-Kontext. Everything else in the category needs 3-4 bit quantisation to fit, which means judging the model well below its best.

Not a lesser feature set for being small: the card states text-to-image and image-to-image multi-reference editing in one unified model.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`fluxkleingraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * group_norm
  * VAE decoder
  * sampler loop (host-side)

See `models-to-port.md` for the state of this one, and `tools/export_fluxklein.py`
for the export that feeds it.
"""
module FluxKleinRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using DNNKernels: loadgraph, readsafetensors

export fluxkleingraph, fluxkleinweights, assetdir

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Throws. FLUX.2-klein-4B is **not ported yet**, so there is no artifact to read from and
nothing on disk that a user of this package would have.

Porting it means, in order: export it with `uv run tools/export_fluxklein.py`, bind
the result with `julia --project=. tools/make_artifacts.jl fluxklein`, and
replace this definition with `@artifact_str("fluxklein")`. Assets come from the
artifact and from nowhere else — see `DNNKernels/src/assets.jl`.
"""
assetdir() = error(
    "FluxKleinRunner: FLUX.2-klein-4B is not ported yet, so no artifact is bound. " *
    "Export it with `uv run tools/export_fluxklein.py`, bind it with " *
    "`julia --project=. tools/make_artifacts.jl fluxklein`, then set " *
    "`assetdir() = @artifact_str(\"fluxklein\")`.")

"""
    fluxkleingraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function fluxkleingraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "fluxklein.json")
    isfile(p) || throw(ArgumentError(
        "FLUX.2-klein-4B graph not found at $p. Generate it with " *
        "`uv run tools/export_fluxklein.py`, or set JULIA_FLUXKLEIN_ASSETS."))
    return loadgraph(p)
end

"""
    fluxkleinweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function fluxkleinweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("FLUX.2-klein-4B weights not found at $p"))
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
            graph = fluxkleingraph()
            weights = fluxkleinweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: prompt + 0..n reference images -> latent, denoised 4 steps, VAE decoded
                nothing
            end
        catch err
            @warn "FluxKleinRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "FluxKleinRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
