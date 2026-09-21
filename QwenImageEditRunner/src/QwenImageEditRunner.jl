"""
Qwen-Image-Edit-2511 — instruction image editing (blocked).

The model we wanted first, and it does not fit. 40.9 GB of transformer plus a 16.6 GB Qwen2.5-VL text encoder against a ~16 GB budget.

Only Q4_0 (11.9 GB) and below leave room for the encoder and the editor, and Q4_K_M — where quantised quality usually stops hurting — is 13.2 GB and already too big. So this is blocked on an int4 dequant epilogue in the GEMM, which is real engine work rather than a model port.

Kept in the registry because it is the target once quantisation exists.

**Not ported yet.** This package is the place the port lands, committed ahead of
the work so the graph path, the asset lookup and the workload guard are settled
and everything after this is model code. What exists: [`assetdir`](@ref) resolves
the export, [`qwenimageeditgraph`](@ref) loads it if it is there, and precompilation is
inert until it is. What does not: the workload body, and whatever ops the export
turns out to need.

Upstream: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
License: **Apache-2.0**

Ops `DNNKernels` does not have yet:
  * int4 dequant in the GEMM epilogue
  * group_norm
  * VAE decoder

See `models-to-port.md` for the state of this one, and `tools/export_qwenimageedit.py`
for the export that feeds it.
"""
module QwenImageEditRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors

export qwenimageeditgraph, qwenimageeditweights, assetdir

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Throws. Qwen-Image-Edit-2511 is **not ported yet**, so there is no artifact to read from and
nothing on disk that a user of this package would have.

Porting it means, in order: export it with `uv run tools/export_qwenimageedit.py`, bind
the result with `julia --project=. tools/make_artifacts.jl qwenimageedit`, and
replace this definition with `@artifact_str("qwenimageedit")`. Assets come from the
artifact and from nowhere else — see `DNNKernels/src/assets.jl`.
"""
assetdir() = error(
    "QwenImageEditRunner: Qwen-Image-Edit-2511 is not ported yet, so no artifact is bound. " *
    "Export it with `uv run tools/export_qwenimageedit.py`, bind it with " *
    "`julia --project=. tools/make_artifacts.jl qwenimageedit`, then set " *
    "`assetdir() = @artifact_str(\"qwenimageedit\")`.")

"""
    qwenimageeditgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function qwenimageeditgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "qwenimageedit.json")
    isfile(p) || throw(ArgumentError(
        "Qwen-Image-Edit-2511 graph not found at $p. Generate it with " *
        "`uv run tools/export_qwenimageedit.py`, or set JULIA_QWENIMAGEEDIT_ASSETS."))
    return loadgraph(p)
end

"""
    qwenimageeditweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function qwenimageeditweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Qwen-Image-Edit-2511 weights not found at $p"))
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
    # `isdefined`, because `use_frozen_kernels` lives in Mantle's VULKAN tree: without a
    # Vulkan driver it does not exist, and an unguarded call here is an `InitError` that
    # stops `using` this package at all. Nothing to read is not an error, it is no cache.
    isdefined(Mantle, :use_frozen_kernels) &&
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
            graph = qwenimageeditgraph()
            weights = qwenimageeditweights()
            @compile_workload KERNELS_VERSION begin
                # Inputs: prompt + image -> edited image
                nothing
            end
        catch err
            @warn "QwenImageEditRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "QwenImageEditRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
