"""
BasicVSR++ — video upscaling.

Temporally consistent 4x upscaling of a short clip: BasicVSR++ trained on REDS4,
7.3M parameters. The footprint is activations rather than weights: it is
recurrent over the clip, so VRAM scales with its length.

**Ported and verified.** [`basicvsrppmodel`](@ref) loads it and
[`upscale`](@ref) runs it. Against upstream's own model files run with the same
weights, the output matches to **1.1e-5** (max abs, five 256x256 frames of a real
clip); `test/runtests.jl` pins that on a synthetic clip written by
`tools/verify_basicvsrpp.py`. The export is static at five 64x64 frames.

**The input degradation matters.** REDS4's low-resolution frames are MATLAB
`imresize(x, 1/4, 'bicubic')`, and that is what the model is trained to invert.
An area-averaged clip is a different degradation and comes back with ringing and
vertical streaks; `docs/examples/basicvsr.jl` has a `bicubicdown` that matches
the training data.

Flow-guided deformable alignment is the unusual part for the engine: DCNv2 is an
irregular per-pixel gather with no clean cooperative-matrix mapping, and it runs
as DNNKernels' `deform_conv2d`.

Upstream: https://github.com/ckkelvinchan/BasicVSR_PlusPlus (via open-mmlab/mmagic)
License: **Apache-2.0**

See `tools/export_basicvsrpp.py` for the export that feeds it.
"""
module BasicVSRRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, toback, Model, call

export basicvsrppgraph, basicvsrppweights
export basicvsrppmodel, upscale, BasicVSRPP

const KA = KernelAbstractions

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    assetdir() -> String

Where the model's graph and weights live: its artifact, downloaded on first use
and cached across every environment on this machine. 26 MiB — the fp32 export of
BasicVSR++ REDS4.

**Changing these assets means re-binding the artifact**, not editing a directory.
Re-export, then `julia --project=. tools/make_artifacts.jl basicvsrpp` — that
hashes the new content and rewrites `../Artifacts.toml`, so this call resolves to
it immediately. Uploading is only needed to publish it to anyone else.
"""
assetdir() = @artifact_str("basicvsrpp")

"""
    basicvsrppgraph() -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function basicvsrppgraph()
    dir = assetdir()
    p = joinpath(dir, "basicvsrpp.json")
    isfile(p) || throw(ArgumentError(
        "BasicVSR++ graph not found at $p. Generate it with " *
        "`uv run tools/export_basicvsrpp.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl`."))
    return loadgraph(p)
end

"""
    basicvsrppweights() -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function basicvsrppweights()
    dir = assetdir()
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("BasicVSR++ weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready() -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready() =
    isfile(joinpath(assetdir(), "basicvsrpp.json")) && isfile(joinpath(assetdir(), "weights.safetensors"))

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
# The workload drives `upscale`, the whole call tree a caller uses: a workload
# that runs a different path leaves the first real call compiling, which is the
# entire cost this package exists to remove.
# ------------------------------------------------------------------- the model

"""
    BasicVSRPP

A loaded 4x video upscaler: the rewritten graph, its weights on the device, and
the backend they belong to. Build one with [`basicvsrppmodel`](@ref) and hand it
to [`upscale`](@ref).
"""
struct BasicVSRPP{B,M}
    backend::B
    model::M
end

"""
    basicvsrppmodel(; backend = Mantle.defaultbackend()) -> BasicVSRPP

Load the upscaler. Downloads the 26 MiB artifact on first use.

Not cached in a module global: a `Model` holds device buffers, and a global
holding one is baked into the package image with a `VkContext` that is dead by
the time anyone loads it.
"""
function basicvsrppmodel(; backend = Mantle.defaultbackend())
    dir = assetdir()
    ready() || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_basicvsrpp.py`"))
    BasicVSRPP(backend, Model(Dict("basicvsrpp" => basicvsrppgraph()),
                              basicvsrppweights(); backend))
end

"""
    upscale(m::BasicVSRPP, lqs) -> AbstractArray

4x a short clip. `lqs` is `(W, H, 3, T, 1)` — the export baked `T = 5` at
`64 x 64`, so that is the shape it takes — and the result is `(4W, 4H, 3, T, 1)`
on the device.

**The extents are baked into the export.** A different clip length or frame size
is a re-export, not an argument; this is why `framesize`-style introspection
belongs here rather than a resize.

Frames are RGB in [0, 1]. The output is not clamped: the model overshoots at
edges, to -0.08 and 1.04 on the fox in `docs/examples/basicvsr.jl`.
"""
function upscale(m::BasicVSRPP, lqs)
    out, = call(m.model, "basicvsrpp", toback(m.backend, lqs); dims = (;))
    return out
end

@setup_workload begin
    if ready()
        try
            backend = Mantle.defaultbackend()
            # Inside `@compile_workload`, not in front of it: `Model`'s last pass
            # folds constant subgraphs by running them on the device, and building
            # it outside leaves those dispatches unfrozen (RIFERunner measured
            # exactly that: 9 misses on a fresh process, every time).
            @compile_workload KERNELS_VERSION begin
                m = basicvsrppmodel(; backend)
                lqs = KA.allocate(backend, Float32, 64, 64, 3, 5, 1)
                fill!(lqs, 0.5f0)
                upscale(m, lqs)
                KA.synchronize(backend)
            end
        catch err
            @warn "BasicVSRRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "BasicVSRRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
