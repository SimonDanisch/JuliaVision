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
using LazyArtifacts
using DNNKernels: loadgraph, execute!, readsafetensors, toback, Model, call

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
# The graph is bound and every op it uses is implemented (2290 ops, `coverage`
# reports none missing). What has NOT been done is a numerical parity run — there
# is no `tools/verify_basicvsrpp.jl`, so "it loads and dispatches" is the whole of
# the claim here.
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "basicvsrpp.json")) && isfile(joinpath(dir, "weights.safetensors"))

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
    basicvsrppmodel(; backend = LavaBackend(), dir = assetdir()) -> BasicVSRPP

Load the upscaler. Downloads the 26 MiB artifact on first use.

Not cached in a module global: a `Model` holds device buffers, and a global
holding one is baked into the package image with a `VkContext` that is dead by
the time anyone loads it.
"""
function basicvsrppmodel(; backend = LavaBackend(), dir::AbstractString = assetdir())
    ready(; dir) || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_basicvsrpp.py`"))
    BasicVSRPP(backend, Model(dir, joinpath(dir, "weights.safetensors");
                              names = ["basicvsrpp"], backend))
end

"""
    upscale(m::BasicVSRPP, lqs) -> AbstractArray

4x a short clip. `lqs` is `(W, H, 3, T, 1)` — the export baked `T = 5` at
`64 x 64`, so that is the shape it takes — and the result is `(4W, 4H, 3, T, 1)`
on the device.

**The extents are baked into the export.** A different clip length or frame size
is a re-export, not an argument; this is why `framesize`-style introspection
belongs here rather than a resize.

**Not verified against PyTorch.** There is no `tools/verify_basicvsrpp.jl`, so
what is known is that all 2290 ops are implemented, it dispatches, and the output
is the right shape and finite. Numerical parity is unmeasured — see
`gen/basicvsrpp/refs.safetensors`, which is what a verifier would read.
"""
function upscale(m::BasicVSRPP, lqs)
    out, = call(m.model, "basicvsrpp", toback(m.backend, lqs); dims = (;))
    return out
end

@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
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
