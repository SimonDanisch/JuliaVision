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

using Lava, DNNKernels, KernelAbstractions, GPUFiltering
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, assetpath, toback,
                  Model, planslab, fusableset, Workspace, Ctx, value
using GPUFiltering: resizeplanar!
using ColorTypes: AbstractRGB, RGB

export depthanythinggraph, depthanythingweights, assetdir
export depthanything, depthmap!, DepthAnything

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

# ------------------------------------------------------------------- the model

"""
    INPUT_RES

The square side the graph was exported for. Not a free parameter: DINOv2 splits
the input into 14x14 patches, and the export bakes 37x37 of them. A different
resolution is a different export (`--size`), not a runtime argument.
"""
const INPUT_RES = 518

"""
    IMAGENET_MEAN, IMAGENET_STD

The normalisation upstream's `image2tensor` applies before the network sees a
frame. Restated here because the export deliberately leaves it out of the graph —
see `tools/export_depthanything.py` — so this is the only place that knows it,
and getting it wrong produces a plausible depth map rather than an error.
"""
const IMAGENET_MEAN = (0.485f0, 0.456f0, 0.406f0)
const IMAGENET_STD = (0.229f0, 0.224f0, 0.225f0)

"""
    DepthAnything

A loaded model: the prepared graph, its weights on the device, and the scratch
the graph needs.

Built through `DNNKernels.Model` rather than `loadgraph`, which is what runs the
host-side preparation passes — on this model they are worth 311 ops down to 290
and **763 ms down to 421**, so skipping them is not a detail. The planned slab is
the other half: 956 buffers left unplanned would each stay live for the whole
graph.
"""
struct DepthAnything{B,G,W,S,P,I}
    backend::B
    graph::G
    weights::W
    slab::S
    plan::P
    ws::Workspace
    lazy::Set{String}
    input::I
end

"""
    depthanything(; backend = LavaBackend(), dir = assetdir()) -> DepthAnything

Load the model. Separate from [`depthmap!`](@ref) so the workload can build it in
`@setup_workload`, where the loading is not what is being cached.
"""
function depthanything(; backend = LavaBackend(), dir::AbstractString = assetdir())
    ready(; dir) || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_depthanything.py`"))
    model = Model(dir, joinpath(dir, "weights.safetensors");
                  names = ["depthanything"], backend)
    graph = model.graphs["depthanything"]
    plan = planslab(graph, (;))
    slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
    input = KA.allocate(backend, Float32, INPUT_RES, INPUT_RES, 3, 1)
    return DepthAnything(backend, graph, model.weights, slab, plan,
                         Workspace(backend), fusableset(graph), input)
end

"""
    depthmap!(model, img) -> depth

Estimate depth for `img`, a frame of any resolution on `model`'s backend.

Returns a `(518, 518, 1, 1)` device array of **inverse relative depth** — larger
is nearer, and there is no metric scale, so the only meaningful operations on it
are comparisons and a normalisation against its own range. The caller resamples
it to the frame; it is deliberately not resampled here, because a depth-keyed
grade wants it at whatever resolution the effect runs at and an intermediate
resize would cost a full pass for nothing.

`img` is scaled into a square, not letterboxed. Upstream keeps aspect ratio and
pads to a multiple of 14, which makes the tensor shape depend on the clip and a
static graph cannot have that. The distortion is uniform across the frame and the
network is scale-tolerant, but it is a real difference from upstream's own
`infer_image` and is worth knowing when comparing against it.

Costs ~421 ms at 518² on an RTX 3070 laptop, against PyTorch's 27.6 ms for the
same forward — this port is correct but far off its `≥ PyTorch` target, and the
gap is convolution throughput (`plans/projects/small-models/REPORT.md`).
"""
function depthmap!(model::DepthAnything, img::AbstractMatrix{<:AbstractRGB})
    resizeplanar!(model.input, img; mean = IMAGENET_MEAN, std = IMAGENET_STD)
    vals = execute!(model.graph, Dict{String,Any}("x" => model.input), model.weights;
                    dims = (;), backend = model.backend,
                    slab = model.slab, plan = model.plan,
                    ws = model.ws, lazy = model.lazy)
    # The output is `unsqueeze`, a view rather than an op result, so it has no
    # entry of its own in the value table. `value` resolves it against the buffer
    # it is a view of — the same thing `wan.jl`'s `rungraph` does.
    return value(Ctx(vals, model.graph, (;), model.backend), only(model.graph.outputs))
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# The workload drives `depthmap!`, which is the whole call tree the editor uses —
# the measurement that matters is `Lava.frozen_stats().misses == 0` on a *fresh*
# process, and a workload that runs a different path than the editor does leaves
# the editor compiling on first use, which is the entire cost this package exists
# to remove. SAM2Runner learned that the expensive way: its `runsam2` workload
# still left 45 s on the first click because the editor goes through a closure
# `runsam2` never touches.
#
# A small source frame on purpose: the graph's own input is fixed at 518², so the
# only thing the frame size changes is the resize in front of it, and that kernel
# is compiled per element type, not per resolution.
@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
            # Model construction inside the workload, not in front of it:
            # `Model`'s last pass folds constant subgraphs by *running* them on
            # the device, so building it outside leaves those dispatches
            # unfrozen. RIFE showed this as a hard `misses == 9`; this graph has
            # no constant subgraph today, and the placement is what keeps that
            # from silently mattering after a re-export.
            @compile_workload KERNELS_VERSION begin
                model = depthanything(; backend)
                img = KA.allocate(backend, RGB{Float32}, 256, 256)
                fill!(img, RGB{Float32}(0.3f0, 0.5f0, 0.7f0))
                depthmap!(model, img)
                KA.synchronize(backend)
            end
        catch err
            @warn "DepthAnythingRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "DepthAnythingRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
