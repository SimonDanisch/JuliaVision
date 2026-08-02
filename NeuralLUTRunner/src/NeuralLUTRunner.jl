"""
Image-Adaptive 3D LUT — style / mood grading.

A look, applied as a grade. Under 600K parameters: a small CNN predicts blend weights over a handful of basis 3D LUTs, and the result is a LUT.

That output shape is the whole argument for it. Per-frame diffusion style transfer flickers and cannot be edited; a predicted LUT is a first-class grading object — it drops into the inspector, the user can push it around, and it keyframes with the machinery already built. Temporal stability comes from smoothing the predicted weights across frames rather than from the network.

Almost nothing new for the runtime, which is why it is second in the order: it is the fastest visible result in the set. The LUT apply is trilinear interpolation and belongs in GPUFiltering, not in the graph.

**Ported and verified** against upstream on 2026-08-02. [`neurallut`](@ref) loads
the model, [`predictlut`](@ref) turns a frame into a table and [`grade!`](@ref)
applies it. The graph's blended LUT matches PyTorch to **3.6e-7** and the apply
matches upstream's own `trilinear_kernel.cu` to **3.0e-7**; see
`plans/projects/small-models/REPORT.md` for the numbers and the caveats.

**Two halves, on purpose.** The graph ends at the LUT, not at an image: what the
network produces is a *grading object*, so the editor can hold it, show it in the
inspector, keyframe it and let the user push it around. Applying it is a
`GPUFiltering` kernel ([`lut3d!`](@ref)) that does not care whether a network or
a `.cube` file on disk produced the table. That split is also what keeps the
graph resolution-independent — the tensor leaving it is 3x33x33x33 whatever the
frame size is.

The consequence for cost is worth stating up front, because the two halves are
paid at different rates: applying a look is **0.79 ms at 4K** on this machine,
while re-predicting one is ~1.8 ms. Both fit a 60 fps frame at 4K, so the
look can be re-predicted per frame if the edit wants it to be.

Upstream: https://github.com/HuiZeng/Image-Adaptive-3DLUT
License: **Apache-2.0**

Ops `DNNKernels` did not have: none. `_native_batch_norm_legit.no_stats` (what
`nn.InstanceNorm2d` decomposes to) was added for this port and is general.

See `models-to-port.md` for the state of this one, and `tools/export_neurallut.py`
for the export that feeds it.
"""
module NeuralLUTRunner

using Lava, DNNKernels, KernelAbstractions, GPUFiltering
using Lava: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  Model, planslab, fusableset, Workspace
using GPUFiltering: lut3d!, resizeplanar!
using ColorTypes: AbstractRGB, RGB

export neurallutgraph, neurallutweights, assetdir
export neurallut, predictlut, grade!, NeuralLUT

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
and cached across every environment on this machine.

**Changing these assets means re-binding the artifact**, not editing a directory.
Re-export, then `julia --project=. tools/make_artifacts.jl neurallut` — that hashes
the new content and rewrites `../Artifacts.toml`, so this call resolves to it
immediately. Uploading is only needed to publish it to anyone else.
"""
assetdir() = @artifact_str("neurallut")

"""
    neurallutgraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function neurallutgraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "neurallut.json")
    isfile(p) || throw(ArgumentError(
        "Image-Adaptive 3D LUT graph not found at $p. Generate it with " *
        "`uv run tools/export_neurallut.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl neurallut`."))
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

# ------------------------------------------------------------------- the model

"""
    CLASSIFIER_RES

The side length the classifier's input is resized to, and not a free parameter:
upstream's `Classifier` begins with `nn.Upsample(size=(256,256))`, so the network
has only ever seen 256x256. The export strips that layer (it would bake a frame
size into a graph that immediately discards it) and [`predictlut`](@ref) does the
resize instead, which is why the constant has to live on this side too.
"""
const CLASSIFIER_RES = 256

"""
    NeuralLUT

A loaded predictor: the graph, its weights on the device, and the classifier
input buffer that [`predictlut`](@ref) reuses.

The input buffer is held rather than allocated per call because it is the same
1.5 MB every time and a look gets re-predicted whenever the user scrubs to a new
shot. Nothing else here is stateful — the predicted table is returned, not
stored, so the caller decides whether this frame's look replaces the last one or
is smoothed against it (`models-to-port.md` wants temporal stability to come from
smoothing the prediction, not from the network).
"""
struct NeuralLUT{B,W,S,P,T}
    backend::B
    graph::DNNKernels.Graph
    weights::W
    slab::S
    plan::P
    ws::Workspace
    lazy::Set{String}
    input::T
end

"""
    neurallut(; backend = LavaBackend(), dir = assetdir()) -> NeuralLUT

Load the model. Separate from [`predictlut`](@ref) so the workload can build it
in `@setup_workload`, where the loading is not what is being cached.
"""
function neurallut(; backend = LavaBackend(), dir::AbstractString = assetdir())
    ready(; dir) || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_neurallut.py`"))
    # `Model`, not `loadgraph`: it runs the host-side preparation passes, and the
    # planned slab is what keeps every intermediate from being a fresh
    # allocation. Together they are worth 14.08 ms -> ~1.8 ms on this classifier,
    # and a runner that skipped them would be slower than its own benchmark.
    model = Model(dir, joinpath(dir, "weights.safetensors");
                  names = ["neurallut"], backend)
    graph = model.graphs["neurallut"]
    plan = planslab(graph, (;))
    slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
    input = KA.allocate(backend, Float32, CLASSIFIER_RES, CLASSIFIER_RES, 3, 1)
    return NeuralLUT(backend, graph, model.weights, slab, plan,
                     Workspace(backend), fusableset(graph), input)
end

"""
    predictlut(model, img) -> lut

Predict a grading LUT for `img`, a frame of any resolution on `model`'s backend.

Returns a `(33, 33, 33, 3)` device array indexed `[r, g, b, channel]` — the
argument [`grade!`](@ref) and `GPUFiltering.lut3d!` take. It is a fresh table per
call, so keeping one is the caller's business.

Costs about 1.8 ms at 4K on an RTX 3070 laptop, nearly all of it the
classifier's six convolutions; the resize into 256x256 is 0.03 ms of it. That is a per-shot
cost, not a per-frame one — see [`grade!`](@ref).
"""
function predictlut(model::NeuralLUT, img::AbstractMatrix{<:AbstractRGB})
    resizeplanar!(model.input, img)
    vals = execute!(model.graph, Dict{String,Any}("img" => model.input), model.weights;
                    dims = (;), backend = model.backend,
                    slab = model.slab, plan = model.plan,
                    ws = model.ws, lazy = model.lazy)
    # The exporter names the blend's output; reading it from the graph rather
    # than hardcoding "sum_1" means a re-export that renumbers cannot silently
    # return the wrong buffer.
    return vals[only(model.graph.outputs)]
end

"""
    grade!(out, img, lut) -> out
    grade!(model, out, img) -> out

Apply a look. The three-argument form takes a table already predicted (or
authored, or dragged in from disk); the `model` form predicts one from `img`
first.

Applying is 0.80 ms at 4K on this machine and predicting is ~1.8 ms, so
re-predicting per frame costs 2.6 ms all in and fits a 60 fps frame. Prefer the
first form anyway when the look is constant across a shot — it is 3x cheaper and
says what it means — but the `model` form is no longer a budget decision.
"""
grade!(out::AbstractMatrix{<:AbstractRGB}, img::AbstractMatrix{<:AbstractRGB},
       lut::AbstractArray{Float32,4}) = lut3d!(out, img, lut)

grade!(model::NeuralLUT, out::AbstractMatrix{<:AbstractRGB},
       img::AbstractMatrix{<:AbstractRGB}) = lut3d!(out, img, predictlut(model, img))

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# The workload drives `predictlut` and both `grade!` methods, which is the whole
# call tree the editor uses — the measurement that matters is
# `Lava.frozen_stats().misses == 0` on a *fresh* process, and a workload that
# runs a different path than the editor does leaves the editor compiling on first
# use, which is the entire cost this package exists to remove. SAM2Runner learned
# that the expensive way: its `runsam2` workload still left 45 s on the first
# click because the editor goes through a closure `runsam2` never touches.
#
# 256x256 rather than a real frame: `lut3d!` and `resizeplanar!` are compiled per
# element type and backend, not per resolution, so a small image freezes the same
# kernels in a fraction of the time and none of the VRAM.
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
                model = neurallut(; backend)
                img = KA.allocate(backend, RGB{Float32}, 256, 256)
                out = similar(img)
                fill!(img, RGB{Float32}(0.3f0, 0.5f0, 0.7f0))
                lut = predictlut(model, img)
                grade!(out, img, lut)
                grade!(model, out, img)
                KA.synchronize(backend)
            end
        catch err
            @warn "NeuralLUTRunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "NeuralLUTRunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
