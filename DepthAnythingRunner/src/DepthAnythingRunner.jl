"""
Depth Anything V2 Small — monocular depth.

Depth per frame, which buys fake shallow depth of field, depth-keyed grading, parallax push-ins and sky masks. 25M parameters.

**Small specifically.** The Base and Large checkpoints are CC-BY-NC-4.0; only Small is Apache-2.0. Large is better and can be added later as its own non-commercial package, the way MatAnyone already is.

A ViT with scaled-dot-product attention and nothing else unusual, so the runtime needs nothing new. Pure editor value, no engine work — which is why it sits where it does in the order.

**Ported and verified** against upstream on 2026-08-02. [`depthanything`](@ref)
loads the model and [`depthmap!`](@ref) turns a frame into inverse relative
depth. The map matches PyTorch to **4.2e-5**, which is 0.0025% of its own range.

**Correct, and off its target.** The `≥ PyTorch` target is not met, and the gap
is throughput in `DNNKernels` rather than anything in this package — see
`plans/projects/small-models/REPORT.md`.

Most of that gap was matrix multiply, in two layers.

`aten::bmm` with no capability dispatch runs one thread per output element with
the K-loop in global memory, which is **79.6% of the forward pass** on this
model. Routing each batch plane through `matmul!` (`DNNKernels`'
`batchedmatmul!`) takes a 518² map from 972 ms to 233 ms, and that moves the
cost onto Lava's fp32 `mul!`, which has the same defect one level down: no
shared memory at all. The scalar branch of llama.cpp's `mul_mm.comp` (Lava
already ran the cooperative-matrix branch of that same shader) takes it from
0.447 to 5.432 TFLOP/s at 2048³ and the frame from 233 ms to **115 ms** on a
Radeon 8060S. Output is unchanged to 1.0e-6 of its own range across both,
against a model verified to 4.2e-5 of PyTorch.

Convolution is what is left, and the largest single op family at 30.2% of the
frame. Numbers from another machine do not compare (GUARDRAILS §6), so this file
quotes none.

**The attention decomposes even though the export is from CUDA.** Unlike Whisper,
DINOv2 falls back to a manual `bmm` + `softmax` when xFormers is absent, so the
graph carries 24 `bmm` and 12 `_softmax` instead of a fused
`_scaled_dot_product_*`. The export-from-CUDA rule is about SDPA *dispatch*, not
a guarantee that attention survives whole.

Upstream: https://github.com/DepthAnything/Depth-Anything-V2
License: **Apache-2.0**

Ops `DNNKernels` did not have: none, as predicted.

See `models-to-port.md` for the state of this one, and `tools/export_depthanything.py`
for the export that feeds it.
"""
module DepthAnythingRunner

using Lava, DNNKernels, KernelAbstractions, GPUFiltering
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, toback, Model
using GPUFiltering: resizeplanar!
using ColorTypes: AbstractRGB, RGB

export depthanythinggraph, depthanythingweights
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

Where the model's graph and weights live: its artifact, downloaded on first use
and cached across every environment on this machine.

**Changing these assets means re-binding the artifact**, not editing a directory.
Re-export, then `julia --project=. tools/make_artifacts.jl depthanything` — that hashes
the new content and rewrites `../Artifacts.toml`, so this call resolves to it
immediately. Uploading is only needed to publish it to anyone else.
"""
assetdir() = @artifact_str("depthanything")

"""
    depthanythinggraph() -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function depthanythinggraph()
    dir = assetdir()
    p = joinpath(dir, "depthanything.json")
    isfile(p) || throw(ArgumentError(
        "Depth Anything V2 Small graph not found at $p. Generate it with " *
        "`uv run tools/export_depthanything.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl depthanything`."))
    return loadgraph(p)
end

"""
    depthanythingweights() -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function depthanythingweights()
    dir = assetdir()
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("Depth Anything V2 Small weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready() -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready() =
    isfile(joinpath(assetdir(), "depthanything.json")) && isfile(joinpath(assetdir(), "weights.safetensors"))

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Mantle.use_frozen_kernels(KERNELS_VERSION)
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
and nearly 2x in wall clock, so skipping them is not a detail. The planned slab is
the other half: 956 buffers left unplanned would each stay live for the whole
graph.
"""
# `plan` is a `DNNKernels.RecordedPlan`: one Mantle plan, declared and recorded
# at load, replayed per frame. It replaced four fields — a `planslab` slab, a
# `Workspace` arena, the lazy-broadcast set and the values table — each of which
# was recovering something the graph had already stated.
struct DepthAnything{B,M,I}
    backend::B
    model::M
    # The staging frame `resizeplanar!` writes into. Separate from the plan's
    # own input buffer, which `call` copies into — the same copy `replay!` did
    # before, since these were never the same array.
    input::I
end

"""
    depthanything(; backend = Mantle.defaultbackend(), dir = assetdir()) -> DepthAnything

Load the model. Separate from [`depthmap!`](@ref) so the workload can build it in
`@setup_workload`, where the loading is not what is being cached.
"""
function depthanything(; backend = Mantle.defaultbackend())
    dir = assetdir()
    ready() || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_depthanything.py`"))
    model = Model(Dict("depthanything" => depthanythinggraph()),
                  depthanythingweights(); backend)
    graph = model.graphs["depthanything"]
    # `emitgraph` DECLARES the ops and runs nothing; `Plan` then runs all seven
    # of Mantle's phases over the whole graph, so placement, aliasing and
    # barriers are decided before a byte is touched. `planfor` is the same one
    # `Model`'s `call` uses, so a runner cannot plan differently from the driver.
    # `planahead!` and not a lazy first call: the latency test asserts that the
    # first frame in a fresh process refuses zero pipeline compiles, so the
    # compiling has to happen here. It builds the plan `call` would build, under
    # the key `call` looks up, so the first frame finds it and replays.
    DNNKernels.planahead!(model, "depthanything")
    input = KA.allocate(model.backend, Float32, INPUT_RES, INPUT_RES, 3, 1)
    return DepthAnything(model.backend, model, input)
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

Costs 115 ms at 518² on a Radeon 8060S (RADV), against PyTorch's 24.7 ms for the
same forward on an RTX 3070. Different machines, so those two do not compare;
what the second number fixes is the order of magnitude the target implies. Still
short of `≥ PyTorch`, and what is left is convolution rather than matrix multiply
(`plans/projects/small-models/REPORT.md`, and the note at the top of this file).
"""
function depthmap!(model::DepthAnything, img::AbstractMatrix{<:AbstractRGB})
    resizeplanar!(model.input, img; mean = IMAGENET_MEAN, std = IMAGENET_STD)
    # `call`: the plan was recorded at load by `planahead!`, so this finds it in
    # the model's cache, writes the input into the buffer it was declared
    # against and submits one recording. It used to hand-roll `planfor` and
    # `replay!`, which is what `call` does and is the only way to drift from it.
    #
    # The output is an `unsqueeze`, a view rather than an op result — which used
    # to mean it had no entry of its own and `value` had to resolve it against
    # its parent. Declared, `emitgraph` resolves every output before it returns,
    # so the plan's own output IS that view and there is nothing to chase.
    return first(DNNKernels.call(model.model, "depthanything", model.input; dims = (;)))
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
# NOT `frozen_stats().misses == 0` alone, which reads stronger than it is: it
# cannot distinguish the frozen cache working from the driver's own shader cache
# having served everything, and its miss report identifies modules by the
# *sampling* hash, so two differing in one byte count as one (`STATUS.md`,
# cross-project). The claim this package makes is `Mantle.no_pipeline_compilation`
# reporting **0 refusals** — it empties `PIPELINE_CACHE` first, so a Julia-side
# hit cannot mask a cold `VkPipelineCache`. Pair it with a control whose kernel
# body is novel per RUN (a `Val{K}` from `RandomDevice`) or a green means
# nothing; verified firing here at refused = 1.
#
# A small source frame on purpose: the graph's own input is fixed at 518², so the
# only thing the frame size changes is the resize in front of it, and that kernel
# is compiled per element type, not per resolution.
@setup_workload begin
    if ready()
        try
            backend = Mantle.defaultbackend()
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
