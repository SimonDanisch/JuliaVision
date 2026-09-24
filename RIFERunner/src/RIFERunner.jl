"""
RIFE 4.x (Practical-RIFE) — frame interpolation.

Slow motion, framerate conversion, and smoothing a retime — about 10 MB of weights for all of it.

Cheap for the runtime: the warping is `grid_sampler_2d`, which is already implemented and already exercised by the optical-flow path in GPUFiltering.

**Ported and verified** against upstream on 2026-08-02. [`rife`](@ref) loads the
model and [`interpolate!`](@ref) produces one frame between two, at any `t` in
[0, 1]. The interpolated frame matches PyTorch to **3.3e-4** max and 4.9e-7 mean.

**Correct, and far off its target.** One 1080p frame costs ~320 ms on an RTX 3070
laptop against the 16.67 ms that 1080p60 needs. The port is not what is slow: the
graph is **149 GFLOP of convolution per frame**, so even a kernel sustaining a
plausible 10 TFLOP/s would spend 14.9 ms on convolution alone. 1080p60 is not
reachable here in fp32 at this model size, which is a target-setting finding
rather than an optimisation task — see `plans/projects/small-models/REPORT.md`.

**Resolution is baked into the export**, and the frame is *padded* to it rather
than resized: RIFE's flow field is in pixels, so scaling the input would silently
rescale every motion vector the network predicts. [`framesize`](@ref) reports the
padded size the installed export was built for.

Upstream: https://github.com/hzwer/Practical-RIFE
License: **MIT**

Ops `DNNKernels` did not have: none, as predicted — including the 18
`grid_sampler_2d` warps and 7 transposed convolutions.

See `models-to-port.md` for the state of this one, and `tools/export_rife.py`
for the export that feeds it.
"""
module RIFERunner

using Lava, DNNKernels, KernelAbstractions, GPUFiltering
import Mantle
using Mantle: @setup_workload, @compile_workload
using LazyArtifacts
using DNNKernels: loadgraph, readsafetensors, toback, Model
using GPUFiltering: tofloat, topixel
using ColorTypes: AbstractRGB, RGB

export rifegraph, rifeweights
export rife, interpolate!, framesize, RIFE, release!

const KA = KernelAbstractions
# Through DNNKernels rather than a direct dependency: this package already has
# it, and `KernelInterface` is not in RIFERunner's Project.
const KI = DNNKernels.KI

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
Re-export, then `julia --project=. tools/make_artifacts.jl rife` — that hashes
the new content and rewrites `../Artifacts.toml`, so this call resolves to it
immediately. Uploading is only needed to publish it to anyone else.
"""
assetdir() = @artifact_str("rife")

"""
    rifegraph() -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function rifegraph()
    dir = assetdir()
    p = joinpath(dir, "rife.json")
    isfile(p) || throw(ArgumentError(
        "RIFE 4.x (Practical-RIFE) graph not found at $p. Generate it with " *
        "`uv run tools/export_rife.py` and bind it with " *
        "`julia --project=. tools/make_artifacts.jl rife`."))
    return loadgraph(p)
end

"""
    rifeweights() -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function rifeweights()
    dir = assetdir()
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("RIFE 4.x (Practical-RIFE) weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready() -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready() =
    isfile(joinpath(assetdir(), "rife.json")) && isfile(joinpath(assetdir(), "weights.safetensors"))

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Mantle.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ------------------------------------------------------------------- the model

# Two frames into the graph's single 6-channel input, and one frame back out.
#
# RIFE **pads** rather than resizes: upstream's `inference_video.py` rounds the
# frame up to a multiple of `max(128, 128/scale)` and crops the result back, so
# 1080p runs as 1920x1152 with 72 rows of padding. A resize would be wrong in a
# way that is hard to see — the flow field is in pixels, so scaling the input
# silently rescales every motion vector the network predicts.
#
# `GPUFiltering.resizeplanar!` therefore does not fit, and this is its own kernel
# rather than a generalisation of that one: the two differ in what happens
# outside the source, which is the whole point of each.
# Macro-free, over `KernelInterface`'s intrinsics. `@index(Global, Cartesian)`
# becomes `KI.get_global_id()`, whose axes are 1-based; `@Const` was an identity
# adaptor on this backend and drops.
#
# THE BOUNDS CHECK the macro used to insert. The `x <= w` below is a CONTENT
# test — inside the real frame or in the pad — and every thread writes on both
# sides of it, so it is not an ndrange guard. `launchgroup` need not divide
# `(pw, ph)`, and the surplus threads wrote past `dst`.
function frames_kernel!(dst, a, b, w::Int32, h::Int32)
    g = KI.get_global_id()
    x, y = g.x, g.y
    (x <= size(dst, 1) && y <= size(dst, 2)) || return nothing
    if x <= w && y <= h
        @inbounds begin
            ca = tofloat(a[x, y])
            cb = tofloat(b[x, y])
            dst[x, y, 1, 1] = ca.r; dst[x, y, 2, 1] = ca.g; dst[x, y, 3, 1] = ca.b
            dst[x, y, 4, 1] = cb.r; dst[x, y, 5, 1] = cb.g; dst[x, y, 6, 1] = cb.b
        end
    else
        # Zero, which is what `F.pad` defaults to and therefore what the network
        # was traced against. Writing it every call rather than once at
        # allocation because the buffer is reused and a previous frame's edge
        # would otherwise persist into the pad.
        @inbounds for c in 1:6
            dst[x, y, c, 1] = 0.0f0
        end
    end
    return nothing
end

# The interpolated frame back out of the graph's (W, H, 3, 1), cropped to the
# real frame — the padded rows are network output over zeros and are not part of
# the picture.
function unpack_kernel!(out, src)
    g = KI.get_global_id()
    x, y = g.x, g.y
    (x <= size(out, 1) && y <= size(out, 2)) || return nothing
    @inbounds out[x, y] = topixel(eltype(out), src[x, y, 1, 1],
                                  src[x, y, 2, 1], src[x, y, 3, 1])
    return nothing
end

"""
The device frames a stream owns, and the two plans that move them.

`pack` fills the graph's padded input from `a` and `b`; `unpack` crops the
graph's result into `out`. Both are recorded ONCE per stream — against these
buffers, at this frame size, for these pixel types — and replayed per frame.

The buffers belong to this and not to the caller for the reason every recorded
plan has: a plan is packed with the ADDRESS of what it reads, so the frames have
to land at an address that does not move between frames. `interpolate!` copies
the caller's frames into them, which is the upload a host frame needed anyway.
"""
struct FrameIO{P,U,A,O}
    pack::P
    unpack::U
    a::A
    b::A
    out::O
end

function freeframeio!(io::FrameIO)
    Mantle.free!(io.pack); Mantle.free!(io.unpack)
    Mantle.free!(io.a); Mantle.free!(io.b); Mantle.free!(io.out)
    return nothing
end


"""
    RIFE

A loaded interpolator: the prepared graph, its weights, the scratch it needs, and
the padded input buffer.

Built through `DNNKernels.Model`, which runs the host-side preparation passes
the editor's own path gets.
"""
# `plan` is a `DNNKernels.RecordedPlan`: one Mantle plan, declared and recorded
# at load, replayed per frame. It replaced four fields — a `planslab` slab, a
# `Workspace` arena, the lazy-broadcast set and the graph's own values table —
# each of which was recovering something the graph had already stated.
struct RIFE{B,M,I,T,A}
    backend::B
    model::M
    input::I
    timestep::T
    padded::Tuple{Int,Int}
    # The arguments `call` takes, in the graph's own input order, resolved once
    # at load. The order is a property of the export and not of a frame.
    args::A
    # The pack and unpack plans, per [`FrameIO`](@ref) key. A `Dict` because the
    # frame size and the pixel type are the CALLER's, not the export's, and the
    # plans are recorded against both: a stream records one entry and replays it
    # for every frame in it.
    frameio::Dict{Any,FrameIO}
end

"""
    release!(model::RIFE) -> nothing

Give back everything the model owns: the recorded plan for every frame geometry
it has been called at, the device frames behind them, and the graph's own plan
and weights.

**A dropped model frees nothing.** Mantle frees on a verb and never from a
finalizer, so without this a caller that interpolates a 1080p stream and then a
720p one keeps the first stream's plans and its three frame buffers for the life
of the process — and a `RIFE` that goes out of scope keeps all of it.
"""
function release!(model::RIFE)
    for io in values(model.frameio)
        freeframeio!(io)
    end
    empty!(model.frameio)
    DNNKernels.releaseplans!(model.model)
    DNNKernels.releaseweights!(model.model.weights)
    return nothing
end


"""
    framesize(model) -> (w, h)

The **padded** frame size the installed export was built for. A frame handed to
[`interpolate!`](@ref) may be smaller — it is padded up to this — but not larger,
because the graph's shape is baked.
"""
framesize(model::RIFE) = model.padded

"""
    rife(; backend = Mantle.defaultbackend(), dir = assetdir()) -> RIFE

Load the model. Separate from [`interpolate!`](@ref) so the workload can build it
in `@setup_workload`, where the loading is not what is being cached.
"""
function rife(; backend = Mantle.defaultbackend())
    dir = assetdir()
    ready() || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_rife.py`"))
    model = Model(Dict("rife" => rifegraph()), rifeweights(); backend)
    graph = model.graphs["rife"]
    # The export baked one resolution; read it back rather than assume 1080p, so
    # a `--height/--width` export is picked up without editing this file.
    shape = graph.buffers["imgs"].shape          # torch (1, 6, H, W)
    w, h = Int(shape[4]), Int(shape[3])
    # `emitgraph` DECLARES the ops and runs nothing; `Plan` then runs all seven
    # of Mantle's phases over the whole graph, so placement, aliasing and
    # barriers are decided before a byte is touched. `planfor` is the same one
    # `Model`'s `call` uses, so a runner cannot plan differently from the driver.
    # Planned HERE and not on a lazy first call: the latency test asserts that
    # the first frame in a fresh process refuses zero pipeline compiles, so the
    # compiling has to happen at load. `recordedplan` builds the plan `call`
    # would build, under the key `call` looks up, and hands it back.
    mp = DNNKernels.recordedplan(model, "rife")
    # The plan's OWN inputs, and not two arrays of ours for `call` to copy in:
    # identical objects are the case `replay!` skips, so a frame no longer moves
    # the padded 1920x1152x6 input twice. That is 53 MB per frame for nothing,
    # and the pack pass below writes the same buffer the graph reads.
    args = Tuple(Mantle.storage(r) for r in mp.inputs)
    tsi = findfirst(==("timestep"), graph.inputs)
    tsi === nothing && throw(ArgumentError(
        "the rife export has no `timestep` input; its inputs are $(graph.inputs)"))
    input = args[findfirst(!=("timestep"), graph.inputs)]
    return RIFE(model.backend, model, input, args[tsi], (w, h), args,
                Dict{Any,FrameIO}())
end

"""
    frameio!(model, w, h, Pin, Pout) -> FrameIO

The pack and unpack plans for frames of this size and these pixel types,
recorded on first use and replayed after.

Recorded HERE and not at load, because none of the four is a property of the
export: the caller's frame may be any size up to [`framesize`](@ref) and any
`AbstractRGB`, and a plan is recorded against the buffers, the geometry and the
types it will run with. The workload interpolates one frame, so a stream that
matches it finds every pipeline in the frozen cache and this records without
compiling anything — which is what the latency test asserts.
"""
function frameio!(model::RIFE, w::Int, h::Int, ::Type{Pin}, ::Type{Pout}) where {Pin,Pout}
    get!(model.frameio, (w, h, Pin, Pout)) do
        dev = Mantle.todevice(model.backend)
        pw, ph = model.padded
        a = Mantle.Buffer(dev, Pin, (w, h))
        b = Mantle.Buffer(dev, Pin, (w, h))
        out = Mantle.Buffer(dev, Pout, (w, h))
        gp = Mantle.Graph(dev)
        Mantle.dispatch!(gp, frames_kernel!, (model.input, a, b, Int32(w), Int32(h)),
                         (pw, ph); group = DNNKernels.launchgroup((pw, ph)),
                         name = "frames")
        pack = Mantle.Plan(gp)
        Mantle.record!(pack)
        # The graph's result, at the address the model plan recorded it at: the
        # unpack reads where the network writes, with no copy in between.
        result = first(DNNKernels.recordedplan(model.model, "rife").outputs)
        gu = Mantle.Graph(dev)
        Mantle.dispatch!(gu, unpack_kernel!, (out, result), (w, h);
                         group = DNNKernels.launchgroup((w, h)), name = "unpack")
        unpack = Mantle.Plan(gu)
        Mantle.record!(unpack)
        FrameIO(pack, unpack, a, b, out)
    end
end

"""
    interpolate!(out, model, a, b; t = 0.5) -> out

Interpolate a frame between `a` and `b` at time `t`, into `out`.

All three frames are the same size and no larger than [`framesize`](@ref); they
are zero-padded up to the graph's baked resolution and the result is cropped
back. `t` is any value in `[0, 1]` — it is a graph input rather than a baked
constant, which is what makes retiming and 4x slow motion three calls that differ
only in one scalar rather than three exports.

Costs ~320 ms per frame at 1080p on an RTX 3070 laptop against a 16.67 ms budget
for 1080p60. The port is correct and the target is not close; see
`plans/projects/small-models/REPORT.md` for where the time goes and why fp32 on
this card cannot reach it.
"""
function interpolate!(out::AbstractMatrix{<:AbstractRGB}, model::RIFE,
                      a::AbstractMatrix{<:AbstractRGB}, b::AbstractMatrix{<:AbstractRGB};
                      t::Real = 0.5)
    size(a) == size(b) || throw(DimensionMismatch("frames differ: $(size(a)) vs $(size(b))"))
    size(out) == size(a) ||
        throw(DimensionMismatch("out $(size(out)) does not match the frames $(size(a))"))
    w, h = size(a)
    pw, ph = model.padded
    (w <= pw && h <= ph) || throw(ArgumentError(
        "frame $(size(a)) is larger than the export's $(model.padded); re-export with " *
        "`uv run tools/export_rife.py --height $h --width $w`"))
    0 <= t <= 1 || throw(ArgumentError("t must be in [0, 1], got $t"))

    fill!(model.timestep, Float32(t))
    io = frameio!(model, w, h, eltype(a), eltype(out))
    # `frames_kernel!` reads `a` and `b` ON THE DEVICE, and this function's
    # signature takes `AbstractMatrix{<:AbstractRGB}` with no backend in the
    # type: every real caller starts from a host frame, because that is what a
    # decoder produces. Passing one straight through handed GPUCompiler a
    # `::Matrix{RGB{N0f8}}` argument and failed at `check_invocation` — the
    # documented entry point could not be called at all. (Same defect as
    # `GPUFiltering.resizeplanar!` had, from the same commit, in a second
    # package: a host-typed API in front of a device-only kernel, with no test
    # that runs a forward pass to notice.)
    #
    # So the frames are copied into the buffers the pack plan was recorded
    # against. `RGB{N0f8}` is isbits, so for a host frame this is the upload it
    # always needed, and for a device frame it is a device copy.
    copyto!(Mantle.storage(io.a), a)
    copyto!(Mantle.storage(io.b), b)
    Mantle.run!(io.pack)
    # `call`: the plan was recorded at load, so this finds it in the model's
    # cache and submits one recording. `model.args` IS that plan's own input
    # tuple, in `g.inputs` order, so nothing is copied into it here — the pack
    # pass above already wrote it.
    DNNKernels.call(model.model, "rife", model.args...; dims = (;))
    Mantle.run!(io.unpack)
    # Cropped back on the device, then handed over. A host `out` is the download
    # it always was; a device `out` is one device copy, which buys the unpack a
    # destination whose address a recorded plan can hold — a caller's array is
    # not that, since nothing stops the next frame passing a different one.
    copyto!(out, Mantle.storage(io.out))
    return out
end

# ---------------------------------------------------------------- the workload
#
# Guarded on the assets and on a working device: precompilation must not fail on
# a machine without either, it should just produce a package with nothing cached.
#
# The workload drives `interpolate!`, which is the whole call tree the editor
# uses — the measurement that matters is `Lava.frozen_stats().misses == 0` on a
# *fresh* process, and a workload that runs a different path than the editor does
# leaves the editor compiling on first use, which is the entire cost this package
# exists to remove. SAM2Runner learned that the expensive way: its `runsam2`
# workload still left 45 s on the first click because the editor goes through a
# closure `runsam2` never touches.
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
# The graph's resolution is baked, so the workload has to run at whatever the
# installed export was built for — there is no smaller stand-in. That makes this
# the most expensive workload of the three: one 1080p interpolation.
@setup_workload begin
    if ready()
        try
            backend = Mantle.defaultbackend()
            # `rife` is inside the workload, not in front of it, and that is not
            # tidiness. `Model`'s last pass is `hoistconstants(graphs, weights,
            # backend)`, which folds constant *subgraphs* by running them on the
            # device — and RIFE has two, the `arange` pair that builds the warp
            # sampling grid. Building the model outside `@compile_workload` left
            # those dispatches unfrozen: `frozen_stats().misses == 9` on a fresh
            # process, every time, no matter what the frame size was.
            #
            # The frame is deliberately *smaller* than the padded size too, so
            # the zero-fill branch of `frames_kernel!` and the crop in
            # `unpack_kernel!` are both on the compiled path. That is what every
            # real frame takes — 1080p is padded to 1152.
            @compile_workload KERNELS_VERSION begin
                model = rife(; backend)
                w, h = framesize(model)
                a = KA.allocate(backend, RGB{Float32}, w, max(h - 72, 1))
                b = similar(a)
                out = similar(a)
                fill!(a, RGB{Float32}(0.3f0, 0.5f0, 0.7f0))
                fill!(b, RGB{Float32}(0.4f0, 0.5f0, 0.6f0))
                interpolate!(out, model, a, b)
                KA.synchronize(backend)
            end
        catch err
            @warn "RIFERunner: workload skipped; first use will compile" exception = err
        end
    else
        @info "RIFERunner: no export at $(assetdir()) — nothing precompiled"
    end
end

end # module
