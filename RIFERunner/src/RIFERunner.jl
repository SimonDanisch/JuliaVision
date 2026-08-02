"""
RIFE 4.x (Practical-RIFE) — frame interpolation.

Slow motion, framerate conversion, and smoothing a retime — about 10 MB of weights for all of it.

Cheap for the runtime: the warping is `grid_sampler_2d`, which is already implemented and already exercised by the optical-flow path in GPUFiltering.

**Ported and verified** against upstream on 2026-08-02. [`rife`](@ref) loads the
model and [`interpolate!`](@ref) produces one frame between two, at any `t` in
[0, 1]. The interpolated frame matches PyTorch to **3.3e-4** max and 4.9e-7 mean.

**Correct, and far off its target.** One 1080p frame costs ~500 ms on an RTX 3070
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
using Lava: @setup_workload, @compile_workload
using DNNKernels: loadgraph, execute!, readsafetensors, assetpath, toback,
                  Model, planslab, fusableset, Workspace
using GPUFiltering: tofloat, topixel
using ColorTypes: AbstractRGB, RGB

export rifegraph, rifeweights, assetdir
export rife, interpolate!, framesize, RIFE

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

Three places, in order: `JULIA_RIFE_ASSETS` if it is set, the generated
directory if this is a checkout that has run the exporter, and the lazy artifact
otherwise. A developer re-exporting a graph gets their own copy without touching
the artifact; a plain `Pkg.add` gets the published one.

The artifact is bound in `../Artifacts.toml` by `tools/make_artifacts.jl`, and it
carries the graph and weights only — `reference*.safetensors` is what
`tools/verify_rife.jl` diffs against and the exporter regenerates it in one
command, so it is not something a caller should have to download.
"""
assetdir() = assetpath(; artifact = "rife",
                       toml = joinpath(@__DIR__, "..", "Artifacts.toml"),
                       generated = joinpath("gen", "graphs", "rife"),
                       env = "JULIA_RIFE_ASSETS", from = @__DIR__)

"""
    rifegraph(; dir = assetdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function rifegraph(; dir::AbstractString = assetdir())
    p = joinpath(dir, "rife.json")
    isfile(p) || throw(ArgumentError(
        "RIFE 4.x (Practical-RIFE) graph not found at $p. Generate it with " *
        "`uv run tools/export_rife.py`, or set JULIA_RIFE_ASSETS."))
    return loadgraph(p)
end

"""
    rifeweights(; dir = assetdir()) -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it.
"""
function rifeweights(; dir::AbstractString = assetdir())
    p = joinpath(dir, "weights.safetensors")
    isfile(p) || throw(ArgumentError("RIFE 4.x (Practical-RIFE) weights not found at $p"))
    return readsafetensors(p)
end

"""
    ready(; dir = assetdir()) -> Bool

Whether an export is installed. The workload and the tests both branch on this,
because neither may fail on a machine that has not run the exporter.
"""
ready(; dir::AbstractString = assetdir()) =
    isfile(joinpath(dir, "rife.json")) && isfile(joinpath(dir, "weights.safetensors"))

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Lava.use_frozen_kernels(KERNELS_VERSION)
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
@kernel function frames_kernel!(dst, @Const(a), @Const(b), w::Int32, h::Int32)
    I = @index(Global, Cartesian)
    x, y = I[1], I[2]
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
end

# The interpolated frame back out of the graph's (W, H, 3, 1), cropped to the
# real frame — the padded rows are network output over zeros and are not part of
# the picture.
@kernel function unpack_kernel!(out, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds out[I] = topixel(eltype(out), src[I[1], I[2], 1, 1],
                               src[I[1], I[2], 2, 1], src[I[1], I[2], 3, 1])
end

"""
    RIFE

A loaded interpolator: the prepared graph, its weights, the scratch it needs, and
the padded input buffer.

Built through `DNNKernels.Model`, which runs the host-side preparation passes.
Note that on *this* graph they appear to cost rather than save time (~500 ms
against 327 ms without them) — that is recorded and unexplained in
`plans/projects/small-models/REPORT.md`; the driver path is kept because it is
the one the editor uses and correctness is not in question.
"""
struct RIFE{B,G,W,S,P,I,T}
    backend::B
    graph::G
    weights::W
    slab::S
    plan::P
    ws::Workspace
    lazy::Set{String}
    input::I
    timestep::T
    padded::Tuple{Int,Int}
end

"""
    framesize(model) -> (w, h)

The **padded** frame size the installed export was built for. A frame handed to
[`interpolate!`](@ref) may be smaller — it is padded up to this — but not larger,
because the graph's shape is baked.
"""
framesize(model::RIFE) = model.padded

"""
    rife(; backend = LavaBackend(), dir = assetdir()) -> RIFE

Load the model. Separate from [`interpolate!`](@ref) so the workload can build it
in `@setup_workload`, where the loading is not what is being cached.
"""
function rife(; backend = LavaBackend(), dir::AbstractString = assetdir())
    ready(; dir) || throw(ArgumentError(
        "no export at $dir — generate it with `uv run tools/export_rife.py`"))
    model = Model(dir, joinpath(dir, "weights.safetensors"); names = ["rife"], backend)
    graph = model.graphs["rife"]
    # The export baked one resolution; read it back rather than assume 1080p, so
    # a `--height/--width` export is picked up without editing this file.
    shape = graph.buffers["imgs"].shape          # torch (1, 6, H, W)
    w, h = Int(shape[4]), Int(shape[3])
    plan = planslab(graph, (;))
    slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
    input = KA.allocate(backend, Float32, w, h, 6, 1)
    timestep = KA.allocate(backend, Float32, 1, 1, 1, 1)
    return RIFE(backend, graph, model.weights, slab, plan, Workspace(backend),
                fusableset(graph), input, timestep, (w, h))
end

"""
    interpolate!(out, model, a, b; t = 0.5) -> out

Interpolate a frame between `a` and `b` at time `t`, into `out`.

All three frames are the same size and no larger than [`framesize`](@ref); they
are zero-padded up to the graph's baked resolution and the result is cropped
back. `t` is any value in `[0, 1]` — it is a graph input rather than a baked
constant, which is what makes retiming and 4x slow motion three calls that differ
only in one scalar rather than three exports.

Costs ~500 ms per frame at 1080p on an RTX 3070 laptop against a 16.67 ms budget
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
    frames_kernel!(model.backend)(model.input, a, b, Int32(w), Int32(h);
                                  ndrange = (pw, ph))
    vals = execute!(model.graph,
                    Dict{String,Any}("imgs" => model.input, "timestep" => model.timestep),
                    model.weights; dims = (;), backend = model.backend,
                    slab = model.slab, plan = model.plan,
                    ws = model.ws, lazy = model.lazy)
    unpack_kernel!(model.backend)(out, vals[only(model.graph.outputs)]; ndrange = (w, h))
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
# The graph's resolution is baked, so the workload has to run at whatever the
# installed export was built for — there is no smaller stand-in. That makes this
# the most expensive workload of the three: one 1080p interpolation.
@setup_workload begin
    if ready()
        try
            backend = LavaBackend()
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
