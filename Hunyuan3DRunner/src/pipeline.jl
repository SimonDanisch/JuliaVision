"""
The loop around the four graphs.

`hunyuan3d_cond`, `hunyuan3d_dit`, `hunyuan3d_vae` and `hunyuan3d_geo` are
ordinary DNNKernels graphs and each one matches PyTorch on its own. What is not a
graph, and is what this file is, is everything between them: the sigma schedule,
the classifier-free-guidance combination, the `1/scale_factor` in front of the
VAE, and the sweep of the geometry decoder over a dense query grid.

The shape is `DNNKernels/src/wan.jl`'s — a flow-matching ODE stepped on the host
with the model evaluated inside the loop — with two differences that come from
the model rather than from taste. Wan evaluates its transformer twice per step,
once conditional and once not; Hunyuan3D's denoiser is exported at **batch 2** and
does both halves in one forward, so `cfg` splits a result instead of combining two.
And Wan's decoder is a single graph, while here the decode is 7134 calls of a
20-op graph over a 385^3 grid, which is where the wall clock goes.

Reference: `tools/dump_hunyuan3d_pipeline.py` runs upstream's own pipeline and
writes every intermediate this file has to reproduce. It is the only check that
covers the *glue*; the per-graph `reference.safetensors` cannot, by construction.
"""

# ------------------------------------------------------------------- the layout

"""
    PARTS

Directory name for each of the four graphs, relative to the export root, and the
graph name inside it. `tools/export_hunyuan3d.py --part cond` writes
`gen/graphs/hunyuan3d-cond/hunyuan3d_cond.json` and a `weights.safetensors`
beside it, so pointing `root` at `gen/graphs` is what makes a working copy
loadable without a second layout to keep in step. An artifact is these four
directories and nothing else.

Four separate models rather than one, because they carry different weights and
run at wildly different rates: the conditioner once per generation, the denoiser
fifty times, the VAE once, and the geometry decoder some seven thousand times.
"""
const PARTS = (cond = "hunyuan3d-cond", dit = "hunyuan3d",
               vae = "hunyuan3d-vae", geo = "hunyuan3d-geo")

"""
    SCALE_FACTOR

`vae.scale_factor` from the checkpoint's `config.yaml`. The sampler works in a
scaled latent space and `_export` divides by this before decoding. Small enough
(1.004) that dropping it produces a slightly wrong mesh rather than a broken one,
which is exactly why it is named here instead of inlined.
"""
const SCALE_FACTOR = 1.0039506158752403

"""
    TRAIN_TIMESTEPS

`scheduler.num_train_timesteps`. The denoiser is conditioned on `t / 1000`, not
on `t`, and the schedule is in units of `t`.
"""
const TRAIN_TIMESTEPS = 1000

"""
    GridChunk(first, G, lo, step, N)

Which points of the sweep a launch of [`gridqueries_kernel!`](@ref) writes.

One struct and not five arguments because it is read through a `GPURef`, and
what that buys is a single recorded plan for ANY `octree` and `box_v`: the grid
is a per-run value rather than something baked when the decoder's graph is
recorded. `first` changes per chunk and the other four per call; both are the
same store.
"""
struct GridChunk
    first::Int          # 0-based index of this chunk's first point
    G::Int              # points per axis
    lo::Float64
    step::Float64
    N::Int              # points in the whole grid, for the tail clamp
end

"""
A loaded shape pipeline: the four graphs with their weights on the device.

Held together rather than passed separately because the denoiser runs once per
step off the same weight table, and rebuilding it per call would dominate the
loop.
"""
# `M<:Model` and not `Model{B}`: `Model` is parameterised on its DEVICE first and
# its backend second, so `Model{B}` named the device type and no four models
# could satisfy it beside a `backend::B` of the same `B`. Parameterising on the
# model type says what was meant — four models of one kind — without this file
# having to track `Model`'s parameter list.
struct Hunyuan3D{B,M<:Model}
    backend::B
    cond::M
    dit::M
    vae::M
    geo::M
    "query points per `hunyuan3d_geo` call — a static shape, read off the graph"
    chunk::Int
    # Which points the geometry decoder's query pass is writing. Here rather than
    # in `occupancy` because the plan is packed with its ADDRESS at `record!` and
    # cached for the model's life: a ref built per call would be one the plan
    # does not read, and freeing it at the end of a call would leave the plan
    # reading memory that is back in the pool.
    querychunk::Mantle.GPURef{GridChunk}
end

"""
    hunyuan3d(; backend = Mantle.defaultbackend(), root = assetdir()) -> Hunyuan3D

Load all four graphs. `root` holds the four directories [`PARTS`](@ref) names.

`maxpasses` is how many passes of the DENOISER go in one submission; see the
comment in the body for why it is not a tuning knob.

Separate from [`generate`](@ref) so a workload can build it in `@setup_workload`,
where the loading is not what is being cached.
"""
function hunyuan3d(; backend = Mantle.defaultbackend(), maxpasses::Integer = 64)
    # `weights` is explicit because the denoiser's do not sit beside its graph:
    # they are four shard artifacts merged by `hunyuan3dweights`, while the other
    # three parts each keep a `weights.safetensors` in their own artifact.
    #
    # `record_maxpasses` on the DENOISER, and it is a correctness fix rather than
    # a tuning knob — the same one `kokorovoc` and Qwen-Image's transformer carry.
    # One step of the 21-block model at batch 2 is seconds of device time, and a
    # single submission that long is killed by the driver: RADV reports "the CS
    # has been cancelled because the context is lost, this context is guilty of a
    # hard recovery", the next call dies with VK_ERROR_DEVICE_LOST, and nothing
    # in the Julia frame names the cause. The completion points cost nothing
    # measurable and the barriers between the pieces are still the ones the graph
    # derived. `maxpasses = 0` restores the single submission.
    load(dir, name, weights = nothing; split = 0) = begin
        isfile(joinpath(dir, "$name.json")) || throw(ArgumentError(
            "Hunyuan3D-2.1: no $name.json in $dir. Re-export with " *
            "`uv run tools/export_hunyuan3d.py` and re-bind with " *
            "`julia --project=. tools/make_artifacts.jl`."))
        w = weights === nothing ?
            readsafetensors(joinpath(dir, "weights.safetensors")) : weights
        Model(Dict(name => loadgraph(joinpath(dir, "$name.json"))), w; backend,
              record_maxpasses = Dict(name => Int(split)))
    end
    geo = load(geodir(), "hunyuan3d_geo")
    # The chunk is whatever the export was built at, not a constant here: passing
    # a different one is a silently truncated sweep, not an error.
    chunk = Int(geo.graphs["hunyuan3d_geo"].buffers["queries"].shape[2])
    return Hunyuan3D(backend, load(conddir(), "hunyuan3d_cond"),
                     load(ditdir(), "hunyuan3d_dit", hunyuan3dweights(); split = maxpasses),
                     load(vaedir(), "hunyuan3d_vae"),
                     geo, chunk,
                     Mantle.GPURef(Mantle.todevice(backend), GridChunk(0, 0, 0.0, 0.0, 0)))
end

"""
    splitpart(dir) -> String

The `--part` flag that writes directory `dir`, for the error message above.
`"hunyuan3d"` is the default part and is spelled `dit`.
"""
splitpart(dir::AbstractString) = dir == PARTS.dit ? "dit" : replace(dir, "hunyuan3d-" => "")

# --------------------------------------------------------------- the conditioner

"""
    encodeimage(m, image) -> cond

DINOv2-large over one normalised 518x518 image, paired with its unconditional
half. `image` is `(518, 518, 3, 1)` — the reversed layout of torch's
`(1, 3, 518, 518)` — already through [`normalizeimage`](@ref).

Returns `(1024, 1370, 2)`: **column 1 is the image condition and column 2 is the
null**, matching `encode_cond`'s `cat([cond, un_cond])`. Swapping them flips the
sign of the guidance and still produces a mesh, so the order is asserted in the
tests rather than left to reading.

The null half is exact zeros, not a learned embedding — upstream's
`unconditional_embedding` is a `torch.zeros` call — so the encoder runs once per
generation, not twice.
"""
function encodeimage(m::Hunyuan3D, image)
    c = only(call(m.cond, "hunyuan3d_cond", toback(m.backend, image); dims = (;)))
    cond = KA.allocate(m.backend, eltype(c), size(c, 1), size(c, 2), 2)
    fill!(cond, zero(eltype(c)))
    copyto!(view(cond, :, :, 1), reshape(c, size(c, 1), size(c, 2)))
    return cond
end

# ------------------------------------------------------------------ the sampler

"""
    hunyuanschedule(steps) -> (; sigmas, timesteps)

The noise levels the sampler steps through, `steps + 1` of them, and the
timestep each step conditions on; built by DNNKernels' shared `flowschedule`.

Upstream's own comment: *this is slightly different from common usage, we start
from 0*. `Hunyuan3DDiTFlowMatchingPipeline.__call__` passes
`np.linspace(0, 1, steps)` as an explicit `sigmas` override, so the schedule runs
**up** from 0 to 1 rather than down, and `set_timesteps` appends a trailing 1.0.
The last step therefore has `sigma_next - sigma == 0` and moves nothing — a
50-step generation does 49 Euler steps and one no-op, which is upstream's
behaviour and not a bug to round away.

The checkpoint's scheduler has `shift = 1.0` and `use_dynamic_shifting = false`,
which is the identity: `NoShift`. A checkpoint with a different shift needs it
back.

**Float32, and narrowed before the differences are taken.** `set_timesteps` does
`torch.from_numpy(sigmas).to(dtype=torch.float32)`, so the fp64 ramp is rounded
to fp32 *first* and `sigma_next - sigma` is an fp32 subtraction. Keeping fp64
here and narrowing at the end is a different number in the last bit, which is far
below an fp16 ULP on its own and still shows up: it is a step's worth of rounding,
taken fifty times, into a state that is stored back as fp16 each time.

The timesteps are the sigmas on the training horizon, one per step. They are
*not* what reaches the model — see [`conditioningtime`](@ref), which is where the
units are undone again.
"""
hunyuanschedule(steps::Integer) =
    flowschedule(linspace(0.0, 1.0, steps), NoShift(); terminal = 1f0,
                 trainsteps = TRAIN_TIMESTEPS)

"""
    conditioningtime(T, t) -> T

The 0..1 scalar the denoiser is actually conditioned on, from a timestep in the
schedule's units.

The rounding order is upstream's and it is not the obvious one:

    timestep = t.expand(batch).to(latents.dtype)          # fp32 -> fp16 FIRST
    timestep = timestep / self.scheduler.config.num_train_timesteps

so the timestep is narrowed to fp16 and *then* divided, rather than divided in
the schedule's precision and narrowed once. At step 2 that is `Float16(20.408)`
= 20.41, over 1000 — a different fp16 value from `Float16(20.408 / 1000)`, and it
is an input to a 754-op forward, not a rounding at the end of one.
"""
conditioningtime(::Type{T}, t::Real) where {T} = T(t) / TRAIN_TIMESTEPS

"""
    denoise(m, cond, latents; steps, guidance, progress) -> latents

The sampling loop: `steps` evaluations of the denoiser at batch 2, guidance, and
an Euler step, exactly `Hunyuan3DDiTFlowMatchingPipeline.__call__`'s body.

`latents` is the starting noise, `(64, 4096, 1)`. It is an argument rather than
something drawn here because reproducing PyTorch's Philox stream is not what this
port is for, and starting from the same noise is the only way a per-step
comparison against the reference means anything. [`randomlatents`](@ref) draws one
when the caller has no reference to match.
"""
function denoise(m::Hunyuan3D, cond, latents; steps::Integer = 50,
                 guidance::Real = 5.0, progress = nothing)
    (; sigmas, timesteps) = hunyuanschedule(steps)
    x = toback(m.backend, latents)
    T = eltype(x)
    # The batch-2 input the denoiser is exported for: the same latents twice, one
    # row against the image condition and one against the null.
    xin = KA.allocate(m.backend, T, size(x, 1), size(x, 2), 2)
    tin = KA.allocate(m.backend, T, 2)
    for k in 1:steps
        copyto!(view(xin, :, :, 1), x)
        copyto!(view(xin, :, :, 2), x)
        # The model takes `t / num_train_timesteps`, i.e. a 0..1 scalar broadcast
        # over the batch — not the timestep itself.
        fill!(tin, conditioningtime(T, timesteps[k]))
        pred = only(call(m.dit, "hunyuan3d_dit", xin, tin, cond; dims = (;)))
        # Column 1 of the batch is the image-conditioned prediction, column 2 the null one.
        p = reshape(pred, size(x, 1), size(x, 2), 2)
        x = eulerstep(x, cfg(view(p, :, :, 1), view(p, :, :, 2), guidance),
                      sigmas[k], sigmas[k + 1])
        progress === nothing || progress(k, steps)
    end
    return x
end

"""
    randomlatents(m; batch = 1, T = Float16) -> latents

Starting noise at the shape the denoiser was exported for. Standard normal and
unscaled: the scheduler has no `init_noise_sigma`, so `prepare_latents`'
multiplication by it is by 1.
"""
function randomlatents(m::Hunyuan3D; batch::Integer = 1, T = Float16)
    b = m.dit.graphs["hunyuan3d_dit"].buffers["x"]
    x = randn(T, Int(b.shape[3]), Int(b.shape[2]), batch)
    return toback(m.backend, x)
end

# ------------------------------------------------------------------- the decode

"""
    shapelatents(m, latents) -> latents

`_export`'s first half: undo the sampler's scaling, then `post_kl` and the
16-layer decoder transformer. `(64, 4096, 1)` in, `(1024, 4096, 1)` out.
"""
function shapelatents(m::Hunyuan3D, latents)
    T = eltype(latents)
    z = T.(Float32.(latents) .* Float32(1 / SCALE_FACTOR))
    return only(call(m.vae, "hunyuan3d_vae", z; dims = (;)))
end

"""
    gridqueries_kernel!(q, chunk)

Fill `q`, the decoder's `(3, chunk, 1)` query input, with the points
[`GridChunk`](@ref) names.

The addressing is upstream's `generate_dense_grid_points(..., indexing="ij")`
followed by `reshape(-1, 3)`, which is row-major: the flat index runs over z
fastest, then y, then x. Getting this transposed produces a mesh that is a
transpose of the right one, which looks like a plausible object.

Indices past the end are **clamped** rather than skipped. The graph's query count
is a static shape, so the final chunk has to be run full and its tail discarded;
clamping evaluates a duplicate point, which is cheaper than a second export and
cannot read out of bounds.
"""
# Macro-free, over `KernelInterface`'s intrinsics. The guard is the one the
# macro used to insert: `ndrange` is `m.chunk` and the workgroup need not divide
# it, so the surplus threads wrote past `q`.
function gridqueries_kernel!(q, chunk)
    c = KI.get_global_id().x
    c <= size(q, 2) || return nothing
    g = @inbounds chunk[1]
    n = min(g.first + c - 1, g.N - 1)        # 0-based, clamped for the tail
    k = n % g.G
    j = (n ÷ g.G) % g.G
    i = n ÷ (g.G * g.G)
    T = eltype(q)
    # Three indices because `q` is the decoder's own `(3, chunk, 1)` input and
    # not a scratch array of this function's shape — see `occupancy`.
    @inbounds q[1, c, 1] = T(g.lo + g.step * i)
    @inbounds q[2, c, 1] = T(g.lo + g.step * j)
    @inbounds q[3, c, 1] = T(g.lo + g.step * k)
    return nothing
end

"""
    occupancy(m, latents; box_v, octree, progress) -> field

The implicit field over a dense `(octree + 1)^3` grid, as `VanillaVolumeDecoder`
computes it: chunk the query points, run the geometry decoder on each, and
concatenate.

Returns a `(G, G, G)` **device** array of fp32 logits in the runtime's reversed
layout — `field[k, j, i]` is upstream's `grid_logits[i, j, k]`, the same
convention every other array here uses. fp32 because upstream's `.float()` is,
and because the level set is read off these values.

This is the expensive call in the pipeline and it is not close: at the default
`octree = 384` the grid is 57.07M points, 7134 chunks, and about 16.8 MFLOP of
cross-attention per point against the 4096 shape latents.
"""
function occupancy(m::Hunyuan3D, latents; box_v::Real = 1.01, octree::Integer = 384,
                   progress = nothing)
    G = Int(octree) + 1
    N = G^3
    lo, hi = -Float64(box_v), Float64(box_v)
    # `np.linspace(lo, hi, G)`: G points inclusive of both ends, so the spacing
    # divides by G-1 and NOT by the octree resolution.
    step = (hi - lo) / (G - 1)
    nchunks = cld(N, m.chunk)

    field = KA.allocate(m.backend, Float32, N)
    # The queries are DECLARED INTO the decoder's own graph, so a chunk is one
    # submission of one plan: the pass that writes `queries` and the twenty that
    # read it, ordered by Mantle rather than by the queue. As a graph of its own
    # it was a second submission per chunk, and `call` then copied the latents
    # into the plan for every one of the 7134 — half a megabyte each, for a value
    # that does not change over the sweep.
    #
    # Which points a run writes is the only thing that changes, and it is a
    # `GPURef`: one small store per run into an address the plan was packed with,
    # so a second sweep at a different `octree` replays this same plan.
    mp = DNNKernels.recordedplan(m.geo, "hunyuan3d_geo"; before = ctx ->
        Mantle.dispatch!(ctx.g, gridqueries_kernel!,
                         (DNNKernels.operand(ctx, "queries"), m.querychunk),
                         m.chunk; group = 256, name = "gridqueries"))
    lat = Mantle.storage(latents)
    size(mp.inputs[2]) == size(lat) || throw(DimensionMismatch(
        "hunyuan3d_geo declares latents $(size(mp.inputs[2])) and got $(size(lat))"))
    copyto!(mp.inputs[2], lat)                        # once, not per chunk
    logits = vec(first(mp.outputs))
    for c in 1:nchunks
        first0 = (c - 1) * m.chunk                    # 0-based
        m.querychunk[] = GridChunk(first0, G, lo, step, N)
        Mantle.run!(mp.plan)
        len = min(m.chunk, N - first0)
        view(field, (first0 + 1):(first0 + len)) .= Float32.(view(logits, 1:len))
        progress === nothing || progress(c, nchunks)
    end
    return reshape(field, G, G, G)
end
