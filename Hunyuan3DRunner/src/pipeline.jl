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
A loaded shape pipeline: the four graphs with their weights on the device.

Held together rather than passed separately because the denoiser runs once per
step off the same weight table, and rebuilding it per call would dominate the
loop.
"""
struct Hunyuan3D{B}
    backend::B
    cond::Model{B}
    dit::Model{B}
    vae::Model{B}
    geo::Model{B}
    "query points per `hunyuan3d_geo` call — a static shape, read off the graph"
    chunk::Int
end

"""
    hunyuan3d(; backend = LavaBackend(), root = assetdir()) -> Hunyuan3D

Load all four graphs. `root` holds the four directories [`PARTS`](@ref) names.

Separate from [`generate`](@ref) so a workload can build it in `@setup_workload`,
where the loading is not what is being cached.
"""
function hunyuan3d(; backend = LavaBackend(), root::AbstractString = assetdir())
    load(part, name) = begin
        dir = joinpath(root, part)
        isfile(joinpath(dir, "$name.json")) || throw(ArgumentError(
            "Hunyuan3D-2.1: no $name.json in $dir. Generate it with " *
            "`uv run tools/export_hunyuan3d.py --part $(splitpart(part))`."))
        Model(dir, joinpath(dir, "weights.safetensors"); names = [name], backend)
    end
    geo = load(PARTS.geo, "hunyuan3d_geo")
    # The chunk is whatever the export was built at, not a constant here: passing
    # a different one is a silently truncated sweep, not an error.
    chunk = Int(geo.graphs["hunyuan3d_geo"].buffers["queries"].shape[2])
    return Hunyuan3D(backend, load(PARTS.cond, "hunyuan3d_cond"),
                     load(PARTS.dit, "hunyuan3d_dit"), load(PARTS.vae, "hunyuan3d_vae"),
                     geo, chunk)
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
    flowsigmas(steps) -> Vector{Float64}

The noise levels the sampler steps through, `steps + 1` of them.

Upstream's own comment: *this is slightly different from common usage, we start
from 0*. `Hunyuan3DDiTFlowMatchingPipeline.__call__` passes
`np.linspace(0, 1, steps)` as an explicit `sigmas` override, so the schedule runs
**up** from 0 to 1 rather than down, and `set_timesteps` appends a trailing 1.0.
The last step therefore has `sigma_next - sigma == 0` and moves nothing — a
50-step generation does 49 Euler steps and one no-op, which is upstream's
behaviour and not a bug to round away.

The checkpoint's scheduler has `shift = 1.0` and `use_dynamic_shifting = false`,
so `shift * s / (1 + (shift - 1) * s)` is the identity and is not applied here.
A checkpoint with a different shift needs it back.

**Float32, and narrowed before the differences are taken.** `set_timesteps` does
`torch.from_numpy(sigmas).to(dtype=torch.float32)`, so the fp64 ramp is rounded
to fp32 *first* and `sigma_next - sigma` is an fp32 subtraction. Keeping fp64
here and narrowing at the end is a different number in the last bit, which is far
below an fp16 ULP on its own and still shows up: it is a step's worth of rounding,
taken fifty times, into a state that is stored back as fp16 each time.
"""
flowsigmas(steps::Integer) = Float32[collect(range(0.0, 1.0; length = steps)); 1.0]

"""
    flowtimesteps(sigmas) -> Vector{Float32}

What the denoiser is conditioned on: the noise level scaled by the training
horizon. One entry per step, so the trailing sigma has none.

This is *not* what reaches the model — see [`conditioningtime`](@ref), which is
where the units are undone again.
"""
flowtimesteps(sigmas::AbstractVector) = sigmas[1:(end - 1)] .* TRAIN_TIMESTEPS

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
    cfg(pred, scale) -> v

Classifier-free guidance over the denoiser's batch-2 output. `pred` is
`(64, 4096, 2)` with the conditional prediction in column 1, so this is
`uncond + scale * (cond - uncond)`.

Evaluated in the prediction's own dtype, which is fp16: upstream combines the two
halves before the scheduler's `.to(torch.float32)`, and doing it wider here would
be a different arithmetic to compare against.

**Three broadcasts, not one, and that is the whole point of writing it this way.**
Fused into a single expression, Lava keeps the product in fp32 and rounds once at
the store, while PyTorch rounds to fp16 after each of the subtract, the multiply
and the add. Lava's is the more accurate of the two — but it is not the one this
port is reproducing, and the difference is real: measured over the denoiser's
262144 outputs, 5110 of them land on a different fp16 value, and replaying
upstream's own per-step predictions through a fused `cfg` drifts 1.5 ULP off the
reference trajectory by step 50 instead of matching it bit for bit. Each
statement below materialises a fp16 array, which is what forces the rounding.
"""
function cfg(pred::AbstractArray{T,3}, scale::Real) where {T}
    c, u = view(pred, :, :, 1), view(pred, :, :, 2)
    d = c .- u
    d = T(scale) .* d
    return u .+ d
end

"""
    eulerstep(x, v, sigma, sigma_next) -> x'

One Euler step of the flow ODE:

    prev_sample = sample.to(float32) + (sigma_next - sigma) * model_output

**The step size is taken in fp16, and that is not a rounding detail.** Reading
that line as fp32 arithmetic is wrong in a way that changes the integration, not
just the last bit. `sigma_next - sigma` is a 0-dimensional fp32 tensor and
`model_output` is a dimensioned fp16 one, and PyTorch's promotion gives a
dimensioned operand priority over a 0-dimensional one of the same category — so
the *scalar* is narrowed to fp16 and the product is fp16. At 50 steps the
difference is `Float16(1/49)` = 0.0204 against 0.020408163, a step size 0.04%
short, taken every step. Only the sum with the sample is fp32, and the result
narrows again on the way out.

Getting this wrong is not visible in one step — it moves 3969 of 262144 elements
by one fp16 ULP — and it does not announce itself later either; it just walks a
slightly different trajectory. It was found by dumping torch's own
`(sigma_next - sigma) * v` and comparing dtypes, which is the only way this kind
of thing gets found.

`sigma`/`sigma_next` are fp32 because [`flowsigmas`](@ref) returns fp32; the
narrowing below is from there, not from an fp64 schedule.
"""
function eulerstep(x::AbstractArray{T}, v, sigma::Float32, sigma_next::Float32) where {T}
    step = T(sigma_next - sigma) .* v
    return T.(Float32.(x) .+ Float32.(step))
end

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
    sigmas = flowsigmas(steps)
    ts = flowtimesteps(sigmas)
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
        fill!(tin, conditioningtime(T, ts[k]))
        pred = only(call(m.dit, "hunyuan3d_dit", xin, tin, cond; dims = (;)))
        x = eulerstep(x, cfg(reshape(pred, size(x, 1), size(x, 2), 2), guidance),
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
    gridqueries!(q, first, G, lo, step, N)

Fill `q`, a `(3, chunk)` device array, with the query points at flat indices
`first .+ (0:chunk-1)`.

The addressing is upstream's `generate_dense_grid_points(..., indexing="ij")`
followed by `reshape(-1, 3)`, which is row-major: the flat index runs over z
fastest, then y, then x. Getting this transposed produces a mesh that is a
transpose of the right one, which looks like a plausible object.

Indices past the end are **clamped** rather than skipped. The graph's query count
is a static shape, so the final chunk has to be run full and its tail discarded;
clamping evaluates a duplicate point, which is cheaper than a second export and
cannot read out of bounds.
"""
@kernel function gridqueries_kernel!(q, first, G, lo, step, N)
    c = @index(Global)
    n = min(first + c - 1, N - 1)            # 0-based, clamped for the tail
    k = n % G
    j = (n ÷ G) % G
    i = n ÷ (G * G)
    T = eltype(q)
    @inbounds q[1, c] = T(lo + step * i)
    @inbounds q[2, c] = T(lo + step * j)
    @inbounds q[3, c] = T(lo + step * k)
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
    q = KA.allocate(m.backend, Float16, 3, m.chunk)
    kern = gridqueries_kernel!(m.backend)
    for c in 1:nchunks
        first = (c - 1) * m.chunk                     # 0-based
        kern(q, first, G, lo, step, N; ndrange = m.chunk)
        logits = only(call(m.geo, "hunyuan3d_geo", reshape(q, 3, m.chunk, 1), latents;
                           dims = (;)))
        len = min(m.chunk, N - first)
        view(field, (first + 1):(first + len)) .= Float32.(view(vec(logits), 1:len))
        progress === nothing || progress(c, nchunks)
    end
    return reshape(field, G, G, G)
end
