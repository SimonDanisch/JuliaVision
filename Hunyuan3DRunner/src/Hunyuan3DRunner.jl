"""
Hunyuan3D-2.1 — a single image to a 3D mesh.

One 7.37 GB fp16 checkpoint holds three models: the shape DiT (3.051B
parameters), a VAE (0.328B) and a DINOv2-large image conditioner (0.304B). This
package is the shape branch. `hunyuan3d-paintpbr-v2-1`, the multi-view PBR
texture painter, is a different model and a separate port.

The shape denoiser is a flow-matching diffusion transformer over a **set** of
4096 latents — the config turns positional embedding off, so the latents carry no
order — with 21 blocks, U-Net skip connections on the last ten and a top-2-of-8
mixture of experts on the last six. Fifty Euler steps at batch 2 for
classifier-free guidance produce the latents; a VAE transformer and a
cross-attention geometry decoder turn those into an implicit field over a dense
grid, and marching cubes turns that into a mesh.

## State: image to mesh runs end to end on Lava

[`imagetomesh`](@ref) takes an RGBA cut-out and returns a triangle mesh. On an
RTX 4000 Ada, `assets/demo.png` at the pipeline's defaults — 50 steps, guidance
5.0, `octree_resolution = 384` — is **188.6 s** to a 346 474-vertex, 692 956-face
watertight mesh, spent 100 s in the denoiser, 88 s in the 7134 geometry-decoder
calls and 0.5 s in the surface extraction.

Measured against `tools/dump_hunyuan3d_pipeline.py`, which runs upstream's own
pipeline on the same image and dumps every intermediate:

| stage | vs upstream | note |
|---|---|---|
| `prepareimage` mask | **bit-exact** | pins the framing geometry |
| `prepareimage` image | 0.0052% of range | OpenCV's fixed-point resize, ~2/255 |
| `normalizeimage` | 0.0033% of range | from upstream's own 512 tensor |
| `encodeimage` | 0.011% of range | including this port's preprocessing |
| one denoiser step | 0.0029% of range | corr 0.99999994 |
| **the whole 50-step sampler** | **bit-exact** | replaying upstream's predictions |
| 50 steps with the model in the loop | 0.26% of range | corr 0.9963 |
| `shapelatents` | 0.0003% of range | corr 0.9999932 |
| `occupancy`, 65³ | 0.0054% of range | corr 0.99999994 |
| `marchingcubes` vs `skimage` | 2.4e-7 | every vertex this produces |

Two rows deserve reading together. **The sampler is bit-exact** — fed upstream's
own per-step predictions it walks upstream's trajectory with zero differing
elements across all 50 steps — and the 0.26% on the row below it is therefore
*not* glue error. It is the denoiser's own fp16 rounding, 2.7e-4 per step,
amplified by fifty steps of a locally expanding ODE. What that costs where it
matters: the sign of the field, which is all marching cubes reads, agrees on
**99.93%** of grid points, and the 185 that disagree out of 274 625 are all
within one voxel of the surface.

Getting the sampler bit-exact took four separate rounding facts that are not
visible in upstream's source; `test/runtests.jl` pins each with the values that
discriminate. In order of how much they moved the result: the Euler step size is
**fp16**, because `sigma_next - sigma` is a 0-dim fp32 tensor against a
dimensioned fp16 one and PyTorch's promotion narrows the scalar — so the step is
`Float16(1/49)` = 0.0204 and not 0.020408163; the timestep is narrowed to fp16
*before* being divided by 1000, not after, which lands on a different value at 9
of the 50 steps; guidance rounds to fp16 after each of its three operations,
where a single fused Lava broadcast keeps the product in fp32 and rounds once;
and the sigma ramp is fp32 before the differences are taken, not fp64 narrowed
after.

**Done.** `tools/export_hunyuan3d.py` produces all four graphs and every one runs
on Lava against the PyTorch reference at the **fp16 noise floor**:

| graph | ops | weights | mean abs err | % of range | corr |
|---|---|---|---|---|---|
| `hunyuan3d_cond` — DINOv2-large | 293 | 439 | 2.1e-3 | 0.0043% | 0.9999985 |
| `hunyuan3d_dit` — the denoiser | 754 | 752 | 7.2e-4 | 0.0120% | 0.9999972 |
| `hunyuan3d_vae` — `post_kl` + 16 layers | 177 | 242 | 1.6e-3 | 0.0004% | 1.0000000 |
| `hunyuan3d_geo` — one occupancy chunk | 20 | 25 | 1.5e-4 | 0.0075% | 0.9999973 |

The denoiser's median error is exactly one fp16 ULP, and its error grows with
magnitude rather than with position — accumulated relative rounding through 21
residual blocks, which is not what a wrong kernel looks like.

Three ATen ops were missing and are now in `DNNKernels`: `_fused_rms_norm`,
`topk` and `scatter.src`, all three in the denoiser. The conditioner, the VAE and
the geometry decoder needed nothing new at all.
`DNNKernels/test/test_hunyuan3d_ops.jl` pins them.

**Not done:**

  * **artifacts.** There is no `gen/` fallback by design
    (`DNNKernels/src/assets.jl`), so until the four exports are bound with
    `tools/make_artifacts.jl` this package has nothing to load and
    [`assetdir`](@ref) says so. `hunyuan3d(; root = ...)` takes an explicit path
    meanwhile. The denoiser's 6.1 GB is the awkward one.
  * **the workload.** `@setup_workload` precompiles nothing, because the call it
    would have to drive needs an artifact. Until then every session pays first-use
    compilation.
  * **background removal.** [`imagetomesh`](@ref) needs a real alpha channel;
    upstream reaches for `rembg` when there is none, and there is no equivalent
    here. An opaque image is one whose bounding box is the whole frame, which is
    not what the model was trained on.
  * **capacity-based MoE routing**, below — the dense form is 4x the routed FLOPs
    on 6 of the 21 blocks, and is most of what the denoiser's 2 s per step is.

## Mesh clean-up

[`imagetomesh`](@ref) returns raw marching-cubes output, as upstream's pipeline
does; the three filters run after it and are separate calls.
[`removedegenerate`](@ref), [`removefloaters`](@ref) and [`decimate`](@ref) are
`postprocess.jl` and `decimate.jl`, ported from `postprocessors.py`'s `pymeshlab`
filters — with the filters' **semantics determined by running MeshLab**, since
three things about `nbfaceratio` are not settled by its name and guessing gets at
least two of them wrong. See `postprocess.jl`.

On `assets/demo.png`, 692 956 faces → 40 000, in **2.2 s**, staying watertight
and keeping the genus (Euler characteristic −4 before and after). Measured
against `pymeshlab` running upstream's exact flags on the same mesh, with
`pymeshlab`'s own Hausdorff filter judging both:

| | faces | mean dist | RMS | max | time |
|---|---|---|---|---|---|
| `pymeshlab` | 39 996 | 9.1e-5 | 1.22e-4 | 6.06e-4 | 5.99 s |
| this port | 40 000 | **2.0e-5** | 4.0e-5 | 6.35e-4 | **2.2 s** |

Closer to the original surface on average, the same worst case, and faster. It is
*not* bit-identical and cannot be: a collapse's cost depends on quadric weighting
and on the order a heap pops ties in, neither of which is part of the algorithm.

The floater filter removes **nothing** on this mesh — it is a single connected
component — so it is a guard, not a fix. The degenerate filter removes four
faces, all around one point where three grid edges crossed the level at the same
grid vertex; welding before dropping them is what keeps the mesh closed, and
skipping the weld tears six edges.

**Marching cubes is built here rather than taken from a package**, from face
contours instead of a 256-entry case table — see `mesh.jl`. That keeps the
package's dependencies to the runtime, and it is the version that can move to the
GPU later. `Meshing.jl` was the alternative and would have been a different
algorithm from upstream's anyway (Lorensen against `skimage`'s Lewiner).

**The mixture of experts is dense, and that is deliberate.** Upstream's
`moe_infer` cannot be exported: it calls `.cpu().numpy()` on the expert
histogram, loops over experts in Python with data-dependent bounds, `continue`s
on empty ones, and gathers a dynamically-sized slice per expert. The exporter
replaces it with an all-experts sum weighted by the scattered top-k mask — the
same function with static shapes, measured **bit-identical** on all six blocks
with real weights at fp16. It costs 4x the routed FLOPs on 6 of the 21 blocks.
Capacity-based routing is the optimisation once the whole pipeline is
numerically right; it is not a correctness fix and must not be done first.

Upstream: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
License: **TENCENT HUNYUAN NON-COMMERCIAL** — like `MatAnyoneRunner`, this
cannot ship in a commercial build. The weights are gated: accept the licence,
then `huggingface-cli download tencent/Hunyuan3D-2.1`.

See `models-to-port.md` for the state of this one, and
`tools/export_hunyuan3d.py` for the export that feeds it.
"""
module Hunyuan3DRunner

using Lava, DNNKernels, KernelAbstractions
import Mantle
using Mantle: @setup_workload, @compile_workload
using DNNKernels: loadgraph, readsafetensors, toback, Model, call
using DNNKernels: linspace, flowschedule, NoShift, eulerstep, cfg
using LazyArtifacts
using Artifacts: artifact_hash, artifact_exists

# `@artifact_str` finds this itself; `ready` needs the path explicitly because it
# asks whether an artifact is present WITHOUT resolving it.
const ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

export hunyuan3dgraph, hunyuan3dweights, conddir, ditdir, vaedir, geodir
export hunyuan3d, Hunyuan3D, imagetomesh
export prepareimage, normalizeimage, encodeimage, denoise, shapelatents, occupancy
export latents2mesh, marchingcubes
export removefloaters, removedegenerate, decimate, weld, writeobj

const KA = KernelAbstractions
# Through DNNKernels rather than a direct dependency: this package already has
# it, and `KernelInterface` is not in its Project.
const KI = DNNKernels.KI

"""
    KERNELS_VERSION

`DNNKernels.KERNELS_VERSION`, shared with every other model on this runtime so a
kernel frozen by one is a hit for the rest. Bump it there, not here.
"""
const KERNELS_VERSION = DNNKernels.KERNELS_VERSION

"""
    LATENTS, LATENT_CHANNELS, COND_TOKENS, COND_DIM

The shapes the exported graph is fixed at, from the checkpoint's own
`config.yaml` rather than from the architecture's defaults — the two disagree, so
reading the class signature gives the wrong answer.

`COND_TOKENS` is 1370 because DINOv2-large at 518 pixels with a patch size of 14
gives 37x37 patches plus the class token. The unconditional half of the
classifier-free-guidance pair is not a learned null embedding: upstream's
`unconditional_embedding` returns exact zeros, so the image encoder runs once per
generation, not twice.
"""
const LATENTS = 4096
const LATENT_CHANNELS = 64
const COND_TOKENS = 1370
const COND_DIM = 1024

"""
    conddir(), ditdir(), vaedir(), geodir() -> String

Where each part's graph export lives. The conditioner, VAE and geometry decoder
also keep their weights beside the graph; the denoiser's weights are the four
artifacts in [`DIT_SHARDS`](@ref).

Four artifacts rather than one tree, because the parts carry different weights
and run at wildly different rates: the conditioner once per generation, the
denoiser fifty times, the VAE once and the geometry decoder some seven thousand
times over the grid chunks. A caller that only wants the geometry decoder
fetches 24 MB rather than 6.8 GB. The denoiser is also 5.7 GiB by itself, past
what a single GitHub release asset can hold, so it has to be published
somewhere without that cap whatever the layout here.

This replaced a single `assetdir()` that threw: there was no artifact bound at
all, and the whole export lived only on the machine that produced it."""
conddir() = @artifact_str("hunyuan3d-cond")
ditdir()  = @artifact_str("hunyuan3d")
vaedir()  = @artifact_str("hunyuan3d-vae")
geodir()  = @artifact_str("hunyuan3d-geo")

"""
    hunyuan3dgraph(; dir = ditdir()) -> Graph

The exported ATen graph. Throws with the path it looked in rather than returning
`nothing` for the caller to trip over later.
"""
function hunyuan3dgraph(; dir::AbstractString = ditdir())
    p = joinpath(dir, "hunyuan3d_dit.json")
    isfile(p) || throw(ArgumentError(
        "Hunyuan3D-2.1 shape graph not found at $p. Generate it with " *
        "`uv run tools/export_hunyuan3d.py`."))
    return loadgraph(p)
end

"""
    DIT_SHARDS

The denoiser's weights, as four artifacts. 5.7 GiB of fp16 does not fit in one:
a GitHub release asset caps at 2 GiB, so the tensor file is split byte-for-byte
by `tools/shard_safetensors.jl` into pieces that do, and each piece is bound
separately. The split is by size and follows file order; which tensor lands in
which shard carries no meaning and may change on a re-export.
"""
const DIT_SHARDS = (("hunyuan3d-dit-w1", "weights-1of4.safetensors"),
                    ("hunyuan3d-dit-w2", "weights-2of4.safetensors"),
                    ("hunyuan3d-dit-w3", "weights-3of4.safetensors"),
                    ("hunyuan3d-dit-w4", "weights-4of4.safetensors"))

"""
    hunyuan3dweights() -> Dict

The exported state dict, keyed the way the graph's `:weight` buffers name it,
merged from the four shard artifacts.

Merging here is the whole reason `Model` takes loaded weights rather than a
path: sharding is this package's business, and `DNNKernels` never learns that a
model's weights arrived in more than one file. The merge is a plain `merge` of
four `Dict`s — the shards partition the tensors, so no key appears twice — and
peak memory is what one file would have cost, because `readsafetensors`
materialises every tensor either way.
"""
function hunyuan3dweights()
    dirs = (@artifact_str("hunyuan3d-dit-w1"), @artifact_str("hunyuan3d-dit-w2"),
            @artifact_str("hunyuan3d-dit-w3"), @artifact_str("hunyuan3d-dit-w4"))
    return merge((readsafetensors(joinpath(d, f))
                  for (d, (_, f)) in zip(dirs, DIT_SHARDS))...)
end

"""
    ready() -> Bool

Whether all four graph parts and all four denoiser weight shards are already in
the artifact store.

Deliberately does NOT go through `@artifact_str`: that downloads, and this is
what `@setup_workload` and the tests branch on — a guard that fetches 6.8 GB to
answer "do we have it" would be worse than no guard.
"""
ready() = all(("hunyuan3d", "hunyuan3d-cond", "hunyuan3d-vae", "hunyuan3d-geo",
               first.(DIT_SHARDS)...)) do n
    h = artifact_hash(n, ARTIFACTS_TOML)
    h !== nothing && artifact_exists(h)
end

include("preprocess.jl")
include("pipeline.jl")
include("mesh.jl")
# `decimate.jl` after `postprocess.jl`: it reaches `compact` and `MAX_FACES`.
include("postprocess.jl")
include("decimate.jl")

"""
    imagetomesh(m, rgba; steps, guidance, octree, box_v, level, latents, progress)
        -> (vertices, faces)

One image to one mesh — `Hunyuan3DDiTFlowMatchingPipeline.__call__` end to end.

`rgba` is `(W, H, 4)` `UInt8` with a real alpha channel: the framing in
[`recenter`](@ref) is driven by it, and an opaque image is one whose bounding box
is the whole frame, which is not what the model was trained on. Upstream reaches
for `rembg` when the alpha is missing; there is no equivalent here, so the caller
supplies a cut-out.

Returns `(3, V)` vertices in the sampling box and `(3, F)` one-based triangles.

`latents` defaults to fresh noise, so two calls differ. Passing the same starting
noise is what makes a run reproducible — and is how the suite compares against
the dumped reference, since matching PyTorch's RNG is not something this port
attempts.

The time is almost entirely in two places: `steps` denoiser evaluations at batch
2, and `(octree + 1)^3 / 8000` geometry-decoder calls. At the defaults that is 50
and 7134.
"""
function imagetomesh(m::Hunyuan3D, rgba::AbstractArray{UInt8,3}; steps::Integer = 50,
                     guidance::Real = 5.0, octree::Integer = 384, box_v::Real = 1.01,
                     level::Real = 0.0, latents = randomlatents(m), progress = nothing)
    img, _ = prepareimage(rgba)
    cond = encodeimage(m, normalizeimage(img))
    x = denoise(m, cond, latents; steps, guidance, progress)
    return latents2mesh(m, shapelatents(m, x); box_v, octree, level, progress)
end

"""
    writeobj(path, vertices, faces)

The mesh as a Wavefront OBJ, which is the least ceremonious thing that every
viewer opens. Deliberately not a `GeometryBasics.Mesh`: this package's
dependencies are the runtime and nothing else, and a caller that wants one can
build it from these two arrays in a line.
"""
function writeobj(path::AbstractString, verts::AbstractArray{<:Real,2},
                  faces::AbstractArray{<:Integer,2})
    open(path, "w") do io
        println(io, "# Hunyuan3D-2.1 shape, $(size(verts, 2)) vertices, $(size(faces, 2)) faces")
        for i in 1:size(verts, 2)
            println(io, "v ", verts[1, i], " ", verts[2, i], " ", verts[3, i])
        end
        for j in 1:size(faces, 2)
            println(io, "f ", faces[1, j], " ", faces[2, j], " ", faces[3, j])
        end
    end
    return path
end

function __init__()
    # Read the entries the workload froze. Recording stays off: a session that
    # hits a kernel the workload missed should compile it and carry on, not
    # quietly rewrite the frozen set under a version it was not built for.
    Mantle.use_frozen_kernels(KERNELS_VERSION)
    return nothing
end

# ---------------------------------------------------------------- the workload
#
# Nothing to precompile yet: the workload has to drive the call the caller makes,
# and the call the caller makes — an image to a mesh — does not exist. Writing
# one against the denoiser alone would freeze a path nobody takes, which is the
# mistake SAM2Runner made and paid 45 s of first-click latency for.
#
# When it is written, the measurement that matters is
# `Mantle.no_pipeline_compilation` reporting 0 refusals in a *fresh* process, NOT
# `frozen_stats().misses == 0` — that cannot distinguish the frozen cache working
# from the driver's own shader cache having served everything. Pair it with a
# negative control whose kernel body is novel per run.
@setup_workload begin
    @info "Hunyuan3DRunner: no artifact bound — nothing precompiled"
end

end # module
