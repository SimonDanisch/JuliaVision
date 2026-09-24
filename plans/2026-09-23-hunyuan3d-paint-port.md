# Porting Hunyuan3D-2.1's texture branch

The shape branch ships: `imagetomesh` runs end to end and the sampler is
bit-exact against upstream. The mesh comes out **bare**, because
`hunyuan3d-paintpbr-v2-1` is a different model and was never started.

`tools/project_texture.jl` is the stopgap — it projects the generating photo onto
the mesh and invents the rest. It is not a texture pipeline and this document is
the plan to replace it.

Everything below marked **[src]** was read out of upstream on 2026-09-23;
everything marked **[open]** is a question with a named way to answer it, not a
guess. That split is the point of the document: the shape port's schedule slipped
where it assumed, and held where it measured.

## What the pipeline is  [src]

`hy3dpaint/textureGenPipeline.py`, in order:

1. `mesh_uv_wrap` — **xatlas** `parametrize`, remapping vertices and faces.
2. `bake_view_selection` — from `azims = [0, 90, 180, 270, 0, 180]` and
   `elevs = [0, 0, 0, 0, 90, -90]` plus 30° increments at ±20° elevation.
   Primary views weight 1.0, the rest 0.01-0.5.
3. `render_normal_multiview` / `render_position_multiview` at **2048²**.
   Orthographic by default, scale ±1.2 x 0.5, camera distance 1.45.
4. The **multiview PBR UNet**: `diffusers.UNet2DConditionModel` with
   `in_channels` 4 -> **12**, every `BasicTransformerBlock` replaced by
   `Basic2p5DTransformerBlock`, which adds four attentions:
   * **MDA**, material-dimension: separate projection weights per PBR material
     (albedo, metallic-roughness)
   * **RA**, reference attention on the input image's features
   * **MA**, multiview attention with **RoPE over voxel-quantised positions**
   * **DINO attention**, cross-attention on `facebook/dinov2-giant` patch features
   Conditioning is the style image + the normal/position maps + **fixed text
   embeddings** for the string "high quality".
5. `imageSuperNet` — RealESRGAN_x4plus to **4096²**, on albedo *and* MR.
6. `bake_from_multiview` -> `back_project` per view -> `fast_bake_texture`:
   cosine weighting `cos_map = weight * cos_map^exp` with `exp = 6`, visibility
   cut at `cos(75°) = 0.259`, boundary pixels marked unreliable by Canny on
   normalised depth plus a dilation of `(scale/512) * max_resolution`. Three
   back-projection methods: `linear`, `mip-map` (min res 128) and `back_sample`
   (per-texel, depth threshold 3e-3).
7. `texture_inpaint` — `meshVerticeInpaint` over mesh connectivity, then
   **OpenCV Navier-Stokes** (`cv2.INPAINT_NS`) for what is left.

Upstream wants **21 GB of VRAM** for texture generation alone. **NON-COMMERCIAL**
licence, like the shape branch.

## What that costs us, by piece

| piece | kind | state here |
|---|---|---|
| UV unwrap | the only external dependency | **nothing in the General registry** unwraps a mesh: no xatlas, no LSCM, nothing |
| view selection | host code | trivial |
| normal/position/alpha/uvpos render | our own rasteriser | Mantle already renders: `render!`/`draw!` into an `OffscreenTarget` whose colour format is a Julia eltype, so `RGBA{Float32}` gives a float position buffer, and the framebuffer carries a depth attachment for visibility. GLMakie, Raycore and Hikari are the alternatives. **This is work, not a missing capability.** |
| multiview PBR UNet, 2B | model export | the work |
| DINOv2-giant | model export | **we already ship DINOv2-large** for the shape conditioner: same architecture, bigger weights |
| SD VAE encode/decode | model export | routine, but see the group-norm question |
| RealESRGAN x4plus | model export | small, convolution-only |
| back-projection + bake | compute | ours to write |
| inpaint | compute | `cv2.INPAINT_NS` has no equivalent here, but it is a small algorithm and it is upstream's FALLBACK — `meshVerticeInpaint` over mesh connectivity runs first |
| text encoder | — | **none needed**, the embeddings are fixed |

## The order, and why it is this order

The shape branch's lesson: the graphs were the easy half. `dump_hunyuan3d_pipeline.py`
exists because *the glue* is where a diffusion port goes wrong — the schedule,
the guidance, the batch order, the chunking. Texture adds a second class of glue
that the shape branch never had, a **renderer in the loop**, and its conventions
(camera, depth, cosine cut, boundary dilation) decide whether a correct model
produces a correct texture. So the non-model pieces come first, and the model
lands into a harness that is already verified.

### Phase 0 — inputs. Blocked on a human.

The weights are gated and **nothing is on this box**: no HF cache entry, no
`gen/hunyuan*`, no upstream checkout. Accept the licence, then

    huggingface-cli download tencent/Hunyuan3D-2.1
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 dev/Hunyuan3D-2.1

plus `facebook/dinov2-giant` and `RealESRGAN_x4plus.pth`.

**Gate: upstream's own `demo.py` textures our teapot on this machine.** Every
comparison below is against that output, so if it will not run here the plan
stops at phase 0 rather than proceeding on a reference nobody has seen.

### Phase 1 — the parity spine

`tools/dump_hunyuan3d_paint.py`, transcribed from `textureGenPipeline.py` line
for line the way the shape one was transcribed from `__call__` — upstream's own
objects, `_save` calls interleaved, anything that reads like a simplification is
a bug. It dumps: selected views and their weights, every normal and position
map, the UNet's input latents and per-step output, the decoded albedo and MR
views, the SR results, the per-view back-projected textures with their cosine
maps, the merged texture, the unfilled mask, and the final texture.

Nothing else in this plan can be checked without it.

### Phase 2 — the pieces that are not models

**2a. UV unwrap.** [open] Wrap xatlas (MIT, self-contained C++) as a jll, or
write charts + LSCM + packing ourselves. Recommendation: wrap it. A
parameterisation is a research project and a wrong one is invisible until the
bake smears.
*Gate:* same mesh in; compare chart count and per-triangle UV area distortion
against upstream's `mesh_uv_wrap` output.

**2b. Rasteriser.** `render_normal`, `render_position`, `render_alpha`,
`render_uvpos` at 2048², orthographic, upstream's exact camera.
*Gate:* buffer comparison against phase 1's dumps — **alpha masks identical**,
normals and positions within fp16 tolerance, at every selected view. A
half-pixel offset here silently rotates the whole texture.

**2c. Bake and inpaint.** `back_project` (start with `linear`, then
`back_sample` with its 3e-3 depth threshold), `fast_bake_texture`, the cosine
cut, the boundary dilation, `meshVerticeInpaint`.
*Gate:* feed phase 1's **upstream-generated views** into our bake and compare
textures texel by texel. This isolates the bake from the model completely.
[open] `cv2.INPAINT_NS` for the residual holes: port Navier-Stokes inpainting,
or accept a different fill and quantify the difference in the render.

### Phase 3 — the models, cheapest first

**3a. DINOv2-giant.** Same architecture as the ported large. Export, verify
against the dump's patch features. Cheap, and it de-risks the conditioning.

**3b. SD VAE.** Export encode and decode.
[open] Does `run_decompositions()` decompose `native_group_norm`? **We have no
group-norm op** — the 61 ATen ops `emit.jl` dispatches on do not include it, and
Qwen-Image's VAE dodged the question by using an RMS norm instead. Answer it with
a graph-only export and the op histogram the exporter already writes; if it
survives undecomposed, that is one new op with its own test.

**3c. RealESRGAN x4plus.** RRDBNet, convolutions only. Export and verify.

**3d. The 2B multiview UNet.** Graph-only export first, then
`op_histogram.json` **diffed against the 61 ops we have** — that diff *is* the
work list, the same way the shape branch's `_fused_rms_norm`, `topk` and
`scatter.src` were found. Then implement, then export weights, then verify the
graph against the exporter's reference.
The four attention processors are the substance and each gets its own fixture in
`DNNKernels/test/`: MDA's per-material projections, RA against the reference
features, MA's RoPE over voxel-quantised positions, and the DINO cross-attention.
Verifying only the whole model would leave a wrong processor hiding inside a
plausible image.

### Phase 4 — the loop

The 12-channel input packing, the per-material batching, the scheduler and
guidance, against the dump **per step**. The shape branch's sampler came out
bit-exact; expect this one to need the same four rounding facts to be found.

### Phase 5 — shipping

Runner (extend `Hunyuan3DRunner` or add `Hunyuan3DPaintRunner`), a
`@compile_workload` that drives the real entry point so the first call compiles
nothing, artifacts via `tools/make_artifacts.jl`, and the demo becomes image ->
mesh -> texture.

## Risks, named

* **The rasteriser is the hidden half.** Not because we lack one — Mantle draws
  into float targets today — but because `custom_rasterizer` is upstream's CUDA
  extension and everything downstream inherits ITS conventions: which pixel
  centre, which depth epsilon, how the boundary is dilated. The port is cheap and
  matching it exactly is not.
* **Memory.** 21 GB upstream. Our pattern — load, run, release per stage, weights
  streamed from GTT — is what makes the 27B Bonsai and the 7.26 GB Qwen denoiser
  work here, and the same applies, but the UNet plus DINOv2-giant plus the VAE
  have to be sequenced deliberately.
* **Exact parity may not be reachable at the inpaint step**, and that is the one
  place to accept a measured difference rather than chase it.
* **Scale.** Four model exports against the shape branch's four, plus a
  rasteriser we write on our own stack and one external dependency — xatlas, or
  whatever replaces it. Treat the shape branch as the yardstick and add the
  renderer on top.

## The one dependency, and how to not have it

xatlas is the only piece of this that is not already in the tree in some form.
Three ways out, in order of preference:

1. **Wrap it.** MIT, self-contained C++, a `parametrize` call. A jll through
   BinaryBuilder and it is done, at the cost of a binary dependency in a tree
   that currently has none of its own.
2. **Write it.** Chart segmentation, LSCM, packing. This is a research project
   and a subtly wrong parameterisation is invisible until the bake smears, which
   is the worst failure mode available.
3. **Do not need it.** The atlas exists so the texture can live in a 2D image.
   A mesh of 416k vertices could carry colour per VERTEX instead — which is what
   `project_texture.jl` does — and skip UV entirely. That trades the 4096²
   texture for whatever the mesh's vertex density resolves, and it means the
   result is no longer an OBJ+PNG anyone else can open. Worth pricing before
   assuming the atlas is required: subdividing to match a 4096² texture's detail
   is the comparison to make.
