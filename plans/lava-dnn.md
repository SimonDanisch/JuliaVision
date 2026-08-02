# MatAnyone2 on Lava.jl

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


Run MatAnyone2 inference in the same execution graph as graphics and post-processing
passes. No Python at runtime.

## Target models

MatAnyone2 is the first, not the only one. The roadmap (2026-07-28):

| area | model | status |
|---|---|---|
| matting | MatAnyone2 | running, 102.3 steps/s, wired into the editor's matte tool |
| segmentation / tracking | **SAM3** | blocked: `facebook/sam3` is a gated HF repo, needs Simon's token + approval |
| upscaling / restoration | **BasicVSR++** | running on CPU + GPU, 1.8e-5; wired into the editor's Restore effect and tool |
| video generation / editing | **Wan 2.x** | all three graphs run (DiT 4.8e-5 with the real checkpoint, VAE 7.3e-6, umT5 4.5e-7); sampler + end-to-end generation on GPU; wired into the editor's **Generate** tool (`src/generate.jl` + `examples/wan.jl`, 23 assertions green incl. a real window) |

All are to be integrated into the video editor the way MatAnyone2 drives the
matte tool, and **each must ship as a fully precompiled Julia package that is
instant to load and apply** — which is what the per-model package layout below
is for.

### Wan 2.2 (TI2V-5B): VAE decoder RUNS on DNNKernels; transformer block exports

**VAE decoder: max 7.27e-6 / mean 7.5e-7 against PyTorch**, all 1326 ops, after
adding 3-D convolution, `linalg_vector_norm`, `index.Tensor`, `mul.Scalar` and a
no-op `_assert_tensor_metadata` (DNNKernels `96a6161`).

**Diffusion transformer.** Forward runs once `attention`/`flash_attention` are
patched to `scaled_dot_product_attention` — Wan's dispatch reaches flash-attn
even when it is absent, and flash-attn asserts CUDA. The patch must be applied to
`attention.py`'s module *before* `model.py` imports the names from it.

Exporting the **whole** model fails with `GuardOnDataDependentSymNode`
("Could not guard on data-dependent expression 22*u4 < 2"): the forward pads a
list of variable-length latents up to `seq_len`, which export cannot trace. The
repeated `WanAttentionBlock` — 163.7M parameters, 30 of them, i.e. essentially
the whole model — has no such padding and **exports cleanly: 31 ops, 232 nodes**.

**The WHOLE MODEL runs on DNNKernels**: 209 ops, relative error 5.1e-5 — patch
embedding, sinusoidal time embedding, text embedding, the attention blocks with
rotary position encoding, head and unpatchify. `tools/wan_static.py` is what made
it exportable; `tools/export_wandit.py --full` writes the graph.

Four things had to be right, and each failed in a way that pointed elsewhere:

* **`rope_apply` calls `grid_sizes[i].tolist()`.** A tensor built inside
  `forward` is traced, so `.tolist()` yields unbacked symints and export dies on
  `GuardOnDataDependentSymNode`. Built once in `__init__` it is a constant. This
  is the *real* blocker — rewriting the list padding first (the obvious suspect)
  changed nothing.
* **`m.freqs` and `_grid_sizes` are plain attributes**, not registered buffers,
  so they are absent from `named_buffers` yet lifted to graph inputs. They must
  be saved explicitly.
* **`freqs` is complex128.** `view_as_real` gives Float64, and reinterpreting
  that to `ComplexF32` is the same 8 bytes — the pair axis silently survives and
  surfaces 40 ops later as a `cat` BoundsError. Use `Complex{eltype(v)}`.
* **`WanModel` zero-inits its output head**, as diffusion models do, so a
  randomly-initialised model emits exactly zeros and the first comparison passed
  with `max|Δ| = 0.0` — vacuously. The exporter now gives the head real weights.

**The block RUNS on DNNKernels**: 90 ops, max 7.87e-4 / mean 4.65e-5 against
PyTorch on an output whose peak is 7.31 — i.e. **1.1e-4 relative**.

That looked an order worse than MatAnyone and BasicVSR++ (both ~1e-5), so it was
traced per node: **no jump anywhere, worst per-node absmax deviation 5.6e-5.**
There is no bug. Wan's layers are 3072 and 14336 wide, so one dot product
accumulates about sqrt(14336)*eps ~ 1.4e-5 relative in fp32 and the block chains
several; the other two models sit at 1e-5 because their layers are 256-1024 wide.
Judge this model against its width, not against the others.

(The first "7e-4 relative" reading compared max *absolute* error against the
output's *rms* — two different scales. Against the peak it is 1.1e-4.)

Op gap was two things: `view_as_complex`/`view_as_real` (the RoPE rotary
embedding) and `gelu`, both added. `erf` turned out to be referenced by the
elementwise table and never defined — no earlier model contained one — so
`erf.default` was latently broken too.

Two export traps, both of which fail confusingly: `grid`/`seq_lens` must be plain
attributes, not `register_buffer`, or export traces them and the block's grid
unpacking reads as data-dependent control flow; and the flash-attn replacement
must be installed on `attention.py`'s module *before* `model.py` binds the names
from it.

So the path for the transformer is: implement complex views, run the block
against PyTorch, then replace the model's dynamic padding with a static wrapper
(the shapes are fixed in practice) so the full graph exports.

### Wan 2.2: the whole pipeline, on the GPU

Encoder -> sampler -> VAE decode runs end to end on `LavaBackend`.
`tools/wan_generate.jl` drives it; `dev/JuliaVision/DNNKernels/src/wan.jl` holds
the sampler.

| stage | time (warm, RTX 4000 Ada) |
|---|---|
| one DiT call (2-layer export, 209 ops) | **2.06 s** |
| VAE decode, 9 frames at 256x256 (1326 ops) | **30.6 s** |
| 4-step generation (8 DiT calls + decode) | **47.9 s** |

The decode dominates by an order of magnitude, so that is where to look first.

**VRAM: 11.3 GB of the card's 20**, and getting there took a third fix. The
accounting did not add up — weights 4.3 GB (DiT 1.6, VAE decoder 2.7) plus both
planned slabs 1.2 GB is 5.5 GB, yet the run peaked at 14.8 GB and then OOM'd once
the desktop took its 3.3 GB. The missing 9 GB was ops allocating *outside* the
plan: `constant_pad_nd` and `copy.default` still called `similar`, and the VAE
decoder pads before each of its 116 3-D convolutions at 256x256x9. Both now take
`dest(ctx, ...)` — the same conversion `slice_scatter` had already had, and the
`dest` docstring names as in progress. Worth checking the remaining `similar(`
call sites when a graph looks memory-hungry; the zero-length ones are placeholders
for tuple outputs and are free.

**Accuracy is unchanged from the CPU path** — DiT 5.1e-5 relative against
PyTorch, BasicVSR++ 1.8e-5, MatAnyone suite 61/61. The GPU is not a second
implementation to keep in step; it is the same graph on a different backend.

Three things were broken and none was in a model:

* **`LavaArray` had no aliasing answer.** `dest .= src` where both sides are
  views makes Base ask whether the parents overlap, and for `DenseArray` parents
  it answers with `pointer(A) == pointer(B)`. `LavaArray <: AbstractGPUArray <:
  DenseArray` and has no host pointer, so *every* `view(a) .= view(b)` between
  device arrays threw `conversion to pointer not defined`, whatever the shapes —
  which is what `cat` compiles to. Identity for a LavaArray is its backing buffer
  plus byte offset, which is what a pointer would encode anyway;
  `Base._parentsmatch` now says so and the rest of Base's machinery (index-range
  comparison, the temporary for a genuine overlap) works unchanged. Do not answer
  this with `Base.dataids`: DNNKernels plans every transient into one slab, so a
  buffer-level `dataids` would make every slab-to-slab broadcast take a
  defensive copy.
* **Weights and sampler state must live on the backend.** `wanpipeline` uploads
  once in the constructor and the sampler's own `cfg`/`eulerstep` run there too —
  a host latent against a device velocity fails at kernel compilation rather than
  falling back. `toback` is now a no-op for an array already resident, judged by
  the *kind* of backend: KA's `get_backend` rebuilds the descriptor, so
  `get_backend(a) == backend` is false even for an array allocated on that exact
  device.

**What the output is not.** The mechanics are right; the pictures are not
meaningful yet, and for reasons that have nothing to do with the runtime:

1. **2 of 30 blocks.** fp32 at 30 layers is ~20 GB, which does not fit alongside
   the 2.8 GB VAE. Real output needs an fp16 export (~9.8 GB) — note that
   `--precision autocast` is *not* that: autocast keeps parameters in fp32 and
   inserts `_to_copy`, so it buys compute precision, not weight memory.
2. **The context is noise**, not a umT5 embedding of a prompt. The exported
   encoder is the small proxy built to prove op coverage; umT5-xxl is 11 GB fp32
   and will not be resident at the same time as the transformer, so the encode
   has to run and free first.
3. Four sampler steps.

And, until 2026-07-28, a fourth that mattered more than the other three and was
not written down: **the transformer's weights were random.** `build_full` built
`WanModel(**cfg)` and never touched the shipped shards, so "the DiT runs at
5.1e-5" was a statement about the arithmetic and nothing else. `load_trained`
now fills the blocks that are kept from `diffusion_pytorch_model-*.safetensors`
(69 tensors for the 2-block export), reading tensor by tensor through
safetensors' mmap so peak memory is the model, not the 18.6 GB checkpoint — which
is also what makes the fp16 30-layer export tractable later. Parity is unchanged
with real weights: **max|Δ| 1.35e-4 on a peak of 2.83, i.e. 4.8e-5 relative**.
Random weights are still available (`trained=False`) and the exporter *prints
which it used*, because that distinction is invisible in the graph and in the
error figures.

So the honest reading of `gen/wan_sample.mp4` is "the pipeline is plumbed and
each stage matches PyTorch", not "Wan generates video here".

### Wan 2.2 in the editor: the Generate tool

`dev/VideoEditor/src/generate.jl` + the `:generate` GUI tool, driven by
`dev/VideoEditor/examples/wan.jl` (`usewan!()`), same opt-in glue shape as
`matanyone.jl` and `basicvsrpp.jl`.

Generation does not fit the shape the other two models have. A matte and a
restoration *modify* a clip, so they live on the clip — a track plus an effect
the render path samples. A generator has no input clip at all: it produces
**media**. So `generateclip!` writes a file and hands it to `addsource!`, the
same door dropped files come through, and from that point the result is an
ordinary clip that trims, keyframes, stabilizes and exports with no special case
anywhere. Nothing was added to the effect stack, the graph or the project format.

Four decisions worth keeping:

* **The sequence decides the frame rate, not the model.** `addsource!` refuses a
  source whose rate does not match — the edit model is frame-exact — so the
  frames are encoded at the sequence's rate. A 16 fps model on a 24 fps timeline
  plays fast, which is visible, rather than being resampled behind the user.
* **Cached by (prompt, seed)** in the package scratch space, because a run takes
  minutes: the same request re-adds its clip instead of re-running, and the
  panel lists every generation with ⇄ to add it again and × to drop it.
* **A focused text box must swallow the keyboard.** `EditableText` consumes the
  keys it uses (priority 60) and lets the rest through, and the editor's
  shortcuts are global at priority 0 — so typing a prompt containing "s" splits
  the clip under the playhead. `tooltextbox!` consumes everything else at
  priority 50: below the editor that owns the text, above the editor that owns
  the timeline.
* **The "which model" line is an Observable, not a string.** The documented way
  to install one is `include("examples/wan.jl"); usewan!()` in a REPL that
  already has an editor open — the panel was built before that, so a plain
  string leaves it reading "no model installed" for the rest of the session.
  Caught by looking at the screenshot, not by a test.

`grep -r Wan dev/VideoEditor/src` being empty is the design, not a gap: `grep -r
MatAnyone` is empty there too. The editor's `src/` owns the *mechanism* — the
registry, the file, the panel — and names a model only in a docstring pointer;
the model itself lives in `examples/`, so VideoEditor never depends on DNNKernels.

`tools/test_generate.jl` runs the assertions on their own (12 headless + 11
driving a real GLMakie window), because one failing `@testset` in `runtests.jl`
aborts the whole file and a marginal tolerance elsewhere would hide them.

One bug worth remembering: the file was written to `<key>.mp4.part` and moved
into place, so a killed run could not leave a truncated file in the cache — but
**ffmpeg picks its muxer from the extension**, and `.part` is not one, so
`open_video_out` failed with "Could not allocate AVFormatContext … Invalid
argument". That reads like a bad frame array, not a bad filename. The temp name
is `<key>.part.mp4` now.

The panel says what the model can actually do. With no umT5 export installed the
label reads *"the prompt seeds the sample, it does not describe it"* — the prompt
picks the noise (so a generation is reproducible and addressable) but does not
condition the transformer. Passing `encdir` switches to real conditioning as soon
as an encoder export exists.

### The PyTorch baseline (2026-07-28)

`tools/wan_pytorch_baseline.py` runs Wan's own `generate.py` on the real 30-layer
TI2V-5B. This is the number DNNKernels has to be measured against; per-op parity
says nothing about throughput.

| run | sampling | total | fits? |
|---|---|---|---|
| 720P (1280x704), 5 frames, 50 steps | **59 s (1.19 s/step)** | — | **no** — decode OOMs |
| 640x352, 5 frames, 50 steps | **14 s (3.39 steps/s)** | 1 m 38 s incl. loading 20 GB | yes, all on GPU |

`gen/wan_small.mp4` is the first real generation on this machine, and it looks
like the prompt.

**720P does not fit, in either memory.** Three attempts, each ruling out one
theory: the first OOM'd wanting a contiguous 1.72 GiB with 2.05 GiB free, which
looked like fragmentation; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
duly cut reserved-but-unallocated from 1.30 GiB to 250 MiB and it died anyway,
deeper in, with 13.81 GiB genuinely allocated; decoding on the host was then
killed by the OOM killer at 24.8 GB RSS against 31 GB of system RAM. Wan's own
table wants 22.9 GB for this model at 720P on a 24 GB 4090, and this card is
20.4 GB with ~3.3 GB held by the desktop. The transformer is offloaded to CPU
before the decode (`textimage2video.py:396`), so that 14 GB is the decoder's own
activations — the sampling half was never the problem.

Both extents must be divisible by 32 (VAE stride 16 x patch 2). TI2V-5B ships
only the two 720P presets and `generate.py` asserts on them, so the wrapper
registers a custom size — a knob for fitting on a smaller card, not a free win,
since 720P is what the model was trained at.

**Two things the wrapper has to fix before any of this runs**, both documented in
it: `model.py` does `from .attention import flash_attention` and calls that
directly (lines 145, 175) rather than the `attention()` next to it that *does*
fall back to SDPA — so the fallback exists and is unreachable, and the run dies
2.5 minutes in at the first self-attention. And `wan/__init__.py` imports the
speech2video and animate paths, so `decord`, `librosa`, `peft` and `dashscope`
must be installed for a text-to-video run that uses none of them.

**Where DNNKernels stands against it.** PyTorch does one forward of the *30-layer*
model at 440 tokens in roughly 0.14 s (0.28 s/step, CFG being two forwards);
DNNKernels takes 2.06 s for one forward of a *2-layer* model at 192 tokens. Normalising
for layers and tokens that is a gap of order 500x — call it two to three orders of
magnitude, since attention is superlinear in tokens and the ratio is not exact.
The VAE side is the same story: 30.6 s for 256x256x9 here against a 640x352x5
decode that disappears into PyTorch's 14 s.

None of that is surprising — PyTorch is bf16 through cuDNN/cuBLAS, DNNKernels is
fp32 through kernels tuned against MatAnyone's 256-1024-wide convolutions, not a
3072/14336-wide transformer. But it is the first honest target, and it is a far
more useful one than the accuracy numbers, which were all green while throughput
was never measured.

### Where the throughput gap actually is (2026-07-28)

Measured on the Wan DiT forward, 2 layers, 192 tokens, fp32, RTX 4000 Ada.
The whole gap is one op, and most of it is one array wrapper.

    clean forward                    1980 ms
      addmm.default   26 calls       1962 ms   96.9%
      convolution      1 call          20 ms    1.0%
      everything else 200+ calls       40 ms    2.1%

`OPTIMES` inflates absolute numbers with a sync per op, but 26 large ops is not
640 small ones - instrumented total 2024 ms against a clean 1980 ms, so here it
can be read directly.

**The same 26 GEMMs, run standalone on contiguous operands, cost 315 ms.** The
missing 1.6 s is not the multiply. It is that `mul!` with a `Transpose` left
operand takes a different, far worse kernel in Lava:

    mul!(C, transpose(A), B)   392 ms     55 GFLOP/s
    mul!(C, A, B)               39 ms    563 GFLOP/s     10.2x

and every `addmm` weight in an exported graph arrives permuted (`matmul.jl`
says so in its own docstring - `astranspose` exists precisely to turn those into
a `Transpose` that BLAS will dispatch on, which fixes *correctness* and leaves
the speed on the floor).

The weights are constant, so the transpose can be paid once: 44.6 ms per weight
at pipeline construction, then every call takes the 39 ms path instead of the
392 ms one, **bit-exact** (max|Δ| = 0 against the transposed-operand result).

Three levers, in order of cost to implement:

1. **Materialise permuted constant operands once.** ~10x on 97% of the forward.
   1980 ms -> roughly 350 ms expected. Cheap, bit-exact, benefits every model
   whose exporter emits permuted weights, which is all of them.
2. **fp16 so the cooperative-matrix path fires.** `matmul_coopmat!` is already
   written and `Lava.coopmat_gemm_available()` returns **true** on this card -
   `mm_coopmat_applicable` just gates on `LavaArray{Float16,2}` operands and Wan
   is exported fp32, so all 26 matmuls take the scalar kernel. The fp16 export is
   also what is needed to fit 30 layers in VRAM: one change, two problems.
3. **The GEMM kernel itself.** Contiguous fp32 is 563 GFLOP/s against ~26.7
   TFLOPS peak, i.e. ~2%. These shapes are skinny (m = 192) so peak is not the
   target, but that is a lot of headroom. Only worth attacking after 1 and 2, and
   only with a number to beat.

The lesson for the other models: nobody would have found this from the accuracy
work, and `astranspose` looks like the fix for exactly this problem while being
only half of it. Measure the wrapper against a standalone call of the same shape.

### Wan 2.2 (TI2V-5B): loading notes

`Wan-AI/Wan2.2-TI2V-5B` is ungated, 34 GB across 23 files: a 20 GB diffusion
transformer, an 11 GB umT5 text encoder, and a 2.8 GB VAE.

The VAE is the tractable first piece and is independently useful — it is what
turns latents into video. Loading it standalone needs the same by-path trick as
BasicVSR++: `wan/modules/vae2_2.py` itself only wants torch and einops, but
importing it through the package drags in `decord`, `diffusers`, `transformers`
and more. Two config values are not the file's defaults and the errors do not
say so: **`z_dim=48`** (default 16, fails as a channel mismatch) and
**`temperal_downsample=[False, True, True]`** (default is three temporal
downsamples; the config's `vae_stride=(4,16,16)` means two, and the mismatch
shows up as missing `time_conv` keys).

With those: **704.7M parameters, encode (1,3,9,64,64) -> (1,48,3,4,4) -> decode
back**, and `torch.export` gives **32 distinct ops, 1825 nodes** for the decoder.

Op gap against `ops.jl`:

| op | count | work |
|---|---|---|
| **`convolution` with 3-D spatial dims** | 116 | **new kernel** — `Conv3d`; DNNKernels has 2D and 1D |
| **`linalg_vector_norm`** | 90 | **new** — the RMS norm; a reduction |
| `index.Tensor`, `any.dim` | 12 | small |
| `_assert_tensor_metadata` | 95 | no-op, drop in the converter |
| everything else | ~1500 | already implemented |

So the VAE is two kernels away, not a rewrite — the encoder/decoder are
convolutional, which is why this is far smaller than "video generation model"
suggests. The diffusion transformer is the separate, larger problem: it needs a
sampler loop outside the graph, and `wan/modules/attention.py` reaches for
flash-attn (absent here, `scaled_dot_product_attention` is the fallback).

### BasicVSR++: exports, and the op gap is two kernels

`tools/basicvsrpp.py` loads the upstream model **without mmcv**. Upstream is an
OpenMMLab package whose network is plain PyTorch but imports four mmcv symbols,
and whose `ModulatedDeformConv2d` is a custom CUDA op `torch.export` cannot trace.
Both problems have one answer: shim the mmcv names (`ConvModule`,
`constant_init`, `kaiming_init`, `_BatchNorm`) and swap the deformable conv for
`torchvision.ops.deform_conv2d`, which is the same modulated v2 operator, native,
and traceable. The upstream files are then loaded *unchanged* by path, so they do
not drift from the repo.

Verified: 7.32M parameters, state dict loads with no missing/unexpected keys,
`(1,5,3,64,64) -> (1,5,3,256,256)` — the 4x REDS4 model. SPyNet weights are
bundled in the same checkpoint (62 keys), so no second download.

`torch.export(...).run_decompositions()` gives **32 distinct ops, 3169 nodes**.
Against what `ops.jl` already implements:

| op | count | work |
|---|---|---|
| `leaky_relu` | 89 | trivial, one broadcast |
| **`grid_sampler_2d`** | 52 | **new kernel** — bilinear sampling; this is `flow_warp` |
| `flip`, `full_like`, `eq.Scalar`, `where.self`, `full`, `copy` | ~70 | trivial |
| `repeat`, `upsample_bilinear2d.vec` | 53 | exist; `.vec` overload needed |
| `avg_pool2d` | 20 | sibling of the existing adaptive pool |
| **`torchvision.deform_conv2d`** | 16 | **new kernel** — the real work |

Everything else (510 convolutions, the elementwise ops, cat/view/select/permute)
is already covered. So the port is: two kernels, a handful of one-line ops, and a
`driver.jl` equivalent — no new infrastructure.

Note the shape: 510 convolutions in one graph against MatAnyone's 134, and a
recurrent structure over frames. The conv path is where this model will live or
die, which makes the coopmat guard fix (`5b4a591`) worth more here than it was
for MatAnyone.

### What a second model actually costs — measured, not estimated

Counting MatAnyone-specific references per source file (`MemoryBank`,
`lastmskvalue`, `sensory`, the graph names):

| model-specific | agnostic |
|---|---|
| `driver.jl` (26) — `step!`, `initstate`, the 8 hardcoded graph names | `graph.jl`, `execute.jl`, `ops.jl`, `plan.jl`, `fuse.jl`, `dce.jl`, `foldbn.jl`, `foldrelu.jl`, `hoistcasts.jl`, `launch.jl`, `workspace.jl`, `safetensors.jl`, `verify.jl`, all of `kernels/` |
| `memory.jl` (8) — the `MemoryBank` | |

**14 of 16 sources are already generic.** So a second model is not a rewrite: it
needs its own `driver.jl` equivalent (the inference loop and whatever recurrent
state it carries) and nothing else on the Julia side, provided its ops are
already in `ops.jl`.

The Python side is the opposite and is the real work: **every** file in `tools/`
hardcodes MatAnyone (`common.py` 12 references, `graphs.py` 8, `patches.py` 7,
`enumerate.py` 6). Exporting a second model means turning those into a
model-config-driven pipeline first — that is the actual first task, not any
Julia work.

Per-model op gaps to expect: BasicVSR++ needs deformable convolution (new
kernel); SAM3 is attention-heavy and `sdpa`/`bmm`/`native_layer_norm` already
exist; Wan 2.x is a diffusion transformer with a sampler loop and a VAE, so it is
not one graph but many invocations plus scheduler logic living outside the
exported graph — a project of its own.

**The blocker for "instant" is measured and is not shader compilation.** Lava's
SPIR-V disk cache works (683 entries; a fresh session adds none). The ~99 s first
call is per-session Julia type inference and codegen over the graph's execution
paths. The generated package therefore needs a `PrecompileTools.@compile_workload`
running a real step, on the *Lava* backend so `LavaArray` specializations are
covered — i.e. a GPU at package-precompile time.

**The premise is already verified: the cost is ~90% shape-independent.** Warming
the matte propagator at 64x48 absorbed **97.9 s of the 99 s**, leaving 12.5 s on
the first real 240x136 clip; warming at the real resolution took that to 1.0 s.
So a workload built around *any* representative graph shape recovers the bulk of
it, and a second pass at the shapes the caller will actually use finishes the
job. That is what `warmmatte!(w, h)` does in the editor today, and it is the
shape `emit_model.py` should generate: one workload at the model's canonical
resolution, plus a runtime warm-up hook for other resolutions.

Which leaves exactly one decision, and it is a packaging-policy one: whether a
model package may *require* a GPU to precompile (clean, but breaks GPU-less CI
and installs), or whether the workload is guarded and falls back to the runtime
warm-up. Everything technical behind it is settled.

---

## Layout

```
tools/                      Python, dev-only, never shipped
  convert_weights.py
  enumerate.py
  dump_plan.py
  dump_refs.py
  emit_kernels.py           reads ALL model JSONs -> DNNKernels
  emit_model.py             reads one model JSON  -> model package

DNNKernels.jl/
  src/kernels/extern/       hand-written: conv, matmul, attention
  src/kernels/fused/        generated from the union of all model JSONs
  gen/kernel_manifest.json  which hashes exist, and from which models
  src/runtime.jl            arena, dispatch, recording, instantiate

MatAnyone2.jl/              generated package, no kernel sources
  gen/graphs/v*.json        committed, regenerable
  src/passes.jl             generated, references DNNKernels kernels by name
  src/driver.jl             generated
  spirv/                    blobs for this model's instantiations
  Artifacts.toml            weights, content-hashed
```

Kernel **sources** live in DNNKernels and are shared. Kernel **instantiations**
(concrete type params, and therefore SPIR-V) live in the model package, so
installing one model does not precompile another's.

---

## Two sources, separated

`torch.export` -> ATen graph, shapes, dtypes, dataflow. Stable API. **Correctness
depends only on this.**

Inductor debug dump -> fusion groups, buffer liveness, layouts. Internal, fragile.
**Advisory.** If it breaks, emit unfused; still correct.

---

## Python side

### `convert_weights.py`
`.pth` -> `weights.safetensors`. Fold norms into preceding convs, cast to working
dtype, permute to target layout (keep the permutation configurable).

### `enumerate.py`
Trace the inference path. Where a buffer grows or varies in length, **pad to fixed
capacity + validity count + mask in the consumer** - the memory bank especially.
Re-trace. Log recompiles and guards.

Closure check: second longer clip with different content, zero new recompiles.

Output: one `gen/graphs/v*.json` per variant. Expect 2-3.

Escalate if any guard depends on tensor *contents* rather than frame index or shape.

### `dump_plan.py`
Inductor debug output -> fusion groups, buffer liveness intervals, layouts.
Merge into the variant JSONs.

### `dump_refs.py`
Per-layer activations for a fixed 3-5 frame clip -> `refs.safetensors`.
Also record: seconds/frame and peak VRAM on both cards, from PyTorch.

---

## JSON schema

Shapes are **symbolic in H, W** - resolution is not a variant axis.

```json
{
  "guards": ["frame_idx % mem_every == 0"],
  "buffers": [
    {"id":"buf0","shape":[1,64,"H/2","W/2"],"dtype":"f16","kind":"transient",
     "live":[3,17]},
    {"id":"w_stem","dtype":"f16","kind":"weight","key":"enc.stem.weight"},
    {"id":"mem_k","shape":[1,64,"CAP"],"dtype":"f16","kind":"crossframe"}
  ],
  "ops": [
    {"id":"c0","aten":"convolution","in":["in0","w_stem"],"out":"buf0",
     "attrs":{"stride":[2,2],"padding":[3,3],"groups":1,"C_in":3,"C_out":64,"K":7},
     "extern":true},
    {"id":"a1","aten":"add","in":["buf0","w_bias"],"out":"buf1"},
    {"id":"r2","aten":"relu","in":["buf1"],"out":"buf2"}
  ],
  "fusion_groups": [["a1","r2"]],
  "barriers": []
}
```

Lifetime kinds: `weight` (persistent), `transient` (arena), `crossframe`
(excluded from arena, versioned), `external` (imported).

Liveness intervals and barrier positions are computed in `emit_julia.py`, not at
runtime. They are resolution-independent; only the size formulas evaluate per resolution.

---

## `emit_kernels.py`

Reads every model's `gen/graphs/*.json`, emits the deduped union into
`DNNKernels/src/kernels/fused/`, updates `kernel_manifest.json`.

Generated, never hand-curated. Additive: adding a model is a DNNKernels minor release.
Regenerating from the union of live models prunes dead kernels; model packages carry
a DNNKernels compat bound.

## `emit_model.py`

### Kernel bodies
One template per ATen op - a few lines of scalar code over an index. Fusion is
concatenating a group's bodies with SSA renaming and dropping the intermediate
store/load. No IR, no pass.

`"extern": true` emits a call to a DNNKernels kernel instead of a body.

### Specialization

**Type parameters** - everything from the architecture: channels, kernel size,
stride, padding, groups, dtype, head count. Julia folds these.

**Push constants** - H, W, and anything derived. Nothing else.

Resolution is never a type parameter. Changing it must not invoke the Julia compiler.

### Indexing
3D dispatch (`@index(Global, NTuple)`) by default - no divmod on runtime values.
Where linear indexing is needed, magic-number multiply-shift computed host-side and
passed as push constants. No integer division by a runtime value in a kernel body.

### Dedup
Key fused kernels on `(template chain, dtypes)` -> `fused_<hash>`, generic over type
params. Not shapes. The ~40 `conv+bias+relu` sites collapse to one kernel; a later
model hitting the same chain is a manifest hit, not a new kernel.

### Output
`passes.jl` (const pass lists per variant, referencing `DNNKernels.fused_<hash>`),
`driver.jl` (guards -> variant selection). No kernel sources in the model package.

Emit a shared trunk + suffix where `v1.ops[1:n] == v0.ops[1:n]`.

---

## DNNKernels.jl

Runtime: arena, dispatch, command recording, integration with the graphics graph.

Owns all kernel sources, shared across models:
- `kernels/extern/` - conv, matmul, attention. Hand-written, type-parameterized.
  Capability dispatch (scalar / cooperative-matrix) selected at instantiate from a
  device query.
- `kernels/fused/` - generated union. Tuning one benefits every model using it.

Holds no SPIR-V and no per-model instantiation list.

`instantiate(variants, spirv, weights, dev; resolution)`:
- evaluate size formulas, compute arena offsets from shipped intervals
- allocate arena + weight buffer
- create pipelines from precompiled SPIR-V (disk pipeline cache via Scratch.jl)
- write descriptors, record command buffers

One arena sized to the max across variants.

Resolution change re-runs: arena, descriptors, recording. No shader work.

---

## Model package

```julia
module MatAnyone2
using DNNKernels, KernelAbstractions, PrecompileTools

include("passes.jl"); include("driver.jl")     # kernels come from DNNKernels
const SPIRV = DNNKernels.load_spirv(@__DIR__)     # this model's instantiations only

setup(dev; resolution) = DNNKernels.instantiate(VARIANTS, SPIRV, weights(), dev; resolution)

@setup_workload begin
    @compile_workload begin
        DNNKernels.specialize_kernels(VARIANTS)   # Julia -> SPIR-V, no device needed
    end
end
end
```

SPIR-V generation is pure and happens at package precompile; blobs ship in the package.

---

## Driver

```julia
const CG = [compile(PASSES_V0, dev), compile(PASSES_V1, dev)]
select_variant(i) = (i % MEM_EVERY == 0) ? 2 : 1

for (i, f) in enumerate(frames)
    cg = CG[select_variant(i)]
    bind_external!(cg, :frame, f)
    submit!(cg, params)
end
```

Guards are functions of frame index or shape, known before submission -> ordinary
Julia conditionals. No conditional rendering, no indirect dispatch.

Cross-frame state versioning must be consistent across variants.

---

## Verification order

1. CPU backend, same KA source, layer-by-layer vs `refs.safetensors`
2. GPU, same comparison, both cards
3. Record achieved bandwidth/FLOPs vs peak per kernel, both cards
4. Peak arena vs the Inductor plan's figure
5. Full clip vs PyTorch output; seconds/frame and peak VRAM vs baseline

Layer-by-layer, not end-to-end. The first mismatching layer is the bug.

---

## Integration

Model passes and graphics passes in one graph. Frame in as imported external image,
alpha matte out as a texture consumed by the compositing pass.

Verify by inspecting the recorded command stream for transfers between inference
output and composite input. There should be none.

---

## Constraints

Optimize for low compile times. Sacrifice them only where measured.

Do not build: tensor IR, fusion pass, layout solver, buffer aliasing, autotuning,
quantization, ONNX import.

Build from the start: views (parent + offset + strides), lifetime kinds on every
buffer, opaque scratch declarations, params passed explicitly not captured.

Never guess an API - read the source, cite `file:line`.
