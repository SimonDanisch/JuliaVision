# QwenImageRunner

Qwen-Image 2.1 text-to-image on Lava, from the compact Comfy-Org checkpoints.

## Generating an image

```julia
using QwenImageRunner

img = generate("a red fox sitting in a snowy forest at sunrise, photorealistic")
size(img)   # (1024, 1024, 4): RGBA in [0, 1], the fourth channel a real alpha matte

wide = generate("a steampunk harbor at golden hour"; width = 1664, height = 928,
                steps = 40, seed = 7)
```

Or from the shell, which writes a PPM, or a PAM when the alpha is not opaque:

```sh
julia --project=. QwenImageRunner/examples/generate.jl \
    "a red fox sitting in a snowy forest at sunrise, photorealistic" fox.ppm
QWENIMAGE21_SIZE=1664x928 julia --project=. QwenImageRunner/examples/generate.jl \
    "a steampunk harbor at golden hour" harbor.ppm
```

The defaults are the reference pipeline's (`QwenImage21Pipeline`): 1024x1024, 40
steps and no classifier-free guidance, which the model is meant to be sampled
without.

- **Size:** any `width` and `height`, floored to multiples of 32 as the reference
  does, from 64x64 up to 16384 image tokens (2048x2048 square, or e.g. 2560x1600).
  The checkpoint is trained around 1024x1024 and is at its best near that area;
  1664x928 and 928x1664 are the reference's 16:9 sizes. `qwen_resolution` says
  what a size will run as, and refuses one the denoiser graph does not take
  before anything is built.
- **Prompt:** up to 1024 tokens, the reference's own limit. The text encoder
  graph was static at 64 tokens until 2026-09-27, which after the chat template
  left **42 tokens** of prompt.
- **Memory:** the encoder (6.9 GB) and the denoiser (7.26 GB) run one after the
  other and each is released before the next is built. Above about one megapixel
  the decoder's buffers need more at their peak than one Vulkan allocation holds
  (4 GB on RADV; 5.8 GB at 1664x928), and Mantle places them in several.

## Examples

Generated with the calls above, 40 steps, seed 42, on a Radeon 8060S through
Vulkan (2026-09-27). Warm, in a session that has generated once, a 1024x1024
image takes **221 s** and a 928x1664 one **302 s**, prompt to pixels; the first
call in a session also compiles, and took 350 s at 1024x1024.

<table>
<tr>
<td><img src="../media/qwenimage/fox.jpg" width="320"></td>
<td><img src="../media/qwenimage/apple.jpg" width="320"></td>
</tr>
<tr>
<td><em>a red fox sitting in a snowy forest at sunrise, photorealistic</em><br>1024x1024</td>
<td><em>A glossy red apple, isolated on a transparent background.</em><br>1024x1024, RGBA, shown over a checkerboard</td>
</tr>
<tr>
<td colspan="2"><img src="../media/qwenimage/harbor.jpg" width="640"></td>
</tr>
<tr>
<td colspan="2"><em>A sprawling steampunk harbor city at golden hour, seen from a hillside: brass
airships with patched canvas balloons moor at iron towers, cobblestone streets wind between
crooked timber houses with copper roofs gone green, a crowded fish market spills onto the
docks, gulls circle above tall ships whose sails glow orange in the low sun, steam vents from
chimneys and drifts across the water, and in the foreground a young mechanic in goggles and a
leather apron repairs a small clockwork bird on a workbench covered in gears, highly detailed,
cinematic lighting, shallow depth of field.</em><br>1664x928, a 129-token prompt, three times
what the encoder held before</td>
</tr>
<tr>
<td><img src="../media/qwenimage/poster.jpg" width="320"></td>
<td><em>A vintage travel poster for Lisbon with the word "LISBOA" in bold art deco letters at
the top, a yellow tram climbing a steep street between pastel houses, flat illustration
style</em><br>928x1664</td>
</tr>
</table>

## Transparent backgrounds, natively

**The image comes back as RGBA, and the alpha is real.** The decoder's fourth
channel is a soft matte: ask for a transparent background in the prompt and the
model cuts the object out itself. No segmentation step, no background removal.
This is one of the main reasons to use Qwen-Image 2.1.

```sh
julia --project=. QwenImageRunner/examples/generate.jl \
    "A glossy red apple, isolated on a transparent background." apple.ppm
# alpha is not opaque (min -1.0): wrote apple.pam, RGBA
```

`generate` returns `(width, height, 4)` already in [0, 1]. The lower-level
`decode!` and `generate!` return `(width, height, 4, batch)` in [-1, 1], and
`x / 2 + 0.5` maps all four channels to [0, 1]; the alpha is straight, not
premultiplied, exactly what diffusers' `postprocess` hands to PIL. **Keep the
fourth channel:** `img[:, :, 1:3, :]` or an RGB writer throws the matte away, and
the colour under a transparent pixel is arbitrary, often purple blotches.

This needs the VAE at **fp32**, which is what ships since 2026-09-26. The fp16
decoder before it overflowed fp16's 65504 inside the network: a
transparent-background prompt could decode to 60% NaN, exactly the background,
while opaque backgrounds stayed in range, so it went unnoticed. The exporter had
also rounded the fp32 checkpoint to bf16 on load. Now the decode matches
diffusers at fp32 to a mean alpha error of 1.4e-6, and
`test/test_transparency.jl` pins it on a real transparent-background generation.

fp32 convolutions miss the fp16 cooperative-matrix path, so since 2026-09-27 the
graph stays fp32 and only the convolutions' operands are fp16
(`qwenimagevae(; halfconvs = true)`, the default). What overflowed was never a
normalised convolution input but the residual stream, which reaches 3.5e5 on
unscaled `randn` latents; the five convolutions that read it are scaled by a
power of two at run time or left in fp32. The decoder's 3x3 convolutions gather
the image inside the GEMM rather than materialising im2col. A warm 1024² decode takes about
2.0 s on Vulkan and 2.05 s on ROCm, against 3.44 s and 3.38 s at fp32, with no NaN on
either input and a maximum alpha error of 3.2e-3 against diffusers at fp32 (one
8-bit step is 7.8e-3 on the [-1, 1] scale). `halfconvs = false` is the fp32
decoder.

Three components run in sequence, and the sequence is not optional: the denoiser
decodes to 7.26 GB of INT8 on the device and the Qwen3-VL-8B conditioner to
another 6.9 GB, which do not fit at once. Each is released as soon as its output
is in hand.

Measured, 20 steps at 1024x1024 (`examples/generate.jl` with `QWENIMAGE21_STEPS=20`;
its default is the reference's 40 since 2026-09-27), one run of the script
against another:

| stage | was | then | now |
| --- | --- | --- | --- |
| prompt encoding, including building the 36-layer encoder | 74.1 s | 55.1 s | 60.7 s |
| denoiser build and record | 56.5 s | 43.0 s | 44.2 s |
| 20 denoising steps | 138 s (6.89 s/step) | 100.5 s (5.02 s/step) | **89.8 s (4.49 s/step)** |
| VAE decode, including its build | 40.2 s | 34.4 s | 40.0 s |
| **total** | **314.9 s** | **234.6 s** | 239.2 s |

**The three columns are not all the same machine, so read the last one against
the A/B below rather than against the one beside it.** Re-measured on the same
day, same session, warm, with only the attention change stashed and unstashed:

| stage | without | with |
| --- | --- | --- |
| prompt encoding | 60.6 s | 60.7 s |
| denoiser build and record | 45.5 s | 44.2 s |
| 20 denoising steps | 100.1 s (5.01 s/step) | **89.8 s (4.49 s/step)** |
| VAE decode | 40.9 s | 40.0 s |
| **total** | **251.8 s** | **239.2 s** |

So the attention change is **-12.6 s**, effectively all of it in the denoising
steps, and it costs nothing in the three build rows.

That A/B isolates the attention kernel. The convolution work from the same day
— `im2col_kernel!`'s two integer divisions per element, and chunks that put the
whole remainder in the last one at full cost — is 10% to 36% per VAE layer and
does **not** show up here, which is the expected result rather than a
disappointment: the VAE decode row is ~34 s of compilation around ~2-3 s of
convolution, so a fifth off the convolution is a fraction of a second under a
row that swings by several. It is measured where it can be seen, per kernel, in
`plans/2026-09-21-rocm-baseline.md`, and it is worth having for the
convolution-heavy models rather than for this one. The middle column of the
first table was recorded in an earlier session; this machine is about 17 s
slower across every row today than it was then, which is why the two tables
disagree about the totals and agree about the change.

What moved: the attention kernel reads its K and V operands as cooperative
matrices straight from the tensors instead of staging them through shared
memory, and the tiling that is fastest then is a different one —
`plans/2026-09-21-rocm-baseline.md` has the measurements.

Most of what is left in the three build rows is still **Julia inference and
codegen**, not work: `emitgraph` is 97% compilation and `Mantle.record!` 99.6%,
and the second call to each in one process is 0.4 s and 0.0.
`plans/2026-09-21-qwen-denoiser-layer.md` has the measurement and why a plain
`precompile` directive does not fix it.

**Both checkpoints reach the device through a pack kernel that was a transpose
done a byte at a time**, and rewriting it takes the denoiser's weight upload
from 39.2 s to 10.3 and the conditioner's from 14.2 s to 1.0, measured back to
back in one process with bit-identical output. That is ~42 s off a generation,
from two kernels that never ran during one.

**And 82% of the VAE decode is convolution that was not on the tensor cores.**
Twenty-seven of its forty-five convolutions were refused because their im2col
matrix is gigabytes, and ran at about 1 TFLOP/s where the eighteen that fit
reach 19-22; its channel counts (144, 288, 576) also miss the column tile the
staged GEMM needs, which is worth a factor of thirteen on its own. Chunking the
pixel axis and padding the channels takes the decode **12.24 s to 5.50**
interpreted and 16.0 s to 5.41 recorded. Two things the convolutions had been
hiding go with them: the `expand` that broadcasts a plane across channels was
being materialised 43 times (639 ms, one of them 1.2 GB) and is now read as the
zero-strided view it is, and the forty explicit `F.pad`s in front of
zero-padded convolutions are now the convolution's own padding (661 ms). The
decode ends at **4.79 s**, bit-identical to before the last two.

The denoiser row is the one the changes below move, and re-measured on the real
model after them it is **95 s (4.74 s/step)** — 43 s off a generation. The
other three rows are from the original run and are untouched by this work.

A denoising step was 12.5 s when the model first ran. Where the rest went:

| | s/step |
| --- | --- |
| first run | 12.5 |
| declared GEMM reads packed int8 instead of dequantising | 10.0 |
| ConvRot in two register passes instead of four | 9.8 |
| attention operands copied rather than read interleaved | 8.4 |
| a 64-row attention tile, which a 128-wide head has room for | 7.1 |
| the padded product's destination declared padded, so nothing discards it | 6.9 |

Then seven more, measured on ONE LAYER rather than on the step. The layer is
cut out of the exported graph and given random weights of the declared shapes,
which replays in 232.8 ms against the real step's 215.6 ms a layer — 8% high,
and it iterates in a quarter of a second instead of a minute. It went to
**167.1 ms**, which is the same 28% off a step if it carries:

| | layer, ms |
| --- | --- |
| the harness, matching the row above | 232.8 |
| Qwen's per-head q/k RMS norm fused | 222.1 |
| the rotary interleave in one dispatch | 205.7 |
| the ragged key axis split at the last whole tile | 189.5 |
| permuted copies walked in the source's order | 180.7 |
| a contiguous slab copied as one run | 174.6 |
| two passes over the scores where the head is 96 wide or more | 171.1 |
| the interleaved rotary fused | 167.1 |
| the stacked-projection tile, and the SwiGLU reading in place | **156.0** |

A ninth change holds the attention's output in cooperative-matrix fragments
rather than shared memory, which also lets a 32-wide key block fit: the pass
goes 44.0 ms to 36.7 and the real model 5.59 s/step to **5.37**, back to back.

Then three more, all bit-identical to what they replace: compiling the softmax
form into the attention kernel as a `Val` rather than passing it, which deletes
the one-pass loop and the deferred rescale a two-pass plan never reaches (5.85
s/step against 6.02 interleaved); merging the two launches of a tail-split key
axis over a flat range instead of a three-dimensional one (that pass 3.645 ms
to 2.615); and choosing a permuted copy's walk order by how many memory streams
it leaves open rather than by source stride alone, which takes the two
attention-output permutes a layer from 1.474 ms to 0.552 each. The last two
together are the step's serialised total 4864.6 ms to 4808.9.

**And one that is not a code change at all.** Every number above was measured
in a Julia session that had been building and freeing plans for hours, and such
a session hands new buffers memory that runs the same tiled GEMM **2.4x
slower** — seventeen of the thirty-two gate+up products at 77-94 ms where the
other fifteen ran at 41-47, reproducible to `cor = 0.9998`, with identical
streaming bandwidth. In a fresh process every one of them runs at 39-44 ms and
**the step is 4.74 s**. See `plans/2026-09-21-qwen-denoiser-layer.md`; the
short version is to check `free -g` before believing a GPU timing.

On the real model the whole of it is 6.89 s/step to **4.74**. The layer harness
shows a larger share (-33% against -20%) because it runs plain int8 weights:
the compact checkpoint's four ConvRot transforms a layer, and the step's
non-layer work, are untouched by any of this and dilute it.

The last row is two changes, A/B'd together in one session: 169.1 ms with
neither, 163.9 with the tile alone, 164.3 with the SwiGLU alone, 156.0 with
both. They are super-additive because they relieve the same thing — the arena
the placer has to fit, which falls from 391 MB to 353.

The harness runs plain int8 weights, so it does NOT include the four ConvRot
transforms a layer (~5 ms) that the compact checkpoint adds. `plans/2026-09-21-qwen-denoiser-layer.md`
has the pass-by-pass breakdown and what is left.

What remains, measured pass by pass on the real step in a fresh process
(4858 ms serialised over 4372 passes):

| | ms | |
| --- | --- | --- |
| the products | 2963 | **61%**, and 22.5 of the device's 24.4 TOP/s |
| the attention | 1336 | **27%**, at ~7.5 TFLOP/s |
| everything else | 530 | 11%, all of it at memory bandwidth |

The products are done: sweeping every registered int8 tile at the biggest one
(`24576 x 4096 x 4224`) puts the shipped pick first at 22.54 TOP/s, and the
128x128 tile that moves 40% fewer bytes is second at 21.91 — so they are
compute-bound at 92% of the cooperative-matrix peak, and only int8
*activations* go past it (`plans/2026-09-20-int8-tensor-cores.md` says what
that costs in accuracy).

The attention is bound by re-reading K and V once per query block: two
diagnostics in the kernel price the softmax at 6% of it and those reads at 29%,
and the only lever on the 29% is a taller query tile, which is inadmissible at
this head width for two different reasons. The plan file has both.

## What is where

- **Tokenizer** (`src/tokenizer.jl`) — Qwen's byte-level BPE from the
  checkpoint's `processor/` directory. Matches `AutoTokenizer` id for id.
- **Conditioner** (`src/textencoder.jl`) — Qwen3-VL-8B in Comfy's asymmetric
  W4A8 with a 16-value codebook, FP8 group scales, per-channel scales and
  ConvRot. The embedding table stays on the host: it is a fifth of the
  checkpoint and a prompt reads forty rows of it. Checked end to end by
  decoding one token through `lm_head` — "The capital of France is" -> " Paris".
- **Denoiser** — the 32-layer transformer in tensor-wise INT8 with ConvRot.
- **VAE** — the 64-channel Wan-derived decoder, **fp32** with fp16 convolution
  operands, four output channels: RGB and a real alpha matte. See "Transparent
  backgrounds" above for why it is not fp16.
- **Host pipeline** — architecture constants, unpatched stride-16 latent
  flattening, the dynamic-shift FlowMatch schedule and its Euler update.

## Neither the prompt length nor the resolution is bound at export

```sh
uv run tools/export_qwenimage21.py --component text_encoder
uv run tools/export_qwenimage21.py --component transformer --graph-only
uv run tools/export_qwenimage21.py --component vae
```

Every graph carries its lengths as symbols: the text encoder's prompt axis `t`,
the denoiser's prompt axis `t` and image axis `i`, the decoder's latent grid `h`
and `w`. `--prompt-tokens`, `--height` and `--width` only size the example each
graph is traced at. The bounds each symbol was checked for are in the two
`*_export.json` files, and the runner reads them rather than assuming.

The denoiser graph carries a `t` symbol for the prompt axis, and
`qwenimagetransformer(; context_tokens)` binds it when the plan is recorded —
once per generation, which is when the prompt is already known. That is what
the reference does: `QwenImagePipeline` pads nothing for a single prompt and
`encode_prompt` drops the mask outright when it is all ones, so
`QwenImage21Transformer2DModel.forward` runs at whatever length it is given.
`--context-tokens` now only sizes the tracing example.

**The rotary tables are graph inputs, not constants.** They are not independent
of the prompt length: `QwenImage21Rope` advances a shared position one step per
text token and then freezes the image block's frame axis at the position the
text reached, so the image's rotary depends on how long the text was. Earlier
exports lifted them as constants, which silently bound the whole graph to one
prompt. `rotarytables` is the port of that loop and runs on the host per
generation, as the reference's `self.pos_embed(...)` does per forward. It
matches PyTorch to 5.96e-8 — half a Float32 ULP — over every element at both
lengths that were checked.

**The text encoder's length is a symbol too**, from 2 up to the 14-token system
turn plus the 1024 prompt tokens the denoiser takes. It was the last static
graph, at 64 tokens, which after the template left 42 tokens of prompt. A plan is
recorded per length and its GEMMs are compiled for it, so a prompt is right-padded
to the next of 64, 128, 256, 512 or the maximum, and at most five plans are ever
built. That is exact because the encoder is causal: padding with a different
token leaves the kept rows bit-identical, which `test/test_prompt_and_size.jl`
checks. Running the same prompt at a different bucket is not bit-identical, since
the kernels are tiled for the length, and moves the rows at fp16 rounding level
(0.3% rms).

## The weights ship as artifacts

Nothing has to be fetched or configured by hand. `assetdir()`, `vaedir()`,
`processordir()`, `compact_denoiser()` and `compact_encoder()` resolve through
`Artifacts.toml`, and `ready()` answers whether they are downloaded without
downloading them.

| artifact | size | holds |
| --- | ---: | --- |
| `qwenimage21` | 22 MB | denoiser and encoder graphs, their lifted constants, the tokenizer tables |
| `qwenimage21-vae` | 1.35 GB | VAE decoder graph and fp32 weights |
| `qwenimage21-dit-w1..w6` | 6.8 GB | the INT8 ConvRot denoiser checkpoint |
| `qwenimage21-enc-w1..w5` | 5.9 GB | the W4A8 Qwen3-VL conditioner checkpoint |
| `qwenimage21-refs` | 3 MB | test fixture only: transparent-background latents and diffusers' alpha for them |

Thirteen for the pipeline rather than one, for two reasons. A caller that only wants prompt
embeddings has no reason to fetch the denoiser, and a GitHub release asset caps
at 2 GiB — the two compact checkpoints are 7.26 GB and 6.31 GB, so they are
split byte-for-byte by `tools/shard_safetensors.jl` and merged back on load.
The split was checked by comparing every tensor of the merge against the
original: 649 and 1762 tensors, zero mismatches.

Re-pack and re-bind them with:

```sh
julia --project=. tools/make_artifacts.jl qwenimage21 qwenimage21-vae \
    qwenimage21-dit-w{1,2,3,4,5,6} qwenimage21-enc-w{1,2,3,4,5}
```

## Licence

The weights are **not** ours to relicense. Qwen-Image 2.1 is released under the
Qwen Research License Agreement, which permits redistribution (section 3) on
three conditions: every recipient gets a copy of the Agreement, modified files
say they were modified, and the attribution notice from 3(c) travels with them.
Every artifact above therefore carries `LICENSE` and `NOTICE`, copied by
`make_artifacts.jl` from `tools/licenses/qwenimage21/`; the `NOTICE` also lists
what was changed and how.

Two terms bind anyone who downloads them. Section 2(b) grants a
**non-commercial** licence only, for research and academic use; commercial use
needs a separate licence from Hangzhou Tongyi Laboratory. Section 4(c) forbids
using "Qwen" as the primary name or identifier of a derivative work, while
allowing descriptive use.

## Not ported

- Condition images (`Qwen-Image-Edit`): needs the vision tower and the
  image-conditioned template.
- Classifier-free guidance. The checkpoint's default is `true_cfg_scale = 1.0`
  — "Qwen-Image 2.1 is meant to be sampled without guidance" — so a generation
  is one denoiser evaluation per step, and a negative prompt would be two.
- A recorded VAE decode — because replaying one is SLOWER than interpreting it,
  not because it cannot be done. At 1024² on an 8060S: 12.8 s interpreted
  against 16.0 s replayed, and the plan costs 0.5 s to build on top, for one
  decode an image. The two agree to 9.3e-5 rms of a [-1, 1] range.
  `qwenimagevae(record = true)` builds it anyway, with the mid-block attention
  left unfused — it is a single head 1152 wide, which `flashcm_plan` declines
  and `coopmat_sdpa_plan` takes, and that plan has no declared form.

Upstream model: <https://huggingface.co/Qwen/Qwen-Image-2.1>
Compact checkpoint: <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>
License: Qwen Research License Agreement.
