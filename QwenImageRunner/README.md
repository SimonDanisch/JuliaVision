# QwenImageRunner

Qwen-Image 2.1 text-to-image on Lava, from the compact Comfy-Org checkpoints.

A 1024x1024 image, end to end in Julia on a Radeon 8060S:

```sh
julia --project=. QwenImageRunner/examples/generate.jl \
    "a red fox sitting in a snowy forest at sunrise, photorealistic" fox.ppm
```

Three components run in sequence, and the sequence is not optional: the denoiser
decodes to 7.26 GB of INT8 on the device and the Qwen3-VL-8B conditioner to
another 6.9 GB, which do not fit at once. Each is released as soon as its output
is in hand.

Measured, 20 steps at 1024x1024 (`examples/generate.jl`, 314.9 s total):

| stage | time |
| --- | --- |
| prompt encoding, including building the 36-layer encoder | 74.1 s |
| denoiser build and record | 56.5 s |
| 20 denoising steps | 138 s (6.89 s/step) |
| VAE decode, including its build | 40.2 s |

Those four rows are one run and predate the seven changes below, which have not
been re-measured end to end — the denoiser row is the one they move.

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

The last row is two changes, A/B'd together in one session: 169.1 ms with
neither, 163.9 with the tile alone, 164.3 with the SwiGLU alone, 156.0 with
both. They are super-additive because they relieve the same thing — the arena
the placer has to fit, which falls from 391 MB to 353.

The harness runs plain int8 weights, so it does NOT include the four ConvRot
transforms a layer (~5 ms) that the compact checkpoint adds. `plans/2026-09-21-qwen-denoiser-layer.md`
has the pass-by-pass breakdown and what is left.

What remains, per layer, is the products at the device's fp16 ceiling (~53% of
it) and the attention at ~30%, which runs at about 6 TFLOP/s where the products
in the same layer reach 21-26. Its key length still divides no tiling, but that
no longer costs the whole kernel: the bounds check is compiled into a second
launch over the ragged remainder, and the 257 blocks that fill a tile run
without it.

## What is where

- **Tokenizer** (`src/tokenizer.jl`) — Qwen's byte-level BPE from the
  checkpoint's `processor/` directory. Matches `AutoTokenizer` id for id.
- **Conditioner** (`src/textencoder.jl`) — Qwen3-VL-8B in Comfy's asymmetric
  W4A8 with a 16-value codebook, FP8 group scales, per-channel scales and
  ConvRot. The embedding table stays on the host: it is a fifth of the
  checkpoint and a prompt reads forty rows of it. Checked end to end by
  decoding one token through `lm_head` — "The capital of France is" -> " Paris".
- **Denoiser** — the 32-layer transformer in tensor-wise INT8 with ConvRot.
- **VAE** — the 64-channel Wan-derived decoder, fp16, four output channels
  (the fourth is alpha).
- **Host pipeline** — architecture constants, unpatched stride-16 latent
  flattening, the dynamic-shift FlowMatch schedule and its Euler update.

## The exports are static

`tools/export_qwenimage21.py` binds resolution and prompt length at export:

```sh
uv run tools/export_qwenimage21.py --component text_encoder --prompt-tokens 64
uv run tools/export_qwenimage21.py --component transformer --graph-only \
    --height 1024 --width 1024 --context-tokens 22
uv run tools/export_qwenimage21.py --component vae --height 1024 --width 1024
```

The encoder's length is a maximum: it is causal, so a shorter prompt is padded
on the right and the kept rows are bit-identical. The denoiser's context length
is **exact** — its text stream is part of a joint attention, so a padded prompt
would change the image — and it is the token count of the prompt template minus
the system turn the pipeline drops (14 tokens). `examples/generate.jl` reports
the number to re-export with when they disagree.

Set `JULIA_QWENIMAGE21_ASSETS` to the export directory,
`JULIA_QWENIMAGE21_COMPACT` to the Comfy-Org checkpoint directory, and
`JULIA_QWENIMAGE21_PROCESSOR` to the tokenizer's `processor/` directory.

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
