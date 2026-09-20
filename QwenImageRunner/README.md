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

Measured, 20 steps at 1024x1024 (`examples/generate.jl`, 413.6 s total):

| stage | time |
| --- | --- |
| prompt encoding, including building the 36-layer encoder | 71.0 s |
| denoiser build and record | 55.8 s |
| 20 denoising steps | 250.2 s (12.5 s/step) |
| VAE decode, including its build | 34.9 s |

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
- A recorded VAE decode. Its mid-block attention is a single head 1152 wide,
  which has no declared plan, and the unfused form then runs one submission past
  the driver's limit. Interpreted it is 34.9 s of the 413.6.

Upstream model: <https://huggingface.co/Qwen/Qwen-Image-2.1>
Compact checkpoint: <https://huggingface.co/Comfy-Org/Qwen-Image-2.1>
License: Qwen Research License Agreement.
