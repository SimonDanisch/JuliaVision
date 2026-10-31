"""Diffusers FLUX.2-klein from the exact noise and prompt embeddings the Julia port used,
so the transformer and VAE paths are compared and not two RNGs or two text encoders.

    .venv/bin/python tools/fluxklein_parity.py 4B out.png noise.f32 embeds.f32 LH LW [reference.png]

`noise.f32` is the Julia `(128, LH*LW, 1)` Float16 start noise written as Float32 (column-major:
channel fastest, so it reads as a `(LH*LW, 128)` row-major matrix); `embeds.f32` the Julia
encoder's `(3*hidden, 512, 1)` output the same way. With a reference picture the edit path runs,
the picture pre-sized the way `FluxKleinRunner.fitreference` sizes it. bf16 weights, each
component straight onto the GPU; the 9B transformer is 17 GB and has the card to itself.
Prints the peak VRAM, which is what a quantised Julia run is up against.
"""

import sys
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from transformers import Qwen2TokenizerFast

REPOS = {"4B": "black-forest-labs/FLUX.2-klein-4B", "9B": "black-forest-labs/FLUX.2-klein-9B"}


def main():
    model, out, noisef, embf, lh, lw = sys.argv[1:7]
    reference = sys.argv[7] if len(sys.argv) > 7 else None
    lh, lw = int(lh), int(lw)
    repo = REPOS[model]
    # `prepare_latents` takes the UNPACKED (B, C, H, W) grid and packs it itself in row-major token order.
    noise = torch.from_numpy(np.fromfile(noisef, dtype=np.float32).reshape(lh * lw, 128)).T.reshape(1, 128, lh, lw)
    emb = torch.from_numpy(np.fromfile(embf, dtype=np.float32)).reshape(512, -1)[None]
    tok = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    tr = Flux2Transformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                 device_map="cuda", low_cpu_mem_usage=True)
    vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch.bfloat16, device_map="cuda",
                                             low_cpu_mem_usage=True)
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    pipe = Flux2KleinPipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=tok, transformer=tr)
    kw = dict(prompt_embeds=emb.to("cuda", torch.bfloat16), latents=noise.to("cuda", torch.bfloat16),
              width=16 * lw, height=16 * lh, num_inference_steps=4, guidance_scale=1.0)
    if reference:
        kw["image"] = [Image.open(reference).convert("RGB")]
    with torch.inference_mode():
        img = pipe(**kw).images[0]
    img.save(out)
    print("ok", img.size, "peak VRAM GiB", round(torch.cuda.max_memory_allocated() / 2**30, 1))


if __name__ == "__main__":
    main()
