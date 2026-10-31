"""FLUX.2-klein through Diffusers, as the reference the FluxKleinRunner port is checked against.

Runs the upstream pipeline on CUDA and writes the pictures with their prompts,
seeds and timings next to them, so a quality judgement and the later numerical
parity checks start from the same, reproducible outputs:

    .venv/bin/python tools/fluxklein_reference.py --model 4B --out /tmp/fluxklein
    .venv/bin/python tools/fluxklein_reference.py --model 9B --out /tmp/fluxklein

`--edit IMAGE PROMPT` runs one reference-image edit instead of the prompt list.
The 9B weights are gated: the account behind ~/.cache/huggingface/token has to
have accepted the FLUX Non-Commercial License on the model page first.

Memory: the first version of this script used `enable_model_cpu_offload`, which
parks every component in host RAM (16 GB in bf16) and froze a 32 GB box that
had 13 GB free (2026-10-12). Now each component is loaded straight onto the GPU
with `device_map`, the text encoder is released before the transformer is
built, and the host never holds more than one safetensors mmap. Run it under
tools/memguard.sh, which kills it before the box starts swapping.
"""

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

REPOS = {"4B": "black-forest-labs/FLUX.2-klein-4B", "9B": "black-forest-labs/FLUX.2-klein-9B"}

# Figures the Odyssee series needs, worded the way the Qwen sketches were (plain
# background instead of transparent: FLUX has no alpha channel).
PROMPTS = {
    "agamemnon": "A smooth rounded 3D character render: Handsome Squidward from SpongeBob as a tall ancient Greek "
                 "high king and warrior, about eight heads tall, broad shoulders, big muscular chest, long strong legs, "
                 "huge smooth bald cranium wider at the top, heavy half-closed eyelids, bored arrogant look, long droopy "
                 "nose, very long face with a huge rounded jaw and bulging chin, small pink pouty kiss lips, thick neck, "
                 "pale turquoise skin, polished bronze muscle cuirass, long royal purple cloak, purple pteruges skirt, "
                 "bronze greaves and sandals, standing straight facing the camera, full body, plain light grey background",
    "helena": "Simple low-poly 3D game character of a very beautiful slender Mycenaean Bronze Age Greek queen with light "
              "olive Mediterranean skin, very long black curly hair in many ringlets with a thin gold diadem, a fitted "
              "short-sleeved deep blue bodice, a long flounced tiered skirt in saffron yellow and blue, gold bracelets, "
              "barefoot, flat colours without texture, like a casual indie game, full body, standing straight, facing "
              "the camera, plain light grey background, no shadow",
    "sailor": "Simple low-poly 3D game character of a wiry thin ancient Greek sailor with a long pointy nose, ochre tunic, "
              "spiky black hair, barefoot, completely blank face without eyes, without eyebrows and without mouth, only a "
              "small nose, flat colours without texture, like a casual indie game, full body, standing straight, facing "
              "the camera, plain light grey background, no shadow",
    "zeus": "A fluffy white cumulus cloud shaped like a relaxed bro man sitting cross-armed on a smaller cloud, puffy "
            "cloud muscles, a cloud beard, two small dark unimpressed eyes, one eyebrow raised, deep blue sky background, "
            "photorealistic volumetric cloud, high detail",
}


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}  (peak RSS {rss_gib():.1f} GiB, "
          f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB)", flush=True)


def load_gpu(cls, repo: str, subfolder: str, **kw):
    """One component straight onto the GPU: the safetensors are mmapped and copied
    tensor by tensor, so the host holds page cache, not a second copy."""
    return cls.from_pretrained(repo, subfolder=subfolder, torch_dtype=torch.bfloat16, device_map="cuda",
                               low_cpu_mem_usage=True, **kw)


def release():
    """After the caller has dropped its last reference: collect, and hand the cache back."""
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(REPOS), default="4B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edit", nargs=2, metavar=("IMAGE", "PROMPT"))
    ap.add_argument("--only", nargs="*", help="prompt names to run (default: all)")
    args = ap.parse_args()
    repo = REPOS[args.model]
    out = args.out / args.model
    out.mkdir(parents=True, exist_ok=True)

    jobs = ([("edit", args.edit[1], (832, 1216))] if args.edit else
            [(n, p, (928, 1664) if n == "zeus" else (832, 1216)) for n, p in PROMPTS.items()
             if not args.only or n in args.only])

    # 1. Prompts through the text encoder, then it is gone before the transformer arrives.
    tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    text_encoder = load_gpu(Qwen3ForCausalLM, repo, "text_encoder")
    log("text encoder loaded")
    embeds = {}
    # inference_mode: called outside the pipeline, the encoder would otherwise build an autograd
    # graph over all 36 layers (9.5 GiB of activations at 512 tokens, measured).
    with torch.inference_mode():
        for name, prompt, _ in jobs:
            embeds[name] = Flux2KleinPipeline._get_qwen3_prompt_embeds(text_encoder, tokenizer, prompt, device="cuda").cpu()
    del text_encoder
    release()
    log("prompts encoded, encoder released")

    # 2. Transformer and VAE, with a pipeline around them that has no text encoder.
    transformer = load_gpu(Flux2Transformer2DModel, repo, "transformer")
    vae = load_gpu(AutoencoderKLFlux2, repo, "vae")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    pipe = Flux2KleinPipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer,
                              transformer=transformer)
    log("transformer and VAE loaded")

    runs = []
    for name, prompt, (width, height) in jobs:
        g = torch.Generator("cpu").manual_seed(args.seed)
        torch.cuda.synchronize()
        t = time.perf_counter()
        kw = dict(prompt_embeds=embeds[name].to("cuda"), width=width, height=height, num_inference_steps=4,
                  guidance_scale=1.0, generator=g)
        if args.edit:
            kw["image"] = [Image.open(args.edit[0]).convert("RGB")]
        picture = pipe(**kw).images[0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        picture.save(out / f"{name}.png")
        runs.append({"name": name, "prompt": prompt, "seed": args.seed, "size": [width, height],
                     "seconds": round(seconds, 2), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        log(f"{name}: {seconds:.1f} s")
        (out / ("edit.json" if args.edit else "runs.json")).write_text(json.dumps(runs, indent=1))


if __name__ == "__main__":
    main()
