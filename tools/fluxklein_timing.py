"""Where Diffusers' FLUX.2-klein spends its time, part by part, as the yardstick for the Julia port:

    tools/memguard.sh -- .venv/bin/python tools/fluxklein_timing.py 4B 832 1216 reference.png

Prints, after a warm-up of each: the prompt encoding (512 tokens), one transformer evaluation
text to image and with the reference, the VAE decode and the reference's VAE encode, and the
whole pipeline call both ways. bf16, every component straight onto the GPU, the text encoder
released before the rest is loaded (as tools/fluxklein_reference.py does it).
"""

import gc
import sys
import time

import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline, Flux2Transformer2DModel
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

REPOS = {"4B": "black-forest-labs/FLUX.2-klein-4B", "9B": "black-forest-labs/FLUX.2-klein-9B"}
PROMPT = "Handsome Squidward as a towering ancient Greek warrior king in a 3D animated feature film"


def timed(f, repeat=3):
    """Median seconds of `f()` after one warm-up call, the GPU drained around each."""
    f()
    out = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t = time.perf_counter()
        f()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    return sorted(out)[len(out) // 2]


class Clock:
    """Per-call GPU time of a module's forward, from CUDA events around it."""

    def __init__(self, module):
        self.pairs = []
        module.register_forward_pre_hook(self.pre)
        module.register_forward_hook(self.post)

    def pre(self, *_):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.pairs.append([start, None])

    def post(self, *_):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.pairs[-1][1] = end

    def take(self):
        torch.cuda.synchronize()
        ms = [s.elapsed_time(e) / 1000 for s, e in self.pairs]
        self.pairs = []
        return ms


def main():
    model, width, height, reference = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    repo = REPOS[model]
    load = dict(torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    tok = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer")
    t = time.perf_counter()
    enc = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", **load)
    print(f"load encoder {time.perf_counter() - t:.1f} s")
    with torch.inference_mode():
        print(f"encode prompt {timed(lambda: Flux2KleinPipeline._get_qwen3_prompt_embeds(enc, tok, PROMPT, device='cuda')):.3f} s")
        emb = Flux2KleinPipeline._get_qwen3_prompt_embeds(enc, tok, PROMPT, device="cuda")
    del enc
    gc.collect()
    torch.cuda.empty_cache()

    t = time.perf_counter()
    tr = Flux2Transformer2DModel.from_pretrained(repo, subfolder="transformer", **load)
    vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", **load)
    print(f"load transformer + vae {time.perf_counter() - t:.1f} s")
    sch = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler")
    pipe = Flux2KleinPipeline(scheduler=sch, vae=vae, text_encoder=None, tokenizer=tok, transformer=tr)
    steps, decodes, encodes = Clock(tr), Clock(vae.decoder), Clock(vae.encoder)
    image = Image.open(reference).convert("RGB")
    kw = dict(prompt_embeds=emb, width=width, height=height, num_inference_steps=4, guidance_scale=1.0)
    with torch.inference_mode():
        for name, extra in (("text to image", {}), ("with reference", {"image": [image]})):
            whole = timed(lambda: pipe(**kw, **extra, generator=torch.Generator("cpu").manual_seed(61)))
            s, d, e = steps.take(), decodes.take(), encodes.take()
            n = len(s) // 4                                   # warm-up + 3 timed calls, 4 steps each
            per_step = sorted(s[4:])[len(s[4:]) // 2]
            print(f"{name}: whole call {whole:.2f} s, transformer step {per_step:.3f} s ({n} calls), "
                  f"vae decode {sorted(d)[len(d) // 2]:.3f} s" + (f", vae encode {sorted(e)[len(e) // 2]:.3f} s" if e else ""))
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
