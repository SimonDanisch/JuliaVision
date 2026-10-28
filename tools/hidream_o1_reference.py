"""HiDream-O1-Image (MIT) through its own reference code, to judge it against Qwen-Image 2.1
and FLUX.2-klein on the same inputs before anything is ported:

    tools/memguard.sh -- ~/.venvs/hidream-o1/bin/python -I tools/hidream_o1_reference.py \
        --code /sim/hf/code/HiDream-O1-Image --out ../../renders/hidream_o1_20261012

The reference code pins transformers 4.57.1 and fails under this repo's 5.17 (the rotary
registry lost its "default" entry), so it runs from its own venv: torch 2.11 cu128,
transformers 4.57.1, diffusers, accelerate, einops, scipy.

`--code` is a checkout of github.com/HiDream-ai/HiDream-O1-Image with flash attention
switched off in `models/pipeline.py` (the README's instruction when flash-attn is not
installed); the weights come from the Hugging Face cache. The model is a Qwen3-VL whose
language model also denoises 32x32 pixel patches, so there is no VAE: the whole 8B net
runs every step, at its trained size of about 2048x2048, and the reference code snaps
every request to its table of such sizes. bf16 on the GPU in one piece (17.6 GB of
weights); the host only holds safetensors mmaps. The language head (1.2 GB) only serves
text generation and is dropped after loading: with it, Agamemnon from the reference sheet
(about 10k tokens) ran out of the RTX 4000's 20 GB in the MLP. Run under tools/memguard.sh.

Jobs, in the order they run: the fox from the QwenImageRunner README; Agamemnon from the
Handsome Squidward sheet as one reference (the input Qwen and FLUX got), then from the
three pictures of that sheet as three references (HiDream's subject mode); four plain
prompts of the FLUX comparison; then the other two Agamemnon seeds.
"""

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# Fragmentation decides whether a 10k-token step fits next to 16.4 GiB of weights.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fluxklein_reference import PROMPTS  # noqa: E402

CRAW = Path(__file__).resolve().parents[3]
SKETCHES = CRAW / "renders/troja_sketches_20261008"
SHEET = SKETCHES / "hs_reference_sheet.png"
# The three Handsome Squidward pictures the sheet was built from (head, bust, full body).
THREE = [CRAW / ".bt-attachments" / n for n in
         ("2026-10-10_102803_653b423a.png", "2026-10-10_102803_5aa2d469.png", "2026-10-10_102803_570853e6.png")]

# `HS_WARRIOR` of tools/troja_sketches.jl with a grey background instead of a transparent one, as FLUX got it.
WARRIOR = ("A smooth rounded 3D character render with soft natural curves and real volume, no polygons, no facets: "
           "Handsome Squidward from SpongeBob as in image 1, as a tall ancient Greek high king and warrior, about eight "
           "heads tall, heroic proportions, broad shoulders and a big muscular chest like in the full-body picture of "
           "image 1, long strong legs. His head exactly like in image 1: a huge smooth bald cranium that is wider at the "
           "top, a heavy brow ridge, heavy half-closed eyelids with pale green eyeshadow, pale yellow eyes with red "
           "pupils, a bored arrogant look, a long droopy nose, bulging cheekbones, a very long face with a huge rounded "
           "jaw and a bulging chin, small full pink pouty kiss lips, a thick neck as wide as his jaw, pale turquoise "
           "skin. He wears a polished bronze muscle cuirass, a long royal purple cloak, a purple pteruges skirt, bronze "
           "greaves and sandals. Standing straight, facing the camera, full body, plain light grey background, no shadow.")
WARRIOR_THREE = (WARRIOR.replace("the full-body picture of image 1", "the full-body reference picture")
                 .replace("as in image 1", "as in the reference images")
                 .replace("like in image 1", "like in the reference images"))
FOX = "a red fox sitting in a snowy forest at sunrise, photorealistic"


def jobs():
    yield dict(name="fox", prompt=FOX, refs=[], size=(1024, 1024), seed=42)
    yield dict(name="agamemnon_sheet_s61", prompt=WARRIOR, refs=[SHEET], size=(832, 1216), seed=61)
    yield dict(name="agamemnon_three_s61", prompt=WARRIOR_THREE, refs=THREE, size=(832, 1216), seed=61)
    for name, prompt in PROMPTS.items():
        yield dict(name=name, prompt=prompt, refs=[], size=(928, 1664) if name == "zeus" else (832, 1216), seed=61)
    for seed in (67, 71):
        yield dict(name=f"agamemnon_sheet_s{seed}", prompt=WARRIOR, refs=[SHEET], size=(832, 1216), seed=seed)
        yield dict(name=f"agamemnon_three_s{seed}", prompt=WARRIOR_THREE, refs=THREE, size=(832, 1216), seed=seed)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="checkout of HiDream-O1-Image")
    ap.add_argument("--repo", default="HiDream-ai/HiDream-O1-Image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="job names to run (default: all)")
    args = ap.parse_args()

    sys.path.insert(0, args.code)
    from transformers import AutoProcessor
    from models.pipeline import generate_image
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = snapshot_download(args.repo, local_files_only=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(weights)
    model = Qwen3VLForConditionalGeneration.from_pretrained(weights, dtype=torch.bfloat16, device_map="cuda").eval()
    del model.lm_head  # generation returns `x_pred` before the logits are computed
    torch.cuda.empty_cache()
    tok = processor.tokenizer
    tok.boi_token, tok.bor_token, tok.eor_token = "<|boi_token|>", "<|bor_token|>", "<|eor_token|>"
    tok.bot_token, tok.tms_token = "<|bot_token|>", "<|tms_token|>"
    print(f"loaded in {time.time() - t0:.0f} s, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
          f"host peak RSS {rss_gib():.1f} GiB", flush=True)

    runs_file = out / "runs.json"
    runs = json.loads(runs_file.read_text()) if runs_file.exists() else []
    for job in jobs():
        if args.only and job["name"] not in args.only:
            continue
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        width, height = job["size"]
        # The full model's settings from the reference inference.py: 50 UniPC steps, guidance 5,
        # shift 3; subject personalisation (two or more references) is run with shift 1 there.
        img = generate_image(model=model, processor=processor, prompt=job["prompt"],
                             ref_image_paths=[str(p) for p in job["refs"]], height=height, width=width,
                             num_inference_steps=50, guidance_scale=5.0, shift=1.0 if len(job["refs"]) > 1 else 3.0,
                             timesteps_list=None, scheduler_name="default", seed=job["seed"])
        img.save(out / f"{job['name']}.png")
        run = dict(name=job["name"], prompt=job["prompt"], references=[str(p) for p in job["refs"]],
                   seed=job["seed"], requested=list(job["size"]), size=list(img.size),
                   seconds=round(time.time() - t, 1),
                   peak_vram_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        runs = [r for r in runs if r["name"] != job["name"]] + [run]
        runs_file.write_text(json.dumps(runs, indent=1))
        print(json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
