"""Does Kokoro survive fp16? And bf16? Answered before porting either.

    uv run tools/precision_probe.py

## Why this runs before any kernel work

Measured on this device, `VK_KHR_cooperative_matrix` offers **no fp32 A/B
configuration at all** — the tensor cores cannot take fp32 operands. Its shape
table is:

    float16        x float16        -> float16 / float32
    bfloat16       x bfloat16       -> float32
    uint8 / sint8                   -> uint32 / sint32
    float8_e4m3_pk / float8_e5m2_pk -> float16 / float32

so an fp32 export is structurally locked out, and it shows: a 4096-cubed GEMM
runs at 42.4 TF/s in fp16 and **0.71 in fp32**, a 60x gap that is not a missing
optimisation but the absence of a path.

That leaves two routes for an fp32 model, and they cost very differently:

  * **fp16** — Lava has it today. The export becomes an autocast export, which
    SAM 2's decoder already is.
  * **bfloat16** — the device supports it, Lava does **not**: no component type,
    no Julia type, nothing in the emitter. A real port.

bf16 is the safer numeric choice (8 exponent bits, the same range as fp32, so
nothing overflows), and this project has been bitten by fp16's range before —
20% of one model's rows fell outside its window, and Whisper in fp16 came out 5x
worse than PyTorch's own fp16. But it is only worth porting if fp16 is actually
insufficient *for these models*, which is a measurement, not a judgement.

## What is measured

The whole model, end to end, at each precision, against the fp32 run — cosine on
the audio and the predicted durations. The durations matter separately because
they decide the output LENGTH: a precision that changes them by one frame makes
the audio comparison meaningless rather than merely worse.

The vocoder's noise is zeroed throughout (`tools/verify_kokoro.py` explains why),
so what is left is the arithmetic.
"""

import json
from pathlib import Path

import torch

from common import find_root

ROOT = find_root()
OUT = ROOT / "gen" / "graphs" / "kokoro-dyn" / "precision_probe.json"

PHRASES = {
    "short": "hˈɛlO",
    "medium": "həlˈO wˈɜɹld, ðɪs ɪz ɐ tˈɛst ʌv spˈiʧ sˈɪnθəsɪs ɪn ʤˈuljə.",
    "long": ("ðə nˈʊɹəl nˈɛtwɜɹk pɹˈɑsɛsᵻz ˈɔdiO ˈæt twˈɛnti fˈɔɹ kˈɪlOhɜɹts, "
             "ænd ðə hˈOl θˈɪŋ ɹˈʌnz ˈɑn ðə ɡɹˈæfɪks pɹˈɑsɛsəɹ."),
}


def nonoise():
    """Zero the vocoder's two random draws; returns the originals to restore."""
    rand, randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *a, **k: torch.zeros(*a, **k)
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    return rand, randn_like


def roundtrip(model, dtype):
    """Round every parameter and buffer through `dtype` and back to fp32.

    Deliberately NOT `model.to(dtype)`. Running the whole model in fp16 also
    changes the accumulate precision, the softmax, the norms — everything — and
    would answer a different question. The GEMM path on this device accumulates
    in **fp32** whatever the operands are, so what a coopmat export actually
    changes is the precision of the OPERANDS, and that is what this simulates.

    Rounding back to fp32 also keeps the rest of the graph bit-identical, so a
    difference in the output is attributable to the rounding and nothing else.
    """
    with torch.no_grad():
        for p in list(model.parameters()) + list(model.buffers()):
            if p.is_floating_point():
                p.copy_(p.to(dtype).to(torch.float32))
    return model


def run(model, ps, packs, voice="af_heart"):
    ids = [model.vocab.get(c) for c in ps]
    ids = torch.LongTensor([[0, *[i for i in ids if i is not None], 0]])
    ref_s = packs[voice][len(ps) - 1]
    with torch.no_grad():
        audio, dur = model.forward_with_tokens(ids, ref_s, 1.0)
    return audio.float(), dur


def cosine(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[:n], b[:n]
    return (a @ b / (a.norm() * b.norm())).item()


def main(out: Path):
    from kokoro import KModel
    from huggingface_hub import hf_hub_download
    gen = ROOT / "gen" / "kokoro"

    def fresh():
        return KModel(repo_id="hexgrad/Kokoro-82M",
                      config=str(gen / "config.json"),
                      model=str(gen / "kokoro-v1_0.pth")).eval()

    packs = {"af_heart": torch.load(
        hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"),
        map_location="cpu", weights_only=True)}

    rand, randn_like = nonoise()
    results = {}
    try:
        base = fresh()
        ref = {k: run(base, ps, packs) for k, ps in PHRASES.items()}

        for name, dtype in (("float16", torch.float16), ("bfloat16", torch.bfloat16)):
            m = roundtrip(fresh(), dtype)
            row = {}
            for k, ps in PHRASES.items():
                a, d = run(m, ps, packs)
                a0, d0 = ref[k]
                row[k] = {
                    "cos": cosine(a, a0),
                    "frames": int(d.sum()), "frames_ref": int(d0.sum()),
                    "dur_exact": bool(torch.equal(d, d0)),
                }
                print(f"  {name:9s} {k:7s} cos {row[k]['cos']:.6f}  "
                      f"frames {row[k]['frames']} vs {row[k]['frames_ref']}"
                      f"{'' if row[k]['dur_exact'] else '   <- DURATIONS DIFFER'}")
            results[name] = row
    finally:
        torch.rand, torch.randn_like = rand, randn_like

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main(OUT)
