"""Reference audio from upstream Kokoro, for the same phonemes the Julia runner uses.

    uv run tools/verify_kokoro.py

Writes `refs_speak.safetensors`: for each phrase, the audio `KModel` produces and
the durations it predicted, both with the vocoder's RNG **zeroed**.

## Why the noise is zeroed rather than seeded

`SineGen` draws `torch.rand` for its initial phase and `torch.randn_like` for the
noise floor of the harmonic-plus-noise source, so Kokoro is stochastic: two calls
on identical inputs differ. Seeding makes PyTorch repeatable but not comparable —
the Julia side draws from Julia's RNG, so a seeded reference measures the
difference between two random streams and nothing about the model.

Zeroing both on this side (and on the Julia side, via `speak(..., noise=false)`)
removes that term entirely and leaves a comparison of the deterministic path,
which is what a port can actually be held to. The noise is a real part of the
output — this is a measurement instrument, not the way the model is meant to run.

The durations are the other half and are checked separately, because they decide
the audio's *length*: if they disagree, every sample after the first divergence
is compared against the wrong instant and the correlation is meaningless whatever
the vocoder did.
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from common import find_root

ROOT = find_root()
OUT = ROOT / "gen" / "graphs" / "kokoro-dyn" / "refs_speak.safetensors"

# The Julia side phonemises these itself; the exact strings it produces are
# written next to the audio so a G2P difference cannot be mistaken for a model
# difference. Lengths deliberately spread: the export was done at 30 tokens.
PHRASES = {
    "short": "hˈɛlO",
    "medium": "həlˈO wˈɜɹld, ðɪs ɪz ɐ tˈɛst ʌv spˈiʧ sˈɪnθəsɪs ɪn ʤˈuljə.",
    "long": ("ðə nˈʊɹəl nˈɛtwɜɹk pɹˈɑsɛsᵻz ˈɔdiO ˈæt twˈɛnti fˈɔɹ kˈɪlOhɜɹts, "
             "ænd ðə hˈOl θˈɪŋ ɹˈʌnz ˈɑn ðə ɡɹˈæfɪks pɹˈɑsɛsəɹ."),
}
VOICES = ["af_heart", "am_michael", "bf_emma"]


def nonoise():
    """Replace the vocoder's two random draws with zeros, globally.

    `SineGen` calls them as bare `torch.rand`/`torch.randn_like`, so there is no
    argument to pass — the functions themselves are swapped for the duration of
    the reference run and restored after.
    """
    rand, randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *a, **k: torch.zeros(*a, **k)
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    return rand, randn_like


def main(out: Path):
    from kokoro import KModel
    gen = ROOT / "gen" / "kokoro"
    model = KModel(repo_id="hexgrad/Kokoro-82M",
                   config=str(gen / "config.json"),
                   model=str(gen / "kokoro-v1_0.pth")).eval()

    packs = {}
    from huggingface_hub import hf_hub_download
    for v in VOICES:
        packs[v] = torch.load(hf_hub_download("hexgrad/Kokoro-82M", f"voices/{v}.pt"),
                              map_location="cpu", weights_only=True)

    rand, randn_like = nonoise()
    tensors, meta = {}, {}
    try:
        with torch.no_grad():
            for pname, ps in PHRASES.items():
                ids = [model.vocab.get(c) for c in ps]
                ids = [i for i in ids if i is not None]
                ids = torch.LongTensor([[0, *ids, 0]])
                # `pack[len(ps)-1]`, exactly as `KPipeline` indexes it — by the
                # PHONEME count, not the token count. Off by two from the
                # token count, and silent if wrong.
                ref_s = packs[VOICES[0]][len(ps) - 1]
                for v in VOICES:
                    ref_s = packs[v][len(ps) - 1]
                    audio, dur = model.forward_with_tokens(ids, ref_s, 1.0)
                    key = f"{pname}/{v}"
                    tensors[f"{key}/audio"] = audio.float().contiguous()
                    tensors[f"{key}/dur"] = dur.contiguous()
                    meta[key] = {"phonemes": ps, "ntokens": int(ids.shape[1]),
                                 "nframes": int(dur.sum()), "nsamples": int(audio.shape[-1])}
                    print(f"  {key:24s} {ids.shape[1]:3d} tokens -> "
                          f"{int(dur.sum()):4d} frames, {audio.shape[-1]:6d} samples")
    finally:
        torch.rand, torch.randn_like = rand, randn_like

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"  -> {out}")


if __name__ == "__main__":
    main(OUT)
