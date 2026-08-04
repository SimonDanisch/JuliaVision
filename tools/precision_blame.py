"""WHICH module loses accuracy in fp16 — one at a time.

    uv run tools/precision_blame.py

`precision_probe.py` says rounding every weight to fp16 costs Kokoro 0.134 of
cosine on a long utterance, and that bf16 is far worse (0.446). Neither number
says *where*, and a whole-model verdict is the wrong unit of decision: autocast
does not round everything, it rounds the GEMM and convolution operands and leaves
norms, biases and accumulations alone.

So this rounds **one submodule at a time**, with everything else exact, and
attributes the loss. A module that costs nothing can go to fp16 and reach the
tensor cores; one that costs a lot stays fp32 and stays slow, which is a fine
trade if it is a small part of the arithmetic.

The complement is also run — everything EXCEPT the named module — because the two
together say whether the loss is concentrated (one module explains it) or diffuse
(no module does, and the whole model is simply at its precision limit).
"""

import json
from pathlib import Path

import torch

from common import find_root

ROOT = find_root()
OUT = ROOT / "gen" / "graphs" / "kokoro-dyn" / "precision_blame.json"

PHRASE = ("ðə nˈʊɹəl nˈɛtwɜɹk pɹˈɑsɛsᵻz ˈɔdiO ˈæt twˈɛnti fˈɔɹ kˈɪlOhɜɹts, "
          "ænd ðə hˈOl θˈɪŋ ɹˈʌnz ˈɑn ðə ɡɹˈæfɪks pɹˈɑsɛsəɹ.")


def nonoise():
    rand, randn_like = torch.rand, torch.randn_like
    torch.rand = lambda *a, **k: torch.zeros(*a, **k)
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    return rand, randn_like


def run(model, ps, pack, voice="af_heart"):
    ids = [model.vocab.get(c) for c in ps]
    ids = torch.LongTensor([[0, *[i for i in ids if i is not None], 0]])
    with torch.no_grad():
        audio, dur = model.forward_with_tokens(ids, pack[len(ps) - 1], 1.0)
    return audio.float(), dur


def cosine(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[:n], b[:n]
    return (a @ b / (a.norm() * b.norm())).item()


def roundnamed(model, pred, dtype=torch.float16):
    """Round every float parameter whose qualified name satisfies `pred`.

    Returns how many tensors and elements were touched, so a module that appears
    harmless can be distinguished from one that was never reached.
    """
    n = nel = 0
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p.is_floating_point() and pred(name):
                p.copy_(p.to(dtype).to(torch.float32))
                n += 1
                nel += p.numel()
    return n, nel


def main(out: Path):
    from kokoro import KModel
    from huggingface_hub import hf_hub_download
    gen = ROOT / "gen" / "kokoro"

    def fresh():
        return KModel(repo_id="hexgrad/Kokoro-82M",
                      config=str(gen / "config.json"),
                      model=str(gen / "kokoro-v1_0.pth")).eval()

    pack = torch.load(hf_hub_download("hexgrad/Kokoro-82M", "voices/af_heart.pt"),
                      map_location="cpu", weights_only=True)

    base = fresh()
    # The top-level submodules, plus a split of the decoder, which is most of the
    # arithmetic and worth resolving further.
    groups = {
        "bert": lambda n: n.startswith("bert."),
        "bert_encoder": lambda n: n.startswith("bert_encoder."),
        "predictor": lambda n: n.startswith("predictor."),
        "text_encoder": lambda n: n.startswith("text_encoder."),
        "decoder.generator": lambda n: n.startswith("decoder.generator."),
        "decoder.decode": lambda n: n.startswith("decoder.decode"),
        "decoder.encode": lambda n: n.startswith("decoder.encode"),
        "decoder.F0_conv": lambda n: n.startswith("decoder.F0_conv"),
        "decoder.N_conv": lambda n: n.startswith("decoder.N_conv"),
        "decoder.asr_res": lambda n: n.startswith("decoder.asr_res"),
    }

    rand, randn_like = nonoise()
    results = {}
    try:
        ref, refdur = run(base, PHRASE, pack)
        print(f"  reference: {ref.shape[-1]} samples, {int(refdur.sum())} frames\n")
        print(f"  {'module':24s} {'tensors':>8s} {'Melem':>8s}   {'cos':>9s}  {'complement':>10s}")
        for name, pred in groups.items():
            m = fresh()
            n, nel = roundnamed(m, pred)
            a, d = run(m, PHRASE, pack)
            c = cosine(a, ref)

            mc = fresh()
            roundnamed(mc, lambda x, p=pred: not p(x))
            ac, dc = run(mc, PHRASE, pack)
            cc = cosine(ac, ref)

            results[name] = {"tensors": n, "elements": nel, "cos": c,
                             "cos_complement": cc,
                             "dur_exact": bool(torch.equal(d, refdur))}
            print(f"  {name:24s} {n:8d} {nel/1e6:8.2f}   {c:9.6f}  {cc:10.6f}"
                  f"{'' if results[name]['dur_exact'] else '  DUR!'}")
    finally:
        torch.rand, torch.randn_like = rand, randn_like

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main(OUT)
