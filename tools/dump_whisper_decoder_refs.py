"""Per-node reference activations for Whisper large-v3-turbo's DECODER step.

Same contract as `dump_whisper_refs.py` — `"<graph>/in<i>"` and
`"<graph>/node/<name>"` in one safetensors file — so `DNNKernels.verifygraph`
reads it without knowing which model produced it. Covers both graphs the decoder
artifact carries, `whispercross` and `whisperdec`.

    uv run tools/dump_whisper_decoder_refs.py

**The step is recorded at cache position 3, not 0.** A step at position 0 has a
mask that admits exactly one slot and an `index_put` that writes slot 0, so a
mask off by one, an `index_put` that writes the wrong slot, and an attention that
ignores the cache entirely all produce identical output. Three real tokens are
decoded first — the actual forced prefix, through the actual graph — and the
fourth is what gets recorded, with a populated cache behind it.

**The input is the real encoder output**, not `randn`: cross-attention reads it
directly, and its scale is what sets the range of every cross-attention
activation. The audio is `dump_whisper_refs.speechlike`, the same deterministic
speech-shaped signal the encoder reference uses, so the two files describe one
pipeline rather than two unrelated experiments.

Also saved, for the acceptance check that is not a per-node diff: a token
sequence `WhisperRunner.greedy` has to reproduce exactly. A per-node tolerance
says the arithmetic is right; only a token sequence says the *decoding* is —
whether the loop feeds the right id back, advances the cache, and stops on the
right token.

**Two sequences, because they test different things.**

`whisperdec/tokens` is a pure argmax loop over this same `DecoderStep`. That is
the reference for the loop: same arithmetic, same prompt, nothing else in the
way, so a difference is unambiguously ours.

`whisperdec/generate_tokens` is what `WhisperForConditionalGeneration.generate`
produces. It is *not* the same thing and `greedy` will not match it, by
construction: `generate` runs Whisper's logits processors — `SuppressTokens`,
`SuppressTokensAtBegin`, the no-timestamps and timestamp-ordering rules — which
change which token wins the argmax. It is recorded as the target for
`plans/whisper-decoder.md` D6, where those processors get implemented, not as a
gate on the machinery underneath them.

(On this deliberately speech-*shaped* but meaningless audio `generate` emits five
tokens of Japanese. That is the model hallucinating on noise, not a bug — and it
is a fine determinism reference either way.)
"""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

import common
from dump_whisper_refs import Recorder, speechlike, SR
from export_whisper_decoder import CrossKV, DecoderStep, SRC_LEN, STORAGE

ROOT = common.ROOT
GEN = common.GEN

# <|startoftranscript|><|en|><|transcribe|><|notimestamps|>. Four tokens, so the
# recorded step is the fourth and the cache holds three.
#
# Pinned rather than detected. `generate` left to itself runs language detection
# and picks the prefix from the audio, so the prompt would depend on the acoustic
# model — and a reference whose *inputs* move when the model changes is not a
# reference. Whisper's own tokens, spelled out.
PROMPT = [50258, 50259, 50360, 50363]
EOT = 50257


def keepset(graph):
    """fx node names to record: everything the graph computes, no weights.

    `Recorder` with `keep=None` stores every node including the *placeholders*,
    and `DecoderStep` holds the whole `WhisperForConditionalGeneration` — so the
    first run of this wrote an 8 GiB reference file that was mostly the 32-layer
    encoder's weights, which are already in the encoder's own artifact and which
    `verifygraph` never looks at.

    Taken from the pruned graph JSON, so it follows `export_whisper_decoder.py`'s
    own idea of what the graph contains. `.0`-`.3` are added blind: ops that
    return tuples (`native_layer_norm`, `_scaled_dot_product_efficient_attention`)
    are recorded per element, and a name that never occurs costs nothing.
    """
    keep = set(graph["outputs"])
    keep |= {b["id"] for b in graph["buffers"] if b.get("kind") != "weight"}
    keep |= {o["out"] for o in graph["ops"]}
    return keep | {f"{n}.{i}" for n in set(keep) for i in range(4)}


def greedyref(step, ck, cv, L, H, hd, device, dt, maxtarget, maxtokens=224):
    """Pure argmax over the reference step — the sequence `greedy` must match.

    No logits processors, no timestamp rules, no suppression list. That is the
    point: this isolates the decode *loop* from Whisper's sampling policy, so a
    mismatch here is the cache, the mask or the feedback, and never a rule
    nobody has implemented yet.
    """
    sk = torch.zeros(L, 1, H, maxtarget, hd, device=device, dtype=dt)
    sv = torch.zeros_like(sk)
    logits = None
    for t, tok in enumerate(PROMPT):
        logits, sk, sv = step(torch.tensor([[tok]], device=device), sk, sv, ck, cv,
                              torch.tensor([t], device=device))
    out = []
    for t in range(len(PROMPT), len(PROMPT) + maxtokens):
        nxt = int(logits.argmax())
        if nxt == EOT or t >= maxtarget:
            break
        out.append(nxt)
        logits, sk, sv = step(torch.tensor([[nxt]], device=device), sk, sv, ck, cv,
                              torch.tensor([t], device=device))
    return out


def run(out_path, gdir, precision, device, maxtarget, seed):
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA — the graph this diffs against was exported "
                         "from CUDA, and a CPU trace decomposes attention "
                         "differently (see export_whisper.py)")

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    import export_graphs as EG

    weights = GEN / "whisper"
    dt = STORAGE[precision]
    model = WhisperForConditionalGeneration.from_pretrained(
        str(weights), torch_dtype=dt, attn_implementation="sdpa",
        local_files_only=True).to(device).eval()
    cfg = model.config
    L, H, D = cfg.decoder_layers, cfg.decoder_attention_heads, cfg.d_model
    hd = D // H

    audio = speechlike(seed=seed)
    fe = WhisperFeatureExtractor.from_pretrained(str(weights), local_files_only=True)
    mel = fe(audio, sampling_rate=SR, return_tensors="pt").input_features.float()

    step = DecoderStep(model, H, hd).eval()
    cross = CrossKV(model, H, hd).eval()

    tensors = {"whisperdec/audio": torch.from_numpy(audio).clone(),
               "whisperdec/mel": mel.cpu().contiguous().clone()}

    gcross = json.loads((gdir / "whispercross.json").read_text())
    gdec = json.loads((gdir / "whisperdec.json").read_text())

    with torch.no_grad(), EG.precision_ctx(precision):
        enc = model.model.encoder(mel.to(device=device, dtype=dt)).last_hidden_state

        # ── whispercross: the window's cross-attention K/V, node by node
        epc = torch.export.export(cross, (enc,), strict=False).run_decompositions()
        recc = Recorder(epc.module(), keepset(gcross))
        recc.run(enc)
        ck, cv = cross(enc)
        tensors["whispercross/in0"] = enc.cpu().contiguous().clone()
        for k, v in recc.results.items():
            tensors[f"whispercross/node/{k}"] = v

        # ── three real tokens first, so the recorded step has a live cache
        sk = torch.zeros(L, 1, H, maxtarget, hd, device=device, dtype=dt)
        sv = torch.zeros_like(sk)
        for t, tok in enumerate(PROMPT[:-1]):
            _, sk, sv = step(torch.tensor([[tok]], device=device), sk, sv, ck, cv,
                             torch.tensor([t], device=device))

        args = (torch.tensor([[PROMPT[-1]]], device=device), sk, sv, ck, cv,
                torch.tensor([len(PROMPT) - 1], device=device))
        ep = torch.export.export(step, args, strict=False).run_decompositions()
        rec = Recorder(ep.module(), keepset(gdec))
        t0 = time.perf_counter()
        rec.run(*args)
        dt_s = time.perf_counter() - t0

        for i, a in enumerate(args):
            tensors[f"whisperdec/in{i}"] = a.cpu().contiguous().clone()
        for k, v in rec.results.items():
            tensors[f"whisperdec/node/{k}"] = v

        # ── the loop reference, and separately the policy target
        tensors["whisperdec/prompt"] = torch.tensor(PROMPT, dtype=torch.int64)
        toks = greedyref(step, ck, cv, L, H, hd, device, dt, maxtarget)
        tensors["whisperdec/tokens"] = torch.tensor(toks, dtype=torch.int64)
        ids = model.generate(mel.to(device=device, dtype=dt),
                             do_sample=False, num_beams=1, max_new_tokens=224)
        tensors["whisperdec/generate_tokens"] = ids[0].cpu().to(torch.int64).clone()

    # ── tokenizer acceptance data, as JSON beside the tensors
    #
    # A BPE encoder that is subtly wrong still produces plausible ids, and the
    # only thing that catches it is HuggingFace's own answer on strings chosen to
    # exercise the merge order: leading spaces (`Ġ`), contractions and digits
    # (separate regex classes in the pre-split), non-ASCII and emoji (multi-byte,
    # so the byte map matters and a single character spans several tokens),
    # newlines and runs of whitespace.
    from transformers import WhisperTokenizer
    tok = WhisperTokenizer.from_pretrained(str(weights), local_files_only=True)
    probes = [
        "Hello world", " Hello world", "hello", " the", "The quick brown fox.",
        "I don't think it's 42 degrees.", "1234567890", "a  b\n\nc",
        "Ich hörte Grüße aus München", "naïve café", "\u4f60\u597d\u4e16\u754c",
        "\U0001f600 emoji \U0001f680", "   leading and trailing   ",
        "Mixed123Case456", "'s 't 're 've 'm 'll 'd",
    ]
    tokrefs = {
        "encode": {t: tok.encode(t, add_special_tokens=False) for t in probes},
        "decode_tokens": tok.decode(toks, skip_special_tokens=True),
        "decode_generate": tok.decode(ids[0].tolist(), skip_special_tokens=False),
        "prompt_ids": PROMPT,
    }
    (out_path.parent / "tokenizer_refs.json").write_text(json.dumps(tokrefs, indent=1))
    print(f"  tokenizer refs: {len(probes)} probe strings")
    print(f"  argmax loop decodes to: {tokrefs['decode_tokens'][:80]!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    nb = sum(v.numel() * v.element_size() for v in tensors.values())
    print(f"{len(tensors)} tensors, {nb / 2**20:.0f} MiB -> {out_path}")
    print(f"  whisperdec step recorded in {dt_s:.2f} s at cache position "
          f"{len(PROMPT) - 1}")
    print(f"  argmax loop: {len(toks)} tokens, first 12 {toks[:12]}")
    print(f"  generate:    {len(ids[0])} tokens, "
          f"first 12 {ids[0][:12].tolist()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", type=Path, default=GEN / "graphs" / "whisperdec")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--precision", choices=sorted(STORAGE), default="fp32")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-target", type=int, default=448)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.out or (a.graphs / "refs.safetensors"), a.graphs,
        a.precision, a.device, a.max_target, a.seed)
