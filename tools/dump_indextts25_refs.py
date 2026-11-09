"""
Stage-wise PyTorch references for `IndexTTS25Runner`.

    ~/tools/index-tts/.venv/bin/python tools/dump_indextts25_refs.py            # every case
    ~/tools/index-tts/.venv/bin/python tools/dump_indextts25_refs.py --case en_greedy

Runs upstream's own `IndexTTS2.infer` (`indextts/infer_v2_5.py`, index-tts at
d9e41aa) and records what each stage hands the next, through hooks on the
modules and wrappers on the methods it calls. Nothing of the pipeline is
re-implemented here, so the references cannot drift from upstream: what is
recorded is what upstream computed.

Writes `gen/indextts25/refs/<case>.safetensors` and `<case>.json` (the texts,
token lists and settings). fp32 throughout, TF32 off, on CUDA.

## Sampling that Julia can reproduce

Upstream samples with `torch.multinomial`, both alone (`num_beams = 1`) and
inside beam search (`num_beams = 3`, `do_sample = True`, its default). Without
replacement, or for one sample, ATen draws it as an exponential race
(`aten/src/ATen/native/Distributions.cpp`, `multinomial_out`):

    q ~ Exp(1);  sample = topk(p / q, n).indices      (argmax for n = 1)

`counter_multinomial` below is that algorithm with the noise taken from a
counter-based hash instead of the CUDA generator: `q = -log(u)`,
`u = ((h >> 8) + 1) / 2^24`, `h = lowbias32(index ^ lowbias32(call ^
lowbias32(seed)))`. The distribution is the same; the stream is one Julia can
compute too (`IndexTTS25Runner`'s sampler uses the same formula), so a sampled
decode is comparable token for token.

## Cases

Each case is one `infer` call. `decode` selects the generation:

  * `greedy`: `do_sample = False, num_beams = 1`. The parity case for the GPT.
  * `beam`: `do_sample = False, num_beams = 3`.
  * `sample`: `do_sample = True, num_beams = 1`, the German model card's setting.
  * `beamsample`: upstream's default, `do_sample = True, num_beams = 3`.

The flow-matching noise is `torch.randn` after `torch.manual_seed(seed)`, and
is recorded (`*/cfm/z`) so Julia can start from the same point.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path.home() / "tools" / "index-tts"
BASE = Path.home() / "models" / "IndexTTS-2.5"
GERMAN = Path.home() / "models" / "index-tts-2.5-german"
EXAMPLES = REPO / "examples"

sys.path.insert(0, str(REPO))
sys.path.insert(1, str(GERMAN))

from common import find_root  # noqa: E402

OUT = find_root() / "gen" / "indextts25" / "refs"

CASES = {
    "en_greedy": dict(spk="voice_01.wav", text="The quick brown fox jumps over the lazy dog, and nobody seems to mind.",
                      lang="en", decode="greedy"),
    "en_emoclip": dict(spk="voice_01.wav", emo="emo_sad.wav", emo_alpha=0.65,
                       text="I waited all evening, but the letter never came.",
                       lang="en", decode="greedy"),
    "en_emovec": dict(spk="voice_02.wav", emo_vector=[0, 0.8, 0, 0, 0, 0, 0.1, 0], emo_alpha=0.9,
                      text="Get out of my house, and take your dog with you!",
                      lang="en", decode="greedy"),
    "en_emotext": dict(spk="voice_02.wav", use_emo_text=True, emo_text="I am so scared, something is in the dark.",
                       text="Did you hear that? I think there is someone downstairs.",
                       lang="en", decode="greedy"),
    "en_duration": dict(spk="voice_01.wav", text="Slow and steady wins the race.", lang="en",
                        decode="greedy", duration_factor=1.4),
    "en_beam": dict(spk="voice_01.wav", text="The quick brown fox jumps over the lazy dog.",
                    lang="en", decode="beam"),
    "en_beamsample": dict(spk="voice_01.wav", text="The quick brown fox jumps over the lazy dog.",
                          lang="en", decode="beamsample", seed=7),
    "zh_greedy": dict(spk="voice_03.wav", text="欢迎大家来体验indextts2，并给予我们意见与反馈，谢谢大家。",
                      lang="zh", decode="greedy"),
    "en_long": dict(spk="voice_01.wav",
                    text=("It was the best of times, it was the worst of times, it was the age of wisdom, "
                          "it was the age of foolishness, it was the epoch of belief, it was the epoch of "
                          "incredulity, it was the season of Light, it was the season of Darkness, it was the "
                          "spring of hope, it was the winter of despair, we had everything before us, we had "
                          "nothing before us, we were all going direct to Heaven, we were all going direct the "
                          "other way."),
                    lang="en", decode="greedy", max_text_tokens_per_segment=60),
    "de_greedy": dict(spk="voice_01.wav", german=True, text="Dazu kommen 3 Gigabyte Speicher, sagte der Kapitän am 12.03.2024.",
                      lang="de", decode="greedy"),
    "de_sample": dict(spk="voice_01.wav", german=True,
                      text="Der große fröhliche Bär trägt einen grünen Regenschirm.",
                      lang="de", decode="sample", seed=1234,
                      gen=dict(temperature=0.7, top_p=0.8, top_k=30, repetition_penalty=10.0)),
    "de_emoclip": dict(spk="voice_01.wav", emo="emo_hate.wav", emo_alpha=0.8, german=True,
                       text="Verschwinde aus meinem Haus, und nimm deinen Hund mit!",
                       lang="de", decode="greedy"),
}

DECODES = {
    "greedy": dict(do_sample=False, num_beams=1),
    "beam": dict(do_sample=False, num_beams=3),
    "sample": dict(do_sample=True, num_beams=1),
    "beamsample": dict(do_sample=True, num_beams=3),
}


# ── the noise stream ─────────────────────────────────────────────────────────

M32 = 0xFFFFFFFF


def lowbias32(x):
    """Chris Wellons' `lowbias32` on uint32 values held in int64. Products wrap
    in int64 but only their low 32 bits are kept, which wrapping preserves."""
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32
    x = x ^ (x >> 16)
    return x


class CounterNoise:
    """Exp(1) noise from a counter: call `k` of the decode, element `i`."""

    def __init__(self, seed):
        self.seed = seed
        self.calls = 0

    def exponential(self, shape, device):
        n = 1
        for s in shape:
            n *= s
        idx = torch.arange(n, dtype=torch.int64, device=device)
        key = lowbias32(torch.tensor(self.calls, dtype=torch.int64) ^ lowbias32(torch.tensor(self.seed, dtype=torch.int64)))
        h = lowbias32(idx ^ key.to(device))
        u = ((h >> 8) + 1).to(torch.float32) * (2.0 ** -24)
        self.calls += 1
        return (-torch.log(u)).view(shape)


def counter_multinomial(noise):
    def multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
        assert not replacement and generator is None and out is None
        q = probs / noise.exponential(probs.shape, probs.device)
        if num_samples == 1:
            return torch.argmax(q, dim=-1, keepdim=True)
        return torch.topk(q, num_samples, dim=-1).indices
    return multinomial


# ── recording ────────────────────────────────────────────────────────────────

class Recorder:
    """Named tensors in call order. A name recorded twice gets `/1`, `/2`, ...
    after the first, so a per-segment or per-step value keeps every instance."""

    def __init__(self):
        self.tensors = {}
        self.counts = {}

    def __call__(self, name, t):
        k = self.counts.get(name, 0)
        self.counts[name] = k + 1
        key = name if k == 0 else f"{name}/{k}"
        self.tensors[key] = t.detach().float().contiguous().cpu() if t.is_floating_point() \
            else t.detach().contiguous().cpu()


def wrap(obj, attr, fn):
    """Replace `obj.attr` by `fn(original, *args, **kw)`; returns an undo."""
    orig = getattr(obj, attr)
    setattr(obj, attr, lambda *a, **kw: fn(orig, *a, **kw))
    return lambda: setattr(obj, attr, orig)


def instrument(tts, rec, meta):
    import indextts.infer_v2_5 as iv
    import torchaudio
    undo = []
    hooks = []

    def extract_features(orig, audio, **kw):
        rec("ref/audio16k", audio)
        out = orig(audio, **kw)
        rec("ref/features", out["input_features"])
        rec("ref/features_mask", out["attention_mask"])
        return out
    undo.append(wrap(tts, "extract_features", extract_features))

    def get_emb(orig, feats, mask):
        out = orig(feats, mask)
        rec("ref/w2v", out)
        return out
    undo.append(wrap(tts, "get_emb", get_emb))

    def mel_fn(orig, x):
        rec("ref/audio22k", x)
        out = orig(x)
        rec("ref/mel", out)
        return out
    undo.append(wrap(tts, "mel_fn", mel_fn))

    def fbank(orig, wav, **kw):
        out = orig(wav, **kw)
        rec("ref/fbank", out)
        return out
    undo.append(wrap(torchaudio.compliance.kaldi, "fbank", fbank))

    # A forward hook that returns anything REPLACES the module's output, so
    # these are functions returning nothing, never tuple-building lambdas.
    def campplus(m, i, o):
        rec("ref/campplus_in", i[0])
        rec("ref/style", o)
    hooks.append(tts.campplus_model.register_forward_hook(campplus))

    def regulator(m, args, kwargs, out):
        rec("regulator/in", args[0])
        rec("regulator/ylens", kwargs["ylens"])
        rec("regulator/out", out[0])
    hooks.append(tts.s2mel.models["length_regulator"].register_forward_hook(regulator, with_kwargs=True))

    def get_emovec(orig, cond, lens):
        out = orig(cond, lens)
        rec("emo/get_emovec_in", cond)
        rec("emo/get_emovec", out)
        return out
    undo.append(wrap(tts.gpt, "get_emovec", get_emovec))

    def merge_emovec(orig, spk, emo, l1, l2, alpha=1.0):
        out = orig(spk, emo, l1, l2, alpha=alpha)
        rec("emo/merged", out)
        meta.setdefault("emo_alpha_used", []).append(float(alpha))
        return out
    undo.append(wrap(tts.gpt, "merge_emovec", merge_emovec))

    def find_most_similar_cosine(orig, q, mat):
        i = orig(q, mat)
        meta.setdefault("emo_rows", []).append(int(i))
        return i
    undo.append(wrap(iv, "find_most_similar_cosine", find_most_similar_cosine))

    if tts.qwen_emo is not None:
        def qwen_generate(orig, **kw):
            out = orig(**kw)
            rec("qwen/input_ids", kw["input_ids"])
            rec("qwen/output_ids", out)
            return out
        undo.append(wrap(tts.qwen_emo.model, "generate", qwen_generate))

        def qwen_inference(orig, text):
            d = orig(text)
            meta["emo_dict"] = d
            return d
        undo.append(wrap(tts.qwen_emo, "inference", qwen_inference))

    def split(orig, text, max_tokens, lang_prefix=""):
        meta["text_prepared"] = text
        segs = orig(text, max_tokens, lang_prefix)
        meta["segments"] = segs
        return segs
    undo.append(wrap(tts, "split_text_by_tokens", split))

    gpt = tts.gpt

    def inference_speech(orig, spk, text_tokens, langs=None, emo=None, cond_lengths=None,
                         emo_cond_lengths=None, emo_vec=None, campplus_embedding=None, **kw):
        rec("gpt/text_tokens", text_tokens)
        rec("gpt/lang", langs)
        rec("gpt/emovec", emo_vec)
        rec("gpt/campplus", campplus_embedding)
        meta.setdefault("gen_kwargs", []).append({k: v for k, v in kw.items()
                                                  if isinstance(v, (int, float, bool, str))})
        codes, latent = orig(spk, text_tokens, langs, emo, cond_lengths, emo_cond_lengths,
                             emo_vec=emo_vec, campplus_embedding=campplus_embedding, **kw)
        rec("gpt/codes", codes)
        rec("gpt/spk_latent", latent)
        return codes, latent
    undo.append(wrap(gpt, "inference_speech", inference_speech))

    def store_mel_emb(orig, emb):
        rec("gpt/prefix", emb)
        return orig(emb)
    undo.append(wrap(gpt.inference_model, "store_mel_emb", store_mel_emb))

    # The raw logits of every forward: the head's output at the last position.
    def lm_head(m, i, o):
        rec("gpt/logits", o[:, -1, :])
    hooks.append(gpt.inference_model.lm_head.register_forward_hook(lm_head))

    def codec_decode(orig, codes):
        rec("codec/codes", codes)
        out = orig(codes)
        rec("codec/out", out)
        return out
    undo.append(wrap(tts.semantic_codec, "decode", codec_decode))

    cfm = tts.s2mel.models["cfm"]
    # The estimator's first call of each solve, inputs and output: one DiT
    # forward to check the export against before the whole solve.
    state = {"first_step": False}

    def solve_euler(orig, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        state["first_step"] = True
        rec("cfm/z", x)
        rec("cfm/mu", mu)
        rec("cfm/x_lens", x_lens)
        rec("cfm/prompt", prompt)
        rec("cfm/style", style)
        rec("cfm/t_span", t_span)
        meta.setdefault("cfg_rate", []).append(float(inference_cfg_rate))
        out = orig(x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
        rec("cfm/out", out)
        return out
    undo.append(wrap(cfm, "solve_euler", solve_euler))

    def estimator(m, args, out):
        if state["first_step"]:
            for n, a in zip(("x", "prompt_x", "x_lens", "t", "style", "mu"), args):
                rec(f"dit/{n}", a)
            rec("dit/out", out)
            state["first_step"] = False
    hooks.append(cfm.estimator.register_forward_hook(estimator))

    def bigvgan(m, i, o):
        rec("bigvgan/in", i[0])
        rec("bigvgan/out", o)
    hooks.append(tts.bigvgan.register_forward_hook(bigvgan))

    def done():
        for u in reversed(undo):
            u()
        for h in hooks:
            h.remove()
    return done


def load(german):
    from indextts.infer_v2_5 import IndexTTS2
    from indextts.utils.checkpoint import load_checkpoint
    tts = IndexTTS2(cfg_path=str(BASE / "config.yaml"), model_dir=str(BASE), use_bf16=False,
                    device="cuda:0", use_accel=False, use_qwen_emo=True)
    if german:
        # As the German model card does: the merged checkpoint over the base GPT.
        load_checkpoint(tts.gpt, str(GERMAN / "gpt.pth"))
        tts.gpt.eval()
    return tts


def run_case(tts, name, case):
    from german_text import prepare_german
    rec, meta = Recorder(), {"case": name, **{k: v for k, v in case.items()}}
    seed = case.get("seed", 1234)
    noise = CounterNoise(seed)
    done = instrument(tts, rec, meta)
    orig_multinomial = torch.multinomial
    torch.multinomial = counter_multinomial(noise)
    try:
        text = case["text"]
        if case.get("german"):
            text = prepare_german(text)
            meta["text_german_frontend"] = text
        kwargs = dict(DECODES[case["decode"]], **case.get("gen", {}))
        # The CFM noise is the global generator's first draw after this.
        torch.manual_seed(seed)
        sr, wav = tts.infer(
            spk_audio_prompt=str(EXAMPLES / case["spk"]),
            text=text, output_path=None, lang=case["lang"],
            emo_audio_prompt=str(EXAMPLES / case["emo"]) if case.get("emo") else None,
            emo_alpha=case.get("emo_alpha", 1.0),
            emo_vector=case.get("emo_vector"),
            use_emo_text=case.get("use_emo_text", False), emo_text=case.get("emo_text"),
            duration_factor=case.get("duration_factor", 1.0),
            max_text_tokens_per_segment=case.get("max_text_tokens_per_segment", 120),
            text_normalization=not case.get("german", False),
            **kwargs)
    finally:
        torch.multinomial = orig_multinomial
        done()
    rec("out/wav_int16", torch.from_numpy(wav.T.copy()))
    meta["sample_rate"] = sr
    meta["noise_calls"] = noise.calls
    # Reference caches live on the object; drop them so every case recomputes.
    tts.cache_spk_cond = tts.cache_emo_cond = None
    tts.cache_spk_audio_prompt = tts.cache_emo_audio_prompt = None
    return rec.tensors, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", help="case name (repeatable); default all")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    names = args.case or list(CASES)
    args.out.mkdir(parents=True, exist_ok=True)
    for german in (False, True):
        todo = [n for n in names if bool(CASES[n].get("german")) == german]
        if not todo:
            continue
        tts = load(german)
        for n in todo:
            print(f"== {n}", flush=True)
            tensors, meta = run_case(tts, n, CASES[n])
            save_file(tensors, str(args.out / f"{n}.safetensors"))
            (args.out / f"{n}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str))
            print(f"   {len(tensors)} tensors, segments {meta.get('segments')}", flush=True)
        del tts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
