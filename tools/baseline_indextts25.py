"""
IndexTTS-2.5 in eager PyTorch, stage by stage: the denominator for
`IndexTTS25Runner`'s RTF table.

    ~/tools/index-tts/.venv/bin/python tools/baseline_indextts25.py --dtype fp32
    ~/tools/index-tts/.venv/bin/python tools/baseline_indextts25.py --dtype fp16
    ~/tools/index-tts/.venv/bin/python tools/baseline_indextts25.py --dtype bf16   # upstream's default

Measured as `baseline_all.py` measures (its `bench`: warm-up, a clock ramp of
sustained load, the median of the timed runs, TF32 off), on one utterance: the
`en_greedy` case of `dump_indextts25_refs.py`, voice_01.wav, 71 characters.

The stages are upstream's own modules called the way `infer_generator` calls
them, each timed alone:

| stage | what | per |
|---|---|---|
| `voice` | resample, SeamlessM4T features, w2v-BERT, mel, fbank, CAMPPlus, regulator, emotion encoder twice | clip (upstream caches it per clip) |
| `gpt` | `inference_speech`, greedy (deterministic): per-token time over the utterance | utterance |
| `gpt_beam` | `inference_speech` with upstream's default beam search with sampling, 3 beams | utterance |
| `s2mel` | codec decode, regulator, 25 CFM steps with CFG | utterance |
| `bigvgan` | the vocoder | utterance |

`--dtype` sets the GPT's weights (`.half()`/`.bfloat16()` and autocast, as
upstream's `use_bf16`) and autocast for the rest; fp32 is the parity
configuration. Upstream's default is bf16 for the GPT and fp32 for everything
else (`infer_generator` runs s2mel and BigVGAN with autocast off), which
`--dtype bf16` reproduces.

The numbers go to `gen/indextts25/baseline_<dtype>.json`, with the SM clock.
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_all import bench, smclock  # noqa: E402
from common import find_root  # noqa: E402

REPO = Path.home() / "tools" / "index-tts"
BASE = Path.home() / "models" / "IndexTTS-2.5"
sys.path.insert(0, str(REPO))

GEN = find_root() / "gen" / "indextts25"
TEXT = "The quick brown fox jumps over the lazy dog, and nobody seems to mind."
CLIP = REPO / "examples" / "voice_01.wav"
DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16, "fast": torch.float16}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    p.add_argument("--iters", type=int, default=7)
    p.add_argument("--ramp", type=float, default=6.0)
    p.add_argument("--stages", nargs="*", default=["voice", "gpt", "gpt_beam", "s2mel", "bigvgan"])
    p.add_argument("--output", type=Path, help="Checkpoint JSON destination (updated after each stage)")
    args = p.parse_args()
    if args.dtype != "fp32" and {"cached_voice", "complete"}.intersection(args.stages):
        p.error("complete-call benchmarks currently require --dtype fp32")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from indextts.infer_v2_5 import IndexTTS2
    import torchaudio
    dt = DTYPES[args.dtype]
    tts = IndexTTS2(cfg_path=str(BASE / "config.yaml"), model_dir=str(BASE), use_bf16=False,
                    device="cuda:0", use_accel=False, use_qwen_emo=False)
    if dt is not None:
        tts.gpt = tts.gpt.to(dt)
        tts.gpt.post_init_gpt2_config(kv_cache=True, half=True)
    dev = "cuda:0"
    gpt_ctx = (lambda: torch.amp.autocast("cuda", dtype=dt)) if dt is not None else contextlib.nullcontext
    # Upstream runs s2mel and BigVGAN with autocast off whatever `use_bf16` is;
    # fp16 here is the configuration our fp16 exports run.
    rest_ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.float16)) if args.dtype in ("fp16", "fast") \
        else contextlib.nullcontext
    # Match the runner: clip encoders and BigVGAN remain fp32 in :fast.
    clip_ctx = contextlib.nullcontext if args.dtype == "fast" else rest_ctx
    vocoder_ctx = contextlib.nullcontext if args.dtype == "fast" else rest_ctx

    state = {}

    def voice():
        # Include file decoding and both reference rates, as Julia's voice(path)
        # and upstream's fresh complete call do.
        audio, sr = tts._load_and_cut_audio(str(CLIP), 15)
        audio22 = torchaudio.transforms.Resample(sr, 22050)(audio)
        emo16, _ = tts._load_and_cut_audio(str(CLIP), 15, sr=16000)
        audio16 = torchaudio.transforms.Resample(22050, 16000)(audio22)
        with torch.no_grad(), clip_ctx():
            inp = tts.extract_features(audio16, sampling_rate=16000, return_tensors="pt")
            spk = tts.get_emb(inp["input_features"].to(dev), inp["attention_mask"].to(dev))
            mel = tts.mel_fn(audio22.to(dev).float())
            feat = torchaudio.compliance.kaldi.fbank(audio16.to(dev), num_mel_bins=80, dither=0,
                                                     sample_frequency=16000)
            style = tts.campplus_model((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))
            prompt = tts.s2mel.models["length_regulator"](spk, ylens=torch.LongTensor([mel.size(2)]).to(dev),
                                                          n_quantizers=3, f0=None)[0]
            einp = tts.extract_features(emo16, sampling_rate=16000, return_tensors="pt")
            emo = tts.get_emb(einp["input_features"].to(dev), einp["attention_mask"].to(dev))
        with torch.no_grad(), gpt_ctx():
            emovec = tts.gpt.merge_emovec(spk, emo, torch.tensor([spk.shape[-1]], device=dev),
                                          torch.tensor([emo.shape[-1]], device=dev), alpha=1.0)
        state.update(spk=spk, mel=mel, style=style, prompt=prompt, emo=emo, emovec=emovec)

    voice()
    toks = tts.tokenizer.encode("<|en|> " + TEXT.lower(), allowed_special="all")
    text = torch.IntTensor(toks + [1]).unsqueeze(0).to(dev)
    lang = torch.LongTensor([0]).to(dev)

    def gpt(**gen):
        if gen.get("do_sample"):
            torch.manual_seed(1234)
        with torch.no_grad(), gpt_ctx():
            codes, _ = tts.gpt.inference_speech(
                state["spk"], text, lang, state["emo"],
                cond_lengths=torch.tensor([state["spk"].shape[-1]], device=dev),
                emo_cond_lengths=torch.tensor([state["emo"].shape[-1]], device=dev),
                emo_vec=state["emovec"], campplus_embedding=state["style"], wav=str(CLIP),
                top_p=0.8, top_k=30, temperature=0.8, num_return_sequences=1, length_penalty=0.0,
                repetition_penalty=10.0, max_generate_length=1500, **gen)
        state["last_gpt_codes"] = int((codes[0] != tts.stop_mel_token).sum())
        return codes

    codes = gpt(do_sample=False, num_beams=1)
    n = int((codes[0] != tts.stop_mel_token).sum())
    state["codes"] = codes[:, :n]

    def s2mel():
        with torch.no_grad(), rest_ctx():
            S = tts.semantic_codec.decode(state["codes"])
            y = torch.LongTensor([int(S.shape[1] * 1.72)]).to(dev)
            cond = tts.s2mel.models["length_regulator"](S, ylens=y, n_quantizers=3, f0=None)[0]
            cat = torch.cat([state["prompt"], cond], dim=1)
            out = tts.s2mel.models["cfm"].inference(cat, torch.LongTensor([cat.size(1)]).to(dev),
                                                     state["mel"], state["style"], None, 25,
                                                     inference_cfg_rate=0.7)
        state["vc"] = out[:, :, state["mel"].size(-1):]

    s2mel()

    def vocode():
        with torch.no_grad(), vocoder_ctx():
            return tts.bigvgan(state["vc"].float())

    wav = vocode()
    seconds = wav.shape[-1] / 22050
    def infer(*, fresh=False):
        if fresh:
            for field in ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt", "cache_mel",
                          "cache_spk_audio_prompt", "cache_emo_cond", "cache_emo_audio_prompt"):
                setattr(tts, field, None)
        result = tts.infer(str(CLIP), TEXT, None, "en", do_sample=False, num_beams=1)
        assert result is not None and result[0] == 22050
        assert len(result[1]) == wav.shape[-1], "complete-call benchmark changed audio length"
        return result

    def synthesis():
        s2mel()
        return vocode()

    stages = {"voice": voice, "gpt": lambda: gpt(do_sample=False, num_beams=1),
              "gpt_beam": lambda: gpt(do_sample=True, num_beams=3), "s2mel": s2mel, "bigvgan": vocode,
              "synthesis": synthesis, "cached_voice": infer, "complete": lambda: infer(fresh=True)}
    out = {"dtype": args.dtype, "device": torch.cuda.get_device_name(0), "audio_seconds": seconds,
           "codes": n, "sm_clock_mhz": None, "stages": {}}
    output = args.output or GEN / f"baseline_{args.dtype}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        temporary = output.with_name(output.name + ".partial")
        temporary.write_text(json.dumps(out, indent=1))
        temporary.replace(output)

    for name in args.stages:
        started_unix = time.time()
        r = bench(stages[name], iters=args.iters, ramp=args.ramp)
        r["started_unix"] = started_unix
        r["finished_unix"] = time.time()
        duration = seconds
        if name in ("gpt", "gpt_beam"):
            r["codes"] = state["last_gpt_codes"]
            duration = int(2 * r["codes"] * 1.72) * 256 / 22050
        r["audio_seconds"] = duration
        r["rtf"] = r["p50"] / 1000 / duration
        if name == "gpt":
            r["ms_per_token"] = r["p50"] / (n + 1)
        out["stages"][name] = r
        out["sm_clock_mhz"] = smclock()
        r["sm_clock_mhz"] = out["sm_clock_mhz"]
        checkpoint()
        print(f"{name:9s} p50 {r['p50']:9.1f} ms  min {r['min']:9.1f}  spread {r['spread']:.3f}  RTF {r['rtf']:.4f}"
              + (f"  {r['ms_per_token']:.2f} ms/token" if name == "gpt" else ""), flush=True)
    out["sm_clock_mhz"] = smclock()
    total = sum(out["stages"][s]["p50"] for s in ("voice", "gpt_beam", "s2mel", "bigvgan") if s in out["stages"])
    print(f"audio {seconds:.2f} s, {n} codes, sm clock {out['sm_clock_mhz']} MHz, "
          f"voice + beam GPT + s2mel + BigVGAN = {total:.0f} ms, RTF {total / 1000 / seconds:.3f}")
    checkpoint()


if __name__ == "__main__":
    main()
