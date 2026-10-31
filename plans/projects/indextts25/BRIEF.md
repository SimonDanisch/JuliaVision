# Brief: IndexTTS25Runner in JuliaVision

Port **IndexTTS-2.5** (Bilibili) to JuliaVision as `IndexTTS25Runner`. It must produce the same audio as the PyTorch reference, and be as fast as possible on this PC's GPUs.

Why this model: it is the only open-weight TTS trained to take the **voice from one clip and the emotion from a second clip**. CrawCity needs that for German character dialogue (Odyssee). Research: `/sim/Programmieren/CrawCity/reports/IndexTTS2 emotion cloning comparison.md`.

## Where to work

- **Checkout:** `/sim/Programmieren/VideoEdit/dev/JuliaVision` (branch `sd/mantle-dev`), the live JuliaVision.
- **Julia env:** `/sim/Programmieren/VideoEdit`, which devs the JuliaVision packages by path. Use `bt_julia_eval` with `env_path = /sim/Programmieren/VideoEdit`.
- **Default Mantle device:** RX 7900 XTX (RADV). The machine also has an RTX 4000 Ada.

## Read first

- `plans/GUARDRAILS.md`, `plans/lava-dnn.md` (export pipeline), `plans/STATUS.md`, `plans/whisper-decoder.md` (autoregressive decode with a KV cache).
- **Closest runners:**
  - `KokoroRunner`: TTS, vocoder, artifacts.
  - `WhisperRunner`: encoder plus autoregressive decoder.
  - `BonsaiRunner`, `HorizonRunner`: LLM decoding.
  - `Trellis2Runner`, `Hunyuan3DRunner`: flow matching.
- **Tooling:** `tools/export_*.py`, `tools/artifacts.py`, `tools/baseline_all.py` and `tools/bench_all.jl`, which define how references and benchmarks are done.

## PyTorch reference

- **Code:** `~/tools/index-tts`, git `d9e41aa` (2026-09-30), which already has `indextts/infer_v2_5.py`. Its venv `~/tools/index-tts/.venv` was built for v2, so check it covers 2.5. Run export and ref-dump scripts with that venv's python, the way the TRELLIS.2 tools do.
- **Weights:** `hf download IndexTeam/IndexTTS-2.5`. Put them on the root disk (about 300G free), not under `/sim`, which has about 44G free. The v2 weights are at `/sim/hf-cache/IndexTTS-2`, linked as `checkpoints_2`. If 2.5 shares files with v2 (w2v-BERT 2.0, semantic codec, CAMPPlus, BigVGAN), reuse them rather than downloading again. `w2v-bert-2.0/conformer_shaw.pt` is never loaded.
- **German:** the community LoRA `sharrnah/index-tts-2.5-german` (r=64 on all 24 GPT layers plus the text embeddings and heads, with its own German text frontend). The runner must be able to apply it, for example merged at load, and use its frontend. German is the reason for this port.

## Model facts from the research (verify against the code)

- **Pipeline:**
  1. Text frontend.
  2. GPT, about 0.8B, autoregressive. It produces semantic tokens conditioned on speaker and emotion.
  3. s2mel flow matching, Zipformer-based in 2.5.
  4. BigVGANv2 vocoder.
- **Reference audio:** w2v-BERT 2.0 features, the MaskGCT semantic codec and a CAMPPlus speaker embedding.
- **Emotion inputs:**
  - A second clip, blended in with `emo_alpha`.
  - An 8-float vector: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
  - Text, converted to a vector by the bundled Qwen emotion model.
  - The 2.5 Hugging Face card documents only the vector and text paths, while the README keeps the clip path. Confirm in `infer_v2_5.py` that the clip path exists in 2.5.
- **Other controls:** `duration_factor` 0.5–2.0 and the sampling parameters. Languages are zh, en, ja, es and ar, with Japanese kana overrides like `<上手|じょうず>`.
- **Upstream speed:** about 6 GB VRAM and RTF 0.21 in bf16 on an RTX 4090, both self-reported. A vLLM recipe exists.
- **Known weakness:** speaker consistency collapses when generated audio is reused as a reference.

## Definition of done

1. **Stage-wise parity** against references dumped by a new `tools/dump_indextts25_refs.py`:
   - frontend tokens exact;
   - reference features and codes;
   - GPT logits and greedy tokens identical over a full utterance;
   - s2mel output within tolerance for fixed noise;
   - vocoder waveform.
2. **End to end:** the same text, references and seed give audio matching the reference. Check intelligibility with `WhisperRunner` and speaker similarity against the reference clip.
3. **Every control works:** speaker clip, emotion clip with `emo_alpha`, emotion vector, emotion text, `duration_factor`, sampling, and long text split the way upstream does. The German LoRA works, both base-model and LoRA outputs checked against PyTorch.
4. **Shipping:** tests in `IndexTTS25Runner/test`, with artifacts bound like the other runners. Follow the `gen/` conventions: delete tarballs once uploaded and shards once bound, since `/sim` is nearly full.
5. **Top speed, measured like `bench_all.jl` / `baseline_all.py`:** same card, same dtype, clock ramp, median (GUARDRAILS §5 and §6).
   - PyTorch eager on the RTX 4000 Ada is the reference to beat.
   - Report JuliaVision on both the RTX 4000 Ada and the RX 7900 XTX.
   - Give RTF per stage. The GPT decode will dominate, so focus on the KV cache, fused sampling and launch overhead, then the flow-matching steps and BigVGAN.
6. **Report:** keep `plans/projects/indextts25/REPORT.md` current, with parity numbers, the RTF table against PyTorch, and anything still missing.
