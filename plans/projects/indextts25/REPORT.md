# indextts25 — report

IndexTTS-2.5 on Lava as `IndexTTS25Runner`. The state of the port: what the model
is, what matches PyTorch and how closely, what is fast, what is missing.

Reference: `~/tools/index-tts` at d9e41aa (`indextts/infer_v2_5.py`), weights
`IndexTeam/IndexTTS-2.5` at c39ce5b plus the shared IndexTTS-2 files, German
checkpoint `sharrnah/index-tts-2.5-german` at a5ac4e1. References come from
`tools/dump_indextts25_refs.py`, exports from `tools/export_indextts25.py`.

## The model, as the code has it

Several points differ from the research summary in the brief; the code is the
authority.

| stage | upstream | notes |
|---|---|---|
| text | tiktoken BPE (`multilingual_zh_ja_yue_char_del.tiktoken`), `<\|lang\|>` prefix, segments of ≤ 120 tokens | zh/en normalise through WeTextProcessing, ja/es through NeMo; German has its own frontend and disables upstream's |
| speaker | CAMPPlus on a Kaldi fbank (16 kHz) → `style` (192) | the GPT's speaker input is `spk_emb_proj(style)`, not a Conformer over w2v-BERT |
| reference features | w2v-BERT 2.0, `hidden_states[17]` (layers 0..16 of 24), normalised | 50 frames/s |
| emotion | Conformer + one-latent Perceiver over w2v-BERT features → `emo_layer(emovec_layer(·))` (1280) | the second-clip path exists in 2.5: `merge_emovec(spk, emo, alpha)` = `base + α (emo − base)` |
| GPT | GPT-2, 24 layers, 1280 wide, 20 heads; prefix `[spk + emo, 0, 0][text + pos + lang][start_mel]` | semantic codes at 25/s; default decode is **beam search with sampling, 3 beams**; the German card uses 1 beam |
| codec | `EnhancedCodec.decode`: codebook → Vocos decoder → ×2 nearest → conv | codes → (2n, 1024) |
| s2mel | length regulator to `int(2n · 1.72 · duration_factor)` frames, then CFM: **DiT** (13 layers, 512) + WaveNet head, 25 Euler steps, CFG 0.7 | not a Zipformer; the same DiT as IndexTTS-2 |
| vocoder | BigVGAN v2, 22.05 kHz, 80 bands, ×256 | |

Upstream quirks a port has to reproduce, each found by a mismatch:

- **Mel positions skip 1.** The start token sits at mel position 0; the j-th
  generated token gets `j + 1` (`GPT2InferenceModel.forward` reads
  `attention_mask.shape[1] - mel_len`, and the mask has grown by one).
- **Text ids 0 and 1 are stripped** before the prefix is built
  (`prepare_gpt_inputs`' `valid_mask`), then `[0, text, 1]`. The trailing pad id
  1 is always stripped, which leaves upstream's prefix left-padded with a masked
  row; masked, it changes nothing, so the port's prefix has none.
- **Repetition penalty sees the fake prompt.** HF's input ids are 1s and the
  start token, so ids 1 and 8192 are penalised from the first step. On raw
  logits for greedy and sampling, on log-softmax scores in beam search.
- **The German checkpoint is a merged GPT**, not an adapter: the whole
  `gpt.pth` is replaced (`load_checkpoint` over the base model). Its frontend
  lowercases, expands numbers and dates, and runs with `lang = "de"` (row 2 of
  the language embedding, trained by the fine-tune).

## Parity

fp32 throughout, TF32 off in the reference, on the RX 7900 XTX (RADV). Each graph
is checked against PyTorch on the recorded inputs of case `en_greedy`; max and
mean |Δ| relative to the reference's absmax.

| graph | ops | max | mean |
|---|---|---|---|
| `w2v` (layers 0..16) | 872 | 2.1e-5 | 2.8e-7 |
| `campplus` | 1884 | 3.7e-6 | 1.0e-6 |
| `emo` | 261 | 1.7e-6 | 3.9e-7 |
| `regulator` | 67 | 7.3e-7 | 5.9e-8 |
| `codec` | 96 | 9.0e-7 | 1.3e-7 |
| `dit` (one estimator call, batch 2) | 731 | 1.2e-6 | 1.5e-7 |
| `bigvgan` | 2260 | 9.2e-5 | 1.8e-6 |
| `gpt_prefill` logits | 578 | 1.9e-6 | |
| `gpt_decode`, 102 steps | 610 | 1.9e-6 worst step | |

**Greedy decode: all 103 tokens of `en_greedy` identical** to PyTorch's, with
the penalty applied on the host.

What the wrappers in `export_indextts25.py` change, each checked against the
upstream module before tracing:

- w2v-BERT stops after layer 16 and computes the `relative_key` term as 73 dot
  products gathered by distance instead of a (t, t, 64) table per layer.
- CAMPPlus's ceil-mode segment pooling is a running sum indexed by `arange`, its
  unbiased std is written out, and its last (non-affine) BatchNorm gets a unit
  scale and zero shift.
- The regulator's `GroupNorm(1, C)` is written out; its nearest-neighbour
  upsampling takes a host-computed index (ATen's `floorf(dst · in/out)`).
- The DiT drops the all-true key mask, folds its 19 weight norms, and pads with
  flipped slices instead of `reflection_pad1d`.
- The GPT's attention is written out; the decode reorders beams with
  `k_cache[:, beam_idx]`.

## DNNKernels changes this needed

| change | file | what was wrong |
|---|---|---|
| `foldrelu` folds a `gelu` only through views that keep every element | `foldrelu.jl` | through a `split` it activated the whole matmul; GEGLU's `gelu(gate) * x` came out 59% off |
| the portable 1-D convolution applies a folded activation | `emit.jl` | it returned after the body, so CAMPPlus's relus vanished (1e12 off) |
| `log1p.default` in the unary table | `ops.jl`, `fuse.jl` | Mish's softplus |
| `slice.Tensor` without attributes is `dim = 0` | `emit.jl` | the schema default |
| `Eq`, `Ne`, `IntTrueDiv`, `CeilToInt`, `FloorToInt` in shape expressions | `graph.jl` | sympy forms from strided convolutions and `arange` |

## Speed

### Planning investigation, 2026-10-23

Restored the saved setup in `experiments/indextts25/setup.jl`. The session uses
the RX 7900 XTX, `precision = :fast` (fp16 codec/DiT/GPT, fp32 BigVGAN and clip
encoders). The whole synthesis is one graph: codec, regulator, Euler loop and
vocoder. It is planned, recorded, run and freed per call.

**The Euler loop is planned once.** At 25 steps there are 4,296 declared passes
(4,297 with the host-update pass), versus 4,294 at 1 step, and 3,802 transients
in either case. Both reserve 98.45 MiB of transient storage. Loop count does not
explain the multi-second planning cost.

On the 560-frame `en_greedy` synthesis, the planner made 2,354,989 alias handover
pairs. Four host costs dominated: repeatedly rebuilding resource lists while
scoring ready passes, dynamically reading heterogeneous transient metadata in
the pairwise alias walk, hashing long `UnitRange`s while resolving slices, and
formatting usage type names to sort two-element barrier wait lists.

Targeted changes in `Mantle/src/phases.jl` and `src/sync/transition.jl`:

- Cache each pass's resource ids during scheduling; skip memory scoring when
  the higher-priority declaration-order bits already rule out a candidate.
- Read transient lifetimes and byte ends into integer vectors before alias
  comparisons. Keep every handover and its final sorting.
- Resolve existing slice segments by `(first, last)` rather than hashing the
  whole range again.
- Deduplicate hazards by their wait sets, without rendering type names.

Representative warm phase times on the same graph/session, seconds (host
diagnostics, not clock-gated GPU benchmark results):

| phase | before | after |
|---|---:|---:|
| DAG | .0047 | .0042 |
| scheduling | .687 | .057 |
| liveness | .0062 | .0045 |
| placement | .0506 | .0506 |
| aliasing | 1.377 | .165 |
| barriers | .665 | .306 |
| pipelines | .753 | .055 |
| total | 3.54 | .64 |

Warm **fresh-call** synthesis timings: declaration .35–.43 s, `Plan` .59–.63 s,
recording .044 s, execution including PCM readback .665–.681 s. About 1.7 s for
4.063 s of audio, synthesis-only RTF about .42; this excludes the GPT and clip
conditioning. The earlier ~5–6 s fresh-call synthesis was predominantly host
planning. No utterance-length cache or padding was introduced.

Validation:

- Original versus optimized planner: all dependencies, scheduled pass ids,
  offsets, 2,354,989 handover pairs and transition records identical on the
  synthesis graph.
- Ran the complete fp16 synthesis with the original scheduling/alias/barrier
  functions from Mantle HEAD, then the optimized functions: **PCM bit-identical**.
- Compile golden/scheduling tests, repeat tests and the 3-D repeat regression:
  **104 assertions pass**, including new checks for mixed-size alias lifetimes,
  distinct barrier wait sets and replanning partially overlapping slices.
- Added reproducible `planningprobe`, `synthesisprobe` and `recordedvoice` in
  `experiments/indextts25/planning.jl`. Warm the probes once before timing.

Do not confuse precision error with the planner change. Using the recorded
codes, conditioning and Euler noise, current fp32 synthesis differs from the
recorded fp32 BigVGAN waveform by max .00277 / mean 9.87e-6 of its absmax.
The `:fast` synthesis differs by max .489 / mean .00607. The fast result is
unchanged by the planner optimization; it is not sample-parity with fp32.

The PyTorch eager baseline files now exist in `gen/indextts25/baseline_*.json`.
On the RTX 4000 Ada, recorded fp16 medians are GPT 735.7 ms, s2mel 235.5 ms,
BigVGAN 83.8 ms; fp32: 1191.5, 627.1, 133.5 ms respectively. The AMD observations
above are **not a same-card speedup comparison**. Full clock-gated Julia tables
on both cards still need to be produced.

### Earlier implementation work recovered from the preceding session

The front ends, tokenizer, German text preparation, runner and device-side GPT
decode are implemented. Audio decode/downmix/libsoxr matched librosa bit for
bit; the recorded text cases and 14 German battery sentences matched upstream
tokens. The handwritten fp32 GPT decode matched the 102 generated greedy codes.
Sampling and beam bookkeeping currently run on the host, reading logits each
step; the cache/transformer decode runs on the device. These observations were
reported in the preceding session; the planning work above did not rerun that
entire battery.

### Export cleanup, 2026-10-23

The exporter now drops unreferenced lifted weight buffers, following nested view
parents so transposed weights stay live. GPT-2's 24 unused 2420×2420 causal-mask
tables and scalar masked biases were present in every GPT export. Decode also
carried the speaker projection, text/position embeddings and language table,
which it never reads.

Applied the same pruning to the four existing exports: 48 tensors / 134.05 MiB
removed from each prefill, 53 tensors / 433.91 MiB from each decode, **1,135.92 MiB
saved in total**. Every retained tensor was checked byte-for-byte on rewriting
the safetensors. The pruned English prefill then matched the resident unpruned
model's logits and both KV caches bit-for-bit; logits remain within 1.45e-6 of
the PyTorch reference's absmax. `tools/test_indextts25_export.py` covers nested
weight views and weights exposed directly at the graph interface.

## Recovery and packaging, 2026-10-23 evening

The chat restart lost run `r50` during artifact packing, after the first German
GPT shard. The source, graph exports and completed archives survived. Packing
now verifies and reuses completed components, writes archives to `.partial`
files and renames them only when complete. All **16 lazy artifact bindings**
are now installed locally, with SHA-256-checked archives under
`/home/simon/gen/indextts25/artifacts`. Release URLs are prepared; the archives
have not been uploaded yet.

Recovered features now present in the package:

- Native Julia WeText FST interpreter, English/Chinese normalization and the
  upstream pronunciation/technical-term protection rules. The packaged frontend
  suite passes **431 assertions**; host rules pass **17 assertions**.
- Native Qwen3-0.6B emotion classification, including tokenizer and the
  checkpoint's prompt template, exposed by `emotiontext` and `speak`'s
  `emotion_text` / `use_emotion_text` keywords. Full token-generation and
  end-to-end packaged validation are running; results are not yet claimed here.
- `indextts25()` loads the artifact metadata and downloads component weights on
  demand, including separate base/German GPT shards. Normalizer and Qwen loading
  are lazy; the latter is loaded only for text-driven emotion.

The packaged runner suite saves completed PCM metrics to
`gen/indextts25/validation/parity.jsonl` and each waveform beside it; the full log
is `gen/indextts25/validation.log`. This persists completed evidence independently
of the Julia/chat session. The suite tests the packaged loader rather than the
original local export directory.

Upstream's `nemo_tn.py` intentionally passes Japanese through unchanged. Spanish
uses optional NeMo; that dependency is absent in the recorded Python environment,
so it also passes through there. The current Julia path reproduces that fallback;
Spanish normalization with NeMo installed remains outside this validation.

## Recovery, 2026-10-24

Run `r4` from the preceding chat disappeared before saving either remaining
German case. The user journal records graphical-session shutdown at 23:28 on
October 23, when `validation-resume.log` stopped. This establishes a session
shutdown, not its cause; the saved log does not show a failed assertion.

The ten completed packaged comparisons remain in
`gen/indextts25/validation/parity.jsonl`: all nine base-model cases plus German
greedy. Mean PCM errors range from 0.0796 to 2.3491 int16 levels, peak errors
from 4 to 218, with matching output lengths. Separately saved Qwen evidence in
`validation_emotion.json` has all 75 generated tokens identical for each of four
prompts.

The resumed tests write to the workspace's
`experiments/indextts25/recovery/`, with `parity.log`, `guard.log` and
`validation/progress.jsonl`. Host rules (31) and frontend checks (458) passed
again. The two remaining German GPU checks passed all six assertions:

| case | PCM samples | mean absolute error | peak absolute error |
|---|---:|---:|---:|
| `de_sample` | 78,336 | 0.211423 | 17 |
| `de_emoclip` | 45,568 | 0.071059 | 3 |

Both lengths match upstream. Combined with the previous ten saved cases, all
12 packaged audio cases now pass the PCM tolerances. The recovery run completed
in 5m24s, including cold loading and compilation; this is not a benchmark.

The package test harness now filters cases before loading models,
records load/voice/speak checkpoints, and writes each WAV by temporary-file
rename before appending its metrics. `tools/check_indextts25_audio.jl` now
contains the recovered upstream-versus-Julia quality comparison, including text
normalization for WER and per-clip saved results.

### Intelligibility and speaker checks

Whisper large-v3-turbo and CAMPPlus compared five cases across upstream,
`:exact` and `:fast`. All 15 results are saved in
`experiments/indextts25/recovery/validation/quality-comparison.jsonl`.
German dates and numbers are normalized before WER scoring.

| case | WER (all three variants) | speaker cosine upstream | exact | fast |
|---|---:|---:|---:|---:|
| `en_greedy` | 0 | .738084 | .738008 | .791310 |
| `en_emotext` | .100 | .623546 | .623522 | .642217 |
| `de_greedy` | 0 | .780373 | .780430 | .771033 |
| `de_emoclip` | 0 | .534020 | .533827 | .543963 |
| `de_sample` | .125 | .760575 | .760524 | .767942 |

Exact-mode transcripts are identical to upstream for all five cases; the largest
speaker-cosine difference is .0001932. The two nonzero WERs occur in upstream
too: Whisper hears “was” instead of “is” in `en_emotext`, and “Helgenschirm”
instead of “Regenschirm” in `de_sample` (fast yields “Hälgenschirm”). The angry
emotion-clip case's lower speaker score is likewise present upstream.

Fast mode has no additional word errors in these five samples. Its speaker
cosine differs by −.00934 to +.05323 from upstream. These fast calls use the
runner's seeded noise rather than the recorded upstream noise, so this is an
end-to-end sample-quality check, not an isolated precision A/B. All five fast
sample counts match upstream. Automated scores do not establish perceptual
equivalence; the waveform deviation reported above remains real.

All **16 model artifacts are published** in
[indextts25-v1](https://github.com/SimonDanisch/JuliaVision/releases/tag/indextts25-v1)
as of October 24, 2026, following the user's upload approval. The archives total
8,930,193,501 bytes. Every remote size and GitHub SHA-256 digest matches the
verified local manifest and `Artifacts.toml`; all 16 public download URLs return
HTTP 200 with the expected size.

A fresh Julia artifact-install check used an empty, isolated artifact directory
and downloaded the metadata and CAMPPlus weights directly from GitHub, with
the package server disabled. Both compressed SHA-256 and unpacked tree hashes
pass. The downloaded tokenizer, CAMPPlus graph and all 939 weight tensors load
successfully. Existing packaged GPU parity remains applicable to these identical
artifact bytes. Upload and download verification records are saved under
`experiments/indextts25/recovery/`.

After publication and fresh-download verification, the 16 packing tarballs and
four redundant GPT shard files were removed, freeing **13,523,715,245 bytes**.
The shard copies were checked against their installed artifacts before removal.

### Fresh-call benchmarks, October 24 (RX 7900 XTX)

Seven clock-gated samples per stage, with rejected samples retained in the raw
JSONL files under `experiments/indextts25/recovery/`. Calls include declaration,
planning, recording, execution and output readback. Complete calls use greedy
decoding; `cached_voice` reuses a `Voice`, while `complete` decodes and conditions
the reference clip on every call. Model loading and compilation are warmed out.

| stage | exact ms | exact RTF | fast ms | fast RTF |
|---|---:|---:|---:|---:|
| `voice` | 1362.2 | 0.335 | — | — |
| `gpt` | 402.3 | 0.099 | 340.1 | 0.084 |
| `gpt_beam` | 711.4 | 0.184 | 646.9 | 0.167 |
| `s2mel` | 1312.4 | 0.323 | 752.5 | 0.185 |
| `bigvgan` | 1025.8 | 0.252 | — | — |
| `synthesis` | 2868.7 | 0.706 | 2144.1 | 0.528 |
| `cached_voice` | 2891.9 | 0.712 | 2240.4 | 0.551 |
| `complete` | 4303.3 | 1.059 | 3952.0 | 0.973 |

The greedy utterance is 4.06349 s; sampled-beam RTF uses its own generated
duration. Voice conditioning and BigVGAN remain fp32 in fast mode and were not
measured again for that configuration. The runs share one Julia process but
are not interleaved; these are throughput observations, not precise precision
speedup ratios. Exact-mode BigVGAN spread is 38.3%, and cached-call spreads are
20.1% (exact) / 11.6% (fast). The raw files include all clock and GC samples.

An initial PyTorch timing run was provisional: another Julia process was then
observed at 98% SM utilization, and its overlap with the run was unknown. Those
numbers in `recovery/baseline_fp32.json` are superseded by the uncontended runs
below. The other process was left untouched and subsequently exited.

The NVIDIA clock probe in `tools/measure.jl` now samples continuously during a
call (`nvidia-smi --loop-ms=50`). A single probe at the start of a fresh-graph
call can sample only host planning and miss GPU execution. The loaded benchmark
verified the continuous probe: every accepted Julia sample reached 2,265 MHz.

### RTX correctness recovery

The benchmark's preflight PCM check exposed a separate correctness failure:
89,600 samples with mean PCM error 4,070.7 and peak 44,188. All seven neural
graphs passed individually on recorded inputs, and GPT produced the same 102
greedy tokens. The 25-step flow loop also passed when supplied with the recorded
packed conditioning. The failing operation was `stackmu!`: selecting between
prompt and generated-feature pointers wrote an entirely zero semantic tensor
on RTX. A small reproducer with poisoned output and distinct input values
reproduced the failure. The generated SPIR-V contains a pointer phi followed by
pointer arithmetic; the precise compiler-level cause remains unproven.

`stackmu!` now loads two valid, clamped input values and selects the value.
This portable implementation passes all three model-free packing cases on
both RX 7900 XTX and RTX 4000 Ada, checking every output element and both CFG
batches. The full RTX English greedy call now passes: 89,600 samples, mean
PCM error **0.533002**, peak **34**. All **12 RTX audio cases now pass**, including German, emotion from text,
beam search and long-text splitting. Three cases initially encountered a VRAM
budget error while the other Julia job was resident; all passed after it exited.
The host checks (31), frontend checks (458), and packing checks (3 per GPU)
also pass. Reproduction IR, stage checks and metrics are saved under
`experiments/indextts25/recovery/`. The standalone packing reproducer asserts
two distinct compiled pipelines and confirms that the old implementation passes
on AMD but fails on RTX; the replacement passes on both.

| RTX case | samples | mean PCM delta | peak PCM delta |
|---|---:|---:|---:|
| `en_greedy` | 89,600 | 0.533002 | 34 |
| `en_emoclip` | 67,584 | 0.260106 | 22 |
| `en_emovec` | 75,520 | 0.849113 | 48 |
| `en_duration` | 82,432 | 0.194815 | 12 |
| `en_long` | 519,994 | 0.473284 | 157 |
| `zh_greedy` | 104,704 | 0.963172 | 48 |
| `de_greedy` | 124,160 | 0.357257 | 22 |
| `de_sample` | 78,336 | 0.357588 | 22 |
| `de_emoclip` | 45,568 | 0.223293 | 6 |
| `en_emotext` | 63,232 | 0.734059 | 63 |
| `en_beam` | 65,024 | 0.219796 | 30 |
| `en_beamsample` | 60,672 | 0.514554 | 30 |

### Uncontended RTX stage and complete-call benchmarks

Seven samples per stage; values are **milliseconds / RTF**. Fresh Julia graphs
include declaration, planning, recording, execution and readback. Models and
kernel compilation are warm. The PyTorch voice row was remeasured with file
decoding included, matching the scope of `voice(path)`.

| stage | PyTorch fp32 | Julia exact | Julia fast |
|---|---:|---:|---:|
| `voice` | 137.4 / 0.034 | 1471.5 / 0.362 | — |
| `gpt` | 822.6 / 0.202 | 745.2 / 0.183 | 479.3 / 0.118 |
| `gpt_beam` | 1009.0 / 0.253 | 865.2 / 0.224 | 605.8 / 0.157 |
| `s2mel` | 633.3 / 0.156 | 1693.8 / 0.417 | 1247.9 / 0.307 |
| `bigvgan` | 134.3 / 0.033 | 1177.7 / 0.290 | — |
| `synthesis` | 770.8 / 0.190 | 3300.4 / 0.812 | 3025.0 / 0.744 |
| `cached_voice` | 1612.4 / 0.397 | 4132.2 / 1.017 | 3475.2 / 0.855 |
| `complete` | 1735.2 / 0.427 | 5546.1 / 1.365 | 4941.0 / 1.216 |

All **112 Julia timing samples** passed the clock gate (56 exact, 42 fast,
14 repeated exact full calls). Julia samples reached 2,265 MHz; PyTorch stage
clock readings were 1,890–2,265 MHz after a six-second load ramp. Passive
process logs show only this Julia process and the sequential baseline Python
processes as compute jobs, alongside the normal desktop graphics clients.

The sequence was Julia exact → PyTorch fp32 → Julia fast → Julia exact full
calls. Repeated exact medians were 4,117.8 ms cached (initially 4,132.2 ms) and
5,547.6 ms complete (initially 5,546.1 ms): drift below .4%. Precision runs were
not interleaved per sample, so small exact/fast differences are not precise
speedup estimates. Fast remains mixed precision; only exact/fp32 is a matching
precision comparison. Sampled-beam rows use each runtime’s actual generated
duration because the RNG streams differ.

On RTX with matching fp32 precision, a complete Julia call takes **3.20× as
long** as PyTorch; cached-voice calls take **2.55–2.56× as long**. GPT is
competitive; fresh voice conditioning and synthesis account for the large gap. Julia's synthesis
and full calls spend about 7–10% of their time in GC, and include the planner
costs measured earlier. Voice conditioning can be reused with `Voice`; in fast
mode the cached call has RTF .855 on RTX and .551 on RX 7900 XTX. The initial
RTX exact voice spread is 22.3%; fast cached-call spread is 8.5%. These limits
and the wider AMD spreads should accompany any performance claim.

Raw timestamps, clocks, samples, preflight PCM and logs are in
`experiments/indextts25/recovery/nvidia/quiet/`. The original voice-only PyTorch
row in `baseline_fp32.json` excludes file decoding and is superseded by
`baseline_voice_fp32.json`; the table above uses the corrected row.

The RTX fast-mode preflight produced 89,600 samples. Whisper/CAMPPlus evaluated
that saved clip on the RX 7900 XTX: WER **0**, with the same transcript as upstream,
and speaker cosine **.787820** (upstream **.738084**). This adds one RTX-produced
sample to the five-case AMD quality battery; it is not a listening-equivalence
claim. The WAV and scores are in `quiet/validation/`.

### RTX attribution of the performance gap

Three warm phase diagnostics per stage separate graph preparation from recorded
execution. These are diagnostic timings, not a replacement for the clock-gated
benchmarks above. CPU setup includes declaration (and clip decoding for voice),
planning, recording and resource cleanup. Execution includes submission, GPU
completion and output readback; it is not a pure GPU timestamp measurement.

| stage | CPU setup, median s | execution + readback, median s | declared passes |
|---|---:|---:|---:|
| voice | 1.37 | .247 | 4,719 |
| synthesis | 1.55 | 1.811 | 3,972 |
| BigVGAN alone | .91 | .303 | 2,883 |

BigVGAN is part of synthesis; its row must not be added to synthesis. The runner
currently rebuilds and plans the voice and synthesis graphs on every call.
GPT reuses its plans. Roughly three seconds of the profiled full call are CPU
preparation around the acoustic stages. Optimizing the cost of one planning
operation left this repeated work on the steady-state path. Safe plan reuse
must account for input shapes, voice buffers, per-call flow noise, loop state
and memory limits.

The 1.81 s execution/readback bucket is not a pure GPU measurement. These
fresh plans use the default finite submission budget; an unmeasured plan can
submit each pass separately while collecting costs. It is therefore premature
to infer how much kernel time remains from that bucket. Measure replay of one
recorded plan with an infinite budget and graph-declared readback before
attributing the remainder to kernels. These observations come from
`experiments/indextts25/recovery/explain-performance.jl`, with all nine samples
saved in `rtx-phase-diagnostics.jsonl`. No production implementation was changed
for this diagnostic.

## Mantle graph readback futures, 2026-10-24

Implemented `Mantle.Readback(graph_array)` (or `Readback(g, buffer)` for an
external buffer), returning a one-shot future with `fetch`, `wait` and `isready`.
The transfer is a declared dependency in the graph and records into its run;
fetch only waits and makes an owning CPU copy. Ordinary `Array` conversion and
KernelAbstractions' `CPU` keep their existing meaning.

Core owns staging and future lifetime. Unfetched previous results are preserved
before replay overwrites staging, and submitted results survive nonblocking
plan teardown. Pending futures fail on teardown or a failed run. Vulkan records
a native buffer copy into host-cached coherent memory plus a host visibility
barrier. Shared-memory devices use a graph copy kernel; the Metal allocation
path is wired but has not been run on this Linux machine.

Tests cover one submission with queued input stores, multiple outputs, shaped
views, snapshots preceding later writes, replay without recompilation, fetch
after teardown, blocked-task wakeup, cancellation, failures, short element types,
a 25-iteration loop, and refreshing a moved source buffer. Validation and sync
validation are enabled on both RTX and RX GPUs: **453 targeted assertions pass**,
including 53 feature assertions per card; neither device reports validation
messages. The result is saved in `recovery/readback-validation.json`.
The new primitive is in Mantle;
this step does not yet replace the TTS runner's per-call graph construction.

### Runner replay integration (in progress)

The acoustic stages now retain native Mantle plans through DNNKernels'
`RecordedPlan` ownership. Voice plans lease their output buffers to each live
`Voice`, so reusing a shape cannot overwrite another live speaker. Synthesis
replays stage new semantic codes, style and flow noise as queued stores; emotion
clips use the same readback path. GPT prefill, prefix-cache seeding and table reset
are one graph, and each decode step includes its logits readback.

The first integrated RTX check passes 11 assertions: warm voice and synthesis
calls each submit once; GPT submits 103 times for 102 codes (one prefill/seeding
run, then one per decode step); greedy codes match upstream exactly; repeated
fixed-noise synthesis is bit-identical. Complete PCM remains at mean **.533002**,
peak **34** from upstream. Raw checkpoints: `recovery/reuse-check.jsonl`.

This does not yet meet the performance target. An initial unclocked, one-submit
probe puts voice execution near .16 s and synthesis near 1.4 s. GPU timestamps
attribute about 291 ms to loop convolutions, 389 ms to loop matrix products,
230 ms to loop attention and 182 ms to other loop kernels; the vocoder and
one-time work add about 254 ms. These are profiling/provisional timings, not a
new benchmark table. Inspect `recovery/baked-synthesis-profile-timings.json`.
The next experiment compares the dominant FP32 5-tap convolution against
materialized im2col plus staged scalar GEMM, using interleaved A/B and separate
pipeline-count checks.

### FP32 convolution improvement, 2026-10-27

Resumed from the saved replay checks; the earlier Julia process had ended. The
RTX was fully occupied by other jobs, so no new RTX/PyTorch comparison is
reported. The RX 7900 XTX was idle and was used for numerical and relative
performance checks.

The isolated 5-tap 512→1024 convolution's alternative (bounded im2col followed
by Mantle's staged scalar GEMM) passes on AMD as well as RTX. Interleaved
microprobe medians: **1.034→.686 ms** on RX 7900 XTX (34% less time), with maximum
output difference **6.89e-8**. Prior RTX microprobe: **1.982→1.529 ms** (23% less
time). These microprobe timings are screening evidence, not clock-gated model
benchmarks. Both probes check distinct pipeline counts and complete output
coverage.

The Float32 declaration now selects this path for wide, long-reduction dense
convolutions whose im2col fits within 16 MiB. It reuses the existing im2col,
scalar GEMM and convolution epilogue kernels; kernel weights and activations
remain Float32. Native and cooperative-matrix paths retain their precedence.

A complete fixed-noise synthesis A/B on AMD, alternating both orders, passes:
**.960→.896 s** (about 7% less time), with mean waveform difference **.0801 PCM
levels**, peak **13.5** before conversion to Int16. The new output also passes
against recorded upstream PCM: mean **.2865**, peak **71**, over 89,600 samples
(the old path: .2725 / 57). These initial wall times are provisional. Detailed
rows: `recovery/flow-convolution-ab-amd.json`.

The subsequent interleaved, clock-gated run confirms **.9769→.9115 s**, a
**6.7% latency reduction** (1.072× throughput). All five samples per arm pass
the clock gate; spreads are .66% and 1.19%, respectively. Observed clocks are
2,528–2,880 MHz. Raw samples: `recovery/flow-convolution-clocked-amd.json`.
This is an acoustic-stage A/B on AMD, not a complete-call PyTorch speed match.

Eight additional scalar-convolution assertions pass against a Float64 CPU
reference, covering two-image layout, non-tile-sized output, stride, dilation,
padding, bias, ReLU and one-submission readback. The full synthesis comparison
adds 12 passing assertions and the upstream PCM comparison four.

## Remaining work

- Further performance work should target fresh-graph planning/allocation and
  acoustic execution; the measured RTX throughput remains below PyTorch.
- Broader listening coverage remains useful beyond the five automated quality
  comparisons above.
- Device-side sampling/beam bookkeeping, if profiling warrants it.
- The README documents fast precision's waveform deviation and the limits of
  the automated quality checks; no claim of perceptual equivalence is made.
