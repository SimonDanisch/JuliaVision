# IndexTTS25Runner

IndexTTS-2.5 speech synthesis in Julia, with a separate voice clip and emotion
clip, eight emotion weights, or emotion inferred from text. Output is mono
22,050 Hz `Vector{Int16}` PCM. The German checkpoint includes its own text
normalization and GPT weights.

```julia
using IndexTTS25Runner

tts = indextts25(; german = true)
v = voice(tts, "speaker.wav")
try
    pcm = speak(tts, "Verschwinde aus meinem Haus!";
                voice = v, emotion = "angry.wav", alpha = 0.8)
    pcm2 = speak(tts, "Endlich bist du wieder da.";
                 voice = v, emotion_text = "happy and relieved")
finally
    close(v)
    close(tts)
end
```

Models are Julia artifacts downloaded on first use from the
[model release](https://github.com/SimonDanisch/JuliaVision/releases/tag/indextts25-v1)
and reused across sessions.
The base and German GPTs share all seven acoustic networks. The Qwen emotion
classifier loads only when `emotiontext` or the text-emotion options are used.
A `Voice` retains a clip's conditioning for reuse across many utterances.

`precision = :fast` uses fp16 GPT decode, codec and flow-model arithmetic, with
fp32 clip encoders, GPT prefill and vocoder. `precision = :exact` uses fp32
throughout and is the configuration tested against upstream's fp32 PCM. Fast
precision changes the waveform; per-network agreement does not imply identical
audio. See the [validation report](../plans/projects/indextts25/REPORT.md).
Five English/German quality cases have the same Whisper word-error rates in
upstream, exact and fast modes. Exact-mode speaker scores closely match upstream;
fast-mode scores vary. These automated checks do not establish perceptual equivalence.

`Sampling()` follows upstream's sampled beam search with three beams. The
German default uses one beam and temperature 0.7. For deterministic greedy
decoding, pass `Sampling(; do_sample = false, num_beams = 1)`. Sampling and flow
noise are reproducible within this runner; the RNG stream differs from CUDA's.
Reference parity tests inject the recorded flow noise and use the same
counter-based sampling stream in both runtimes.

Useful `speak` keywords:

- `emotion`: an emotion clip path or a precomputed `emotionclip(tts, path)`.
- `weights`: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
- `emotion_text`: a separate emotion description; `use_emotion_text = true`
  classifies the spoken text when no description is supplied.
- `alpha`: the emotion strength; vectors are scaled as upstream scales them.
- `duration`: the upstream duration factor, ordinarily 0.5–2.0.
- `lang`: defaults to `"de"` for the German model and `"en"` for the base model.
- `maxtokens`: segment limit, default 120; `interval`: silence between segments
  in milliseconds, default 200.
- `normalize`: English/Chinese normalization, default true. German always uses
  the checkpoint's frontend. Japanese passes through; optional NeMo Spanish
  normalization is not included.

Text emotion takes precedence over weights, and weights take precedence over
an emotion clip. Text is normalized and split before generation; long outputs
are joined with silence and the final 20 ms are faded out.

Run `test/runtests.jl` in an environment containing this package. The default
suite exercises packaged English, German and Chinese references on the GPU.
Set `INDEXTTS25_TEST_GPU=false` for frontend tests only, or
`INDEXTTS25_TEST_DEVICE=4000` to select the RTX. Set `INDEXTTS25_TEST_OUTPUT` to
a directory to save comparison metrics and waveforms after each completed case.
To resume just the German cases, set `INDEXTTS25_TEST_CASES=de_sample,de_emoclip`
and `INDEXTTS25_TEST_QWEN=false`. Unselected models are not loaded. The output
directory also contains `progress.jsonl` with load, voice and synthesis checkpoints.

`tools/check_indextts25_audio.jl` compares the saved waveforms with upstream
using Whisper transcription and CAMPPlus speaker embeddings. It records both
results so recognition errors shared with upstream are visible.

The bilibili Model Use License Agreement applies to the model and derivatives.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and [APACHE_LICENSE](APACHE_LICENSE).
