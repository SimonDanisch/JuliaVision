# WhisperRunner

[Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
speech recognition on Lava: a log-mel front end on the device, the encoder, and a
4-layer decoder with a KV cache, inside a port of whisper.cpp's 30 s window loop.
MIT.

## Transcribing

```julia
using WhisperRunner

w = whisper()                                      # both halves, fp16
text, segs = transcribe(w, "talk.mp4")             # anything ffmpeg opens
text, segs = transcribe(w, pcm; language = "en")   # 16 kHz mono samples
for s in segs
    println(s.start, " - ", s.stop, ": ", s.text)
end
```

`language = nothing` detects it once from the first window and holds it.
`DecodeOptions` carries whisper.cpp's knobs: the temperature ladder, the
compression and log-probability thresholds, the no-speech threshold and
`timestamps`. `transcribechunk(w, tokenizer(), pcm)` decodes one window without
the loop, for streaming.

A 3 s chunk costs what a 30 s one does: the encoder's extents are baked.

## Examples

The two lines [KokoroRunner](../KokoroRunner/README.md) speaks, transcribed back:
[`docs/examples/speech.jl`](../docs/examples/speech.jl).

| audio | transcript |
|---|---|
| [line1.mp3](../media/kokoro/line1.mp3) | The fox in these pictures was never photographed. A diffusion model drew it, and every other model on this page took it from there. |
| [line2.mp3](../media/kokoro/line2.mp3) | Each network runs on the graphics card through Vulkan from one Julia process with no Python anywhere. |

Both word for word what HF transformers returns for the same files. **1.0 s**
and **0.7 s** a file on a Radeon 8060S (RADV, 2026-09-30), ffmpeg decode
included.

The window loop follows whisper.cpp's `whisper_full_with_state`, and two of its
rules were missing until 2026-09-30. A window whose last segment closes with
nothing after it skips the rest of its span (`single_timestamp_ending`); the loop
used to step to that timestamp and decode the 0.21 s of silence behind line 2 as
a window of its own, which came back as "Yeah." or "考慮". And with
`timestamps = false` the window's text is one segment; the transcript used to be
empty.

## Assets

| artifact | size | |
|---|---|---|
| `whisper` | 1.19 GiB | the fp16 encoder |
| `whisper-decoder` | 386 MiB download | decoder, tokenizer, generation config |
| `whisper-fp32` | 2.37 GiB | the fp32 encoder, `whisper(; precision = :fp32)` |
