# KokoroRunner

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text to speech on Lava:
54 voices from one set of weights, 24 kHz. Apache-2.0. The model runs on the
device; the host turns text into phonemes with a Julia port of misaki's lexicon
lookup.

## Speaking

```julia
using KokoroRunner

k = Kokoro()
audio = speak(k, "The quick brown fox jumps over the lazy dog."; voice = "af_heart")
# Vector{Float32}, mono, SAMPLERATE = 24000

pronounce!(k, "Vulkan" => "vˈʌlkən")      # teach it a word
audio = speak(k; phonemes = phonemize(k, "Hello world"), voice = "bm_george", speed = 1.2)
```

`voices(k)` lists them: `af_*` and `am_*` are American, `bf_*` and `bm_*`
British, and the English lexicon covers those four; any other voice wants
`phonemes =` directly. The lexicon has 90k words and no proper nouns, and a word
it does not know is **dropped** with a warning rather than guessed, which is what
`pronounce!` is for. `speed` scales the predicted durations, so the pitch stays.
`trim = true` cuts leading and trailing silence.

### Word stress

Inline controls use the Kokoro/Misaki demo syntax, e.g.
`"A wider opening, [not](-1) just [better](+2) glass."`:

- `[word](-1)` demotes primary stress to secondary.
- `[word](-2)` removes stress marks.
- `[word](+1)` promotes secondary stress; an unmarked word gains secondary stress.
- `[word](+2)` can give an unmarked word primary stress.
- `0`, `+0.5`, and `-0.5` are also accepted, following Misaki's stress rules.

These are stress levels, not volume controls: raising a word already bearing
primary stress need not change it. Labels can contain multiple words; the level
then applies to each word. The markup is not spoken. Other link targets are
rejected; phoneme overrides still use `pronounce!` or `phonemes =`.

## Examples

[`docs/examples/speech.jl`](../docs/examples/speech.jl), written as MP3 through
the ffmpeg the package already depends on:

| voice | text | audio |
|---|---|---|
| `af_heart` | The fox in these pictures was never photographed. A diffusion model drew it, and every other model on this page took it from there. | [line1.mp3](../media/kokoro/line1.mp3), 7.1 s |
| `bm_george` | Each network runs on the graphics card through Vulkan, from one Julia process, with no Python anywhere. | [line2.mp3](../media/kokoro/line2.mp3), 6.9 s |

**2.1 s** a line on a Radeon 8060S (RADV, 2026-09-30), about 3.4x real time.
The first call at a new length also compiles. [WhisperRunner](../WhisperRunner/README.md)
transcribes both back word for word.

## Assets

| artifact | size | |
|---|---|---|
| `kokoro` | 356 MB | graphs, weights, voices, vocabulary and lexicon |
| `kokoro-refs` | 4.2 MB | PyTorch audio for the parity test |
