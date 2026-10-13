# HorizonRunner

[K2 Horizon 32B](https://huggingface.co/IFM/K2-Horizon-32B) (IFM / MBZUAI), a
dense grouped-query decoder, on Lava: greedy generation with int8 weights (32.4
GiB on the device instead of 65) and a 4096-slot KV cache. Apache-2.0.

**Local only.** The export is 65 GB and is not an artifact, so `horizon32b`
needs `dir` pointing at one (`tools/export_horizon32b.py` writes it). There is no
Julia tokenizer either: ids go in and ids come out, and
`tools/horizon_tokenize.py` wraps the checkpoint's own `transformers` tokenizer
over JSON on stdin.

## Generating

```julia
using HorizonRunner

m = horizon32b(; dir = "gen/graphs/horizon32b")   # int8, recorded plans; ~35 s
ids = [...]                   # 0-based, from tools/horizon_tokenize.py {"chat": "..."}
out = generate(m, ids; maxnew = 160, stop = (1, 250019))
```

Prompts up to 4096 tokens, prefilled in buckets of 16 to 512.
`verbose = true` prints the prefill time and ms a token.

## Example

[`docs/examples/horizon.jl`](../docs/examples/horizon.jl), which drives the
tokenizer through the repo's Python environment. Horizon reasons before it
answers, and 160 tokens is where this stopped:

> **Explain in two sentences what a KV cache is for.**
>
> The user asks: "Explain in two sentences what a KV cache is for." So they want
> a concise explanation in two sentences. [...] Let's craft.
>
> Sentence 1: The KV cache stores the key and value vectors computed for each
> token during the forward pass of a transformer, so that when generating
> subsequent tokens the model can reuse them instead of recomputing them.
> Sentence 2: This dramatically speeds up autoregressive inference and reduces
> redundant computation, making real‑time generation feasible.

160 tokens after a 21-token prompt in 36.7 s, **4.4 tokens/s** all in, on a
Radeon 8060S (RADV, 2026-09-30). `tools/horizon_bucket_validation.md` recorded
154 to 164 ms a token for decode alone, level with llama.cpp.

Until 2026-09-30 this model did not run at all. `fusemaskedattention` turns its
masked attention into `fused.maskedattention`, whose only implementation went
with the eager op library on 2026-09-23; DNNKernels declares it again now. The
test suite missed it because it runs a tiny random model the fusion never fires
on.

## Q4_K, drafts and the weight cache

```julia
m = horizon32b(; dir, quantize = :q4k_m, cache = "gen/q4kcache/horizon32b-q4km")
out = generate(m, ids; maxnew = 128, stop = (1, 250019), draft = 7)
```

Measured on the 8060S (RADV, 2026-10-13). Decode is 128 tokens after the
21-token prompt above; perplexity is over a 261-token English passage; prompts
are timed whole, at 32, 128, 512 and 2048 tokens:

| `quantize` | ms a token | perplexity | prompt (s) |
| --- | ---: | ---: | --- |
| `:int8` | 166.7 | 4.28 | 0.48, 1.02, 3.45, 15.55 |
| `:q4k` | 105.7 | 4.52 | 0.27, 0.61, 2.44, 11.93 |
| `:q4k_m` | 108.7 | 4.42 | 0.35, 0.71, 2.64, 12.9 |
| `:q4k`, `draft = 7` | 95.1 | | |
| `:q4k_m`, `draft = 7` | 98.8 | | |

A load that quantizes reads the 65 GB export: 30 s when it is in the page cache,
60 to 80 s when it is not (int8 32 s). From `cache` it is 13 s from disk and 5 to
10 s from the page cache, the same weights to the bit.

`:q4k` stores the matmul weights as llama.cpp's Q4_K, 17.5 GiB (DNNKernels'
`q4k.jl`); every one of them is read once a token, at about 220 GB/s. `:q4k_m` is
llama.cpp's Q4_K_M: Q6_K (`q6k.jl`) for `lm_head` and half the down projections
(see `q4km`). Keeping those in int8 instead measured 4.40 at 113.3 ms. Prompts
of more than 16 tokens take the K-quant tensor-core GEMM, which is faster than
the int8 one at every width measured.

`cache` names a directory for the quantized weights. The first load writes them
there (18.8 GB for `:q4k`, 20.3 GB for `:q4k_m`); later loads read them instead
of the 65 GB export. An entry the policy would now quantize differently is
quantized again and appended; delete the directory to reclaim the old bytes.

`draft = k` guesses up to `k` tokens by finding the context's last tokens
earlier in itself, and checks them in one pass of `horizon32b_verify_bucket_8`,
whose eight positions cost 123 ms against 102 ms for one. The tokens are the
same as without; what it saves depends on how much the text repeats itself
(here 20 of 204 guesses were kept). The verify bucket is exported with
`tools/export_horizon32b.py --graphs-only --verify 8`.
