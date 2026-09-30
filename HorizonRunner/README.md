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
