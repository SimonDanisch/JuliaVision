# Qwen3EmbeddingRunner

Qwen3-Embedding-0.6B through DNNKernels and Mantle, producing normalized
1024-dimensional vectors for retrieval.

```julia
using Qwen3EmbeddingRunner
m = qwen3embedding()
query = embed_query(m, "macro for data wrangling")
document = embed(m, "Macros for operating on DataFrames")
```

The graph, fp16 weights, host embedding table, retrieval instruction and tokenizer
are the lazy `qwen3embedding` artifact in `Artifacts.toml`. First use downloads it;
later environments on the same machine reuse the cached artifact.
The `qwen3embedding-refs` artifact holds only the fp32 parity-test fixtures.

Upstream: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
License: Apache-2.0; its license and attribution travel with the artifacts.
