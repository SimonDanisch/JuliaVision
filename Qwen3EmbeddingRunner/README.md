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
later environments on the same machine reuse the cached artifact. No Python export,
workspace `gen` directory or manual weight installation is required at runtime.
The `qwen3embedding-refs` artifact holds only the fp32 parity-test fixtures.

To update the model assets, export with `tools/export_qwen3embedding.py`, then pack
and bind both artifacts with `tools/make_artifacts.jl qwen3embedding qwen3embedding-refs`.
Upload the resulting tarballs to the release URLs recorded in `Artifacts.toml`.

Upstream: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
License: Apache-2.0; its license and attribution travel with the artifacts.
