# Making and publishing model artifacts

For maintainers. Users never run any of this: every runner resolves its graphs
and weights through its `Artifacts.toml`, and Pkg downloads them on first use.

The flow is the same for every model:

1. **Export.** `tools/export_<model>.py` reads the upstream checkpoint and writes
   graphs, weights and references to `gen/graphs/<name>`.
2. **Pack and bind.** `julia --project=. tools/make_artifacts.jl <name>...`
   creates each artifact, writes `gen/artifacts/<name>.tar.gz` and binds the
   lazy entry in the runner's `Artifacts.toml`. Reference fixtures only the
   tests read go into a separate `<name>-refs` artifact.
3. **Upload.** `make_artifacts.jl` prints the upload command for the
   `assets-v1` release and does not run it. Commit `Artifacts.toml` once the
   tarballs are attached.

`tools/publish_artifacts.jl` packs and uploads SAM 2 and MatAnyone in one step
(`--dry-run` first). OIDN has its own packer, `tools/make_oidn_artifacts.jl`,
and K2 Horizon its own publisher, `tools/publish_horizon_artifacts.jl`. Each
script's header describes it.

## Per model

**OIDN** (`OIDNRunner`). From the workspace root, with the upstream clone at
`dev/oidn`:

```sh
uv run --no-project --python 3.12 --index https://download.pytorch.org/whl/cpu \
  --with torch==2.9.1+cpu --with safetensors --with numpy \
  python dev/JuliaVision/tools/export_oidn.py --checkout dev/oidn \
  --output dev/JuliaVision/gen/graphs/oidn
julia --project=. dev/JuliaVision/tools/make_oidn_artifacts.jl
```

The exporter verifies the Git LFS weight SHA-256 values; `provenance.json` pins
code commit `6602ee2ca38a1a2a02135beed8f6e68eed630180` and weight commit
`28883d1769d5930e13cf7f1676dd852bd81ed9e7`. The packer keeps the published
download for unchanged content. For new content, upload `oidn-models.tar.gz`
first and pass its URL as the packer's second argument. Test references:
`--references dev/JuliaVision/OIDNRunner/test/references` on the exporter for
the networks, and `tools/reference_oidn.py --library <libOpenImageDenoise.so>
--models <pinned weights> --output <...>/pipeline-reference.safetensors` (add
`--width 815 --height 47` for `tiled-reference.safetensors`) with the released
v2.5.1 library for the pipeline.

**Qwen-Image 2.1** (`QwenImageRunner`):

```sh
uv run tools/export_qwenimage21.py --component text_encoder
uv run tools/export_qwenimage21.py --component transformer --graph-only
uv run tools/export_qwenimage21.py --component vae
julia --project=. tools/make_artifacts.jl qwenimage21 qwenimage21-vae qwenimage21-refs \
    qwenimage21-dit-w{1,2,3,4,5,6} qwenimage21-enc-w{1,2,3,4,5}
```

`--prompt-tokens`, `--height`, `--width` and `--context-tokens` only size the
example each graph is traced at; the bounds each symbol was checked for go into
the two `*_export.json` files. The reference-image path is exported by
`tools/export_qwenimage21_condition.py`. Checkpoints above GitHub's 2 GiB asset
limit are split by `tools/shard_safetensors.jl`.

**K2 Horizon 32B** (`HorizonRunner`). `tools/export_horizon32b.py` writes the
graphs (`--graphs-only --verify 8` adds the eight-position verify bucket).
`tools/publish_horizon_artifacts.jl <export> <tokenizer> <tiny> <workdir>`
builds the artifact trees in the depot, checks every tensor's SHA-256 against
the export, uploads to `assets-v1` with the existing `gh` login, and checks
GitHub's digests before rewriting `Artifacts.toml`. `HORIZON_GZIP` can point
at `pigz`.

**PP-OCRv6** (`PPOCRRunner`):

```sh
uv run --project <env with torch + transformers> tools/export_ppocrv6.py --device cpu \
    --reference-image <a page>
julia --project=. tools/make_artifacts.jl ppocrv6_det ppocrv6_rec ppocr_docori
```

CPU export is fine. `--verify` writes small graphs with every node's PyTorch
result for `DNNKernels.verifygraph`; give it a page with real text.

**PP-DocLayoutV3** (`PPDocLayoutRunner`):

```sh
uv run tools/export_ppdoclayout.py --reference-image page.png
julia --project=. tools/make_artifacts.jl ppdoclayoutv3
```

**UVDoc** (`UVDocRunner`):

```sh
uv run tools/export_uvdoc.py --reference-image curled-page.jpg
julia --project=. tools/make_artifacts.jl uvdoc
```

The exporter reverses the first convolution's channels so the Julia API takes
RGB, checks against the unmodified BGR model, records the upstream revision in
`gen/graphs/uvdoc/provenance.json`, and writes PyTorch's inputs, grid and
full-resolution output to `reference.safetensors`.

**BasicVSR++** (`BasicVSRRunner`): the graph is static, so another clip
length or frame size is `tools/export_basicvsrpp.py --frames T --size S`, then
`julia --project=. tools/make_artifacts.jl basicvsrpp`.

**Qwen3-Embedding** (`Qwen3EmbeddingRunner`): `tools/export_qwen3embedding.py`,
then `julia --project=. tools/make_artifacts.jl qwen3embedding qwen3embedding-refs`.

**TRELLIS.2** (`Trellis2Runner`), not published yet. The graphs come from
`tools/export_trellis2.py --part {ss, ssdec, cond, shape512, shape1024, tex512,
tex1024, sampling}`, run in a TRELLIS.2 checkout's Python environment; the two
sparse decoders are upstream's `shape_dec_next_dc_f16c32_fp16.safetensors` and
`tex_dec_next_dc_f16c32_fp16.safetensors`. Both need binding as artifacts before
the runner can read them.
