# GPUMODE kernelbot corpus

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


581,339 competition submissions from the GPU MODE KernelBot leaderboards, with full
source. Set up as a local DuckDB so we can read what the fastest people wrote for a
given kernel shape before writing our own — the "port SOTA, don't invent" rule
applied to competition kernels rather than to llama.cpp.

## Where it is

    reference/kernelbot-data -> /home/simon/data/kernelbot-data    (9.5 GB + 90 MB db)

Staged off `/sim` because that disk is at 97%. `reference/` is the symlink; every
tool path below is relative to the project root.

Rebuild the DB after re-downloading: `.venv/bin/python tools/kernelbot_build.py --rebuild`
(~20 s; the parquet files are never copied, `submissions` is a view over them).

## Tables

| name | rows | what |
|------|------|------|
| `submissions` | 581,339 | view — every run of every submission, both schema families normalised |
| `kernels` | 165,461 | view — passing `mode='leaderboard'` runs only, one row per (submission, GPU), with code |
| `best_kernels` | 3,258 | table — each user's fastest kernel per (problem, GPU), with code |
| `code_features` | 3,258 | table — technique keyword flags per kernel, no code column, instant to scan |
| `leaderboards` | 39 | problem definitions |
| `amd_successful`, `amd_dedup` | | the upstream pre-filtered AMD exports, untouched |

Scores are wall-clock **seconds, lower is better**. The four `mode` values are
`test` / `benchmark` / `leaderboard` / `profile`; only `leaderboard` carries a score,
which is why `kernels` filters on it.

## CLI

    .venv/bin/python tools/kernelbot.py problems
    .venv/bin/python tools/kernelbot.py top trimul --gpu H100 -n 10
    .venv/bin/python tools/kernelbot.py show 408928              # code to stdout
    .venv/bin/python tools/kernelbot.py dump trimul --gpu H100 -n 5 --out reference/kernelbot-dumps
    .venv/bin/python tools/kernelbot.py search 'cp\.async\.bulk' --problem nvfp4_gemm
    .venv/bin/python tools/kernelbot.py techniques --problem trimul --top 15
    .venv/bin/python tools/kernelbot.py progression trimul --user "Zeyu Shen"
    .venv/bin/python tools/kernelbot.py sql "select ..."

`progression` is the one worth knowing: it lists one author's attempts in order with
the score delta between them, so the diff of two consecutive submissions is a
labelled optimisation. Zeyu Shen's trimul run goes 10.17 ms → 1.14 ms in 15 steps,
each file named after the change (`fused_preprocess_kernel` −41.9 %, `..._fp16`
−50.5 %, `fused_prologue_epilogue` −28.1 %).

## What is actually worth mining for us

Our targets are Vulkan/SPIR-V coopmat on an Ada-class card, so Blackwell-only
instructions (`tcgen05`, TMA, clusters) and MI300 `mfma` don't port directly — the
*structure* does. `pct_handwritten` below is the share of the top 10 that contains a
real kernel rather than a `return a @ b` vendor-library wrapper:

- **trimul** (496, A100/H100/B200/MI300) — 5,036 passing, 51 % hand-written, mostly
  Triton. Closest thing here to our attention/batched-GEMM work, and it exists on
  A100/H100 where the tricks are Ampere-era and portable.
- **nvfp4_gemm / dual_gemm / group_gemm** (595/597/598/697/730) — 117k passing, 100 %
  hand-written CuTe, ~1,900 LOC each. Blackwell-specific in its instructions but the
  best worked GEMM-scheduling source in the set.
- **amd-fp8-mm, amd-mla-decode, amd-moe-mxfp4** — 70-80 % hand-written HIP; MLA decode
  and MoE routing structure carries over even though `mfma` doesn't.
- **qr_v2 / eigh / cholesky** (774-776) — 100 % hand-written but ~8,200 LOC each.
- **pmpp_v2** (537-544: conv2d, grayscale, histogram, matmul, prefixsum, sort,
  vectoradd, vectorsum) — the shapes match our elementwise/reduction kernels, but the
  boards allow arbitrary Python and the winners exploit that: `matmul_v2` is 44 %
  hand-written and rank 2 on L4 is literally `return a @ b`. `sort_v2` is 30 %,
  `amd-identity` 20 %. Filter with `code_features` before reading.

Data caveats, all handled but worth knowing:
- 10 early trimul submissions were scored on more than one GPU, so `submission_id`
  is **not** a key in `kernels`/`best_kernels` — group by `(submission_id, gpu)` or
  you get silent fan-out.
- The AMD legacy export stores `code` as postgres `bytea`; the build decodes it.
- One `eigh` entry scores 4.6e-25 s. Nothing below 100 ns is a real measurement, so
  `best_kernels.plausible` marks it and ranks it last; it is flagged, not dropped.
  Real `eigh` best is 86 µs.
- `leaderboards.parquet` omits 776 (cholesky) and lists the eight pmpp v1 boards
  (339-346) whose submissions were never exported. `problems` is driven off the data
  and prints both facts.

The technique flags are keyword heuristics over source text, not semantic analysis —
good for "which of these 15 kernels are warp-specialised, show me those", not for
counting. Vocabulary is the `FEATURES` list in `tools/kernelbot_build.py`.

## Licence

June 9 Researcher Reciprocity License. Reading these kernels to inform our own is
plain use and only needs attribution. **Training Use** — defined broadly enough to
cover fine-tuning, distillation, synthetic-data generation, embeddings-for-training
and benchmark-for-training — obliges reciprocity: any model improved that way must
not then bar GPU MODE researchers from generating with, evaluating, or publishing on
it. Full text in `reference/kernelbot-data/LICENSE`; attribute as
`GPUMODE/kernelbot-data`.
