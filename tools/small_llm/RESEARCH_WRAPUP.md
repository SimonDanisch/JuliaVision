# Qwen long context and Horizon artifact research wrap up

Closed out on 2026-10-31. Measurements below are from the October 27–28 work;
this closeout adds documentation, without running more GPU experiments.

We achieved usable long-context inference, verified speculative decoding, reusable
kernel improvements, and portable Horizon model assets. We did not establish a
reliable end-to-end MTP speedup, explain the restricted draft head's acceptance
loss, or benchmark every affected runner and backend.

## What is preserved and shipped

| Work | Result | Evidence and limits |
|---|---|---|
| Batched Qwen prefill | Chunk-wide attention, multi-token Gated DeltaNet, batched preparation and grouped expert work | 512-token chunks; measured rates below |
| Long-context decode | Split-K attention and q8_0/q4_0 KV caches | Reached the model's 262,144-position limit; retrieval spot checks passed |
| Prompt-lookup speculation | Verified drafts copied from earlier context | Exact verification reproduced greedy output and a sampled frozen tool-loop receipt; speed depends on copying and GPU load |
| Prefix reuse | Device-side KV and recurrent-state checkpoints | Wide-chunk reuse requires chunk-aligned prefixes for bit-identical output |
| Qwen MTP | Native MTP weights, quantized loader, separate KV, drafts and verification in one plan, tool-loop integration | 26/26 MTP assertions reported passing; receipt and prefix-cache tests passed in the earlier wrapup |
| K-quant staging | Cheaper Float16 staging in DNNKernels batched Q4_K/Q6_K GEMM | Q4_K 15–19% less kernel time; Q6_K 8–16%; Q4_K 22/22 and Q6_K 10/10 tests passed |
| Runner infrastructure | Semantic graph paths and Mantle allocation changes | Earlier wrapup tested Bonsai, Trellis2, PPOCR, NeuralLUT and OIDN semantic graphs; not a benchmark of every runner |
| Horizon packaging | Default loading through Artifacts.toml, sharded mmap weights, graph/tokenizer and tiny fixtures | 48 public lazy artifacts; fresh-manifest tests 67/67; full model generated two tokens |

The K-quant improvement applies to batched products using these kernels. It does
not accelerate single-token integer-dot-product decode. The shared GDN kernel
can benefit other runners, but those gains were not measured here.

## Measurements worth keeping

Qwen3.6-35B-A3B Q4_K_S on Radeon 8060S, q8_0 cache, 512-token chunks:

| Ingested segment | Earlier prefill | Final measured prefill | Decode at segment end |
|---|---:|---:|---:|
| 4k to 16k | 202 tok/s | 838 tok/s | 51.6 tok/s |
| 16k to 64k | 148 tok/s | 627 tok/s | 44.8 tok/s |
| 64k to 128k | 98 tok/s | 431 tok/s | 45.7 tok/s |

128k ingest fell from roughly 22 minutes to under 5 minutes. These were
shared-GPU runs, not controlled hardware comparisons. GPU contention changed
dramatically between runs; treat the numbers as observations, not guaranteed
throughput or evidence of superiority to ROCm or llama.cpp.

With q4_0 KV, a later run reached 261k and retrieved the code word planted at the
start at each tested depth. Decode there was about 20 tok/s under heavier load.
This establishes capacity and a retrieval spot check, not comprehensive long
context quality.

Against a Float16 KV reference after a 32k prefix, over 64 teacher-forced real
tokens: q8_0 mean KL 6.9e-4, same argmax 64/64; q4_0 mean KL 2.1e-3,
same argmax 63/64. These small checks do not establish task-wide equivalence.

MTP acceptance on two short workloads was 89–94% for one draft, 83–90% for two,
and 76–82% for three, yielding roughly 3.3–3.4 tokens per three-draft round.
Combining acceptance with minimum plan times estimated about 104 tok/s at short
depth versus 77 plain, and 81 tok/s at 64k versus 57 plain: about 1.4x.
These are estimates, not a stable end-to-end benchmark. The higher ~1.65x
projection depended on the restricted draft head and is not a shipped result.

## What did not succeed or remains unproven

* The 16,384-token draft head made individual guesses cheaper but reduced
  acceptance: k=3 about 82% to 64%, k=2 about 90% to 72%. Row equality was
  checked, but the loss remains unexplained. It is excluded from the working
  implementation and preserved at `sd/mtp-draft-vocabulary`, commit `c9005d8`.
* GPU contention prevented a trustworthy end-to-end MTP speed comparison. Do
  not quote the estimated gains as measured application speedups.
* Deep-context chunk verification and wide GEMMs change floating-point rounding.
  Exact output claims apply to the tested exact verification path, not every
  fast configuration. Chunk boundaries matter for prefix reuse.
* Grouped expert verification, folding replay into the next plan, parallel
  chunkwise GDN, sparse attention and combined MTP/prompt-lookup acceleration
  were not completed or validated as performance wins.
* Other Qwen sizes, Metal, and our ROCm path were not comprehensively benchmarked.
* OIDN's older quality suite initially lacked generated reference fixtures.
  Remote commit `d99a87c` subsequently adds independent checked-in fixtures;
  the updated suite was not rerun during this documentation closeout.
* The Science4Makie JuliaVision checkout was last observed on the old Horizon
  loader. It still needs updating before artifact-backed loading works there.

## Reproduction and retained evidence

The working changes are on JuliaVision `sd/mantle-dev`:

* `2a922d7`: semantic graphs, GDN chunk kernel, K-quant staging.
* `696c558`: small_llm harness, long-context paths, MTP and fixtures.
* `179571e`: verified Horizon artifacts and portable loading.

Runtime changes were pushed to Mantle and Lava `sd/lava-refactor`, including
Mantle `56d490e` and Lava `b464018`. The latter allows compilation when the
optional CompilerCaching ObjCache API is absent. Use the repository's declared
environment and sources; a copied runner directory is not the full setup.

For MTP, `download_qwen36_mtp.py` fetches just the MTP tensors from the official
checkpoint at revision `995ad96eacd98c81ed38be0c5b274b04031597b0` and verifies
pinned hashes. The weights are about 1.6 GB BF16, quantized at load. The harness
entry points are `loadbackbone(QWEN36_GGUF; device, mtp=true)` and
`tool_julia_answer(...; speculate=:mtp)`. See the downloader's usage and the
test setup for paths and model prerequisites; these are not self-contained
one-line examples. Relevant tests are `test_mtp.jl` and the speculative,
tool-loop and prefix-cache tests in this directory.

[Archived detailed research](LONG_CONTEXT_RESEARCH_ARCHIVE.md) preserves the
original cost models, historical source links, measured results and caveats.
Its literature section and early implementation-status paragraphs are dated
notes; this wrapup supersedes them where implementation status changed.
External links were not revalidated during closeout.

The original workspace also retains `tmp/horizon-weight-cleanup.txt`,
`tmp/horizon-cleanup-work-state.txt`, the Horizon publication log and installation
test logs. Those local logs are supporting evidence, not installation inputs.
Large logit dumps and exploratory output remain excluded from git. No further
files were deleted during this closeout.

### Archived experimental outputs

[Download the October 2026 result archive](https://github.com/SimonDanisch/JuliaVision/releases/tag/small-llm-research-2026-10-31).
It contains 2,522 previously untracked experimental outputs (20,560,673 bytes
before compression), plus a manifest recording every path, size, SHA-256 and
code revision `d37f0000ab9496cfa6a4d3cd95fbaa140df025ae`. Failed and partial
experiments are retained as evidence, not promoted to validated results.

The archive is `small-llm-research-results-2026-10-31.tar.gz`, SHA-256
`1d1ed42c230e6d1cd1066a78c8967cbcd0f8d8362eaf2ac688cabcd6847e857f`.
The release also carries the manifest and checksum as separate downloads.
Extract at the repository root to restore the original `tools/small_llm/`
paths. Tracked sources and fixtures are not duplicated in the archive. Local originals are retained pending separate cleanup approval.

## Horizon assets and storage outcome

HorizonRunner loads its canonical FP16 weights directly from 46 lazy weight
artifacts, plus graph/tokenizer and tiny-fixture artifacts. All 580 tensors were
checked byte for byte; archive hashes and fresh-depot downloads were verified.
No monolithic checkpoint is reassembled in gen. Explicit developer directories
remain supported.

The canonical weights are 64.78 GiB because they retain 16-bit precision. The
34.43 GiB comparison file was an 8-bit GGUF; default GPU Q4_K_M quantization
does not imply the canonical download is 4-bit.

Approved cleanup removed 99.21 GiB of redundant native checkpoint/GGUF files
and 36.45 GiB of Science4Makie quantization caches: approximately 135.66 GiB
net saved. Moving the 64.78 GiB gen export into the shared artifact store is
not an additional 64.78 GiB saving. Deleted quantization caches are regenerable;
the original checkpoint/GGUF would need downloading or rebuilding for future
export or comparison work.

## If this research is resumed

The useful next experiment is one short, isolated-GPU benchmark of the shipped
full-head MTP against plain decode at short depth and 64k, with acceptance and
wall time reported together. Expand quality coverage before changing cache or
verification defaults. Investigate the parked head only if draft cost remains
the bottleneck. No additional optimization is needed to retain the results above.
