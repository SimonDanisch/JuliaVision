# whisper

**Machine** Desktop (RTX 4000 Ada — export needs CUDA) · **Repo** `dev/JuliaVision` @ `sd/whisper`
**Read first** `plans/models-to-port.md`, the Whisper section, then `plans/GUARDRAILS.md`.

The keystone of the model work, and the reason it runs alongside the refactor
rather than after it: it needs **no new intrinsics**, and the FFT is a *new*
kernel rather than a change to an existing one — so it does not collide with
`kernels-refactor`, which is rewriting the existing kernel families.

Biggest editor feature in the set by a distance: transcript-driven editing is
what Premiere and DaVinci sell on. And one FFT unlocks DeepFilterNet, Demucs and
Kokoro — four models' worth of groundwork.

## State

- MIT, ungated, weights fetched (`gen/whisper`, 1.6 GB).
- **The encoder already exports and needs no new ops**: 617 nodes over 11 distinct
  ATen ops, all 11 already implemented — checked against the runtime's op set
  rather than assumed. `WhisperRunner.ready()` is true and `whispergraph()` loads
  it. So first light is a bring-up, not a kernel project.

## Work, in order

1. ~~**Run the encoder graph on Lava and diff against PyTorch.**~~ **Done
   2026-08-02** — all 617 ops node by node, rel rms 6.30e-5 at the output.
   `REPORT.md` has the numbers and the three `verifygraph` fixes it needed.
2. **fp16 export.** *Exported and blocked.* 1.186 GiB against 2.373 GiB, an
   exact halving, and 681 nodes over 12 ATen ops rather than 617 over 11 — but
   13.7% rel rms at the output where PyTorch's own fp16 costs 2.5%. Blocked on
   `lava-core`: Lava's scalar GEMM accumulates in the destination's type, so an
   fp16 destination gives an fp16 accumulator over K = 5120. Do not re-open this
   until that lands.
3. **The mel front end** — an FFT. Its reference already exists: `whisper/audio`
   and `whisper/mel` in `gen/graphs/whisper/refs.safetensors`, written by
   `tools/dump_whisper_refs.py`. Host-side is acceptable to start; it is tiny
   next to the encoder. Device-side is what Demucs later wants.
4. **The decoder** — autoregressive, so it needs a **KV cache**. That is the
   first batch-1, bandwidth-bound GEMV workload this engine will have run, and
   none of the GEMM tiling work applies to it: expect the existing tuning to be
   irrelevant and measure fresh rather than assuming a shape carries over.

   Note this is a *consequence* of picking Whisper, not the reason for it —
   Whisper won on being multilingual, where Parakeet TDT 0.6B is English-only.
   The earlier justification in `models-to-port.md` had that backwards and has
   been corrected.

Word-level timestamps need the cross-attention DTW trick — a second pass, and it
can wait.

## The export trap, already paid for once

**Export from CUDA or the attention decomposes.** A CPU export gave 969 nodes with
64 `bmm`, an explicit `_softmax`, and the `logical_not`/`where`/`any.dim` cluster
that guards fully-masked rows. From CUDA it is 617 nodes with 32
`_scaled_dot_product_efficient_attention` and no `bmm` at all. For 32 blocks of
(1, 20, 1500, 1500) attention that is ~180 MB of materialised matrix per block.
The exporter now refuses a silent CPU export — do not work around that refusal.

## Targets

Two, on purpose. **≥ 5x realtime audio** is the editor budget — what makes the
feature usable. **vs PyTorch** is the engine goal. A model that beats PyTorch and
still misses the editor budget has not bought the feature.

## Interaction with the refactor

The FFT is a new kernel family. Write it in whatever shape `kernels-refactor` has
reached — if plan objects exist by then, it gets a plan; if not, keep it simple
and expect to convert it. Do **not** add a global toggle for it either way
(`GUARDRAILS.md` §2).

## Report

Append to `REPORT.md`: the encoder diff against PyTorch, the fp16 weight size, and
the decoder's measured regime — the KV-cache step is the first batch-1 workload
this engine has run, so whatever it says about launch overhead and bandwidth is
new information for `perf-plan.md`.
