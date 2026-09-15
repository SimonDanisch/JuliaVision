# Recorded execution validation

For subsequent occupied-cache/prompt buckets, selected-position logits, and
the fresh llama.cpp comparison, see [horizon_bucket_validation.md](horizon_bucket_validation.md).

2026-09-12, Radeon 8060S Graphics (RADV STRIX_HALO).

The Whisper failure had two causes. Dtype conversion and output-copy kernels
must be inside launch capture. In addition, Mantle's `packdispatch!` must retain
the original arguments with `holdleaves!`, registering their buffers for
submission stamps and cross-queue waits. Otherwise host-mapped token and position
inputs can be overwritten while replay still reads them.

Horizon prefill compiles the full graph and records it in explicit partitions of
64 passes. Decode keeps one submission. `DNNKernels.Model` defaults to recording
off; Horizon enables it and configures the prefill partition. Whisper supports
the explicit `record=true` option. The diagnostic `BATCHED` global is removed.

Validation after the dispatch-buffer registration fix:

| Check | Result |
| --- | --- |
| WhisperRunner | 18/18; cold recorded model, three distinct audio windows, two full runs, then a shorter input; text, tokens and timestamps match immediate execution |
| DNNKernels suite | Passed; workgroup cooperative-matrix kernel unavailable on this GPU |
| HorizonRunner | 13/13; tiny-model reference parity and repeated generation |
| Mantle recording/lifetime selection | 264/264 across 14 test files |
| Mantle partition/synchronization regressions | 33/33; mapped input overwrite, two recorded queues, partition invalidation and resized input |
| Full Horizon 512-token prefill | Both replays byte-identical to immediate logits and both full KV caches; 4.722 s and 4.715 s; no device loss |
| Full Horizon repeated generation | 6/6; two prompts, each repeated twice, every generated token matches immediate execution |

Run the model suites with the workspace Julia project. The focused Mantle
regressions are in `dev/Mantle/test/vulkan/test_partitioned_recording.jl` and
are included by Mantle's test runner. Full-model reproductions are
`check_horizon_prefill_replay.jl` and `check_horizon_generation_replay.jl` in
this directory; run them in that order in one Julia process to reuse weights.

Decode timing depends on the exported cache length. With the same resident
weights, 12 interleaved samples measured 154.186 ms for 64 slots and 178.229 ms
for 4,096 slots. The raw decode exports were checked to differ only by the
64/4,096 shape substitution. The 64-slot replay also matched its immediate
execution exactly. This accounts for the approximately 24 ms difference
between the small-cache performance measurement and full-cache replay.

The full-cache recording still has 1,804 passes in one submission; its 974
tracked buffers add no per-dispatch work during replay. Median host submission
time was 0.151 ms. A warmed token loop with input updates, readback and host
argmax measured 155.73 ms/token at 64 slots and 179.18 ms/token at 4,096 slots.
Short 12-token generation calls
reported 201–204 ms/token including decode setup and initial effects.

`check_horizon_decode_cache.jl` reproduces the cache-size experiment without
reloading weights. Include it after the generation check and call
`check_decode_cache(recorded_horizon, "<64-slot export>/horizon32b_decode.json")`.
The earlier llama.cpp result (6.46 tokens/s) was not rerun in this validation.
