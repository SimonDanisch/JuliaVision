# Full Horizon prefill replay check

## Validated fix

The full graph now records in explicit partitions of 64 passes: 3,148 passes
in 50 submissions. After the captured-buffer synchronization fix, the
2026-09-12 rerun on the same Radeon 8060S completed:

| Phase | Result |
| --- | --- |
| Immediate reference | 13.486 s; all outputs finite |
| Build and record | 5.716 s |
| First replay | 4.722 s; byte-identical logits and both full KV caches |
| Second replay | 4.715 s; byte-identical logits and both full KV caches |

No device loss occurred. Two different prompts subsequently generated the
same tokens as immediate execution on two runs each. See
[recording_validation.md](recording_validation.md) for the regression results
and the separate decode measurements at 64- and 4,096-slot cache lengths.

## Original single-submission failure

2026-09-12, Radeon 8060S Graphics (RADV STRIX_HALO), full K2 Horizon 32B,
257 of 388 weights stored as int8, 512-token prefill, 4096-token cache.
`BATCHED=false`; model automatic recording disabled to measure the phases separately.

Reproduction: run `tools/check_horizon_prefill_replay.jl` with the VulkanDev
workspace Julia project and the full export in `gen/graphs/horizon32b`.
The script uses 512 deterministic valid token IDs, resets both KV caches before
each execution, and compares SHA-256 hashes of all logits and both full caches.

| Phase | Result |
| --- | --- |
| Immediate reference | Completed in 22.694 s; all three outputs finite |
| Build and record full plan | Completed in 12.162 s (includes first-call compilation) |
| First recorded replay | GPU context lost; output download failed with `VK_ERROR_DEVICE_LOST` |
| Second replay | Not reached |

RADV reported: `The CS has been cancelled because the context is lost. This
context is guilty of a hard recovery.` The error surfaced at the first output
download after replay and device-idle waiting; replay output correctness could
not be checked.

Kernel journal at 10:38:31 Europe/Berlin:

```text
ring gfx_0.0.0 timeout, signaled seq=10936390, emitted seq=10936392
Process julia pid 329647 thread julia pid 329647
Starting gfx_0.0.0 ring reset
Ring gfx_0.0.0 reset succeeded
device wedged, but recovered through reset
```

Single-submission full prefill therefore fails with baking as well as the previously reported
batching failure. The log establishes a GPU timeout; it does not by itself
distinguish excessive submission duration from a GPU execution hang. Passing
decode replay does not establish passing prefill replay. Graph passes do not
inherently split a recording into multiple submissions.

A fresh-process load also exposed an unclosed recording docstring left by the
interrupted previous edit. Closing it restored DNNKernels/HorizonRunner loading.
