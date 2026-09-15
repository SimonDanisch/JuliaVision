# Baseline before "declare, don't capture"

Recorded 2026-09-15, immediately before the refactor deletes DNNKernels' capture
path, its three allocators and Mantle's `record_into`. Every number here is
measured on one machine — Radeon 8060S (gfx1151, Strix Halo, 40 CU / 20 WGP,
32 MB Infinity Cache, 112 GiB unified) — through RADV + Lava on one side and
HIP + AMDGPU.jl on the other. Medians are of 10–15 runs after warm-up unless
stated.

**This file is the acceptance criterion.** The refactor is done when parity is
unchanged, timings are no worse, and the two numbers marked WRONG below have
become real.

## Correctness — SAM 2.1 large

Against the PyTorch reference dump, `maxabs` per encoder output:

| node | Lava | ROCm |
|---|---|---|
| `convolution_5` | — | 0.00159 |
| `convolution_6` | — | 0.00708 |
| `add_129` | 0.3675 | 1.39813 |
| `_to_copy_599` | — | 0.0 |
| `_to_copy_595` | — | 0.0 |
| `repeat_2` | — | 0.0 |

`add_129` is a known shared divergence, not a runtime fault: both backends first
diverge from PyTorch at `add_82`, 7.9% of that node's scale through Lava and
10.5% through ROCm (the ROCm figure predates the one-kernel layer norm reaching
that backend, which took `add_129` from 4.244 to 1.398). `SAM2Runner`'s gate pins
`0.2 < add_129 < 0.6` for Lava only.

Decoder: `verifygraph` reports **ok = true, 0 mismatches on both backends**.

`runsam2` end to end, default prompt on the reference image:

| | score | mask positive | wall |
|---|---|---|---|
| Lava | 0.0403 | 35615 / 65536 | 311.9 ms |
| ROCm | 0.0404 | 35621 / 65536 | 450.9 ms |

Six pixels of 65,536 differ between the backends. Both deterministic run to run.

## Speed — four models, p50

| | ROCm | Lava |
|---|---|---|
| SAM 2.1 encoder (replay) | 322.3 ms | 269–274 ms |
| DepthAnything | 85.07 ms | 131.34 ms |
| RIFE 1920×1152 | 162.04 ms | 141.26 ms |
| NeuralLUT | 1.57 ms | 2.00 ms |

RIFE on Lava spans 140.7–522.7 over 15 runs; one soft-cap collection still lands
inside the 5% `gc_budget`. The others are tight.

## Structure — SAM 2.1 encoder, 548 ops

| aten | ops | ROCm passes | Lava passes |
|---|---|---|---|
| `addmm.default` | 195 | 486 | 243 |
| `_scaled_dot_product_flash_attention` | 48 | 291 | 81 |
| `add.Tensor` | 98 | 98 | 98 |
| `native_layer_norm.default` | 96 | 96 | 96 |
| `clone.default` | 90 | 90 | 90 |
| `convolution.default` | 7 | 7 | 14 |
| `max_pool2d_with_indices` | 6 | 6 | 6 |
| `_to_copy.default` | 6 | 6 | 6 |
| `fused.elementwise` | 1 | 1 | 1 |
| `upsample_nearest2d.vec` | 1 | 1 | 1 |
| **total** | **548** | **1082** | **636** |

ROCm dispatch kinds within those 1082: 1014 `gpu_broadcast_kernel_cartesian`
(elementwise expanded pairwise by `emit`'s lazy `Broadcasted` path), 374
`HIPLaunch` (bare `@roc`, i.e. `mapreduce` partials), 195 `BLASMul`, 119
`gpu_ndmap_flat!`, 51 `gpu_permutedims_kernel!`, 40 + 40 attention, 7 conv.

**1014 broadcast dispatches for one `fused.elementwise` op is the number to
beat.** Group codegen should collapse these toward the number of groups.

## Memory — the two numbers that are WRONG today

```
SAM 2.1 encoder plan:  peakbytes = 0    naivebytes = 0     (both backends)
                       argmemory  = present (Lava) / none (ROCm)
```

Zero because Mantle places nothing: every intermediate comes from DNNKernels'
own slab (`planslab`), op scratch from its `Workspace` bump arena, and the rest
from loose `KA.allocate`. After the refactor these must be the model's real
footprint and a genuine aliasing ratio, with `peak < naive`.

Pool footprint, for comparison against whatever the arena settles at:

| | reserved | blocks |
|---|---|---|
| SAM 2 recorded plan, Lava | 1117.5 MiB | 15 |
| SAM 2 eager, Lava, 20 calls | 1181.5 MiB | 16 |
| RIFE steady state, Lava | 2094.6 MiB | — |

Allocation per replay: **0 bytes** on both backends for a recorded plan, against
10,569,376 bytes/call for the eager path. Zero must stay zero.

## Dispatch cost, 64-pass chain of 4096 elements

| | latency | throughput | per dispatch | allocated |
|---|---|---|---|---|
| ROCm walked | 0.156 ms | 0.133 ms | 2.08 µs | 40,976 B |
| ROCm hipGraph | 0.146 ms | 0.124 ms | 1.93 µs | 0 B |
| Lava recorded | 0.097 ms | 0.061 ms | 0.96 µs | 6,208 B |

Lava's 6,208 B/run is core's `refit!` walking `Vector{TransientResource}` through
a two-method dynamic dispatch — 96 bytes per transient per frame. It will grow
with the transient count this refactor introduces, so it stops being a footnote.

## Test counts to restore

Mantle: arena recording 55, pool 18 testsets, pool trim 16, philosophy guards 8
(0.2 and 0.3 deliberately `@test_broken`), ROCm 54, Vulkan partitioned recording
33. DNNKernels: 48 testsets. SAM2Runner: 80 assertions in 5 testsets.
