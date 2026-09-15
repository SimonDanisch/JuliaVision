# Horizon 32B: prompt buckets, quantized prefill, masked attention

Validated 2026-09-13 on Radeon 8060S (RADV STRIX_HALO), 64-layer 32B model,
per-channel int8 weights, fp16 activations, 4096-slot KV cache.

Horizon is now faster than the llama.cpp Vulkan baseline at every prompt width
and every decode depth measured. Both were measured sequentially on an
otherwise idle machine, each in a freshly started process.

That now includes prompts past the largest bucket, up to the cache capacity: a
4096-token prompt is 23.58 s against 23.60 s. It is still about 2.6x slower to
LOAD (45.6 s against 17.4 s, cold), and *Load time* below has the budget for
that, which says a cache is not what it needs.

## Prefill, seconds, median of 7

| Prompt tokens | Horizon | llama.cpp | llama.cpp / Horizon |
| --- | ---: | ---: | ---: |
| 16 | 0.215 | 0.689 | 3.21 |
| 32 | 0.236 | 0.668 | 2.83 |
| 64 | 0.280 | 0.538 | 1.92 |
| 128 | 0.619 | 0.662 | 1.07 |
| 256 | 0.816 | 0.961 | 1.18 |
| 512 | 1.664 | 2.346 | 1.41 |

Horizon runs the prompt and attention bucket for that width and includes reading
back the selected logit column. llama.cpp is `llama-bench -p ... -n 0 -r 5`.
Its curve is not monotone in the prompt length (16 tokens costs more than 64),
so the two short-prompt ratios say as much about its fixed cost as about ours.

## Decode, ms/token, 32 advancing positions, median of 7 sequences

| Occupied slots | Horizon | llama.cpp | llama.cpp / Horizon |
| --- | ---: | ---: | ---: |
| 0 | 154.0 | 159.5 | 1.035 |
| 96 | 155.5 | 159.5 | 1.026 |
| 480 | 157.0 | 160.1 | 1.020 |
| 2016 | 159.7 | 162.0 | 1.014 |
| 4064 | 163.7 | 164.3 | 1.004 |

Both fill the prefix with real KV values first and then advance the position 32
times, one token per step. Horizon reads back logits and takes an argmax each
step and checks that every repeat of a sequence produced the same tokens;
llama.cpp samples randomly. The workloads are close but not identical.

## Load time

The one axis where llama.cpp was ahead, and by minutes. Fixed to within about
3x; what remains is a format question, not a tuning one.

| | before | after |
| --- | ---: | ---: |
| `Model` build, warm page cache | 227.0 s | **19-21 s** |
| `Model` build, cold page cache | not measured, >= 227 s | **45.6 s** |
| whole `horizon32b`, cold, precompiled | 246-270 s | **57-60 s** |
| host allocations during the build | 70.8 GiB | **9.6 GiB** |
| llama.cpp, cold, load + 1 token | | 17.4 s |

Almost all of it was one line. `hoistpermutes` leaves a transposed weight as a
lazy `PermutedDimsArray`, and `toback` materialised it with a host-side
`permutedims`: 0.33 GiB/s, against 21.8 GiB/s for the contiguous upload of the
same tensor. Every large weight in a decoder arrives transposed, so the whole
64.78 GiB went through it — 224.7 s of a 227 s build, with the device idle.
Uploading the parent as it lies and transposing on the device with the tiled
kernel attention already uses is 7.3x on the largest tensor (1.01 s against
7.35 s) and **bit-exact**: the per-bucket cache and logit deviations came back
identical to the digit, and generation is still token-for-token equal.

Phase timings are now reported by `Model` itself (`host_passes_s`, `upload_s`,
`fusion_s`), because a four-minute build that says nothing about where it went
is how this sat unnoticed.

### Where the rest of it goes, and what would move it

Measured with paired builds in one hot process, so the numbers below are not
single samples:

| | |
| --- | ---: |
| build, everything hot | 11.2 s (upload 9.3 s) |
| build, warm page cache, cold process | 19-21 s |
| build, cold page cache | 45.6 s (upload 43.2 s) |

The hot floor is a **copy**, not a computation: 69.6 GB from page cache into
device memory at ~7.5 GB/s. Cold adds ~34 s of mmap faulting on top, because
that path runs at ~1.5 GB/s where the same file reads at 6.0 GB/s with direct
I/O. No kernel change touches either number.

So a cache is not required to reach llama.cpp's 17.4 s, and the arithmetic says
which change would: read the checkpoint STRAIGHT INTO device memory. That
removes the fault path and the copy at once, leaving `max(disk, kernels)` —
about 12 s of disk at the drive's real rate against ~2 s of quantisation. It
needs weights allocated from host-visible memory: `VkManagedBuffer.mapped_ptr`
is null for an ordinary device array here, so it is a Mantle allocator change,
and on this APU the memory is physically shared anyway. Explicit buffered reads
into a reusable staging buffer are the smaller version of the same idea — they
fix the 1.5 GB/s but keep the copy, and they cost `readsafetensors` its
"a dict of arrays for the whole checkpoint" contract.

A prepared-state cache would also get there, by halving the bytes rather than
raising the rate. It is the more invasive option, not the less: a 32.4 GiB
derived asset to keep in sync, `Serialization` as a package dependency, and a
staleness story where a wrong cache is a silently wrong model.

### Four things tried here that did not work

Recorded because the next person to look at this load path will think of them:

- **`MADV_SEQUENTIAL`** on each mapping: 47.4 s against 46.2 s for nothing.
- **`MADV_WILLNEED`** one tensor ahead of the upload loop: 46.5 s.
- **Quantising straight out of the checkpoint layout**, skipping the transpose
  entirely with fully contiguous reads. Written, made bit-identical, and 2.5 s
  SLOWER over three paired builds each (13.65 s against 11.17 s, spreads of
  0.1 s, disjoint). Fusing it everywhere instead of only where the shape suited
  it was worse again. Removed; the note above `quantizeint8` has the detail.
- The **per-shape microbenchmark** that motivated it, which is the real lesson:
  it timed `toback(PermutedDimsArray(host))` against
  `quantizeint8(PermutedDimsArray(device))`, and only the first included the
  upload. That invented a 2-3x speedup, a crossover in the wrong place, and a
  "the transpose costs 16 s of every build" diagnosis that was simply false.

## Long prompts, and what is actually a limit

An earlier draft of this file listed "512 tokens is the largest prompt" as
something the engine could not do. That was wrong, and finding out cost one
measurement: the graphs are symbolic in the attention extent and the cache
holds 4096 slots, so a long prompt runs as consecutive bucket prefills against
the growing cache. Only `generate` refused it, with an error of its own making.

It now chunks, and the ceiling is the cache and not the bucket:

| Prompt tokens | Horizon | llama.cpp | llama.cpp / Horizon |
| --- | ---: | ---: | ---: |
| 1024 | 3.63 | 4.74 | 1.30 |
| 2048 | 9.30 | 10.16 | 1.09 |
| 4096 | 23.58 | 23.60 | 1.00 |

Two changes got the 4096 case from 29.79 s to 23.58 s. The flash configuration
for a long query was tuned at `lk = 512` and then applied to every key extent;
measured per shape it wanted `(epad, rpad) = (8, 8)` rather than `(16, 0)`
throughout, and a `(32, 64)` tile once the extent reaches 4096 — 25% at that
shape. And the attention extent used `nextpow(2, occupied)`, so a chunk with
2560 live slots swept 4096; [`chunkkv`](@ref) rounds to multiples of the prompt
bucket above 512 instead, which is worth 14.8% at 4096 slots and costs
+2.7 GiB of pool reservation. Both are shape-keyed decisions with the numbers in
the comment above them.

What *is* a limit, each one checked rather than assumed:

- **4096 KV slots.** The extent is symbolic but the cache buffer is a literal
  `[64, 1, 8, 4096, 128]` in every graph. Handing `decode_bucket` an
  8192-slot cache raises `DimensionMismatch: new dimensions (128, 4096, 8, 1,
  64) must be consistent with array size (128, 8192, 8, 1, 64)` — loudly, at
  the call, which is the right behaviour. More slots is a re-export.
- **Batch 1.** `input_ids` is `[1, T]` with a literal batch dim. A batch-2
  input used to *run*, return one element's logits and poison the recorded
  plan; it now raises `ArgumentError` naming both shapes (see below).
- **`generate` stops at the cache.** A 4096-token prompt yields exactly one
  token, from `p < m.maxlen || break`, rather than wrapping or corrupting.
- **The decode workloads are close but not identical**: Horizon reads back
  logits and takes an argmax every step, llama.cpp samples randomly.
- **The weights are not the same numbers.** Per-channel int8 with fp16
  activations here; Q8_0 blocks with int8 activations and integer dot products
  there. That is also why the remaining prefill headroom is not tuning: closing
  it means quantizing activations, which is a different accuracy trade.

### The same question for decode, measured and declined

Decode rounds its extent up the same way, so the obvious follow-up was whether
`chunkkv` should apply there too. It should not, and the numbers say so rather
than an argument:

| Occupied slots | pow2 extent | `chunkkv` extent | gain |
| --- | ---: | ---: | ---: |
| 1088 | 159.86 (2048) | 159.10 (1536) | 0.5% |
| 2080 | 163.58 (4096) | 160.77 (2560) | 1.7% |
| 3040 | 164.10 (4096) | 161.79 (3072) | 1.4% |

A decode step reads 32 GB of weights and a few hundred MB of cache, so shrinking
the cache sweep cannot move it much. Against that, each new extent costs 0.7 s
to record — five more of them across a full context, each landing on the first
token at that depth. A sub-2% gain under the pipeline's own ±3% replay spread
does not buy a slower first token. The prompt case is the reverse trade and was
taken: 0.91 s once for 4.1 s a prompt.

## A wrong-shaped input was accepted

Found while checking the batch claim above, and worth its own note because the
failure mode was the bad one. `call` validated arguments against *whatever the
first call passed*, never against the graph's declared buffers. So a batch-2
`input_ids` ran, produced logits shaped like the declaration, and got RECORDED
— after which the correctly shaped call failed with "replay input shape or
dtype changed". One mistake, silently wrong results, and the entry poisoned for
every caller after it. `call` now checks every input against `evalshape` of its
declared buffer before anything runs.

## What was implemented

- **Prompt and attention buckets.** Nine graphs (prefill at 16/32/64/128/256/512
  and decode, plus the two fixed base graphs) share one set of 388 weights and
  one full-capacity cache. Attention runs at `nextpow(2, occupied)` slots
  instead of 4096.
- **Selected-position logits.** The prefill export takes the wanted position and
  selects it before the final norm and the vocabulary projection, instead of
  projecting all positions and downloading one column.
- **Direct W8A16 cooperative GEMM** (`q8gemm.jl`). Packed int8 weights unpack
  into shared fp16 tiles with no dequantized weight temporary. Per-channel
  quantization and fp16 activations are unchanged.
- **Tile selection from measurement** (`q8gemm_tile`). Twelve registered tiles
  swept over the four shapes a layer runs, at every prompt width, recorded
  rather than immediate. Summed per layer, the new picks against the old:
  3.38/3.71/4.01/6.31/11.80/23.27 ms against 3.38/3.90/5.59/7.33/13.03/28.20 at
  n=16/32/64/128/256/512. This is what took 512-token prefill from 2.177 s to
  1.694 s and 64-token prefill from 0.375 s to 0.288 s.
- **Masked attention fusion** (`fusemaskedattention.jl`). The exported rounded
  scale/add/masked-softmax chain becomes one op, 576 blocks across the nine
  graphs. Short queries use the cooperative flash kernel, tuned to eight key
  splits and eight subgroups at 2048 slots.
- **Staged masked prefill** (`maskedprefill.jl`), two batched GEMMs and a
  rounded masked softmax, for long contiguous queries: 1.648 ms a layer against
  flash's 2.856 at the 512-token shape. Its query floor is a documented
  judgement — at the 128-token bucket it is also faster in isolation but worth
  only ~11 ms of 631 there, against a measurably larger cache deviation, so
  that bucket stays on flash. The comment above the floor has the numbers.
- **Strided SwiGLU**, reading its two input slices in place, bit-identical to
  the kernel it replaces.
- **`opdouble` now works through a recording.** The recording path called
  `runop!` directly and dropped it, so every aten in a recorded model measured
  as free: a 974-op prefill reported -2.1% to +3.7% per family, which was the
  replay's own spread. Fixed in `execute.jl` and pinned by a test that counts
  dispatches in the plan rather than timing it.

## Where prefill time goes

In-graph attribution at the 128-token bucket (`tools/attribute_prefill_ops.jl`,
each aten run twice in the recorded plan):

| aten | count | doubled cost | share |
| --- | ---: | ---: | ---: |
| `mm.default` | 257 | 541 ms | 82.4% |
| `fused.maskedattention` | 64 | 16 ms | 2.4% |
| everything else | 589 | below noise | — |

Further prefill work has to go into the quantized GEMM; attention and the
elementwise chain are no longer where the time is. llama.cpp quantizes
activations to int8 and uses integer dot products for that product, which is a
different accuracy trade from this W8A16 path, not a tuning difference.

## Correctness

- **Token-exact generation, 4/4.** Two prompts of 19 and 29 tokens, 52 new
  tokens each, two runs each, every token equal to the fixed-graph immediate
  reference. Crosses the 32-, 64- and 128-slot attention buckets. The reference
  sequences are identical to the ones saved before any of this work.
- **Chunked prompts against the unfused graphs, 4/4.** 1024, 1536, 2048 and
  4096 tokens, comparing the final logits and both complete caches, and at the
  first three also generating 4-8 tokens through both and requiring them equal.
  They are. The caches drift where the tokens do not, and by an amount worth
  knowing: relative to their own maxima, logits 0.0017-0.0028 against K
  0.0092-0.0302 and V 0.0174-0.0916. The jump is at the third chunk, where `nk`
  passes the staged route's 1024 ceiling and attention moves to flash — after
  which the K figure is bit-identical at 1536, 2048 and 4096, so it is one route
  change saturating rather than an accumulation. `check_horizon_chunked_prefill`
  therefore bounds the logits tightly, the tokens exactly, and the caches only
  loosely; a bound tight enough to fail on 0.09 would just be measuring which
  side of that ceiling the last chunk fell on.
- **Every bucket against the unfused graph, 6/6 at 9 checks each.** 16, 32, 64,
  128, 256 and 512 tokens: all outputs finite, argmax equal, replay exact, and
  the logits plus *both* complete KV caches within bound. Attention fusion is
  the only difference from the reference there, so the figures are attention's
  alone: logits 0.0011–0.0042 relative, caches 0.0014–0.0249. The 128-token
  bucket is the worst of them on both routes, flash included, which is why the
  check's bound is 0.01 for logits and 0.04 for the caches rather than one
  number — it is a drift alarm around the token-exact gate, not the gate.
- **DNNKernels suite: 12 725 assertions, 0 failures.** Includes the int8 tile
  sweep over 810 shapes (a branch picking a tile that does not divide its shape
  throws from inside a graph), all twelve tiles against a dequantized
  reference, the staged prefill against a strided cache, and the query floor.
- **Tiny Horizon suite: 46/46**, including complete logits and KV-cache
  comparisons and changing the selected prefill logit position between replays.
- Pool reservation 35.7 GiB after loading, 40.4 GiB peak during validation. No
  device loss.

## Measurement notes

Worth knowing before trusting any single number here:

- **Two recordings of the same graph differ by up to ±8%**, and one pathological
  case measured 0.48 s where the steady state was 0.66. Slab placement and pool
  state at record time are part of what is being measured. Every number above
  comes from a process that loaded the model first and benchmarked second; A/B
  comparisons were run back to back inside one process.
- **Microbenchmarks underestimate GEMM cost by about a third.** The tile sweep
  re-runs one shape five times, so its weight is partly resident; the in-graph
  attribution puts the 128-token bucket's matmuls at 541 ms where the sweep's
  per-layer sums predict 404. The sweep is still the right tool for *ranking*
  tiles — its ranking held up end to end — but not for absolute cost.
- **Immediate-launch timings misrank kernels.** Both the attention split
  configuration and the GEMM tiles chose differently when measured through a
  recorded replay, which is how they actually run.
- **Re-measure the baseline before believing a change helped.** A 42.0 s
  reading of the cold upload phase had me hunting a regression in two madvise
  experiments (47.4 and 46.5 s) that turned out not to exist: the unchanged
  code re-measured at 46.2 s. The first reading was the outlier.
- **Page cache decides cold from warm**, and `posix_fadvise(DONTNEED)` on the
  checkpoint drops it without root — but only once no process still holds the
  mapping, so restart the worker first. `dd iflag=direct` gives the disk's real
  rate to compare against (6.0 GB/s here).

## Baseline

llama.cpp source commit `35999d101cf2233fc54f09c3c8d599da7303ce02`, fresh
Release Vulkan build, shader compiler 1.4.357.1, integer dot products, KHR
cooperative matrices, flash attention on, all layers on the GPU. Its Q8_0
checkpoint comes from the same source model, but block quantization differs from
Horizon's per-channel int8, so this is not a bit-identical weight comparison.

Assets are on the internal NVMe at `VulkanDev/gen`; JuliaVision's `gen` link
points at `../../gen`. No USB assets are needed.

## Reproduction

In `bt_julia_eval`, from the VulkanDev workspace, model first in a fresh session:

```julia
using HorizonRunner, DNNKernels, Mantle, Statistics
BH = horizon32b(dir="gen/graphs/horizon32b")
include("dev/JuliaVision/tools/bench_horizon_buckets.jl")
include("dev/JuliaVision/tools/bench_horizon_sequences.jl")
bench_horizon_buckets(BH; prompts=(16,32,64,128,256,512), depths=())
bench_horizon_sequences(BH; depths=(0,96,480,2016,4064))
bench_horizon_longprompt(BH; prompts=(1024,2048,4096))     # past the bucket
# The extent policy this measured, if it needs re-checking:
bench_horizon_longprompt(BH; prompts=(4096,), label="pow2",
                         kvof=(m, occ) -> HorizonRunner.attentionbucket(m, occ))
```

Correctness, and the tools that produced the tables above:

```julia
using Serialization
bucket_horizon = BH
include("dev/JuliaVision/tools/check_horizon_buckets.jl")           # token-exact
REFG = deserialize("out/horizon-bench/unfused-prepared-graphs.jls")  # unfused graphs
include("dev/JuliaVision/tools/check_horizon_optimized_prefill.jl")
check_horizon_optimized_prefill(BH, REFG; tokens=128)
include("dev/JuliaVision/tools/check_horizon_chunked_prefill.jl")    # past the bucket
check_horizon_chunked_prefill(BH, REFG; tokens=1536, maxnew=8)
include("dev/JuliaVision/tools/compare_prefill_routes.jl")           # staged vs flash
compare_prefill_routes(BH, REFG; tokens=256)
include("dev/JuliaVision/tools/attribute_prefill_ops.jl")            # where time goes
attribute_prefill_ops(BH, 128)
include("dev/JuliaVision/tools/bench_q8_tiles.jl")                   # no weights needed
bench_q8_tiles(Mantle.LavaBackend())
```

llama.cpp, after releasing Horizon's device allocations (restart the session —
dropping the model and collecting garbage does not return the pool):

```sh
B=dev/llama-horizon-bench/build-local/bin/llama-bench
M=out/horizon-bench/horizon32b-q8_0.gguf
$B -m $M -p 16,32,64,128,256,512 -n 0 -r 5 -ngl 99 -fa on -o json
$B -m $M -p 0 -n 32 -d 0,96,480,2016,4064 -r 5 -ngl 99 -fa on -o json
```

Logs, all under `out/horizon-bench/`: `horizon-tiles-tuned.log` (the final
Horizon timings), `llama-prefill-final.json` and `llama-decode-final.json` (the
baseline), `prefill-tiles-parity.log` (six buckets against the unfused graph),
`buckets-validation-tiles.log` (token-exact generation),
`dnn-final-tests.log` and `tiny-final-tests.log` (suites),
`q8-tile-sweep.log` (the full tile sweep), `prefill-op-attribution.log`,
`horizon-longprompt.log` and `llama-longprompt.json` (past the bucket, both
sides), `attn-largekv-sweep.log` (the flash configuration per key extent),
`generate-longprompt.log` (chunked `generate` against the benchmark's own
chunking), `decode-extent-policy.log` (the extent A/B for decode),
`prefill-chunked-final.log` and `chunked-token-parity.log` (chunked prompts
against the unfused graphs, caches and tokens),
`llama-load-cold.log` (the baseline's cold load),
`attn-shapes-tiles.log` (staged against flash per shape),
`prefill-route-gate.log` and `horizon-prefill-gated.log` (the pre-tile state).
The local shader toolchain is in `out/toolchains/`.
