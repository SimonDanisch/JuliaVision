# Status

One line per project. The coordinator updates this after reading reports; agents
update only their own `REPORT.md`.

Last updated 2026-08-02.

## Phases — foundation before feature work

Bugs, missing instructions and refactors first, so that nobody builds on a wrong
belief or writes code the next phase deletes.

| phase | contents | exit criterion |
|---|---|---|
| **1 — bugs** | Rule-0 re-audit of the two "driver bugs"; the `blockfor` hang; the flush-hang soak | no tracker item says "driver bug" without having passed all three Rule-0 instruments |
| **2 — refactors** | `kernel-library-review.md` steps 2–7 (DNNKernels) and step 8 (Lava) | globals **34 → 0** in DNNKernels ✅ and **84 → 0** module-level in Lava, **and the two-device test passes** ✅ |
| **3 — instructions** | the missing SPIR-V surface, each landing as a plan-type method | section D of `dev/Lava/spirv-intrinsics.md` is empty |
| **4 — the work** | model ports and the kernel items | — |

Phases 1 and 2 overlap where the repos differ: `lava-core` and `kernels-refactor`
are different repos and do not conflict. `whisper` runs alongside from the start
— it needs no new intrinsics and adds a new kernel rather than changing an
existing one, so it does not collide with the refactor.

## Projects

| project | machine | repo / branch | phase | state |
|---|---|---|---|---|
| `lava-core` | Desktop | `Lava.jl` @ `sd/nvidia` | 1 → 2 → 3 | phases 1–2 done; phase 3 (section D) **not started** |
| `kernels-refactor` | Desktop | `JuliaVision` @ `sd/kernels-refactor` | 2 | steps 1–8 done; **step 9 in progress** (moving the model drivers out) |
| `whisper` | Desktop | `JuliaVision` @ `sd/whisper` | 4 (runs early) | encoder **runs and matches** (`fa76347`), `WhisperRunner` packaged; decoder not started |
| `portability` | AMD laptop | both @ `sd/portability` | 1 → 3 | **phases 1–3 done.** Capability dump; 3 hard crashes fixed; the `OpBitcast` family closed (store path has a reproducer, load path is unreachable and recorded as such); `Extruded` does NOT reproduce on RDNA 3.5; coopmat 32-lane pin confirmed; SAM 2 runs at 294 ms; **bank-conflict sweep settled NEGATIVE** — 22–32% in the microbenchmark, nothing on the model, so the defaults stand. Read its report before touching `Device`, any subgroup width, or `flashepad`/`flashrpad` |
| `small-models` | 3070 laptop | `JuliaVision` @ `sd/small-models` | 4 | **done.** Three models export, run and match; LUT meets budget, RIFE and Depth Anything miss by ~30x / ~15x with measured reasons; artifacts refactor landed |

The three desktop projects were all killed mid-flight on 2026-08-02 and recovered
by hand; see the *Process* section of each report, and `GUARDRAILS.md` §9, which
was rewritten because of it.

Step-by-step against `kernel-library-review.md`'s suggested order:

| step | what | state |
|---|---|---|
| 1 | expose the device properties | done — leftovers: `convtiles`/`convsplit`/`FLASH_SHARED_BUDGET`/`FLASHCM_MINGRID` unwired |
| 2 | delete the settled toggles | done |
| 3 | diagnostics onto `Ctx` | done |
| 4 | one plan object per kernel family | done |
| 5 | dispatch the entry functions on the plan type | done |
| 6 | generate the dispatchers alongside the kernels | done — `ATTN_BLOCKS` folds into both arms |
| 7 | type `Ctx`'s remaining fields | done |
| 8 | device caches onto `VkContext` | done — see below |
| 9 | **portability decision + move the model drivers out** | **in progress** — the history answered the "what is DNNKernels" question: `sam2.jl`, `wan.jl` and `driver.jl` arrived wholesale in `7273481` "Import LavaDNN as DNNKernels", so the rename described half the contents and moved nothing. Eleven `*Runner` packages already establish the convention |

Exit criteria: DNNKernels globals **34 → 0** ✅ (verified by mutation, not by the
`= Ref` grep — 20 `const`s, none ever written), two-device test ✅, Lava
module-level device caches **12 → 0** ✅.

**Lava's global count, corrected.** The review's metric was
`grep "^const [A-Z_0-9]* = Ref"`, which cannot see `Threads.Atomic`, `Dict`,
`IdDict` or `UInt64[]`. Counted by mutation the starting point was **70**, not the
77/73 reported against it, and three of the invisible ones were the per-device
defect finding 3 exists to name. Now **27**, and the floor is about three
(`VK_CONTEXT_REF`, the atexit flag, `FROZEN_RT_MEM`) rather than the 1 the review
set — see `kernel-library-review.md`'s correction section.

## Where the numbers stand

Only the desktop produces these. Last measured 2026-08-02.

- SAM 2 encode **100.4 ms** vs PyTorch 87.64 → **87.0%**. Target 90% = 97.4 ms.
- SAM 2 click **2.21 ms** replayed vs 2.10 → **95%**, and decode 3.13 vs 2.10 →
  **61%**. `kernels-to-port.md` item 1 (flash-decoding) is **done**: the
  decoder's cross-attention went 0.4279 → 0.1204 ms, 3.55x, by splitting the key
  axis six ways and merging the partial softmaxes. Ported from llama.cpp's
  `flash_attn_split_k_reduce.comp`; the measured optimum is 8 splits at 0.1096,
  so the reference's "two workgroups per core" constant is within 10% here.
- VRAM **1181 MiB** vs PyTorch 1756 — 33% under, goal met, nothing to do.
- Device cooperative-matrix ceiling **107.3 TF/s** measured; `addmm` at 38.0 = 35%.

## Shipped 2026-08-02, do not redo

- Pipeline cache keyed on every byte (`spirv_content_hash`); the workgroup limit
  256 → 1024. The 256 cap was our hash collision, not the device. (It is
  `Lava.caps(ctx).workgrouplimit` since 2026-08-03 — queried, not a `Ref`.)
- `VK_NV_cooperative_matrix2`, `VK_NV_cooperative_vector`,
  `VK_KHR_shader_maximal_reconvergence`, `..._subgroup_uniform_control_flow`,
  `..._subgroup_rotate` requested and exposed on `VkContext`.
- Subgroup shuffle family + `subgroup_size()` / `subgroup_lane()`.
- `coopmat_perelement` (NV) and `coopmat_mul` (portable KHR, component-wise
  `OpFMul`; with a stride-0 load it applies a per-row factor in one instruction).
- Held `O` **closed**: worth −21% while under 128 registers, +9% once over; six
  routes to stay under measured and all dead. See `FLASHCM_HELD`.
- gelu into the GEMM epilogue; stem convolution padded onto the tensor cores;
  1×1 convolution routed to `matmul!`; decode capture/replay; click 16.7 → 5.05 ms.
- **The scalar GEMM accumulated in the destination's precision** — an fp16
  destination meant an fp16 accumulator over K = 5120. **234×** error, found by
  the Whisper port, fixed by `gemmaccumtype` (Lava `a246295`). Affects every fp16
  matmul missing the coopmat path, which in a raw exported graph is every
  `addmm`. Whisper's encoder matches PyTorch at rel rms 6.30e-5 after it.
- **`clear_kernel_cache!` silently did nothing** — `LAUNCH_PLAN_CACHE` is
  consulted first and was not cleared (Lava `740d982`). It invalidated an A/B
  over six SPIR-V variants. See `GUARDRAILS.md` §4.
- DNNKernels globals **34 → 0**, the refactor's exit criterion. Nine settled
  toggles deleted; diagnostics and the attention clamp onto `Ctx`; the tuning
  constants absorbed into four plan objects; SAM 2's policy onto its struct.
  Entry points dispatch on plan type, so a new kernel path is a new method.
- A device-capability record answers the five hardcoded device literals from a
  live device. `NW * 32` was in three places and is half the workgroup on wave64.
  (It is `Lava.DeviceCaps` since 2026-08-03; `DNNKernels.Device` is deleted.)
- **`test_coopmat_attention.jl`'s main A/B was vacuous** — it flipped a threshold
  to force the two-GEMM path, but the fused path took all four shapes first, so
  it compared a result with itself. Fixed; the pattern is worth grepping for.

## Standing constraint: multi-device — MET

Two Vulkan devices in one process, both computing correctly, with disjoint
caches. `dev/Lava/test/twodevice_probe.jl` is the acceptance test and is in
`runtests.jl`: it builds the real GPU and lavapipe from one loader, runs a
dispatch, a reduction and a split-K GEMM on each, and asserts `PIPELINE_CACHE`
grows by **more than one** — a shared entry can still produce a right answer by
luck.

**RETRACTED (kept for the record; superseded below by step 8 proper).**
This section briefly claimed the
carrier did not exist. `BatchQueue.ctx` is real and populated, so
`backend -> dispatch_bq -> ctx` resolves for every construction form. The claim
came from reading the first half of `BatchQueue`'s field list, where `ctx::Any`
sits sixty-odd lines down. A `ctx` field was added to `LavaBackend` on the
strength of it and has been removed again — a second copy of a fact the queue
already holds can only disagree with it.

What landed instead: `Lava.vk_context(backend)` and `vk_context(array)` name
that path, and **the three pipeline-owning caches are now keyed by device**
(`VkContext.id`, a never-reused counter) — Lava `0aaa7f9`. So two devices
compiling the same kernel no longer collide. `GFX_PIPELINE_CACHE` is still
unkeyed and holds only graphics pipelines.

What remains for phase 2: the 58 direct `vk_context()` calls in `src/`.
`vk_context()` may remain as a convenience default — nothing inside the library
may depend on it. And the two-device acceptance test (real GPU + lavapipe) is
now *possible*, which it was not before, and still unwritten.

**Testable everywhere, no second GPU needed — and now actually written.**
`VkContext(; select)` returns a context without installing it, and the loader
enumerates the GPU and lavapipe from one instance, so `dev/Lava/test/twodevice_probe.jl`
builds the pair on any machine here. It asserts correct results on both *and*
that one kernel compiles **twice** — a single new `PIPELINE_CACHE` entry means
the devices shared a pipeline, which can still produce a right answer by luck.

**It PASSES** (Lava `83b9b10`) and is back in `runtests.jl`.

**The blocker was the allocator, not the caches.** `POOL_BLOCKS` was
module-level and `PoolBlock` carried no device, so a second device's allocation
came out of the first device's 64 MiB block — measured: the block count stayed at
1 across an allocation on each device, and `fill!` on the second context read
back 0.0. All four caches §8 names were already keyed per device by then and
**none of them was what actually broke it**; the allocator is a bigger class,
because it hands out memory rather than handles and so corrupts silently. Now
`DevicePool` per device, with each block back-referencing its pool so the
finalizer free path needs no lookup.

The last piece was six `LavaBackend()` calls *inside* the library — an unpinned
backend resolves through `vk_context()`, so `fill!`/`mul!`/`coopmat_gemm!` and
broadcast dispatched on whichever device was global. All six now derive the
backend from the data.

**Reading the code produced a list of four caches. None of the four was what
actually broke it.** In the order the probe found them, each invisible until the
one above it was fixed: the device **function-pointer table**
(`vkCmdPipelineBarrier`, recorded through the *other* driver), `PREPARE_INDIRECT`,
the subgroup-size properties, the **memory pool** (lavapipe arrays carved out of
NVIDIA's block — silent data corruption, not a bad handle), six `LavaBackend()`s
built inside the library, `GEMM_SPLIT_SCRATCH`, and `_REDUCE_SCRATCH` — which was
already keyed by context and still *allocated* on the global one.

### And then step 8 properly: the caches are FIELDS, not keyed globals

The above was reached by keying twelve module-level `const`s on `ctx.id`. That
satisfied the letter of `GUARDRAILS.md` §8 and missed its point — entries
outlived the device they described, `ctx.id` was a surrogate for the object they
should have been stored on, and `RESET_CALLBACKS` existed mostly to empty them.
Two of the twelve (`BLIT_PIPELINE`, `TIMESTAMP_POOL`) were never keyed at all and
stayed broken on a second device throughout, because the probe's path reaches
neither graphics nor dispatch profiling.

They are now concrete fields in `DeviceCaches`, one per `VkContext`:

| | before | after |
|---|---|---|
| module-level device caches | 12 | **0** |
| `RESET_CALLBACKS` registrations | 10 | 8 (none cache-clearing) |
| `ctx.id` as a cache key | every lookup | never — identity only |
| Lava module-level `Ref`s | 77 | 73 |

The obstacle had been include order: `VkContext` is defined before the cached
value types, so a field would have to be `Any` and cost inference on a
per-dispatch lookup. The answer was to move the nine **type definitions** ahead
of it (`runtime/coretypes.jl`) — methods can stay where they are, since include
order constrains types and not methods. An abstract-typed field would have cost
what `Any` costs; a type parameter would have worked without touching signatures
(`::VkContext` still matches the UnionAll) but is unnecessary once the types
move.

The probe's central assertion changed with it. "One kernel on two devices must
compile twice" counted entries in a shared dict; there is no shared dict, so a
shared pipeline is now unrepresentable rather than merely detected. It asserts
the caches are distinct objects and that each device populated its own.

`POOLS`' reset callback did real work — it destroyed the pool blocks — so that
became `destroy_pool!(ctx)`, called from `vk_reset_device!` where the retiring
context is in scope instead of hunted for in a global.

### The `Ref`s are the rest of finding 3, not a separate question

Calling the remaining 73 "tunables and diagnostics, a different question" was
wrong — they are the bulk of what finding 3 is about, and two of them were
outright correctness bugs rather than untidiness. Progress, by owner:

| batch | count | state |
|---|---|---|
| **pool policy → `DevicePool` fields** | 14 | **done** |
| **queue policy → `BatchQueue` fields** | 6 | **done** |
| diagnostics → a `Diagnostics` struct on the context | 18 | next |
| launch/kernel arguments (`GEMM_*`, `BROADCAST_*`) | 12 | after that |
| device properties → `VkContext` fields | 9 | |
| compiler config → `lava_compiler_config` | 4 | |
| genuinely process-level | 9 | staying |

**73 → 52 so far.** Two of the moves fixed real bugs rather than tidying:

- `last_trim`, `gc_last` and friends were global, so **one device's trim
  suppressed the other's** — the pool was being rate-limited across devices that
  share nothing.
- `NEXT_SKIP_BARRIER` is a **one-shot flag consumed by the next dispatch**. As a
  global, a dispatch on one queue could consume the skip armed for another. It is
  a `BatchQueue` field now.
- Separately, `debug.jl` set `POOL_DISABLED` *before* `vk_reset_device!()`, i.e.
  it configured the pool that was about to be discarded. It now sets it on the new
  context, after.

The precedent for the diagnostics batch is DNNKernels' own: it retired
`OPTIMES`/`OPDOUBLE`/`OPDOUBLEFILTER`/`PLAN_MISSES`/`LAUNCH_PROBE` onto
`Ctx.diag` (review step 3), which is exactly the shape Lava needs.

Two more fell out on 2026-08-02 once the probe ran to completion:

- **A capability query is per device too.** `mul!` derives its backend from the
  array — with a comment warning about exactly this — and then asked
  `coopmat_gemm_available()` with no context. lavapipe (8-lane subgroups) was
  told it had tensor cores because the NVIDIA card does, and returned the right
  answer only because redundant subgroups agreed.
- **A context that is dropped must be retired.** See "Open bugs" below; this was
  the segfault that stopped the Lava suite from ever printing a summary.

`TIMESTAMP_POOL`, `BLIT_PIPELINE` and `GFX_SHADER_CACHE` are fields now, so they
are correct by construction rather than unaudited — but the probe still does not
*exercise* graphics or dispatch profiling, so nothing has been run there on two
devices. `WORKGROUP_LIMIT` remains a genuine global (a policy limit read once).

## Cross-project, act on these first

- ✅ **MERGED.** Lava `sd/portability` → `sd/nvidia` (`708eb20`). DNNKernels
  green against it and SAM 2 unchanged (encode 100.88 ms, click 3.08, VRAM
  1181, masks identical). Everything below this line is settled unless marked
  otherwise; rebase onto `sd/nvidia` before further Lava work.
- **`VK_PIPELINE_COMPILE_REQUIRED` was discarded on Linux — this was live on the
  DESKTOP, not just on AMD.** Fixed. It is a *success*-class code, so Vulkan.jl does not
  raise and Lava caches and binds a NULL pipeline. The check exists but sits
  inside `if LARGE_STACK_PIPELINE`, which is `Sys.iswindows()`. Consequence
  beyond the crash: `PIPELINE_COMPILES_REFUSED` is always 0 here and
  `PIPELINE_COMPILE_MISSES` never fills, so **`no_pipeline_compilation` cannot
  report a miss on Linux** — it either crashes or returns a false green. That is
  the instrument the frozen-kernel-cache workflow verifies with, and it is the
  third instrument-cannot-fire bug found on 2026-08-02.
- **…and its negative control was dead on BOTH vendors, which extends
  portability finding 9.** The control used a fixed "novel" kernel body, novel
  only to *Lava's* cache — but the flag asks the DRIVER, and the driver keeps a
  shader cache across processes that nothing here controls. So it fired exactly
  once per machine, ever. Deleting Lava's `VkPipelineCache` blob does not
  restore it on either vendor (verified by doing it), so this is not RADV-only.
  Note the instrument was behaving *correctly* throughout — cached means no
  compile required — the test's premise was wrong. Now novel per RUN via a
  random `Val{K}` literal; verified firing across two independent sessions.
  **Consequence to carry: "0 misses" was a weaker claim than it read.** It could
  mean the frozen cache worked or that the driver's own cache served everything.
  The miss report also identified modules with `hash(spirv_bytes)` — the
  *sampling* hash — so two modules differing in one byte reported as one miss.
- **Subgroup width is not a device fact — and the coopmat half is already
  handled, which NARROWS portability finding 1.** Lava pins any module declaring
  `CooperativeMatrixKHR` to `COOPMAT_SUBGROUP` (32) at pipeline creation
  (`pipeline.jl`), so the literal `32`s *inside coopmat kernels* are correct
  everywhere, including RDNA 3.5.
  What the finding does still catch: subgroup kernels that do **not** declare
  that capability are unpinned and run at the device default (64 there) — the
  `getcomp` + butterfly fallback for `coopmat_reduce` is exactly one of those.
  And it caught a bug in `DNNKernels.Device` (`4f43ba4`), fixed in `6489d5c`:
  sizing a coopmat workgroup from the device *default* would ask for `NW * 64`
  threads at a kernel the driver runs 32-wide — worse than the literal it
  replaced. `Device` now carries `subgroup` and `coopmatsubgroup` separately.
- **A `Bool` capability predicate cannot express this hardware.**
  `coopmat_gemm_available()` is `true` on RDNA 3.5, but its shape table has no
  `Float32` A/B form at all — so a Float32 GEMM is told yes and emits
  instructions the device does not implement. `portability`'s report sketches the
  capability-as-type replacement (`CoopMatBasic` / `CoopMatMapped`, named after
  capability levels and never after vendors) and shows the coopmat2 fallbacks are
  constructible from KHR `getcomp`/`setcomp` rather than hypothetical.
- **TF32 was on by default in every exporter but `dump_sam2_refs.py`**, so their
  "PyTorch reference" was itself a 10-bit-mantissa approximation. RIFE read as
  7.9e-3 and is actually 3.3e-4. Copy the three lines when writing a new
  exporter: `EG.precision_ctx` looks like it covers precision and does not.
- **Benchmark through `Model`, never `loadgraph` + `execute!`** — the latter
  measures a graph nothing ships. Depth Anything 764 → 421 ms from that alone.
- **A model that no other model's tests touch WILL rot, and the rot is silent.**
  2026-08-05: running all five package suites after an unrelated change found
  SAM 2 **not loading at all** — `Model(...)` threw while constant-folding, and
  it had been broken since Kokoro's `6ddff7a` three commits earlier. Kokoro's
  fix to `index.Tensor` (torch PAIRS index tensors; Julia CROSSES them) required
  every axis to be indexed, which is true of Kokoro's mask and false of SAM 2's
  position-embedding interpolation. DNNKernels' own suite was green throughout:
  its layer-by-layer test uses MatAnyone's graphs, and the only SAM 2 it touches
  is one rewrite pass, not `Model`.
  The same run found `SAM2Runner`'s `@compile_workload` skipping, because the
  GEMV port put an `@eval` back on the runtime path — the trap `FFT_PREGENERATED`
  already exists for. Two regressions, both invisible to every test that ran.
  **Run every suite when a shared file changes**, and prefer a check that walks
  the real graphs (`test_index_tensor.jl` does; it costs no device).

## Open bugs

- **`blockfor` refuses non-square attention** — an open bug with a workaround,
  currently living inside a predicate: blocking the decoder's lopsided shapes
  reproducibly hangs on `vkWaitSemaphores`. Sits under the decode work.
- **Flush hang** — dominant path fixed, one recurrence after ~90 clean trials.
- **Device-reset segfault — FIXED 2026-08-02.** `vk_reset_device!` dropped the
  old context without retiring it. Its comment said pre-reset buffers skip
  Vulkan calls because that context has `device_lost` set, and that only held
  when a device *loss* caused the reset; a voluntary reset left it false. The
  context and its buffers then became garbage in the same collection, where
  Julia does not order finalizers, so `Vulkan.Device`'s finalizer destroyed the
  device and `vk_free!`'s `query_timeline` called into it. Ten-line MWE,
  reproduces back at `046b1ed`, and it is why the Lava suite never printed a
  summary. The AMD laptop filed the same thing from three different call sites
  as "a floating GC race" — it floats because the crash lands wherever the next
  GC does. Now `mark_device_lost!`, with `test_device_reset_finalizer.jl`.
- **`OpUDiv` in a shared-store index — SETTLED 2026-08-02, and the label was
  earned after all.** The file closing it said the question could not be settled
  here because "lavapipe reports `coopmat available: false`". Measurably wrong:
  lavapipe has four 8x8x8 cooperative-matrix shapes, `Float16` among them, so the
  second consumer was on this box the whole time. Running the same kernel and
  geometry on both, with only the tile extent taken from the device:

      device      form       K=32    K=64   K=128   K=256
      NVIDIA      udiv       3072     256     240     240    DROPS
      lavapipe    udiv       3072    3072    3072    3072    exact
      both        fastdiv    3072    3072    3072    3072    exact   (control)

  So the module is runnable as written and NVIDIA's compilation of it loses the
  stores. Limits stated in the file: the modules are not byte-identical (no
  16x16x16 on lavapipe) and llvmpipe is a software rasteriser. The AMD laptop can
  run it byte-identically and would settle the remainder. Now a permanent testset.

- **Narrow index + rank≥3 `Extruded`** — still labelled "driver miscompile"; per
  Rule 0 that label is a suspect, not a finding. Re-open against our own compiler.
  Note the sibling above was settled by varying the consumer, which is the cheap
  move that works, and it is available for this one too.
