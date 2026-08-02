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
| `lava-core` | Desktop | `Lava.jl` @ `sd/lava-core` | 1 → 2 → 3 | phase 1a **done**; phase 2 **started on `sd/nvidia`** — caches keyed per device, the two-device probe exists and found a class §8 missed. 1b and 3 not started |
| `kernels-refactor` | Desktop | `JuliaVision` @ `sd/kernels-refactor` | 2 | steps 1–4 and 6 done, **globals 34 → 0**; step 5 declined with reasons; **multi-device blocked on Lava** |
| `whisper` | Desktop | `JuliaVision` @ `sd/whisper` | 4 (runs early) | encoder **runs and matches** (`fa76347`), `WhisperRunner` packaged; decoder not started |
| `portability` | AMD laptop | both @ `sd/portability` | 1 → 2 | **phase 1 + 2 done.** Capability dump, 3 hard crashes found and fixed, `Extruded` does NOT reproduce on RDNA 3.5, SAM 2 runs at 294 ms. Read its report before touching `Device` or any subgroup width |
| `small-models` | 3070 laptop | `JuliaVision` @ `sd/small-models` | 4 | **done.** Three models export, run and match; LUT meets budget, RIFE and Depth Anything miss by ~30x / ~15x with measured reasons |

The three desktop projects were all killed mid-flight on 2026-08-02 and recovered
by hand; see the *Process* section of each report, and `GUARDRAILS.md` §9, which
was rewritten because of it.

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

- Pipeline cache keyed on every byte (`spirv_content_hash`); `WORKGROUP_LIMIT`
  256 → 1024. The 256 cap was our hash collision, not the device.
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
- A `Device` object answers the five hardcoded device literals from a live
  device. `NW * 32` was in three places and is half the workgroup on wave64.
- **`test_coopmat_attention.jl`'s main A/B was vacuous** — it flipped a threshold
  to force the two-GEMM path, but the fused path took all four shapes first, so
  it compared a result with itself. Fixed; the pattern is worth grepping for.

## Standing constraint: multi-device

Phase 2 must leave the library able to drive **two Vulkan devices at once**. It
cannot today: four caches hold device-owned handles at module scope keyed without
the device (`PIPELINE_CACHE`, `LINKED_KERNEL_CACHE`, `LAUNCH_PLAN_CACHE`,
`GFX_PIPELINE_CACHE`), so a second device running the same kernel is handed the
first device's `VkPipeline`. The review's stated end state — "`VK_CONTEXT_REF` as
the one global that stays" — describes a single-device library and has been
restated in the briefs.

**RETRACTED, and the briefs were right.** This section briefly claimed the
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
`init_vulkan!(; select)` returns a context without installing it, and the loader
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

Still unaudited on a second device: `TIMESTAMP_POOL`, `BLIT_PIPELINE`,
`GEMM_SPLIT_SCRATCH`, `WORKGROUP_LIMIT` — graphics and profiling, which the
probe does not exercise.
Keying the caches was necessary and not sufficient: the first thing it caught was
`CMD_PIPELINE_BARRIER_FPTR`, a module-global **device function pointer**, which
§8's four-cache list does not cover and which is worse than any of them — a stale
handle is undefined behaviour, a foreign function pointer jumps into another
driver. Fixed. The remaining worklist is in `projects/lava-core/REPORT.md` and
includes three device *properties* cached globally, which give wrong answers
rather than crashes.

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

## Open bugs

- **`blockfor` refuses non-square attention** — an open bug with a workaround,
  currently living inside a predicate: blocking the decoder's lopsided shapes
  reproducibly hangs on `vkWaitSemaphores`. Sits under the decode work.
- **Flush hang** — dominant path fixed, one recurrence after ~90 clean trials.
- **`OpUDiv` in a shared-store index** — labelled "driver miscompile"; per Rule 0
  that label is a suspect, not a finding. Still to re-open against our own
  compiler, and it cannot inherit the verdict below: that audit specifically
  **disproved** `OpSDiv`-vs-`OpUDiv` as a mechanism. Needs its own GLSL
  differential.
- **narrow index + rank≥3 `Extruded`** — **settled 2026-08-02** (Lava
  `3f59291`). The driver verdict stands, now on a glslang-produced module rather
  than by elimination, and every previously asserted mechanism is disproved.
  Re-read the test header before touching anything that divides twice: it is a
  scheduling / register-allocation fault around **chained 64-bit division with
  the first quotient re-used**, not a rule about any instruction. Blocks
  narrowing Lava's broadcast kernels, which is worth ~3 ms.
