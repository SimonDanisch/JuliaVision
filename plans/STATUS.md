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
| **2 — refactors** | `kernel-library-review.md` steps 2–7 (DNNKernels) and step 8 (Lava) | globals **34 → 0** in DNNKernels and **84 → 0** module-level in Lava, **and the two-device test passes** |
| **3 — instructions** | the missing SPIR-V surface, each landing as a plan-type method | section D of `dev/Lava/spirv-intrinsics.md` is empty |
| **4 — the work** | model ports and the kernel items | — |

Phases 1 and 2 overlap where the repos differ: `lava-core` and `kernels-refactor`
are different repos and do not conflict. `whisper` runs alongside from the start
— it needs no new intrinsics and adds a new kernel rather than changing an
existing one, so it does not collide with the refactor.

## Projects

| project | machine | repo / branch | phase | state |
|---|---|---|---|---|
| `lava-core` | Desktop | `Lava.jl` @ `sd/lava-core` | 1 → 2 → 3 | phase 1a **done** (`3f59291`); 1b, 2, 3 not started |
| `kernels-refactor` | Desktop | `JuliaVision` @ `sd/kernels-refactor` | 2 | steps 1–4 and 6 done, **globals 34 → 0**; step 5 declined with reasons; **multi-device blocked on Lava** |
| `whisper` | Desktop | `JuliaVision` @ `sd/whisper` | 4 (runs early) | encoder **runs and matches** (`fa76347`), `WhisperRunner` packaged; decoder not started |
| `portability` | AMD laptop | both @ `sd/portability` | 1 → 2 | not started — **its capability dump is an input to the two refactors, so run it early**; it files bugs rather than fixing them |
| `small-models` | 3070 laptop | `JuliaVision` @ `sd/small-models` | 4 | not started |

The three desktop projects were all killed mid-flight on 2026-08-02 and recovered
by hand; see the *Process* section of each report, and `GUARDRAILS.md` §9, which
was rewritten because of it.

## Where the numbers stand

Only the desktop produces these. Last measured 2026-08-02.

- SAM 2 encode **100.4 ms** vs PyTorch 87.64 → **87.0%**. Target 90% = 97.4 ms.
- SAM 2 decode **3.30 ms** replayed vs 2.10 → **64%**. 36% of it is one attention
  shape at 0.10 TF/s, which `kernels-to-port.md` item 1 targets.
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

**The carrier does NOT exist, and both briefs said it did.** `LavaBackend(ctx)`
keeps `ctx.default_bq` and **discards `ctx`**; `BatchQueue` holds a
`Vulkan.Device`, not the `VkContext` that owns it. So there is no path from a
backend to its context, and neither refactor can finish the multi-device work
until Lava keeps that one field. Found on 2026-08-02 while building DNNKernels'
`Device`; see that project's report.

Once it exists, what remains is the 58 direct `vk_context()` calls.
`vk_context()` may remain as a convenience default — nothing inside the library
may depend on it.

**Testable everywhere, no second GPU needed:** a real device plus lavapipe
(`dev/Lava/test/run_lavapipe.sh`) is a two-device pair on all three machines.
Assert correct results on both *and* that the two contexts' caches are disjoint —
a shared entry can still produce a right answer by luck.

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
