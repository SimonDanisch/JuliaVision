# lava-core

**Machine** Desktop (RTX 4000 Ada) · **Repo** `dev/Lava` @ `sd/lava-core`
**Read first** `dev/Lava/spirv-intrinsics.md`, then `plans/GUARDRAILS.md`.

The Lava half of the foundation: settle the two mislabelled bugs, drain the
globals, then add the missing SPIR-V surface. In that order, because each step
makes the next cheaper and the last one is worthless if done first.

## Phase 1 — the two "driver bugs"

Both are labelled *NVIDIA driver miscompile*, both root-caused and mitigated,
both recorded as "cannot be settled on this hardware". That is the exact shape
the workgroup-size cap had, and it was one line of ours. **Rule 0 applies: the
label is a suspect, not a finding.**

- **narrow index + rank≥3 `Extruded`.** Its own notes contain the lead: *"LLVM's
  canonicalisation of `x <s 1` when it believes `x` is non-negative — valid only
  under `nuw`/`nsw` flags that rank 4's extra index arithmetic supplies."* That
  is a description of our compiler emitting a comparison that holds only under an
  assumption. Start there.
- **`OpUDiv` in a shared-store index drops stores.** Mitigated by `splitidx`,
  which means we avoid emitting the shape rather than understand it — so if the
  defect is ours it is still live anywhere else that divides in a store index.
  "lavapipe has no coopmat" does not block the GLSL differential: reproduce the
  divided-index store without a cooperative matrix in scope, and if it cannot be
  reproduced that way, that fact is itself the diagnosis.

Instruments in order, per Rule 0: `spirv-val --target-env vulkan1.3`;
`Lava.enable_gpu_av`; a UB hunt in our own output; then the GLSL differential.

**Exit:** each item is either a located bug in our emitter with a test, or has
passed all four instruments and has a written argument for why it is not ours.

Also in this phase, because they are cheap and prevent the next misdiagnosis:
turn `spirv-val` and `gpu_av` on by default for test runs, and add the
pipeline-count assertion (`GUARDRAILS.md` §4) to the A/B harnesses.

## Phase 2 — globals, and this is where multi-device is won or lost

`kernel-library-review.md` step 8, explicitly independent of the DNNKernels
refactor: move the ~84 `Ref`s onto `VkContext`, starting with the seven-plus that
the fifteen `RESET_CALLBACKS` sites exist to clear. `ctx.compute` is the worked
example already in the tree.

**The review's stated end state — "`VK_CONTEXT_REF` as the one global that stays"
— describes a single-device library, and must not be built.** The direction is
right (caches onto the context *is* the multi-device fix); the goal statement is
what needs restating.

### What is broken today

Four caches hold **device-owned Vulkan handles** at module scope, keyed without
the device:

| cache | holds |
|---|---|
| `PIPELINE_CACHE` (`runtime/pipeline.jl:176`) | `LavaComputePipeline` — shader module, pipeline layout, pipeline, descriptor set layout |
| `LINKED_KERNEL_CACHE` (`runtime/launch.jl:32`) | `LavaLinkedKernel`, which owns a pipeline |
| `LAUNCH_PLAN_CACHE` (`array/ka_backend.jl:689`) | `LaunchPlan`, which owns a pipeline |
| `GFX_PIPELINE_CACHE` (`graphics/pipeline.jl:23`) | `CompiledGraphicsPipeline` |

Two devices running the same kernel produce the **same cache key**, so the second
gets the first's `VkPipeline` and binds it into a command buffer on a different
`VkDevice`. Undefined behaviour, and exactly the class of the `hash(spirv_bytes)`
collision fixed on 2026-08-02.

Two things are already right and should not be disturbed: the **on-disk**
`VkPipelineCache` is keyed by device name and driver version
(`lava_pipeline_cache_path`), and the ray-tracing cache is a **struct field**, so
it is per-object already.

### What makes it tractable

The carrier exists. `LavaBackend(ctx::VkContext)` pins a context, `BatchQueue`
carries `ctx`, and KernelAbstractions hands the backend to every launch — so a
pinned backend already reaches a specific device. What leaks is the **58 direct
`vk_context()` calls** in `src/`, which resolve a global instead.

`vk_context()` may **stay** as a convenience default for the single-device case.
What must change is that nothing inside the library *depends* on it: every path
that touches a device object takes the context from its argument.

`RESET_CALLBACKS` also encodes the single-device model — fifteen sites that
`empty!` global state on the assumption there is one device to reset. Those become
per-context lifetime.

### Exit

- **84 → 0 module-level.** All device state on `VkContext`; `vk_context()`
  survives only as a default, depended on by nothing.
- **The acceptance test, which needs no second GPU:** `test/run_lavapipe.sh` shows
  lavapipe is available, so a real GPU plus lavapipe is a two-device pair on
  *every* machine. Create both contexts, run the same kernel on each, and assert
  the results are correct on both **and that the two contexts' caches are
  disjoint** — a shared entry is the bug, and it would otherwise be invisible
  because the answer can still come out right by luck.

Coordinate the `Ctx`-carries-the-device half with `kernels-refactor`; it is doing
the same thing one layer up.

## Phase 3 — the missing instructions

Ordered by cost, cheapest first. Section D of `spirv-intrinsics.md` has the
verified opcode and capability numbers for each — they were obtained by
disassembling glslang output and by assembling and validating a module with
`spirv-as`, so do not re-derive them.

1. **coopmat2 reductions.** `OpCooperativeMatrixReduceNV` = 5366, capability
   `CooperativeMatrixReductionsNV` = 5430, both already declared in `module.jl`
   but emitting nothing. It takes a function operand exactly like
   `OpCooperativeMatrixPerElementOpNV`, so the marker / `coopmat_keepparam` /
   thunk machinery already exists and is tested — this is a new `op ==` branch
   plus a binding, not new infrastructure.
2. **`OpMemoryBarrier`** — declared, never emitted, and a prerequisite for any
   spin-wait. Plus the `MaximallyReconvergesKHR` execution mode, which is *not*
   declared: the device feature is enabled but without the execution mode in the
   module the reconvergence guarantee does not apply.
3. **fp8 + bf16 types.** `VK_EXT_shader_float8` (Khronos, not NV) and
   `VK_KHR_shader_bfloat16` are both present on the device with their feature
   bits true and neither is requested. `emit_type_float!` needs an FP-encoding
   operand; the Julia side needs a 1-byte primitive type with host-side
   conversion, **not** an arithmetic type — a GEMM loads fp8, muladds,
   accumulates fp32 and stores fp32, so no fp8 value reaches LLVM IR.
   **Gated** on the int8-K32 ceiling probe (`GUARDRAILS.md` §5).
4. **`ballot` binding** (emitter case exists, nothing reaches it — mind the
   wave64 uvec4 trap), and **scans beyond `add`**.
5. **Tensor addressing / block loads** — the largest surface, and the only one
   needing new SPIR-V *types* in `SPIRVTypeContext`. Last.

Each lands with its KHR fallback as a sibling method, which is only cheap because
`kernels-refactor` has landed plan dispatch by then. Coordinate with that project
before adding a capability gate.

**Exit:** section D of `spirv-intrinsics.md` is empty, and nothing in section C is
a capability the code claims to have.

## Report

Append to `REPORT.md`: what each instrument said, the located cause if there was
one, and the global count after phase 2. Negative results in full — if an item
turns out not to be ours, the argument for that is the deliverable.
