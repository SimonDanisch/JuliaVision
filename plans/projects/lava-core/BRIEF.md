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

## Phase 2 — globals

`kernel-library-review.md` step 8, explicitly independent of the DNNKernels
refactor: move the ~84 `Ref`s onto `VkContext`, starting with the seven-plus that
the ten `RESET_CALLBACKS` sites exist to clear. `ctx.compute` is the worked
example already in the tree. Ends with `VK_CONTEXT_REF` as the one global that
stays.

**Exit:** 84 → 1.

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
