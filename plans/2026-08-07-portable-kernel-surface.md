# A portable kernel surface: where Metal and Vulkan diverge

Follow-on to `2026-08-07-dnnkernels-on-mantle.md`, which put DNNKernels'
*execution* on Mantle. This one is about the other half of the Lava dependency:
the 230 references in `DNNKernels/src` that are not execution at all.

## The question this answers

For a GEMM, at what level does the backend get overloaded?

  * **A.** Both backends expose the same cooperative-matrix intrinsics, and one
    GEMM is written against them, parameterised by device capabilities.
  * **B.** Each backend brings its own GEMM kernel.

The answer is **A**, and the reason is that the codebase has already been forced
to solve the harder version of the problem.

## The evidence, not the argument

**`Lava.DeviceCaps` is already portable.** Nine fields, none Vulkan-specific:

    coopmat::Bool  tile::Int  subgroup::Int  coopmatsubgroup::Int
    sharedbudget::Int  workgrouplimit::Int  cores::Int  warps::Int

Every one has a direct Metal equivalent — subgroup/simdgroup,
workgroup/threadgroup, shared/threadgroup memory. The vocabulary is
Vulkan-flavoured; the content is a description of *a GPU*. It is a portable
struct sitting in a Vulkan package.

**The non-portable intrinsic problem is already solved, inside Vulkan.**
`coopmat_perelement` and `coopmat_reduce` are `VK_NV_cooperative_matrix2` —
NVIDIA-only, so not portable even across Vulkan devices. `FlashCMPlan` handles it
today with a `rescale::Symbol` field (`:comp` / `:perelem` / `:fmul`), an
availability predicate, and a default chosen explicitly because `:fmul` "is plain
`SPV_KHR_cooperative_matrix` rather than `VK_NV_cooperative_matrix2`, so it is the
one that also runs on AMD."

That is the mechanism a Metal backend needs. It exists, and it has already been
load-tested against a real portability gap rather than an imagined one.

**Tile shape is a number, not a code path.** Vulkan queries supported M/N/K;
Metal's `simdgroup_matrix` is fixed 8x8. `DeviceCaps.tile` is one `Int`, and the
kernel plans derive their blocking from it. Metal reports 8, RDNA 3.5 reports 16,
and the same kernel body takes different numbers.

## The layering

| package | owns |
|---|---|
| **Mantle** | execution (placement, lifetimes, barriers, recording) — done. Plus `DeviceCaps` under neutral names, and the coop-matrix *surface*: `load`, `store`, `muladd`, `mul`, `add`, `convert`, `getcomp`, `setcomp`. |
| **DNNKernels** | the kernel *bodies* — GEMM, flash attention, conv, gemv, FFT — written once against that surface, with plans choosing tilings from caps. |
| **Lava** | a Mantle backend: the runtime Mantle wraps, plus lowering the intrinsics to SPIR-V. |
| a Metal backend | the same two jobs, in MSL. |

So Lava's `coopmat_gemm_kernel`, `gemv`, `fft`/`stft` move to DNNKernels: they are
bodies, and they belong with the other bodies. What stays is the lowering.

## Why this does not foreclose B

`kernelplans.jl` is built so that "a new path is a new type and a new method" —
`sdpa!(ctx, ::FlashCMPlan, …)`, `matmul!(ctx, ::MMCoopMatPlan, …)`. If Metal's
8x8 blocking turns out to want a bespoke GEMM, that is `MMMetalPlan` plus one
method, and `Decline(reason)` already exists for "this path does not apply here."

The decision is therefore **per kernel and deferrable**, taken on measurements
rather than now. Choosing A costs nothing if B turns out to be right for one
kernel; choosing B first would have meant writing every kernel twice before
knowing whether it was needed.

## Order of work

1. `DeviceCaps` into Mantle under neutral names, with Lava filling it in.
   Mechanical, and the names DNNKernels already uses can be kept.
2. The coop-matrix intrinsic surface into Mantle, with Lava lowering it. The
   `:perelem` / `:fmul` split is the template for anything Metal lacks.
3. Kernel bodies out of Lava into DNNKernels.
4. Only then is DNNKernels free of Lava, and only then is a Metal backend a thing
   somebody could write without touching DNNKernels at all — which is the test of
   whether any of this worked.

Steps 1-3 are each independently useful and independently revertible. Nothing
here needs a Metal backend to exist to be worth doing; the portability is the
motivation, the decoupling is the payment.

## Not in scope

Removing Lava from DNNKernels' `Project.toml` before step 3. The dependency is
honest until the bodies move — `GEMM_TILE` alone is 56 references, and a
re-export that pretends otherwise is worse than the dependency.
