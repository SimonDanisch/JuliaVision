# portability

**Machine** AMD laptop (Ryzen AI MAX, Radeon 8060S, RDNA 3.5, 32 GB unified)
**Repos** `dev/Lava` and `dev/JuliaVision` @ `sd/portability`
**Read first** `plans/GUARDRAILS.md`, then `kernel-library-review.md` finding 8.

This machine is the only one that can run the KHR-only path, a subgroup of 64,
and RDNA3 WMMA. Its first job is not to write fallbacks — it is to find out what
is actually broken, because several things were *claimed* portable on 2026-08-02
and have never run here.

## Phase 1 — run everything, believe nothing

Three claims went into committed docstrings on 2026-08-02 on the strength of "it
is KHR, so it should work". None has executed on AMD:

- **`coopmat_mul`** — `OpFMul` on two cooperative matrices, component-wise, and
  the **stride-0 broadcast load** that pairs with it. Documented as "plain KHR, so
  this one also runs on AMD". Unverified.
- **The subgroup shuffle family and `subgroup_rotate`.** `test_subgroup_shuffle.jl`
  is *written* for `subgroup_size() ∈ (16, 32, 64)` and loops to `subgroup_size()`
  rather than a literal 32 — but it has only ever seen 32. RDNA3 runs compute at
  32 **or** 64 depending on how the driver compiled the shader.
- **`test_coopmat_shape.jl`** — its docstring is *about this device* ("the AMD
  Radeon 8060S lists 16x16x16 four ways") and was written from a report, never
  executed here. It exists because an extent-only match answers "yes" for Float16
  on hardware that only implements the integer forms.

Run: the Lava suite, the DNNKernels suite, and specifically
`test_subgroup_shuffle.jl`, `test_coopmat_shape.jl`, `test_coopmat_perelement.jl`
(its first testset is the KHR-only half — the rest should skip cleanly on a
device without `VK_NV_cooperative_matrix2`), and `test_workgroup_limit.jl`.

Also capture, once, and put it in the report: the full device capability dump —
`coopmat_shapes` with decoded component types, `coopmat2`, `coopvec_available`,
subgroup size and supported operations, `maxComputeSharedMemorySize`,
`maxComputeWorkGroupInvocations`, and the measured shared-memory-vs-residency
table. Everything on the desktop is currently written against *one* device's
answers.

**Exit:** a list of what fails, what skips, and what silently does the wrong
thing. A skip that should have been a run is a finding.

## Phase 2 — answer finding 8, in writing

The review's open question, and it is a decision rather than an implementation:
**what is DNNKernels?**

> (a) a Lava kernel library that uses KA for convenience, (b) a portable library
> with a Lava fast path, or (c) portable-with-Lava-required. It is currently (a)
> documented as (b).

The evidence: **176 direct `Lava.*` references** in DNNKernels, including
`Lava.GEMM_TILE` used as "16, the tile size" in 75 places, and five kernels marked
`@kernel cpu=false` in a package whose verification story depends on running the
same source on the CPU. Some of that is unavoidable — coopmat intrinsics have no
KA equivalent — and some is incidental.

(a) is a perfectly good answer and costs the least. Whatever the answer, fix the
two docstrings that overstate it today: `conv.jl:105` claims "Same kernel on every
backend, CPU included" of a path that reaches `Lava.coopmat_gemm!`, and
`matmul.jl:56` claims "Nothing here knows tensor cores exist" three lines above a
predicate that checks `Lava.coopmat_gemm_available()`.

**Exit:** the decision written into the README, and the three docstrings corrected.

## What NOT to do yet

Do **not** write per-feature fallbacks. "Give every coopmat2-gated kernel a tested
KHR path" *is* `kernel-library-review.md` finding 1, and doing it before
`kernels-refactor` lands plan dispatch means writing more `if has_feature` chains
that the refactor then deletes. Gather the evidence; implement after.

## Note on this machine

It has **no CUDA**, so it can never run a model export — exports are
desktop-or-3070 only, and `models-to-port.md` records that a CPU export decomposes
attention and the exporter now refuses one. It does have the most memory of the
three (32 GB unified), which makes it the right home for the largest model later.

## Report

Append to `REPORT.md`: the capability dump, the pass/fail/skip table, every
portability bug found, and the finding-8 decision with its reasoning.
