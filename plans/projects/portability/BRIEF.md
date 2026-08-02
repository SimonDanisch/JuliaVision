# portability

**Machine** AMD laptop (Ryzen AI MAX, Radeon 8060S, RDNA 3.5, 32 GB unified)
**Repos** `dev/Lava` and `dev/JuliaVision` @ `sd/portability`
**Read first** `plans/GUARDRAILS.md`, then `kernel-library-review.md` finding 8.

This machine is the only one that can run the KHR-only path, a subgroup of 64,
and RDNA3 WMMA. Its first job is not to write fallbacks — it is to find out what
is actually broken, because several things were *claimed* portable on 2026-08-02
and have never run here.

## Where this sits relative to the refactor

Not downstream of it — **upstream**. `kernels-refactor` step 3 has to make plan
objects per-device, and `lava-core` phase 2 has to make the caches per-device;
both are being designed against a single card's answers because that is all
anyone has. Your capability dump is what they need, and the sooner it exists the
less of that design is guesswork.

So: **measure now, do not wait.** But because the two desktop projects are
rewriting the same files:

- **Pin your baseline.** Record the exact commits you tested (`git -C dev/Lava
  rev-parse HEAD` and the same for JuliaVision) at the top of your report. A
  finding without a commit is unactionable once the refactor lands.
- **Do not fix library source.** Test-level fixes are fine — a test that
  hardcodes 32 where it should ask `subgroup_size()`, a skip condition that is
  wrong. Anything in `src/` gets *filed*, not patched: it will land in
  `kernels-refactor` or `lava-core`, and a parallel fix on this branch will
  conflict with a rewrite of the same function.
- **A crash at device creation is the exception** — if Lava will not initialise
  on RDNA3 at all, nothing else can be measured, so fix that minimally and say so
  loudly.

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

## Phase 3 — the refactor has landed, and your segfault is fixed

**Rebase first.** Everything Phase 1 was told to wait for is now on `sd/nvidia`
and `sd/kernels-refactor`, so the two standing constraints above are lifted in
this order: plan dispatch exists, so per-feature fallbacks are now the *right*
shape rather than a chain the refactor would delete; and `src/` is no longer
being rewritten under you, so fix what you find.

What changed that concerns this machine:

- **Your blocker is diagnosed and fixed.** The report's "floating GC race" —
  one segfault, three distinct locations, suite never completes — is
  `vk_reset_device!` failing to retire the old context. Its comment claimed
  pre-reset buffers skip Vulkan calls because the old context has `device_lost`
  set; that only held when a device *loss* caused the reset, and a voluntary
  `vk_reset_device!()` left it false. The context and its buffers then became
  garbage in the same collection, where Julia does not order finalizers:
  `Vulkan.Device`'s finalizer destroys the device and `vk_free!` then calls
  `query_timeline` on it. Ten-line MWE, fixed, and it reproduces back at
  `046b1ed` — so it was never anything the per-device work introduced. **Re-run
  the full suite; it should complete now, which is the first time you get a
  real pass/fail table.**
- **Per-device state is done and has an acceptance test.**
  `dev/Lava/test/twodevice_probe.jl` builds a GPU and a lavapipe context in one
  process and asserts the pipeline cache grows twice. It found seven things
  reading found none of — the function-pointer table and the memory pool
  especially. Run it here: this machine's second device is different from the
  desktop's.
- **A coopmat pipeline now refuses to build where it cannot pin 32 lanes**
  rather than returning wrong numbers quietly. RDNA 3.5 advertises
  `VK_EXT_subgroup_size_control` with `minSubgroupSize == 32`, so this *should*
  be invisible here — if it fires, that assumption is wrong and it is the most
  important thing you can report.
- **`Device` is the object kernels ask.** `coopmat, tile, subgroup,
  coopmatsubgroup, sharedbudget, workgrouplimit, cores, launchgroup` — note
  `subgroup` and `coopmatsubgroup` are deliberately separate, because Lava pins
  coopmat modules to 32 while RDNA's device default is 64. Anything still
  keyed on the wrong one of those two is a bug worth filing.

### The measurement this machine should own

Shared-memory bank conflicts, because the answer is hardware-specific and the
NVIDIA numbers are now written down to compare against. `flashepad`/`flashrpad`
in `DNNKernels/src/kernels/extern/flash.jl` carry both sweeps and both defaults;
`sdpaflashcm!` takes `epad` and `rpad` as keywords, so a sweep needs no edit.

On this desktop `epad` is worth **-31.4% at head dimension 64** and `rpad` is
worth -13.7% in the same microbenchmark but **+6.7% on the real encode**, which
is why it ships off. Both of those are statements about a 32-bank, 4-byte,
wave32 machine. RDNA 3.5's LDS is banked differently and its wave width may not
be 32 — so the defaults are very likely wrong here, in a way that costs a third
of the attention time, and nobody can find that out on the desktop.

Run the `epad × rpad` sweep at `E ∈ (16, 32, 48, 64, 72)`, then check the winner
against a real model rather than the sweep — that is exactly where the desktop's
`rpad` conclusion inverted.

## Note on this machine

It has **no CUDA**, so it can never run a model export — exports are
desktop-or-3070 only, and `models-to-port.md` records that a CPU export decomposes
attention and the exporter now refuses one. It does have the most memory of the
three (32 GB unified), which makes it the right home for the largest model later.

## Report

Append to `REPORT.md`: the capability dump, the pass/fail/skip table, every
portability bug found, and the finding-8 decision with its reasoning.
