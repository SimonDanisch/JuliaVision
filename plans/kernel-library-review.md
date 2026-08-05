# Kernel library design review

> Paths in this file — `dev/Lava`, `dev/JuliaVision`, `tools/`, `gen/` — are relative to the
> **workspace root**: the untracked scratch directory that contains `dev/`. This file used to
> live there; it moved into the repo so all three machines share one copy.


A read of `dev/JuliaVision/DNNKernels` (5,371 lines of runtime + ~3,000 of kernel
sources) and the parts of `dev/Lava` it sits on, against the questions asked:
cross-platform reach, interface cleanliness, usability, maintainability, use of
dispatch and structs, hardcoding discipline, globals discipline, whether shape
specialisations are reachable through one entry point, and whether hardware
capability selection is expressed as types rather than as `if has_feature`.

No code was changed. Every claim below carries a `file:line` so it can be checked
before anything is done about it. Both packages are being actively worked on, so
this is a work list, not a verdict.

## Scorecard

| dimension | grade | one-line |
|---|---|---|
| op-level dispatch | **very good** | 62 `runop!` methods on `Val{Symbol}`, tag precomputed in the `Op` struct |
| entry points per kernel family | **good** | `sdpa`, `convolution!`, `matmul!`, `layernorm!` all exist and `ops.jl` calls only those |
| documentation | **exceptional** | ~25% of lines are prose, and most record a *measurement*, not an intention |
| structs | **mixed** | `Model{B}` and `Op` are right; `Ctx` has six `::Any` fields and four telescoping constructors |
| capability selection | **weak** | boolean predicates + `if` chains; the type-level mechanism exists in Lava and is unused |
| globals | **weak** | 34 mutable toggles in DNNKernels, 84 `Ref`s in Lava; the reachable target is **0** and **1** (the context) — see finding 3 |
| cross-platform | **weak, and partly mislabelled** | KernelAbstractions on the surface, 176 direct `Lava.*` references underneath |
| hardcoding | **mixed** | device constants (48 SMs, 32 lanes, 48 KB) are literals; Lava can now answer all three (9a), the five call sites still don't ask |
| usability | **poor for a second user** | behaviour is configured by mutating unexported globals; nothing is in the exported API |

Short version: the **graph layer is well designed** and the **kernel layer is
well measured but poorly encapsulated**. Nothing here is a rewrite. The two
structural items — capability dispatch and the toggle population — are each a
contained refactor, and doing them in the right order makes the rest fall out.

## What is already right

Worth stating plainly, because the findings below should not disturb any of it.

**Op dispatch is textbook.** `runop!(ctx, op, op.tag)` (`execute.jl:651`) with 62
methods on `::Val{Symbol("aten.name")}` and a single erroring fallback at
`execute.jl:598`. The tag is built once when the graph loads (`graph.jl:33`) with
the comment explaining that interning a Symbol per op per step showed up in the
profile. Adding an op is one method in one place, and an unimplemented one fails
by name. Do not touch this.

**Entry points exist and are the only thing the graph calls.** `ops.jl` reaches
`sdpa` (`ops.jl:716`), `matmul!` (`ops.jl:707`), `convolution!`, `layernorm!` and
nothing below them. The answer to "do the shape specialisations get called
individually" is **no** — `sdpa` picks between the fused coopmat kernel, the
two-GEMM coopmat path and the blocked/plain scalar path; `convolution!`
(`conv.jl:99`) picks between 1×1-as-GEMM, im2col+coopmat, implicit GEMM and the
direct grouped kernel. The layering is right. The complaints below are about
*how* those choices are expressed, not that they are missing.

**The prose is the most valuable asset in the repo.** `flash.jl` is 363 comment
lines in 1,468; `attention.jl` 202 in 802. They record measurements, dead ends
and the reason a form that "looks equivalent" is not — e.g. `gemm.jl:292`, why a
`Val{BLK}`-gated single kernel collapses every shape to 0.15 ms, or
`attention.jl:134`, why the blocked kernels are `@eval`-generated instead of
parameterised. **Any refactor below must carry these forward verbatim.** They are
the record of what was already tried.

**`Model{B}` and `Ctx{B,S}` are parameterised on the backend** with the reason
documented at `driver.jl:19` — `backend::Any` put 4,062 `KA.__run` CodeInstances
into SAM 2's package image. That is precisely the right instinct; it just was not
carried to the other fields.

## Findings

### 1. Capability selection is boolean predicates, not types

**The pattern today.** Four predicates —
`mm_coopmat_applicable` (`matmul.jl:87`), `conv_coopmat_applicable`
(`conv_coopmat.jl:74`), `coopmat_sdpa_applicable` (`attention.jl:655`),
`flashcm_applicable` (`flash.jl:1406`) — each returning `Bool`, each consumed by
an `if ... && return` chain in the entry function. Each one mixes three
independent questions:

    function flashcm_applicable(q, k, v, bias, Lq, Lk)
        FLASHCM[] || return false                          # a toggle
        bias === nothing || return false                   # a problem property
        eltype(q) === Float16 && ... || return false        # a type property
        Lava.coopmat_gemm_available() || return false      # a device property
        flashcm_tiling(...) !== nothing                    # a shape/resource fit
    end

Device capability, element type, problem shape and a debug switch all collapse
into one `Bool`, at runtime, on every call.

**Why it costs.** A new backend, or an older card without coopmat, is expressible
only by making `coopmat_gemm_available()` return `false` — so every path is
selected by negation and nothing can *add* a path. There is no place to hang an
implementation that a different device would prefer. The predicates also cannot
be tested without a device.

**The mechanism already exists, twice, in this codebase.**

- `Lava/src/device/acceleratedmatrix.jl:30` defines
  `MatrixA`/`MatrixB`/`Accumulator <: MatrixUse` and uses them as type parameters
  on `AcceleratedMatrix{T,M,N,Use}`. That is exactly the trait pattern, applied
  correctly, one directory away from the code that does not use it.
- `Lava/src/compiler/target.jl:48` defines
  `Base.Experimental.@MethodTable lava_method_table` with an `@overlay` macro and
  `GPUCompiler.method_table(::LavaCompilerJob)`. The overlay route is available.
- `Lava/src/array/ka_backend.jl:56` defines `LavaBackend <: KA.GPU` — a concrete
  backend type that currently carries no capability information.

**Sketch.** Give the capability set a type and dispatch the entry function on it:

    abstract type MatmulPath end
    struct CoopmatPath{TILE} <: MatmulPath end
    struct ScalarPath       <: MatmulPath end

    matmulpath(::LavaBackend, out, A, B) = ...   # one place decides
    matmul!(out, A, B, bias; kw...) = matmul!(matmulpath(backend, out, A, B), out, A, B, bias; kw...)
    matmul!(::CoopmatPath, ...) = ...
    matmul!(::ScalarPath,  ...) = ...

The selection stays one function (it is still a runtime question — shapes are not
known at compile time), but the *implementations* become methods, a new one is
additive, and each is callable directly in a test by constructing its path type.
Whether the capability travels in `LavaBackend`'s type parameters or in a
`caps(backend)` trait function is the open design question; `CoopMat2Caps`
(`device.jl:225`) is already a struct of eight `Bool`s on the context and is the
natural source.

### 2. Applicability is checked twice, in two different vocabularies

`sdpa` asks `flashcm_applicable` (`attention.jl:736`), then calls `sdpaflashcm!`,
which **re-checks** element types (`flash.jl:1433`), `coopmat_gemm_available`
(`:1434`), extent divisibility (`:1435`), shared-memory fit (`:1436`), a write-out
tiling condition `flashcmfits` cannot see (`:1439`), and strided-root
extractability (`:1449`) — returning `false` from six places to mean "did not
run", which the caller reads as `&& return out` falling through to the next path.

Two overlapping predicates and a `Bool`-means-"declined" convention. If the two
drift, the observable symptom is a silent change of algorithm — the exact class
of bug that is hardest to notice, because the answer stays correct.

**Sketch.** One function that returns a plan or `nothing`:
`flashcm_plan(q,k,v,bias) -> Union{Nothing,FlashCMPlan}`, carrying `(BR, BC, NW,
strides, roots)`. `sdpa` tries plans in order and calls the first non-`nothing`.
Rejection reasons become fields or an enum on the `nothing` branch, which also
makes "why did this shape not take the fast path" answerable without a debugger.

### 3. Thirty-four mutable global toggles in DNNKernels, eighty-four `Ref`s in Lava

Full list at the top of the evidence appendix. In DNNKernels: `FLASHCM`,
`FLASHCM_DENSIFY`, `FLASHCM_REGO`, `FLASHCM_LAZYRESCALE`, `FLASHCM_ONEPASS`,
`FLASHCM_HELD`, `FLASHCM_PERELEM`, `FLASHCM_RESCALE`, `FLASHCM_CLAMP`,
`FLASHCM_MINGRID`, `FLASH_SHARED_BUDGET`, `ATTN_MINL`, `COOPMAT_MINL`,
`COOPMAT_QCHUNK`, `ATTN_SOFTMAX_ROWS`, `CONV_1X1_GEMM`, `CONVT_GEMM`,
`CONV_CRS_PAD`, `MATMUL_FUSED`, `LN_FUSED`, `LAUNCH_FLAT`, `LAUNCH_GROUP`,
`FOLD_OUTCASTS`, `FOLD_CONSTSUBGRAPHS`, `DUPLICATE_FUSION`, `DUPLICATE_MAX`,
`CACHE_DECODER_INPUTS`, `REPLAY_DECODE`, `SEGMENT_TIE`, plus five diagnostic
`Ref{Any}`s.

These are not all the same thing and should not all get the same treatment:

- **Diagnostics** (move onto `Ctx`, do not keep): `PLAN_MISSES`, `OPTIMES`,
  `OPDOUBLE`, `OPDOUBLEFILTER`, `LAUNCH_PROBE`. The usual defence is that they
  must be reachable from outside a call stack that does not thread them — which
  is a fact about today's signatures, not a constraint. `Ctx` already reaches all
  62 `runop!` methods carrying `ws`, `plan`, `lazy`, `rec`. One more field and
  `execute!(...; diag)` retires five globals, makes two concurrent executions
  with different instrumentation possible, and removes the save/restore dance
  from the tests.
- **Device-derived caches in Lava** (move onto `VkContext`): `BLIT_PIPELINE`,
  `PREPARE_INDIRECT_*`, `PIPELINE_CACHE`, `DEVICE_SUBGROUP_SIZE`,
  `SUBGROUP_SIZE_CONTROL`, `LAUNCH_PLAN_CACHE`, `RESERVED_ARG_SLABS`. Each is
  state *belonging to a device* held in a module-level `Ref`, so each must be
  cleared by hand when the device goes away — which is what the **ten**
  `push!(RESET_CALLBACKS, ...)` sites (`device.jl:588`) exist to do. That list is
  correct today, and it is correct by upkeep: the eleventh cache whose owner
  forgets to register one holds a stale value against a new device, silently.
  `ctx.compute` from 9a needs no callback at all — it lives on the context and
  dies with it. That is the argument, and it is already demonstrated in the tree.
- **Settled decisions masquerading as switches** (delete the switch, keep the
  winner): `LN_FUSED`, `MATMUL_FUSED`, `LAUNCH_FLAT`, `CONV_1X1_GEMM`,
  `CONVT_GEMM`, `FLASHCM`, `FLASHCM_ONEPASS`, `FLASHCM_PERELEM`,
  `FLASHCM_LAZYRESCALE`, `GEMM_STAGED`, `GEMM_VEC2`, `GEMM_NARROW`. Each defaults
  to the measured winner, each is read in exactly one file, and the losing branch
  is dead code that still has to compile and still has to be correct.
- **Tuning constants that belong to a shape, not to a session** (per the global
  rule — a const used in one place should be a kwarg): `ATTN_MINL`,
  `COOPMAT_MINL`, `COOPMAT_QCHUNK`, `CONV_CRS_PAD`, `FLASHCM_MINGRID`,
  `FLASH_SHARED_BUDGET`, `LAUNCH_GROUP`. These want to be fields on the
  path/plan object from finding 1, defaulted per device — **and "per device" is
  exactly what a module-level `Ref` cannot express**, see the constraint below.

**The target is zero globals in DNNKernels and one in Lava.** An earlier draft of
this section said "any toggle that survives as a `Ref` must hold a
device-independent value". That concedes too much — it presupposes toggles
survive. Sorted by where the state actually belongs, almost nothing does:

| what it is | where it belongs | globals retired |
|---|---|---|
| settled decision | deleted, winner inlined | 12 |
| per-call tuning | plan object field / kwarg (finding 2) | 7 |
| live experiment | plan object kwarg, literal default | 5 |
| per-run diagnostics | a field on `Ctx` | 5 |
| device-derived cache | a field on `VkContext` | 7+ in Lava |
| **the device itself** | **`VK_CONTEXT_REF` — the one that stays** | — |

That last row is the rule from `CLAUDE.md` applied rather than bent: *if
something is REALLY global state, restrict it to one global.* A Vulkan context is
process-global, everything else in the table is reachable **through** it.

The test for the last row is not "is it awkward to thread" — threading an object
through a call stack is always available in code we own, so awkwardness is a cost
estimate, not a reason. It is **"would two of these in one process be
incoherent?"** A second `VkContext` is a second device and is meaningful; a
second `OPTIMES` is just two runs measuring different things, which is the
feature. Only the context fails that test.

The experiments row is the one that looks hardest and is not.
`sdpaflashcm!` already takes `rego`, `held`, `lazyrescale`, `onepass`, `clamp`,
`rescale` as keyword arguments — the globals are only supplying their *defaults*.
Literal defaults on the plan object do the same job, and the toggle's docstring
(which is the valuable part) moves with it.

**Why this is a constraint and not just tidying.** A `const X = Ref(...)` at
module scope is evaluated when the package *loads*, which includes when it
precompiles. The moment such a default becomes device-derived —
`Ref(Lava.max_shared_memory())` in place of `Ref(48 * 1024)` — the package builds
a Vulkan context during precompilation, and `precompile-bakes-gpu-state` records
what that costs: a dead `VkContext` shipped inside the package image. This is
not hypothetical; it is why two of the five sites in finding 9 were left alone
after the Lava query landed in 9a.

Under the table above the hazard does not need managing, because it cannot
arise: no module-level binding holds a value, so none can hold a device's value.

**There are no exceptions, and an earlier draft claimed two.** Both were
"the current call stack does not carry it", which describes the code rather than
constrains it — this is our code, and threading an object through a call stack is
a refactor, not a barrier. Checked, both dissolve, and both leave the design
better than they found it:

* **`LAUNCH_PROBE`** (`launch.jl:173`). The claim was that `launch!` takes a
  backend rather than a `Ctx` across ~28 call sites. True, and beside the point:
  every one of those sites sits inside a kernel entry function
  (`sdpa`, `upsample_bilinear2d!`, …) that already receives `backend` *and* `ws`
  — both of which are fields of the `Ctx` its caller is holding. Passing `ctx`
  instead of `(backend, ws)` is **one argument in place of two**, and it hands
  those functions the workspace, the diagnostics and the device capabilities from
  9a at the same time. This is a simplification that happens to retire a global,
  not a cost paid to retire one.
* **`VALIDATION_MESSAGES` / `VAL_RING_*`** (`device.jl:422`). The claim was that
  the debug-messenger callback must not allocate, log, or take locks and so
  "has no way to reach a Julia object". The first half is true and is quoted from
  the callback's own comment; the second half is wrong. `debug_callback` already
  takes `p_user_data::Ptr{Cvoid}` (`device.jl:1430`) — Vulkan's own mechanism for
  exactly this — and `setup_debug_messenger` (`:1535`) simply never passes
  anything through it. The ring is already a preallocated raw buffer written by
  `@inbounds` stores, so reaching it through a pointer instead of a module
  binding changes nothing about allocation or locking.

  It also fixes a real lifetime mismatch. The ring's lifetime is currently the
  *module's* while the messenger's is the *context's*, which is why
  `vk_reset_device!` has to hand-patch `VAL_RING_READ[] = VAL_RING_WRITE[1]` to
  stop a new instance replaying the old one's undrained messages. Owned by the
  context, the buffer and the messenger die together and that line goes away.
- **Live experiments** (keep, but mark them): `FLASHCM_REGO`, `FLASHCM_HELD`,
  `FLASHCM_DENSIFY`, `FLASHCM_CLAMP`, `FLASHCM_RESCALE`. Their docstrings already
  say they are measured-and-lost; a naming convention (`EXPERIMENT_*`) or a single
  `mutable struct FlashTuning` would keep them from reading as configuration.

**Concrete harm, not just style.** `sdpaflashcm!` (`flash.jl:1420-1428`) takes
**thirteen** keyword arguments defaulting to seven different globals. Tests
save/restore them by hand (`test_flash.jl:152`, `test_convtranspose_gemm.jl:107`),
which is the classic mutable-global test hazard: a failing test leaves the global
flipped for every test after it, and nothing is parallelisable.

### 4. Runtime `Bool`s converted to `Val` at the call site

`flash.jl:1457` passes `Val(rego), Val(held && !rego), Val(clamp)` — values that
came from mutable globals — into the kernel's type domain. Each independent
`Bool` doubles the number of compilable kernel instances, and the values are not
known until run time, so the frozen SPIR-V cache
(`KERNELS_VERSION`, `DNNKernels.jl:46`) can only hold the combinations that
actually occurred. With five such toggles that is a 32-entry surface for one
kernel. Folding them into a single `FlashVariant` type — or, better, deleting the
settled ones per finding 3 — collapses it.

### 5. A generated kernel family with a hand-written dispatcher

`ATTN_BLOCKS = (2,4,8,16,32)` (`attention.jl:132`) generates ten kernels in a
loop. They are dispatched by `scoresblocked!` / `applyblocked!`
(`attention.jl:228`, `:237`) as a hand-written chain:

    tk == 32 && return attn_scores_b32!(backend)(...)
    tk == 16 && return attn_scores_b16!(backend)(...)
    ...

Adding a block size to `ATTN_BLOCKS` generates the kernel and silently does not
dispatch to it. The generation loop should emit the dispatch method too —
`scoresblocked!(::Val{32}, ...)` per entry, with `blockfor` returning `Val`. Note
the *kernel* genuinely cannot be `Val`-parameterised (`attention.jl:134` records
why: the closure over the accumulator tuple defeats inference and every `muladd`
becomes a dynamic call). That constraint applies to the kernel body, not to the
dispatcher.

Same shape in Lava: `GEMM_BLOCK_KERNELS` (`gemm.jl:359`) and
`GEMM_STAGED_KERNELS` (`:517`) are `Dict`s keyed by tiling, populated by `@eval`,
looked up at run time.

### 6. `Ctx` is six `::Any` fields and four telescoping constructors

`execute.jl:27`. The struct threaded through every one of the 62 `runop!` methods
has `slab::S`, `backend::B` typed — and `plan::Any`, `ws::Any`, `lazy::Any`,
`rec::Any`, plus `values::Dict{String,Any}`. The comment immediately above it
(`execute.jl:20`) documents that leaving `slab` untyped is what put 3,948 CPU
kernel specialisations into the package image. The lesson was applied to one
field out of six.

The four positional constructors at `:40-:47` that fill in trailing `nothing`s
are the usual sign that this wants `Base.@kwdef` with typed defaults, or
`Union{Nothing,Workspace}` etc. rather than `Any`.

### 7. A shape predicate that silently reads a toggle

`onebyone(w, stride, padding, dilation, groups)` (`conv.jl:88`) answers "is this a
1×1 convolution" — and begins with `CONV_1X1_GEMM[] &&`. When the toggle is off,
a 1×1 convolution reports that it is not one. The name lies, and the caller
(`conv.jl:115`) reads as though it is asking about the problem. Same shape in
`flashcm_applicable`'s leading `FLASHCM[]`. Predicates about the problem should
not be able to answer a question about configuration.

### 8. Cross-platform: KernelAbstractions on the surface, Lava underneath

The package `using`s KernelAbstractions and writes `@kernel` bodies, which reads
as backend-portable. It is not, and the gap is not small: **176 direct `Lava.*`
references** in DNNKernels, including `Lava.GEMM_TILE` (75), `Lava.splitidx` (18),
`Lava.AcceleratedMatrix` (13), `Lava.coopmat_gemm!` (6), `Lava.LavaArray` (9),
`Lava.FastDiv`, `Lava.cart`, `Lava.launchgroup`, `Lava.capture`/`replay!`.

Some of this is unavoidable — coopmat intrinsics have no KA equivalent, and that
is a legitimate reason for a backend-specific kernel. But some is incidental:
`Lava.GEMM_TILE` is used as "16, the tile size" in 75 places including pure shape
arithmetic, `Lava.splitidx` and `Lava.FastDiv` are general index utilities that
would work anywhere, and `Lava.cart32` is used in `ndmap_flat!` — the *generic*
elementwise launcher.

Two specific mislabels to fix in the prose regardless of what happens to the code:

- `conv.jl:105` says "Same kernel on every backend, CPU included" of
  `convolution!`, but the path it actually takes calls `matmul_coopmat!` →
  `Lava.coopmat_gemm!`. The claim is true of `convolution_direct!` only.
- `matmul.jl:56` says "Nothing here knows tensor cores exist, which is the
  point" — three lines above `mm_coopmat_applicable`, which checks
  `Lava.coopmat_gemm_available()` and `Lava.GEMM_TILE`. That docstring described
  an earlier design.

Also: `@kernel cpu=false` on `layernorm_kernel!`, `attn_softmax_rows!`,
`attn_flash!`, `attn_flash_cm!` and one more — five kernels with no CPU form, in a
package whose verification story (`verify.jl`) depends on running the same source
on the CPU.

**Decide and write down which of three things this package is:** (a) a Lava kernel
library that uses KA for convenience, (b) a portable library with a Lava fast
path, or (c) portable-with-Lava-required. It is currently (a) documented as (b).
(a) is a perfectly good answer and costs the least — it just means the KA facade
should stop implying otherwise.

### 9. Device constants are literals — Lava side fixed, consumers open

    conv_implicit.jl:69    convtiles(K, NPQ; cores::Int = 48)
    conv_implicit.jl:286   convsplit(nbk, nbn, nblk; cores::Int = 48)
    flash.jl:112           FLASH_SHARED_BUDGET = Ref(48 * 1024)
    flash.jl:399,431,440   NT = NW * 32, tid ÷ 32          # subgroup width, literal
    flash.jl:1395          FLASHCM_MINGRID = Ref(48)

`48` is this card's SM count, `49152` its `maxComputeSharedMemorySize`, `32` its
subgroup width. When this review was written Lava had **no** `shader_core_count`
query and **no** shared-memory-limit query, which is why all three are literals.
`device_subgroup_size` did exist (`pipeline.jl:199`), and `GEMM_SUBGROUP` — and
`flash.jl` still writes `32` as a literal in five places anyway.

These are the constants that most need *not* to be hardcoded, because they are
exactly what a second GPU changes. The connection to the porting plan is why this
was first on the list: `kernels-to-port.md` item 1 wants llama.cpp's split-count
derivation, `split_k = shader_core_count * 2 / total_wgs`, and that line could
not be written at all. **The Lava half is now built (9a); the five sites above
are still literals.**

#### 9a. DONE in Lava, 2026-08-02 — consumers still to wire

Built and tested. `Lava.DeviceCompute` (`runtime/device.jl`) is queried once at
context creation and cached on `VkContext.compute`; three accessors read it:

    Lava.shader_core_count()    -> 48       Union{Nothing,Int}
    Lava.shader_warps_per_sm()  -> 48       Union{Nothing,Int}
    Lava.max_shared_memory()    -> 49152    Int (core limit, always known)

`nothing` rather than llama.cpp's `0` sentinel, deliberately: the number's job is
to be a denominator, and a `0` yields zero splits *silently* — a wrong launch
with a plausible answer. `nothing` cannot be divided, so a missing fallback is a
`MethodError` at the first arithmetic. Callers write
`something(Lava.shader_core_count(), 16)`.

Vendor coverage follows llama.cpp (`ggml-vulkan.cpp:6138-6146`): NVIDIA via
`VK_NV_shader_sm_builtins`, AMD via `VK_AMD_shader_core_properties2`, otherwise
unknown. Neither extension is *enabled* on the logical device — these are
physical-device property queries — but each is guarded by `has_extension`.
Marked `public` (not exported), matching `device_subgroup_size`'s treatment.

`test/test_device_compute.jl`, 14 tests, wired into `runtests.jl`. It asserts
against whatever the running device reports rather than a vendor's numbers, with
one unconditional check: **an advertised extension must yield a non-zero count.**
That is the pNext-chain regression below, which otherwise degrades silently.

Both hardcodes are confirmed equal to the queried values (`cores = 48`,
`48 * 1024 = 49152`), so nothing changes behaviour here. And the line this was
all for now computes:

    total_wgs=  8 -> split_k = 12      # SAM 2 decode's cross-attention
    total_wgs= 48 -> split_k = 2
    total_wgs=512 -> split_k = 1       # encoder: no splitting, no overhead

**Still to do — and one of the two sites has a hazard.** `convtiles` /
`convsplit` (`conv_implicit.jl:69`, `:286`) take `cores::Int = 48` as a *keyword
default*, evaluated per call, so those are a one-line change each. But
`FLASH_SHARED_BUDGET = Ref(48 * 1024)` (`flash.jl:112`) and
`FLASHCM_MINGRID = Ref(48)` (`:1395`) are **module-level** `const`s, evaluated
when the package loads — i.e. during precompilation. Reading a device property
there would build a Vulkan context at precompile time, which is exactly the
failure `precompile-bakes-gpu-state` records. Those two need the query moved to
the use site, which is a small design change to `flash.jl` rather than a
substitution, and `flash.jl` is under active edit. Left deliberately.

#### 9b. Why it was hardcoded — measured 2026-08-02

Probed on this machine before proposing it. Every piece already exists:

    device: NVIDIA RTX 4000 Ada Generation
      shader_sm_count             = 48      <- the hardcoded `cores = 48`
      shader_warps_per_sm         = 48
      maxComputeSharedMemorySize  = 49152   <- the hardcoded `48 * 1024`

- `VK_NV_shader_sm_builtins` is advertised by this device (extension revision 1).
- Vulkan.jl already binds `PhysicalDeviceShaderSMBuiltinsPropertiesNV` with
  `shader_sm_count` / `shader_warps_per_sm` (`generated/linux.jl:11516`).
- Lava already uses the exact query idiom it needs —
  `Vulkan.get_physical_device_properties_2(phys_dev, SomePropertiesStruct)` then
  read `.next` — twice, at `runtime/device.jl:1097` and `:1115`.
- `maxComputeSharedMemorySize` is **core** Vulkan (`VkPhysicalDeviceLimits`), no
  extension at all: `get_physical_device_properties(pd).limits.
  max_compute_shared_memory_size`. Lava can read it today.

So this is roughly twenty lines: query at context creation, cache three fields on
`VkContext`, export accessors. Both hardcoded constants are **confirmed correct
for this device**, so wiring the query changes no behaviour here — it only makes
the numbers travel to the next GPU.

`shader_warps_per_sm = 48` is a bonus nobody asked for: it is the occupancy
denominator. `lava-flash-attention-occupancy` records residency being *measured*
(2 workgroups/SM at 48,904 B); with this it can be computed.

**The trap, and it is a silent one.** A first probe read `shader_sm_count = 0`.
Not an unsupported extension, and *not* the call failing — measured on both
instances, the base `VkPhysicalDeviceProperties` comes back fully populated
either way (device name, `api_version`, limits, all correct). It is **only the
pNext chain** that is dropped, and the rule is opt-in, not version:

    instance built as                              sm_count   rt_handle
    ---------------------------------------------  --------   ---------
    no ApplicationInfo (=> apiVersion 1.0)                0           0
    explicit api 1.0                                     0           0
    explicit api 1.0 + VK_KHR_get_physical_device_properties2
                                                        48          32
    explicit api 1.1                                    48          32
    explicit api 1.3                                    48          32

`vkGetPhysicalDeviceProperties2` is core in Vulkan 1.1 and an instance extension
before that. Chain-filling happens if the application legally opted into it
**either way** — API ≥ 1.1, or 1.0 with `VK_KHR_get_physical_device_properties2`
enabled. Without either, the driver services the base struct and ignores the
chain. The function returns `void`, so there is no status to check even in
principle, and validation layers are off by default here (`LAVA_VALIDATION=1`).

**Lava is on the right side of this** — `runtime/device.jl:567` asks for
`v"1.4.0"`, which is why its RT properties query at `:1097` returns real numbers.
Nothing to fix; the zeros were an artefact of the throwaway probe instance.

The relevance to us is the failure mode, not the bug: a `0` here flows straight
into `split_k = shader_core_count * 2 / total_wgs` and yields 0 splits. Follow
llama.cpp, which stores `0` when unavailable and guards **every** use site —
`shader_core_count ? shader_core_count : 16` (`ggml-vulkan.cpp:10616`) and
`> 0 ? ... : 32` (`:10776`). Port the fallback with the feature, not after it.

**Is this a Vulkan.jl / VulkanCore defect?** The bindings are correct — the
struct is bound, chained and read back properly. Two things make it easy to fall
into, if we ever want to patch our clone: `Vulkan.Instance(layers, exts)` leaves
`application_info` at `C_NULL`, and a NULL `pApplicationInfo` means apiVersion
1.0 per spec, so the shortest way to make an instance is the one that silently
drops chains. And `PhysicalDevice` does carry its parent `Instance`
(`generated/linux.jl:27877`), so `get_physical_device_properties_2` has
everything it needs to warn when the chain will be ignored — it just doesn't.

Vendor coverage, from llama.cpp's `ggml-vulkan.cpp:6138-6146`, if this ever needs
to be more than NVIDIA: `VK_NV_shader_sm_builtins` → `shaderSMCount`;
`VK_AMD_shader_core_properties2` → `activeComputeUnitCount`; Intel → a
device-ID table (`ggml_vk_intel_shader_core_count`); otherwise 0.

### 10. Model drivers live inside the kernel library

`DNNKernels.jl:1` says the package "holds no per-model instantiation list: kernel
*sources* live here and are shared across models". It then includes `sam2.jl`
(362 lines) and `wan.jl` (321), and `Model{B}` (`driver.jl:27`) carries
`memevery`, `memframes`, `topk` — SAM 2's memory-bank parameters — on a struct
named `Model`. `SEGMENT_TIE`, `CACHE_DECODER_INPUTS` and `REPLAY_DECODE` are SAM
2 policy in the kernel package's global namespace.

With SAM3, BasicVSR++ and Wan queued (`models-to-port.md`), this is the seam that
will hurt first: the fourth model adds a fourth `include` and three more fields to
`Model`. Either `Model` becomes generic (`config::C` parameter) with the driver in
the model package, or the package is honestly renamed.

### 11. Usability by a second person

Nothing in finding 3's list is exported (`DNNKernels.jl:21` exports eight names).
Every performance-relevant decision is made by mutating an unexported
`const ... = Ref` found by reading the source. The docstrings are excellent and
`?FLASHCM_MINGRID` does work — but only if you already know the name.

An earlier draft proposed an exported `tuning()` / `withtuning(f; kwargs...)`
pair here, as the cheap fix that gives discoverability and kills the test
save/restore hazard without the larger refactor. **Dropped**, because it is a
nicer interface *to* the globals and finding 3's target is not to have them.
Building it first would mean writing an API whose whole job is to manage state
that steps 3–5 delete, and — worse — publishing it invites callers, which makes
the state harder to remove afterwards.

What replaces it falls out of the refactor rather than preceding it: once the
knobs are fields on a plan object, `flashcm_plan(q, k, v, bias)` *is* the query
("what will this call do, and with what tiling"), constructing one with a keyword
override *is* the experiment, and neither needs a global to be set or restored.
The docstrings — which are the genuinely valuable part of the current toggles —
move onto the fields.

The one thing worth doing before then is documentation, not API: a short section
in `lava-dnn.md` naming the knobs that exist and where they live, so a second
person can find them without grepping for `= Ref(`.

## Do not "fix" these

Recorded because each looks like a defect and is not, and the reason is in the
source at the line given.

- **`@eval`-generated kernel families instead of `Val` parameters**
  (`attention.jl:134`, `gemm.jl:292`). Both record that the parameterised form
  does not just underperform — it miscompiles or deoptimises. `gemm.jl:292`:
  gating tiles on `i <= BLK` collapsed every shape to ~0.15 ms because a
  cooperative matrix defined inside a conditional stops being an SSA value to the
  emitter.
- **`unsafe_indices=true` on the staged GEMM** (`gemm.jl:611`) — the KA bounds
  guard cost 3x (8.8 vs 26.1 TFLOP/s) because the structurizer left the `muladd`
  under an `OpSelectionMerge`.
- **Two reduction passes in `layernorm_kernel!`** rather than `E[x²] − μ²`
  (`layernorm.jl:12`) — deliberate, for numerics.
- **The 90 `clone`s in SAM 2's graph** (`sam2-encode-where-the-time-is`) — a clone
  trades one expensive read for one cheap read plus one cheap write, and measured
  end-to-end it wins.
- **`blockfor` refusing non-square attention** (`attention.jl:191`) — bounded
  rather than diagnosed, and the docstring says so. Blocking the decoder's
  lopsided shapes reproducibly hangs on `vkWaitSemaphores`. **This is an open bug
  with a workaround**, and it should be on a bug list rather than quietly living
  in a predicate.

## Suggested order

Ordered so each step makes the next cheaper, and so nothing blocks the porting
work already planned.

1. ~~**Expose the device properties**~~ — **DONE** (finding 9a):
   `Lava.shader_core_count` / `shader_warps_per_sm` / `max_shared_memory`, cached
   on `VkContext.compute`, 14 tests. `kernels-to-port.md` item 1 is unblocked.
   *Remaining:* wire `convtiles`/`convsplit` (trivial), then `FLASH_SHARED_BUDGET`
   and `FLASHCM_MINGRID` (needs the read moved off module scope first — see 9a),
   and use `device_subgroup_size` where `32` is a literal in `flash.jl`.
2. **Delete the settled toggles** (finding 3, tier two). Twelve of them, each read
   in one file, each with a measured winner. This removes the dead branch, shrinks
   the `Val` explosion in finding 4, and shortens every function that reads them.
3. **Diagnostics onto `Ctx`** (finding 3): a `diag` field and
   `execute!(...; diag)`, retiring `OPTIMES`, `OPDOUBLE`, `OPDOUBLEFILTER`,
   `PLAN_MISSES` and `LAUNCH_PROBE`. `Ctx` already reaches every `runop!`, so
   this is plumbing, not design — and it is what makes two differently-
   instrumented runs possible at once. The one real edit is giving the kernel
   entry functions `ctx` in place of their `(backend, ws)` pair, which is a
   smaller signature, not a bigger one. Do it before the plan objects, because
   it settles the pattern.
4. **One plan object per kernel family** (finding 2): `flashcm_plan`,
   `matmul_plan`, `conv_plan`, each returning a typed plan or `nothing`, absorbing
   the surviving tuning constants as fields. This is where finding 1's dispatch
   naturally lands, and it removes the duplicated predicates. It is also where
   the device-derived defaults go — computed in the constructor, on a live
   context, never captured at module scope (finding 3's constraint).
5. **Dispatch the entry functions on the plan type** (finding 1). Now additive:
   a new path is a new type and a new method.
6. **Generate the dispatchers alongside the kernels** (finding 5).
7. **Type `Ctx`'s remaining fields** (finding 6) — worth doing after 4, because
   the plan objects give `ws`/`plan` real types to be.
8. **Device caches onto `VkContext`** (finding 3, Lava side): the seven-plus
   `Ref`s that the ten `RESET_CALLBACKS` sites exist to clear. Independent of
   1–7 and can be done whenever Lava is quiet; `ctx.compute` is the worked
   example. Ends with `VK_CONTEXT_REF` as the one global that stays.
9. **Decide the portability story and rewrite the three docstrings that overstate
   it** (finding 8), then move the model drivers out (finding 10).

Steps 1–3 are a day's work and independently valuable. Steps 4–6 are the real
refactor and should be one branch, with the flash and GEMM prose carried across
unchanged. The global count is the progress metric, with no carve-outs:
**34 → 0** in DNNKernels and **84 → 1** in Lava.

### Correction: the metric undercounted its own population

The count above, and every progress report against it, came from
`grep "^const [A-Z_0-9]* = Ref"` (see the appendix). That pattern sees a `Ref`
and nothing else — not `Threads.Atomic`, not `Dict`, not `IdDict`, not
`UInt64[]`. Counted by *mutation* instead of by declared type, Lava had **70**
mutable module-level globals at the point this correction was written, after four
batches had already retired 44 by the old metric.

The miss was not uniform: three of the invisible ones were the same
per-device-state defect the review's finding 3 exists to name.

**And the "1" was wrong in the other direction.** The floor is not one global,
it is about three, and none of the three is what the review nominated:

- `VK_CONTEXT_REF` — the review's intended survivor, "which device is current".
- the pipeline-cache atexit-registered flag — `atexit` is process scope.
- `FROZEN_RT_MEM` — a `LavaRTShader` is bytes and metadata with no device handle,
  and its key is device-independent. Its compute sibling `frozen_mem` moved onto
  `DeviceCaches`; this one is documented as staying.

`PIPELINE_THREAD_CFUNC` is a fourth only technically: it is a `Ref` because
`@cfunction` cannot run at precompile time and must be filled in `__init__`, not
because it holds state.

Everything else that looked immovable was not, and the reasons are worth
recording because each was a *category error* rather than a hard constraint:

- **`RESET_CALLBACKS` is device teardown wearing a global.** Every entry existed
  to empty a module-level cache. Move the caches onto the context and the list
  empties itself; it is deleted, not maintained. State a `VkContext` owns dies
  with it and needs nobody to remember.
- **`ctx.id` and `VK_CONTEXT_COUNTER` were made redundant by the refactor's own
  success.** `ctx.id` existed to key the module-level dictionaries. Nothing in
  `src/` reads it any more.
- **`VAL_RING_*` is per-device, not per-process.** `create_vulkan_context` builds
  a fresh `Vulkan.Instance` and `DebugUtilsMessengerEXT` per context and
  `VkContext` already holds both — so two contexts mean two messengers writing
  into ONE ring, and device A's validation errors surface in device B's
  `get_validation_messages()`. `VkDebugUtilsMessengerCreateInfoEXT` carries a
  `pUserData` pointer for exactly this, and `debug_callback` already accepts and
  ignores it. The callback's real constraint — no Julia runtime on a driver
  thread — is satisfied by raw memory the context owns just as well as by module
  globals.
- **The frozen-cache directories, version and recording flag are configuration**,
  read from `ENV` at init or gating a recording pass. A memoized accessor or an
  argument, on the same argument as the GEMM tunables.

- `KERNEL_ITER_PLAN_CACHE::Dict` — keyed by `(kernel type, ndrange,
  workgroupsize)`, a key that does not name the device, while the cached
  `block_dims` is `pad_to_3d(ctx, …)` over `ctx.max_wg_dims`. The first device to
  launch a given kernel shape decided the block grid for every device after it.
- `RESERVED_ARG_SLABS::IdDict{BatchQueue,Int}` and
  `REPLAY_WATERMARK::IdDict{BatchQueue,UInt64}` — per-queue values in
  process-wide dictionaries keyed by the queue, which is precisely the surrogate
  ("the object I should have stored this on") the finding describes.
- `TOUCHED_RANGES` / `DISPATCH_RANGES` — the barrier-elision tracker, shared by
  every queue, so ranges written while recording one command buffer could elide
  a barrier in another.

Use mutation, not declaration shape:

```bash
# every `const NAME = …` that is ever written to, whatever its type
python3 - <<'EOF'
import re, os
decl, text = {}, ""
for root,_,fs in os.walk('.'):
    for f in (x for x in fs if x.endswith('.jl')):
        t = open(os.path.join(root,f), errors='replace').read(); text += t
        for i,l in enumerate(t.splitlines(),1):
            m = re.match(r'^const ([A-Za-z_][A-Za-z_0-9]*)\s*(?:::[^=]+)?=', l)
            if m: decl[m.group(1)] = f'{f}:{i}'
for n,loc in sorted(decl.items()):
    pat = rf'(?<![A-Za-z_0-9]){n}\s*\[[^\]]*\]\s*[-+*]?=[^=]|(?:push!|empty!|delete!|resize!|append!|pop!)\(\s*{n}\b|Threads\.atomic_\w+!\(\s*{n}\b|get!\([^,]*,\s*{n}\b'
    if any(re.search(pat,l) and f'const {n}' not in l for l in text.splitlines()):
        print(f'  {n:36s} {loc}')
EOF
```

## Evidence appendix

Commands used, so any number above can be re-derived:

```bash
cd dev/JuliaVision/DNNKernels/src
grep -rn "^const [A-Z_0-9]* = Ref" .                      # 34 toggles
grep -rhn "Lava\.[a-zA-Z_!]*" -o . | sort | uniq -c       # 176 Lava refs
grep -rn "_applicable\b" .                                # 4 predicates
grep -rc "cpu=false" kernels/*.jl kernels/extern/*.jl     # 5 CPU-less kernels
grep -rn "^struct \|^mutable struct " .                   # 13 structs
cd ../../../Lava/src
grep -rn "^const [A-Z_0-9]* = Ref" --include=*.jl . | wc -l   # 84 — UNDERCOUNTS,
                                                              # see the correction above
grep -rn "@overlay\|@MethodTable" --include=*.jl .            # the unused mechanism
```
