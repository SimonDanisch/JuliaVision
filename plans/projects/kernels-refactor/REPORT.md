# kernels-refactor — report

Append, newest last. One entry per working session: what was done, what was
measured, what was **dis**proved. Negative results in full — roughly half the
value in `plans/perf-plan.md` is knowing which things were tried and lost.

Numbers only from the machine this project is assigned to, and engine
comparisons only from the desktop (`plans/GUARDRAILS.md` §6).

---

## 2026-08-02 — steps 1 and 2 of 6

Desktop, RTX 4000 Ada. `dev/JuliaVision` @ `sd/kernels-refactor`, commits
`07a8211` and `0bce6b8`. Session killed after step 2 (see *Process*); this is
what landed, reconstructed from the commits.

**Globals in DNNKernels: 34 → 25 → 20.**

### Step 1 — delete the settled toggles (`07a8211`)

Finding 3, tier two: a switch whose default is the measured winner and whose
other branch nothing takes is not a switch, it is a dead branch that still has to
compile and still has to be correct.

Deleted, winner inlined: `LAUNCH_FLAT`, `LN_FUSED`, `MATMUL_FUSED`,
`CONV_1X1_GEMM`, `CONVT_GEMM`, `FLASHCM`, `FLASHCM_PERELEM`. Deleted as globals
but kept as `sdpaflashcm!` keywords with literal defaults: `FLASHCM_ONEPASS`,
`FLASHCM_LAZYRESCALE` — the globals were only supplying those defaults, and the
keyword is what the tests A/B.

**Correction to the review:** its list of twelve counts `GEMM_STAGED`,
`GEMM_VEC2` and `GEMM_NARROW`, which are **Lava's**, not this package's. Nine,
not twelve, live here.

Every measurement in the deleted docstrings is carried across verbatim as a
comment at the site the decision now lives at — they are the record of what was
already tried, and the A/B that would overturn each is named next to its number.

Two things fell out of the deletions, both finding 7:

- `onebyone` and `shufflecase` now answer only what their names ask. With the
  switch off, **`onebyone` reported that a 1×1 convolution was not a 1×1
  convolution.**
- `shufflecase`'s docstring was attached to `CONVT_GEMM` — the `const` sat
  between the prose and the function — so `?shufflecase` answered nothing.
  Deleting the const put it back. (Same adjacent-docstring trap that bit three
  edits in Lava the same day; worth grepping for rather than waiting to notice.)

Two tests had been flipping a deleted toggle to produce their reference. They now
call the alternative path by name, which is **stronger**: it asserts the routing
(`onebyone`, `flashcm_applicable`) instead of assuming it.

### Step 2 — diagnostics onto `Ctx` (`0bce6b8`)

Five module-level `Ref`s retired — `OPTIMES`, `OPDOUBLE`, `OPDOUBLEFILTER`,
`PLAN_MISSES`, `LAUNCH_PROBE` — replaced by one `Diagnostics` on the context.

The standing defence for a global was that they must be reachable from a call
stack that does not thread them. That described the old signatures rather than
constraining anything, and the fix is a **smaller** signature, not a bigger one:

```
convolution!(out, x, w, b, s, p, d, g; ws)   ->  convolution!(ctx, out, x, w, b, s, p, d, g)
sdpa(q, k, v, bias, scale; backend, ws)      ->  sdpa(ctx, q, k, v, bias, scale)
scratch!(ws, backend, T, dims...)            ->  scratch!(ctx, T, dims...)
launch!(f, out, args...; backend)            ->  launch!(ctx, f, out, args...)
```

Every entry point that reaches a `launch!` or the workspace now takes the
context. Kernel launchers needing only a backend keep it — they have nothing to
instrument.

`Ctx` moved to its own file (a signature naming a type needs that type) and is
included above every kernel file. `Ctx(backend; ws, rec, diag)` builds one with
no graph behind it, which the driver's between-graph memory read and every direct
call from a test now use. `Model` carries a `Diagnostics` from construction, so
instrumenting a whole `step!` is one field assignment with no signature change
anywhere.

New `test_diagnostics.jl`, 25 asserts, pins the property that replaced the
globals: **two runs in one process, instrumented differently, neither seeing the
other's measurements.** A global cannot do that, so a regression to one fails
here. It also drops the save/restore dance the old tests needed — which a
*failing* test would have skipped, leaving the global flipped for everything
after it.

This is the step that makes the plan objects cheap, and it settles the pattern
(`ctx` carries the per-device thing) that step 3 then follows for the device.

### Behaviour

Suite green and unchanged across both steps:
61/80/4023/149/4511/25/37/18/19, transposeLE 16, coopmat attention 10, fused
attention 216 + 11, replay decode 41.

**Not re-measured:** SAM 2 encode/decode against the 100.4 / 3.30 ms baseline.
The brief requires it and it has not been done — the suite passing is not the
same claim. Do this before step 3, so a number that moves is attributable to two
commits rather than five.

### Next — step 3, and what it inherits

The **20 remaining globals are almost exactly the tuning constants step 3
absorbs as plan fields**, which is the expected shape rather than a coincidence:

```
attention.jl   ATTN_MINL  COOPMAT_MINL  ATTN_SOFTMAX_ROWS  COOPMAT_QCHUNK
flash.jl       FLASH_SHARED_BUDGET  FLASHCM_DENSIFY  FLASHCM_REGO  FLASHCM_HELD
               FLASHCM_RESCALE  FLASHCM_CLAMP  FLASHCM_MINGRID
conv_coopmat   CONV_CRS_PAD
launch.jl      LAUNCH_GROUP
fuse/folds     DUPLICATE_FUSION  DUPLICATE_MAX  FOLD_OUTCASTS  FOLD_CONSTSUBGRAPHS
sam2.jl        CACHE_DECODER_INPUTS  REPLAY_DECODE  SEGMENT_TIE
```

`FLASH_SHARED_BUDGET` (48 KB) and `LAUNCH_GROUP` (256) are finding 9's hardcoded
device literals and must become per-device values from `Lava.max_shared_memory`
and the device's subgroup size — not defaults captured at module scope.

### Process — why this report was nearly lost

The agent doing this work was killed ~80 minutes before anyone noticed. It had
committed both steps (the second at 14:50:41, seconds before it died), so unlike
`lava-core` nothing was lost — but nothing was written here either, and the UI
still showed it running. This report was reconstructed from the commit messages.

`GUARDRAILS.md` now requires appending here at the end of each *step*.


## 2026-08-02 (later) — steps 3, 4 and 6; **globals 34 → 0**

Same machine and branch, commits `4f43ba4` and `f6babca`. The exit criterion is
met: **no module-level mutable toggles remain in DNNKernels**, no carve-outs.

### What each step actually did

**Step 3 — plan objects.** `FlashCMPlan`, `CoopMatSDPAPlan`, `MMCoopMatPlan`,
`ConvCoopMatPlan`. The review's worked example was real and worse than described:
`sdpa` asked `flashcm_applicable`, which ran `flashcm_tiling` and threw the
answer away to return a `Bool`; `sdpa` then ran `flashcm_tiling` **again**; and
`sdpaflashcm!` re-checked six more conditions and could still return `false`
*after* `out` had been allocated. Now one decision, carried, and the runner
cannot decline.

A refusal is `Decline(reason)`, not `nothing`, because two callers need to know
which rule fired. One is live: `:wrapped` means `stridedroot` could not account
for the operand stack, and `sdpa` now densifies and retries — which is exactly
what `FLASHCM_DENSIFY`'s docstring described as the fallback, expressed as a
global nobody sets, so it never happened and the call silently took a slower
path. The other is `blockfor`'s refusal of non-square attention: a known bug,
which must say so rather than look like a design decision.

**Step 4 — dispatch on plan type.** `sdpa!` and `matmul!` have one method per
plan. A new path is a new type and a new method.

**Step 6 — `Ctx`.** Five telescoping positional constructors → one keyword form;
four `::Any` fields → type parameters (the named types are defined in files
included *after* `context.jl`, so parameters were the only option; there are two
live combinations, not a spread).

**Step 5 — NOT done.** After step 4 the dispatchers are three one-line methods,
so generating them saves nine lines and adds a macro. A judgement against the
review, not a completed item.

### Finding 9, and a device the kernels can be asked about

`Device` answers, once per context from a live device: coopmat availability and
tile, subgroup width, shared budget, workgroup limit, shader cores, launch group.
3 µs and one allocation.

Five numbers were hardcoded at their call sites. **`NW * 32` appeared in three
places and names half the workgroup on a wave64 part**, and every tiling decision
keyed on it inherits that. `flashcm_tiling` and `flashcmfits` now *take* a
`Device` rather than reading one, so the AMD laptop's questions can be asked from
here before it answers them:

    flashcm_tiling(Device(true, 16, 64, 65536, 1024, 40, 256), 72, 4096, 4096)

### `FLASHCM_CLAMP` was per process; it is per run

SAM 2 set it around the decode and restored it in a `finally`, on the grounds
that "`flashcm_tiling` reads it six frames down, inside `runop!`". That is the
same defence the five diagnostics `Ref`s made, and it has the same answer: `Ctx`
already reaches every `runop!`. It is now `ctx.clampattn`, threaded from
`call(...; clampattn=true)` — so two contexts with opposite policies can be alive
at once, and there is no state an error can leave switched on for the next
encode.

### ⚠ A test that had stopped testing anything

`test_coopmat_attention.jl`'s "agrees with the three-pass path" A/B'd by flipping
`COOPMAT_MINL` and calling `sdpa` twice. But `sdpa` tries the **fused** path
first, and the fused path takes all four of its shapes — so both halves ran the
same kernel and it compared a result with itself. **Vacuous since flash landed;
four assertions that could not fail.** Verified directly rather than reasoned
about.

The general lesson, and it applies to every A/B in this repo: *a test that forces
a path by moving a threshold stops testing that path the moment something earlier
in the routing changes.* It now calls `sdpa_coopmat!` by name with an explicit
plan against a host reference, which is the same correction step 1 made for two
other tests.

### Behaviour

Suite green from a clean session: 61/80/4023/149/4511/25/37/22/19, transposeLE
16, coopmat attention 11, fused attention 216 + 11 + 14, replay decode 41.

| | baseline | now |
|---|---|---|
| encode p50 | 100.4 ms | **101.06** (three runs: 100.97 / 100.70 / 101.06) |
| click p50 | 3.30 ms | **3.15** |
| VRAM live | 1181 MiB | **1181** |
| masks | — | IoU 1.00000 / 0.99955 / 1.00000, identical every run |

### ⚠ The multi-device exit criterion is BLOCKED, and both briefs are wrong about why

**`BatchQueue` has no `ctx` field.** Both briefs say "the carrier already exists —
`LavaBackend(ctx)` pins a context, `BatchQueue` carries `ctx`". It does not:
`LavaBackend(ctx)` keeps `ctx.default_bq` and **discards `ctx`**, and a
`BatchQueue` holds a `Vulkan.Device`, not the `VkContext` that owns it. There is
no path from a backend to its context.

So `Device` is built from the *default* context, and a second device would
silently receive the first's numbers. The fix is one field in Lava and belongs to
`lava-core` phase 2. `vkcontext` in `context.jl` is the single function here that
changes when it lands, and says so at the point of change rather than in a note
on a branch.

Until then the two-device acceptance test cannot pass — for either project.
