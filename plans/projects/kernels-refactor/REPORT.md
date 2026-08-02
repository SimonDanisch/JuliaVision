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
