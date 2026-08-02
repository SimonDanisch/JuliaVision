# kernels-refactor

**Machine** Desktop (RTX 4000 Ada) · **Repo** `dev/JuliaVision` @ `sd/kernels-refactor`
**Read first** `plans/kernel-library-review.md` in full, then `plans/GUARDRAILS.md`.

Steps 2–7 of the review's own suggested order, as **one branch** — the review says
so explicitly, and it is right: steps 4–6 touch `flash.jl`, `matmul.jl`,
`conv_coopmat.jl` and `attention.jl` together, so splitting them guarantees
conflicts.

This is the project everything else waits on. Every new capability-gated kernel
written before plan dispatch exists is another `if has_feature` chain that this
work then deletes — one was added on 2026-08-02 while the review sat unread.

## The order, and why it is this order

The review's reasoning, not a re-derivation: each step makes the next cheaper.

1. **Delete the settled toggles** (finding 3, tier two). Twelve of them, each read
   in one file, each with a measured winner. Removes the dead branch, shrinks the
   `Val` explosion in finding 4, shortens every function that reads them.
2. **Diagnostics onto `Ctx`** — a `diag` field and `execute!(...; diag)`, retiring
   `OPTIMES`, `OPDOUBLE`, `OPDOUBLEFILTER`, `PLAN_MISSES`, `LAUNCH_PROBE`. `Ctx`
   already reaches every `runop!`, so this is plumbing. It is what makes two
   differently-instrumented runs possible at once. Do it *before* the plan
   objects — it settles the pattern. The one real edit is giving the kernel entry
   functions `ctx` in place of their `(backend, ws)` pair, which is a smaller
   signature, not a bigger one.
3. **One plan object per kernel family** (finding 2): `flashcm_plan`,
   `matmul_plan`, `conv_plan`, each returning a typed plan or `nothing`, absorbing
   the surviving tuning constants as fields. This kills the double-checking —
   today `sdpa` asks `flashcm_applicable`, then `sdpaflashcm!` re-checks six more
   things and returns `false` from six places to mean "declined". Two predicates
   that can drift, where the symptom of drift is a *silent change of algorithm*.
   Device-derived defaults are computed in the constructor on a live context,
   never captured at module scope.
4. **Dispatch entry functions on the plan type** (finding 1). Now additive: a new
   path is a new type and a new method, and each is testable by constructing its
   plan without a device. `CoopMat2Caps` on the context is the natural source for
   the capability part.
5. **Generate the dispatchers alongside the kernels** (finding 5).
6. **Type `Ctx`'s remaining fields** (finding 6) — six `::Any` fields and four
   telescoping constructors. Worth doing after 3, because the plan objects give
   `ws`/`plan` real types to be.

## Multi-device is a hard requirement of this design

The plan objects and `Ctx` are the layer where a second Vulkan device is either
supported or permanently excluded, so decide it here rather than discovering it
later.

- **`Ctx` carries the backend/context.** Every `runop!` already receives `Ctx`, so
  once the device is on it, no kernel needs a global to find its device. This is
  the same move `lava-core` is making one layer down — coordinate, do not
  duplicate.
- **Plan objects are per-device by construction.** The review already requires
  device-derived defaults to be "computed in the constructor, on a live context,
  never captured at module scope". Multi-device makes that mandatory rather than
  stylistic: a plan built for one device's shared-memory budget, subgroup size or
  coopmat shape list is *wrong* on another.
- **Finding 9's literals become per-device values.** `48` SMs, `32` lanes and
  `48 KB` are hardcoded at five call sites; `Lava.shader_core_count`,
  `shader_warps_per_sm`, `max_shared_memory` and `device_subgroup_size` already
  answer all four. On one machine a literal is merely wrong-in-principle; across
  an Ada desktop, an RDNA3 iGPU with a subgroup that may be 64, and lavapipe, it
  is wrong in fact.
- **Nothing in a plan may be cached at module scope** if it was derived from a
  device — see `GUARDRAILS.md` §8.

## Exit

**Globals 34 → 0**, no carve-outs. That is the review's metric and it is
unambiguous.

Plus: no code path reaches a device except through `Ctx` or an explicit backend
argument. The check is the same two-device test `lava-core` sets up — a real GPU
and lavapipe, which needs no second card.

Behaviour must be unchanged: the DNNKernels suite green, and SAM 2 encode/decode
re-measured on this machine and within noise of 100.4 / 3.30 ms. If a number
moves, that is a finding, not a rounding error — say so in the report.

## Two things to carry, not lose

- **`blockfor` refusing non-square attention is an open bug with a workaround**,
  not a design choice: blocking the decoder's lopsided shapes reproducibly hangs
  on `vkWaitSemaphores`. It currently lives quietly inside a predicate. When
  predicates become plans, it must become an explicit, named rejection reason
  pointing at the bug — not a silent `nothing`.
- The review's **"do not fix these"** list. Each looks like a defect and is not,
  with the reason in the source: the `@eval`-generated kernel families (the
  parameterised form *miscompiles* — a cooperative matrix defined inside a
  conditional stops being an SSA value to the emitter), `unsafe_indices=true` on
  the staged GEMM (the KA bounds guard cost 3x), two reduction passes in
  `layernorm_kernel!` (numerics), and the 90 clones in SAM 2's graph.

## Then, and only then

Flash-decoding (`kernels-to-port.md` item 1, ~1.2 ms of a 3.30 ms decode) lands
**as the first new plan type**, not as its own branch. It is the proof that the
refactor made a new path additive.

## Report

Append to `REPORT.md`: the global count after each step, the re-measured
encode/decode, and anything the review got wrong — it was written from a read,
and reading is not running.
