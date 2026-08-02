# lava-core — report

Append, newest last. One entry per working session: what was done, what was
measured, what was **dis**proved. Negative results in full — roughly half the
value in `plans/perf-plan.md` is knowing which things were tried and lost.

Numbers only from the machine this project is assigned to, and engine
comparisons only from the desktop (`plans/GUARDRAILS.md` §6).

---

## 2026-08-02 — phase 1, item 1 of 2; plus two bugs found on the way

Desktop, RTX 4000 Ada, driver 595.336. Baseline `dev/Lava` @ `b8751de`.
Commits `a246295`, `740d982`, `3f59291` on `sd/lava-core`.

This session was killed part-way through (see *Process* at the bottom); what is
below is what had been finished and is now committed, not a plan.

### Phase 1a — narrow index + rank≥3 `Extruded`: verdict stands, mechanism was wrong

The brief said the "NVIDIA driver miscompile" label was a suspect, not a
finding. Re-audited with all four Rule-0 instruments. **The verdict survives —
and is now carried by the strongest instrument instead of by elimination — but
every mechanism the old header asserted is disproved.**

What settles it: the same computation **hand-written in GLSL and compiled by
`glslangValidator` 16.4.0**, over the same argument buffer, run through the same
Lava dispatch, gives the *identical* wrong answer on the Ada and is exact on
lavapipe.

| producer \ consumer | RTX 4000 Ada 595.336 | lavapipe / LLVM 22.1.8 |
|---|---|---|
| Lava (the kernel under test) | **WRONG** | exact |
| `glslangValidator` 16.4.0 | **WRONG** | exact |

The old two-driver argument had a hole nobody had noticed: Lava's module
generation is device-dependent, so "it works on lavapipe" never established that
lavapipe ran the *same bytes*. A hand-written GLSL module is literally the same
file on both, which closes it. `dev/Lava/test/glsl/` holds the two shaders and
the build line.

**Located exactly.** Reading back four intermediates in one dispatch — one
dispatch specifically, so no probe can differ from another by dead-code
elimination:

```
q1 = z ÷ e1           read directly:  CORRECT
q2 = q1 ÷ e2          read directly:  CORRECT
c2 = (q1 + 1) - q2*e2                 WRONG — equals (q2 + 1) - q2*e2
same expression in Int32, same dispatch:  CORRECT
```

`q1` is correct when read on its own and delivers **`q2`'s value** where it is
used again after the second division. For `(5,5,3)` that gives `c2 = 1, -3, -7`
instead of `j`, which is exactly the observed delivered index and flat offset.

**Disproved, each by swapping the emitted module under an otherwise identical
dispatch and re-running.** Every one of these was previously asserted in the
test file as the cause:

- **Not the Int32 narrowing.** `shl 32 / ashr 32` → `shl 0 / ashr 0` (the wide
  computation, same instruction count) is still wrong. "It is `CartesianIndices`
  under a narrow index" was never the mechanism — `cis[I]` escapes because Julia
  emits a *different decomposition* for it (one `udiv` by `d1*d2` plus a `udiv`
  of the remainder, not a chained pair).
- **Not `OpSDiv` vs `OpUDiv`.** Rewriting every division: still wrong.
- **Not the divide-by-zero control flow.** Deleting both guards for one
  straight-line block: still wrong. Keeping them and deleting the redundant
  re-tests in the merge blocks: still wrong.
- **Not the struct layout, not our argument packing.** A GLSL memory-dump shader
  reads the `Broadcasted` at the offsets the module uses and gets dims (5,5,3),
  keeps `0x010101`, defaults (1,1,1), axes (5,5,3), extents (5,5,3), ndrange 75.
- **Not `Extruded`'s selects, not the address arithmetic.** `c1`, `c2`, `c3`,
  `dims[1]`, `dims[2]` each read back correct; three explicit multiplies for the
  address changes nothing.
- **`spirv-val --target-env vulkan1.3` passes.** GPU-AV reports nothing — but
  `Lava.verify_gpu_av()` **fails on this layer**, so GPU-AV cannot see BDA
  accesses here and its silence is *not* evidence. Worth stating plainly,
  because "GPU-AV was quiet" is exactly the kind of non-evidence that built the
  original wrong diagnosis.

**And what does not reproduce it**, which is why it looked like
`CartesianIndices`: the chained-division shape *alone* is not sufficient. A
kernel doing only `q1 = z ÷ e1; q2 = q1 ÷ e2; out[I] = (q1+1) - q2*e2` with
runtime divisors is correct at both widths, in Julia and in hand-written GLSL,
even with `q1` additionally stored. It needs the surrounding register pressure —
which is why it tracked rank ≥ 3 and `Extruded`, and why every attempt to
instrument it made it vanish. Treat it as a scheduling / register-allocation
fault around chained 64-bit division, **not** a rule about any one instruction.

**Consequence for our kernels:** any kernel doing two *dependent* 64-bit integer
divisions and re-using the first quotient afterwards is exposed. `splitidx` /
`cart32` (magic-number division, no `OpSDiv`) is why Lava's broadcast path is
safe. Both cases stay `@test_broken` so a driver fix announces itself.

Worth fixing rather than avoiding: the same narrowing inside
`DNNKernels.im2col_kernel!` moved an inference step 35.7 → 32.7 ms. Lava's
broadcast kernels cannot take it until this is fixed.

### Two bugs found on the way

**1. The scalar GEMM accumulated in the destination's precision** (`a246295`).
`strided_gemm_kernel!` took `T = eltype(C)`, so an fp16 destination gave an fp16
accumulator over K = 1280 or 5120. Found by the Whisper port on block 1's `fc2`,
measured against an fp64 reference over the *same* fp16 inputs:

| | rel rms |
|---|---|
| fp16 destination → fp16 accumulator | 4.83e-2 |
| predicted by `sqrt(K) * eps(Float16) / 2` | 3.49e-2 |
| same kernel, fp32 destination, rounded after | 2.06e-4 |
| Lava's coopmat path (fp32 accumulator) | 2.13e-4 |
| PyTorch's own fp16 `addmm` | 3.96e-4 |

**234×, from one line** — and the agreement between predicted and measured is
what makes it a diagnosis rather than a correlation. Scope is far wider than one
model: every fp16 matmul missing the cooperative-matrix path, and
`mm_coopmat_applicable` declines whenever an operand arrives wrapped, which in a
raw exported graph is every `addmm`.

Fixed by `gemmaccumtype`, with the α/β scaling also staying in the
accumulator's precision so only the single store rounds. After the fix the fp16
result sits **on the representation floor**: 1.9568482863662596e-4 measured vs
1.9568482738730768e-4 for taking the exact answer and merely storing it as
`Float16` — eight significant digits, at both K.

*Trap, recorded because it cost a test failure that read as a failure of the
fix:* `test_gemm_fp16_accum.jl` first asserted `e16 < 20 * e32`. That is
unsatisfiable however wide the accumulator is — the fp32-destination run never
rounds to half, so it sits at 2.5e-8, four orders below. The correct reference is
the store-to-fp16 floor, not another destination type.

**2. `clear_kernel_cache!` silently did nothing** (`740d982`).
`LAUNCH_PLAN_CACHE` holds its own `VkPipeline` and is consulted *before*
`LINKED_KERNEL_CACHE` on every dispatch, so emptying only the latter left the old
pipeline running with no symptom. This made a SPIR-V A/B harness report "no
difference" for **six variants, including one with its `OpStore` deleted** —
i.e. it silently invalidated a set of A/B results. The Revise path happened to
work anyway because a method redefinition moves the world counter and
`launch_plan` rejects plans from a superseded world; a caller who only cleared
the cache had no such luck.

This is the second cache-identity bug in two days, after the
`hash(spirv_bytes)` collision. Both had the same signature: **an A/B that
reports no difference between things that must differ.** `GUARDRAILS.md` §4's
pipeline-count assertion is aimed at exactly this and should be applied to every
A/B harness before the next one is trusted.

### Not done

- **Phase 1b — `OpUDiv` in a shared-store index.** Not started. Note that phase
  1a *disproved* `OpSDiv`-vs-`OpUDiv` as the mechanism of the other bug, so this
  one cannot inherit that finding either way; it needs its own GLSL differential.
- **`spirv-val` / `gpu_av` on by default for test runs**, and the §4
  pipeline-count assertion in the A/B harnesses. Not started, and the second bug
  above is the argument for doing it first.
- **Phase 2 (globals → `VkContext`, multi-device)** and **phase 3 (missing
  instructions)**. Not started. Global count unchanged at 84.

### Process — why this report was nearly lost

The agent doing this work was killed ~80 minutes before anyone noticed, with all
of the above uncommitted in the worktree and nothing written here. It was
recovered by hand. The UI still showed it running, so "the card is ticking" is
not evidence a project is alive; the last write time in its worktree is.

`GUARDRAILS.md` now requires appending here at the end of each *step*. Reporting
last means a killed agent loses everything instead of one step.


## 2026-08-02 (later) — phase 2 started: the two-device probe exists, and it works

Lava `16f193e`. Done by the coordinator rather than this project's agent, which
was killed; recorded here because it is this project's phase 2.

### The acceptance test the brief describes is now possible

`init_vulkan!(; select = pick_physical_device)` chooses the physical device and
still returns a context **without** installing it as the global. The Vulkan
loader enumerates the real GPU *and* lavapipe from one instance, so every machine
here has a two-device pair with no second card. Before this there was no way to
ask for the second device at all — which is why the test had never been written,
and the reason was never the caches.

`test/twodevice_probe.jl`, deliberately **not** in `runtests.jl`: it segfaults,
and a segfault takes the whole suite with it.

### Keying the caches was necessary and not sufficient

`VkContext.id` (a never-reused counter) now leads `PIPELINE_CACHE`'s key, and
`LINKED_KERNEL_CACHE` / `LAUNCH_PLAN_CACHE` are per-device outer dicts. That is
the §8 defect fixed.

The probe then crashed anyway, on the **first** thing §8 does not name.

### `CMD_PIPELINE_BARRIER_FPTR` — a whole class the review missed

A device function pointer is per device: `vkGetDeviceProcAddr` returns one valid
only for the device asked. It was a module global, so creating the second context
overwrote it and the **first** device's command buffers were recorded through the
**second** device's driver. The crash was inside `libvulkan_lvp.so` while
dispatching on the NVIDIA context.

§8 lists four caches holding *handles* and says nothing about the function table,
which is strictly more dangerous: a stale handle is undefined behaviour, a
foreign function pointer is an immediate jump into another driver.

Fixed — `VkContext.cmd_pipeline_barrier_fptr`, reached through `barrier_fptr(bq)`
at the three recording sites. With it, the GPU dispatch goes from segfault to
correct, and the crash moves to the lavapipe dispatch.

### The phase-2 worklist, produced by running rather than reading

The review's list was four caches. The real list is longer, and the probe is how
to enumerate it honestly. In descending order of suspicion:

```
PREPARE_INDIRECT_PIPELINE_REF   a LavaComputePipeline, bound at dispatch
PREPARE_INDIRECT_OFFSETS_REF    its layout, same lifetime
TIMESTAMP_POOL                  a Vulkan.QueryPool
BLIT_PIPELINE                   a graphics pipeline
GEMM_SPLIT_SCRATCH              device memory
DEVICE_SUBGROUP_SIZE   \
SUBGROUP_SIZE_CONTROL   >  device PROPERTIES cached globally
WORKGROUP_LIMIT        /
```

The last three matter differently from the rest and are worth calling out: they
produce a **wrong answer** rather than a crash, so they are the ones that survive
unnoticed. `DEVICE_SUBGROUP_SIZE` cached from the GPU and read on a device whose
subgroup is 64 is exactly the finding `portability` raised from the other side.

### Behaviour on one device

Unchanged. Full DNNKernels suite green after each step —
61/80/4023/149/4511/25/37/22/19/16/11/216/11/15/41, zero failures — and SAM 2
masks identical with VRAM at 1181 MiB.
