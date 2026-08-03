# small-models — report

Append, newest last. One entry per working session: what was done, what was
measured, what was **dis**proved. Negative results in full — roughly half the
value in `plans/perf-plan.md` is knowing which things were tried and lost.

Numbers only from the machine this project is assigned to, and engine
comparisons only from the desktop (`plans/GUARDRAILS.md` §6).

---

## 2026-08-02 — all three models export, run and match; two miss their targets

Machine: RTX 3070 laptop, 8 GB, driver 595.84, SM clock 1815–1935 MHz under load.
Every number below is **this machine's** and no engine comparison has been put in
`perf-plan.md` (`GUARDRAILS.md` §6).

### Where the three ports stand

| model | exports | runs | vs PyTorch | editor budget | target |
|---|---|---|---|---|---|
| Neural 3D LUT | yes, 26 ops | yes | **3.6e-7** on the LUT | apply **0.79 ms** at 4K | < 2 ms — **met** |
| RIFE 4.26 | yes, 366 ops | yes | **3.3e-4** max, 4.9e-7 mean | **~320 ms** at 1080p | 16.67 ms — **missed, ~19x** |
| Depth Anything V2 S | yes, 311 ops | yes | **4.2e-5** (0.0025% of range) | **~380 ms** at 518² | ≥ PyTorch (24.7 ms) — **missed, ~15x** |

Timings are the median of 3 runs on an **uncontended** card; see the correction
below, and treat them as one significant figure (`GUARDRAILS.md` §6).

**No new runtime ops were needed**, as the brief predicted, with one exception:
`_native_batch_norm_legit.no_stats` — what `nn.InstanceNorm2d` decomposes to —
was added to `DNNKernels/src/ops.jl` for the neural LUT and is general. RIFE's 16
ops and Depth Anything's 12 were all already implemented, including transposed
convolution (7 in RIFE) and `grid_sampler_2d` (18 in RIFE).

### The two findings that changed the numbers

**1. TF32 makes a reference that is not a reference.** `EG.precision_ctx` only
chooses autocast; it does not touch TF32, and on Ampere it is on by default for
convolution and matmul. Exports here inherited that, so the "PyTorch reference"
was itself a 10-bit-mantissa approximation. RIFE first measured **max|Δ| 7.9e-3
over 0.05% of samples** — which reads exactly like a broken warp at a few hundred
pixels, and is the kind of thing that costs a day. Turning TF32 off in the
exporter:

| | before | after |
|---|---|---|
| RIFE max\|Δ\| | 7.914e-03 | **3.265e-04** |
| RIFE mean\|Δ\| | 1.118e-05 | **4.897e-07** |
| neural LUT max\|Δ\| | 1.276e-05 | **3.576e-07** |

`dump_sam2_refs.py` already does this and says why; the three lines are now in
all three exporters. **Anyone writing a new exporter here should copy them** —
`precision_ctx` looks like it covers precision and does not.

**2. Benchmarks must go through `Model`, not `loadgraph`.** `Model` runs the
host-side preparation passes (`foldbatchnorm`, `foldrelu`, `hoistcasts`,
`hoistpermutes`, `hoistconstants`, `foldoutcasts`, `dropdead`) that the editor's
path gets. Benchmarking `loadgraph` + `execute!` measures a graph nothing ships:

- Depth Anything **763.56 → 421 ms**, 311 → 290 ops. Nearly 2x. (Both contended;
  the clean figure through `Model` is ~380 ms.)
- The neural LUT classifier **14.08 → ~10 ms**, almost all of it from passing a
  planned slab rather than letting every intermediate allocate.

Both benchmarks originally did the wrong thing and both numbers were wrong in the
pessimistic direction. All three `tools/bench_*.jl` now build a `Model` and a
planned slab.

**RIFE looked like an exception and was not** — see the correction above. The
"passes cost this graph 50%" reading was soak contention, not the passes.

### Correction: the first round of timings was measured against a running soak

Every number in the first version of this entry was taken while
`tools/soak_flush_hang.jl` was still hammering the same GPU. The soak had been
stopped from the harness, and the *shell* died, but the Julia child survived and
kept dispatching — 1 007 MiB resident and a continuous kernel stream. That is
what the unexplained spread was: RIFE read 327 / 357 / 478 / 500 / 516 ms across
runs and I wrote it up as an unexplained property of the preparation passes.

Re-measured with nothing else on the card:

| | contended | clean |
|---|---|---|
| RIFE, one frame | ~500 ms | **307 / 322 / 331 ms** |
| Depth Anything, one map | 421 ms | **364 / 401 ms** |
| Depth Anything, PyTorch | 27.6 ms | **24.7 ms** |
| neural LUT classifier | 7.0 / 8.4 / 8.4 ms | **10.5 / 9.5 / 10.5 ms** |
| neural LUT apply at 4K | 0.88 ms | **0.79 ms** |

**The `Model`-is-slower-than-`loadgraph` finding does not survive this.** RIFE's
327 ms "without the passes" and ~500 ms "with them" were the same code path
measured against different amounts of soak. Clean, through `Model`, it is ~320
ms. There is no evidence the preparation passes cost RIFE anything, and the claim
is withdrawn rather than left standing with a caveat.

Two lessons worth more than the numbers: a stopped background job is not
necessarily a dead one — check `nvidia-smi --query-compute-apps` before timing,
not just the task list — and an unexplained 50% spread is a symptom to chase, not
a footnote to write.

### Calibration: how fast is SAM 2 on this machine?

The obvious question about a 15–30x gap is whether the laptop is simply slow.
It is not, and SAM 2 is the control because the desktop has published numbers for
it.

| | this machine | desktop | ratio |
|---|---|---|---|
| SAM 2.1 encode, 1024² | **182.3 ms** | 100.4 ms | 1.8x |
| SAM 2.1 decode, one click | 17.6 ms | 3.30 ms *(replayed)* | not comparable |
| VRAM | 1 337 MiB | 1 181 MiB | 1.1x |

**This card is ~1.8x the desktop, not 15x.** The desktop has SAM 2 encode at 87%
of PyTorch; scaling both sides by 1.8 leaves it at about 87% here too. Depth
Anything on the same engine, same machine, same session is at **6.5%** of
PyTorch. That is a 13x spread *between models*, which is the finding: the gap is
not the card and not the runtime as a whole — it is which kernels these two
graphs land on. The decode figure is not a fair comparison in either direction:
the desktop's 3.30 ms is a replayed captured command buffer and this is not.

### Why RIFE and Depth Anything miss

Measured, not guessed (`GUARDRAILS.md` §5 — no unmeasured denominators).

RIFE's 63 convolutions are **149.0 GFLOP** per interpolated frame at 1080p, of
which one transposed convolution is 47.9 GF. At the measured ~320 ms that is
**~0.47 TFLOP/s**, a low single-digit percentage of this card's fp32 peak. Even a
conv running at a *plausible* 10 TFLOP/s leaves 14.9 ms of convolution alone
against a 16.67 ms frame — so **1080p60 is not reachable in fp32 on this card
even with a good kernel**, and the target as written needs either fp16/tensor
cores or a smaller model. That is a target-setting finding, not just a slow
kernel.

Dispatch timing on RIFE (`Lava.with_dispatch_timing`, 374 dispatches):

- `conv2d_igemm` ~50% — the largest is 7 dispatches at 128.3 ms total
- `ndmap_flat` ~26% — elementwise over 115M-element tensors, ~92 GB/s against
  the card's ~380, so these are bandwidth-bound and losing 4x
- `grid_sample2d` ~6% — the warp, which was the op this model was picked for,
  is not the problem

The neural LUT classifier shows the *opposite* conv failure: its first
convolution launches **8 workgroups** for a 4.23 ms dispatch, starving a 46-SM
card. Two models, two different conv pathologies, same kernel.

### What is fast enough

The neural LUT port meets its target and is worth shipping. Applying a look at
4K is **0.79 ms**, the 4K→256 reduce is **0.03 ms**, and predicting a new look
is **~1.8 ms** (corrected 2026-08-03 — see below; it was measured inside a
warm-up ramp). Re-predicting *and* grading is 2.6 ms, so both fit a 60 fps frame. That is the right shape for the feature: predict per shot or per
keyframe, grade per frame. `NeuralLUTRunner` has a real workload and
`Lava.frozen_stats().misses == 0` on a fresh process.

### Flush-hang soak

**508,740 trials, zero hangs** (355,510 + 153,230 across two runs, ~24 min total,
`tools/soak_flush_hang.jl`, log in `soak_flush_hang.log`). Every third trial ran
under `with_dispatch_timing`, which is the condition the one post-fix recurrence
carried. Against a bug that previously recurred once after ~90 clean trials, half
a million clean trials is the strongest evidence available that the dominant path
is fixed — though it cannot prove the tail is gone. Stopped twice, deliberately,
so it would not contend with a timed run.

### Negative and incidental results, recorded

- **`models-to-port.md` is wrong about RIFE needing no checkout.** The weights
  archive carries `IFNet_HDv3.py` but not the `model.warplayer` it imports.
  `dev/Practical-RIFE` is now cloned (MIT, two files' worth of use).
- **Upstream RIFE ships a cache bug of the class `GUARDRAILS.md` §8 describes.**
  `warplayer.backwarp_tenGrid` memoises the sampling grid on `(device, size)`;
  `torch.export` fills that key with a FakeTensor and the next real forward
  silently reuses it, failing 100 lines later at `save_file`. The exporter clears
  it.
- **Depth Anything's attention decomposes anyway.** Exporting from CUDA does not
  keep it fused the way it does for Whisper: DINOv2 falls back to manual
  `bmm` + `softmax` when xFormers is absent, so the graph has 24 `bmm` and 12
  `_softmax`. The CUDA-export rule is about *SDPA dispatch*, not a guarantee.
- **`assetpath` reports a wrong path when it finds nothing.** With no export
  present, `DepthAnythingRunner.assetdir()` returns `/gen/graphs/depthanything`
  — root-anchored, not workspace-anchored. It resolves correctly once the
  directory exists, so this only bites the error message a new user sees first.
- **Measurement spread on this laptop is wider than `perf-plan.md`'s 13%.**
  Repeat runs of the same benchmark in the same session varied ~8% (neural LUT
  classifier 7.0/8.4/8.4 ms) and RIFE ~8% (478/500/516). Single numbers here are
  worth one significant figure.

### All three runner packages are now real

`NeuralLUTRunner`, `RIFERunner` and `DepthAnythingRunner` each have a model
struct, a call and a workload, and **all three report
`Lava.frozen_stats().misses == 0` on a fresh process** — the measurement the
scaffolds' own TODO said was the point.

- `neurallut` / `predictlut` / `grade!` — a 4K frame graded from a predicted look
- `rife` / `interpolate!(out, model, a, b; t)` — zero-pads to the export's baked
  size and crops back, any `t` in [0, 1]. Checked: `t = 0.5` between flat frames
  at 0.30 and 0.40 gives 0.3530 against an exact midpoint of 0.35, and `t = 0.25`
  moves toward the first frame.
- `depthanything` / `depthmap!` — ImageNet normalisation folded into the resize

**A workload must build its `Model` *inside* `@compile_workload`.** RIFE sat at
`misses == 9` no matter what frame size the workload used. The cause is that
`Model`'s last pass is `hoistconstants(graphs, weights, backend)`, which folds
constant *subgraphs* by running them on the device — and RIFE has two, the
`arange` pair that builds the warp sampling grid. Constructed in front of the
block, those dispatches are never frozen. Moving one line took it to 0. The other
two packages have no constant subgraph today and measured 0 either way; they were
changed to match anyway, because "today" is doing the work in that sentence.

`NeuralLUTRunner` was also switched from `loadgraph` to `Model` plus a planned
slab — without it the shipped runner would have been ~1.8x slower than the
benchmark that measures it, which is its own kind of wrong.

### Artifacts are bound, uploaded and resolving

All three packages now have an `Artifacts.toml` in the SAM 2 shape — lazy, with a
`git-tree-sha1` and a `sha256` against `assets-v1` — written by
`tools/make_artifacts.jl`, which packs `gen/graphs/<name>` and binds it in one
command so a re-export can redo it.

| artifact | tarball | tree sha1 |
|---|---|---|
| `neurallut` | 2.1 MiB | `69986cb87d2a` |
| `rife` | 20.1 MiB | `b41f6be8ace2` |
| `depthanything` | 87.7 MiB | `f33c510b8044` |

**A ported model now reads its assets from its artifact and from nowhere else** —
`assetdir() = @artifact_str("rife")`, one line, no fallbacks. The first attempt
kept an env-var override and a generated-tree branch in front of it; both are
gone. Nothing in the repo ever set the six environment variables, and the
generated-tree branch is precisely what made a *broken* artifact download
invisible for as long as it was: on any machine with a `gen/` tree, the fallback
always answered, so nobody could tell the artifact path had never worked.

The consequence is a real workflow change and it is the right one: **changing a
model's assets means re-binding its artifact.** Re-export, then
`tools/make_artifacts.jl <name>` — that hashes the new tree and rewrites the
`Artifacts.toml`, so `assetdir()` resolves to the new content immediately;
uploading is only how it reaches anyone else. Verified: the runners now load from
`~/.julia/artifacts/...` and produce byte-identical output to the `gen/`-backed
runs, with `frozen_stats().misses == 0`.

`DNNKernels.assetpath` is **deleted** — it has no callers left. The six runners
for models that are not ported yet do not fall back to a directory either: their
`assetdir()` throws and names the three steps that would fix it (export, bind,
change the line), and `ready()` returns `false` so the workloads still precompile
to nothing rather than failing. None of the six has an export on this machine, so
there was never a directory for them to point at — `models-to-port.md` lists
Whisper and BasicVSR++ as exported, but those were produced on the desktop and
have not been published.

End state across all eleven runner packages: five resolve through
`@artifact_str`, six say plainly that they are not ported, and nothing consults
an environment variable or a working tree.

The tarballs carry the graph, the weights and the op histogram, and deliberately
**not** `reference*.safetensors`: those are what `tools/verify_*.jl` diffs
against and the exporter regenerates them in one command. It matters most for
RIFE, whose references are ~80 MB against 22 MB of weights.

All three are **uploaded to `assets-v1` and verified end to end**: each URL serves
bytes whose sha256 matches what `bind_artifact!` recorded, `Pkg` installs each one
(which checks the `git-tree-sha1`, so that is confirmed too), and each unpacks to
exactly the three intended files. A fresh clone with no `gen/` now gets working
assets rather than an error naming a directory it does not have — which is the
difference between a working port and an installable one.

Note this supersedes `models-to-port.md`'s claim that RIFE is blocked on
re-hosting `train_log/`: the artifact carries the *exported graph*, not upstream's
checkpoint, so nothing needed mirroring first.

### What is not done

- **The conv findings are not fixed**, per the brief — these three are bring-up,
  not kernel work. They are handed to whoever owns `kernels-to-port.md` item 1.

### New in this branch

`GPUFiltering`: `lut3d.jl` (trilinear apply, matches upstream's own
`trilinear_kernel.cu` to 3.0e-7 with full coverage asserted) and `resizeplanar.jl`
(RGB frame → planar NCHW, `align_corners=false` to match `nn.Upsample`, with
optional per-channel normalisation folded into the same pass). `DNNKernels`: the
InstanceNorm op. Three runner packages with workloads. `tools/`: three exporters,
three verifiers, three benchmarks, one PyTorch baseline, and the soak.

RIFE's own padding kernel stayed in `RIFERunner` rather than joining
`resizeplanar!` in `GPUFiltering`: the two differ in what happens outside the
source frame, which is the entire point of each. RIFE pads because its flow field
is in pixels, and resizing the input would silently rescale every motion vector
the network predicts.

---

## 2026-08-03 — SETTLED STATE (read this, then the entries below for how)

The three entries that follow are a working log and they correct each other four
times. This is what is true at the end of the day, so nobody has to reconstruct
it from the sequence.

**The three ports are unchanged and still correct** against `main` at `aefd9e6`,
which carries the DNNKernels refactor, Lava's per-device allocator and
flash-decoding. Parity: neural LUT 3.576e-07, RIFE 3.265e-04 max, Depth Anything
4.232e-05. Runner output byte-identical to 2026-08-02.

**Corrected timings**, after finding that `timed()` warmed up 3 calls into a
24-call ramp and measured inside it:

| | value | note |
|---|---|---|
| LUT apply at 4K | **0.80 ms** | target < 2 ms, met |
| LUT classifier | **1.8 ms** | was reported as ~8–14 ms; that was the ramp |
| LUT re-predict + grade | **2.6 ms** | fits a 60 fps frame — supersedes "predict per shot" |
| RIFE at 1080p | **314 ms** | target 16.67 ms, missed ~19x |
| Depth Anything at 518² | **374 ms** | vs PyTorch 25.3 ms, missed ~15x |
| SAM 2 encode / decode | **173.8 / 17.7 ms** | desktop 100.4; this card is ~1.8x |

**Workload coverage is now a real claim**, not `frozen_stats().misses == 0`:
all three editor paths run under `Lava.no_pipeline_compilation` with **0
refusals**, and the negative control (a `Val{K}` novel per run) refuses 1, so the
instrument fires.

**The VRAM finding, settled.** Loading SAM 2 leaves the pool at **7.9x its
resident set** (63 blocks / 4032 MiB against 505 MiB resident — the resident set
is 505 MiB, *not* the 941 MiB checkpoint, because `dropdead` drops 824 orphaned
weights and `hoistcasts` replaces fp32 masters with fp16).

- **Cause:** `Model`'s `hoistconstants` runs **186 constant-subgraph folds on the
  device at load**. The weights those consume are uploaded into the same blocks
  as the weights that survive, then discarded. MatAnyone folds **0** and does not
  reproduce; the three small ports strand nothing.
- **The fix that works:** one `reclaim_empty_pool_blocks!` after load — **7.9x →
  1.50x, 52 ms, once**. `POOL_SOFT_CAP` does not do this today: past the cap the
  allocator runs a GC (which cannot help — nothing is garbage) and cuts a block
  anyway, and the reclaim scan is on the OOM retry path only.
- **Two fixes I proposed and then disproved — do not attempt either.** Folding
  before the upload is impossible (`constops` seeds from `:weight`, so folding
  consumes what it would precede). A two-phase upload is computable but pointless
  (foldable ops consume 95% of the weight bytes, so phase one is nearly
  everything). The populations are interleaved *in the graph*, so no `Model`-level
  ordering separates them; the residual 1.50x needs the allocator to learn
  lifetimes, which is `lava-core`'s call.
- **Reproducer:** `tools/pool_fragmentation_probe.jl` — no model, no weights, no
  export; exits non-zero when it reproduces.

**The three suites now run, and assert the above.** `Pkg.test` had never worked
for *any* runner package here — none declares a test target, so the suite gets an
environment without `Test` and every `runtests.jl` dies on `using Test` before
asserting anything. `SAM2Runner`, whose test file the scaffolds cite as the shape
to copy, is among them. **The other six packages need the same three lines**
(`[extras] Test` + `[targets] test = ["Test"]`); mine and `newrunner.py` have
them, so nothing scaffolded after this inherits the defect.

Each of the three now carries the latency test the scaffold's docstring promised,
in a subprocess, asserting `no_pipeline_compilation` **and a negative control**:

    NeuralLUT      refused = 0, control = 1, compile = 0.0 s, wall 0.55 s
    RIFE           refused = 0, control = 1, compile = 0.0 s, wall 1.41 s
    DepthAnything  refused = 0, control = 1, compile = 0.0 s, wall 0.74 s

Writing it produced three faults of the kind it exists to catch, which is the
argument for having written it rather than trusting a session's worth of manual
checks: a dep the test env lacks; a testset that passed with **zero** assertions
when the subprocess returned nothing; and stdout-only capture, which made a
crashed subprocess indistinguishable from "no GPU here" — that one would have
gone green forever on a CI machine without a device.

**Soak:** 1 718 240 trials, zero hangs, on top of 508 740 yesterday — and see
the fourth entry below, which makes that number mean much less than it reads.

Owners for what remains: the allocator is `lava-core`'s; the desktop should
re-check its own 1 181 MiB figure, which was measured the same way; and the six
**all eleven runner suites run, and ten pass** — verified in one sweep at the end
of the day, not assembled from runs taken at different points:

    NeuralLUTRunner PASS   SAM2Runner      PASS   KokoroRunner     PASS
    RIFERunner      PASS   BasicVSRRunner  PASS   ProPainterRunner PASS
    DepthAnything…  PASS   DeepFilterRunner PASS  WhisperRunner    PASS
    DemucsRunner    PASS   MatAnyoneRunner FAIL (vkWaitSemaphores hang)

This morning none of them could run at all — no package declared a test target.

**And the monorepo's own dev environment did not resolve.**
`dev/JuliaVision/Project.toml` lists `Lava` in `[sources]` but not in `[deps]`,
which Pkg refuses outright: *"Sources for `Lava` not listed in `deps` or
`extras`"*. So `Pkg.activate(".")` — the workflow written at the top of that very
file — failed for anyone who tried it. Not my change (empty diff against
`origin/main`); found only because I finally checked the repo's env rather than
the ad-hoc workspace one I had been dev'ing into all day. One line, and it
resolves — and `Pkg.test` then works from that environment too, spot-checked on
a ported package and a scaffold, which is the path a contributor actually takes.
The one failure is a real bug with a one-command reproducer and six controls
(fourth entry below), not an infrastructure problem.

---


## 2026-08-03 — still green after the refactor, and "0 misses" made into a real claim

Re-ran everything against `main` at `aefd9e6` — which now carries the DNNKernels
refactor (globals 34 → 0, plan objects, per-device `Device`), Lava's per-device
allocator and flash-decoding, all of which landed *after* `sd/small-models` was
merged.

**Nothing regressed.** All three parity checks pass with the same numbers, and
the three runners produce byte-identical output to 2026-08-02:

| | parity | runner output |
|---|---|---|
| neural LUT | 3.576e-07 | `RGB(0.3627222, 0.78073686, 1.0)` |
| RIFE | 3.265e-04 max, 4.897e-07 mean | centre `RGB(0.35299557, 0.5, 0.6470045)` |
| Depth Anything | 4.232e-05 | depth max 3.4986925 |

The one digit that moved is Depth Anything's mean, 8.481e-07 → 8.485e-07.

### The `misses == 0` claim was weaker than I wrote it, and now is not

`STATUS.md`'s cross-project list says it plainly: **"0 misses" could mean the
frozen cache worked, or that the driver's own shader cache served everything** —
and the miss report identified modules by `hash(spirv_bytes)`, the *sampling*
hash, so two modules differing in one byte counted as one. Yesterday's entry
asserts `Lava.frozen_stats().misses == 0` three times as if it settled the
question. It did not, and that is my error rather than a change of circumstance.

Re-tested with the stronger instrument. `Lava.no_pipeline_compilation` empties
`PIPELINE_CACHE` first — so a Julia-side hit cannot mask a cold `VkPipelineCache`
— and refuses any pipeline that would need compiling:

| editor path | refusals |
|---|---|
| `predictlut` + `grade!` at 4K | **0** |
| `depthmap!` | **0** |
| `interpolate!` at 1080p | **0** |

**With a negative control, because a green from an instrument that cannot fire is
worth nothing** — this repo has now hit that class three times, and the same
`VK_PIPELINE_COMPILE_REQUIRED` bug that made this instrument unable to report a
miss on Linux was live until recently. The control dispatches a kernel whose body
is novel *per run* (a `Val{K}` with `K` from `RandomDevice`), so neither Lava's
cache nor the driver's cross-process one can serve it: **refused = 1, the
instrument fires.** The three zeros above are therefore real.

So the workloads do cover the paths the editor takes, which is what the packages
exist for — but it is worth being exact about what the test now proves: every
pipeline these three paths need was already in the driver's cache, on this
machine, in a fresh process. That is the first-use cost the workload removes.

### Nothing else outstanding

`small-models` is marked done in `STATUS.md` and the three artifacts are live on
`assets-v1`. Both findings from yesterday — TF32 in the exporters, and
benchmarking through `Model` rather than `loadgraph` — are now in the
cross-project "act on these first" list, so they need nothing further here.

---

## 2026-08-03 (second entry) — a warm-up ramp I was measuring inside, and 4 GB of pool for 942 MB of weights

### The classifier is 1.8 ms, not ~10 — my benchmark was measuring its own warm-up

Every classifier number in this report has been unstable: 7.0, 8.4, 8.4, 10.5,
9.5, 10.5, then 2.1, 4.4, 5.5, 11.1 ms on an idle card. I wrote that up twice as
noise and once as a refactor speed-up. It was neither.

Timing 200 single calls and printing them **in sequence** rather than sorted:

    call 1        2 834 ms
    calls 2-24    7-20 ms, decaying
    calls 25-200  ~2 ms, flat

The outlier indices are `[1 … 24]` — contiguous, nothing after. It is a warm-up
ramp, and `timed()` in all three `bench_*.jl` warmed up **three** calls before
measuring, so the median-of-seven landed at a random point on the ramp. Over 60
samples the distribution was min 1.19 / median 1.47 / p95 10.13 / max 569.67 ms —
a heavy right tail that is entirely the ramp, not a stall.

Warm-up is now 30 calls. Three consecutive runs: **1.79 / 1.66 / 1.81 ms**, a 9%
spread instead of 5x.

**This changes the port's advice.** Applying a look is 0.80 ms at 4K and
predicting one is 1.8 ms, so re-predict *and* grade is **2.6 ms** — inside a 60
fps frame. Yesterday's "predict per shot, grade per frame; re-predicting costs
18x" was an artefact of the ramp. Prefer the precomputed-LUT form because it is
3x cheaper and says what it means, not because the budget forbids the other one.

RIFE (314.6 ms) and Depth Anything (374.2 ms, PyTorch 25.3) are unchanged by the
fix, as expected: a 20-call ramp is negligible against a 300 ms call, which is
also why only the small graph ever looked noisy.

### SAM 2 on this machine, after flash-decoding

| | today | 2026-08-02 | desktop |
|---|---|---|---|
| encode | **173.8 ms** | 182.3 | 100.4 |
| decode, one click | **17.7 ms** | 17.6 | 2.21 *(replayed)* |

Decode is unchanged here, which is what should be expected rather than a
disappointment: the desktop's 2.21 ms is a **replayed captured command buffer**
and `segment()` is not replayed, so flash-decoding's 3.55x on cross-attention is
not what this number is dominated by.

### A VRAM regression that only a small card would notice

Measuring the same SAM 2 script as yesterday, `nvidia-smi` reports **4 467 MiB**
for the process where yesterday it reported **1 337 MiB**. `STATUS.md` records the
VRAM goal as met at 1 181 MiB.

Located, not guessed:

- Lava's pool holds **63 blocks x 64 MiB = 4 032 MiB**, against a
  `POOL_SOFT_CAP` of 2 048 MiB — nearly 2x its own cap.
- All 63 blocks are allocated inside **`sam2model()`** — the weight upload.
  `encode` adds **zero** blocks, and a second encode adds zero. So this is
  residency, not transients, and not a leak per call.
- SAM 2's weights are ~942 MB on disk. 4 032 MiB to hold them is 4.3x — but
  both halves of that are wrong and the entries below correct them: the
  blocks are mostly *empty*, and the resident set is 505 MiB, not 942, so
  the real ratio is **7.9x**.

The mechanism is visible in `memory.jl`: past `POOL_SOFT_CAP` the allocator asks
the GC and then **cuts a new block anyway**, and `reclaim_empty_pool_blocks!` is
documented as running "only on the OOM retry path". So the pool ratchets up and
never gives back until an allocation is about to fail. That is invisible on a
desktop and decisive on 8 GB — it is why `sam2model()` OOM'd here earlier today
with 2 246 MiB of budget free.

**Not attributed to a commit.** The only allocator change since yesterday is
`POOL_BLOCKS` → `pool(ctx).blocks` (Lava `83b9b10`), which with one device should
be equivalent; DNNKernels' globals-to-plan-objects refactor landed in the same
window and SAM 2's loader does not go through `Model`. Bisecting wants a machine
that can hold two Lava versions comfortably, and the allocator worklist already
lives in `projects/lava-core/REPORT.md`. Handing it over with the measurement
rather than a guess.

Worth pairing with `GUARDRAILS.md` §6's own advice: this is a *memory* number,
and the only reason it was caught is that this machine has 8 GB. The desktop's
1 181 MiB "goal met, nothing to do" may no longer be true there either.

---

## 2026-08-03 (third entry) — the 4 GB is 81% empty blocks, and `POOL_SOFT_CAP` never reclaims

Chased the VRAM finding above to a mechanism. **It is not fragmentation, not
residency, and not the allocator's packing** — all three of which I named as
suspects, and all three are wrong.

### Pure Lava packs this shape fine

SAM 2's checkpoint is 909 tensors, 941 MiB, median **2 KiB**, largest exactly
64.0 MiB (right at `POOL_LARGE_THRESHOLD`, so it does *not* bypass the pool).
Allocating that exact size distribution through `KA.allocate` and holding every
buffer live:

    909 buffers, 941 MiB, minimum blocks = 15
    all held live: 16 blocks (1024 MiB) -> 1.1x minimum

So the bump allocator packs a 2 KiB median at 1.1x. Nothing wrong with it.

### The blocks are empty

    after load          : 63 blocks (4032 MiB)   nvidia-smi 3591 MiB
    after drain + GC(true): 63 blocks (4032 MiB)   <- neither helps
    reclaim_empty_pool_blocks! freed 51 blocks, 3264 MiB
    after reclaim       : 12 blocks ( 768 MiB)   nvidia-smi  326 MiB
    after encode        : 12 blocks ( 768 MiB)   nvidia-smi  752 MiB

**81% of the pool was empty.** SAM 2's true steady-state footprint is 768 MiB of
pool and 752 MiB by `nvidia-smi` — *under* the 1 181 MiB the goal is stated in,
not 3.8x over it. The number in the entry above is real as a measurement of what
the process holds, and wrong as a description of what SAM 2 needs.

`Model.scratch` is still empty at this point (`n = 0`, the slab is allocated
lazily on first `encode`), so every one of those 63 blocks comes from uploading
734 weight entries — and `encode` afterwards adds **zero** blocks.

### Why nothing gives them back

`reclaim_empty_pool_blocks!` is documented as "called from `vk_alloc` /
`alloc_pool_block` **only on the OOM retry path**, so steady-state allocations
don't pay the scan cost". And past `POOL_SOFT_CAP` the allocator calls
`collect_for_pool!` — a *garbage collection*, which does nothing here because
nothing is garbage — and then cuts a new block regardless.

So the cap does not cap. It gates a GC, and the one thing that would actually
return memory runs only when an allocation is already failing. A bulk upload of
909 buffers walks the pool up to 4 GB and it stays there until something OOMs —
which is exactly what happened here earlier today, at 2 246 MiB of free budget.

### Two things worth doing, neither of them mine

1. **Call `reclaim_empty_pool_blocks!` after a bulk upload.** One line at the end
   of model loading recovers 3.2 GB on an 8 GB card. This is the cheap fix and it
   needs no design.
2. **Make `POOL_SOFT_CAP` reclaim before it collects.** Asking the GC for memory
   that is not garbage cannot work; the empty-block scan is the operation that
   matches the situation. That is a change to Lava's allocator policy and belongs
   to `lava-core`, whose report already owns the worklist.

Filed here rather than acted on because these three models are bring-up, and
because the desktop should confirm the numbers — but the mechanism is not
machine-specific and its own `1 181 MiB` figure was measured the same way.

### Checked, and these three models are not affected

Before changing anything in my own loaders, I measured whether they strand blocks
the way SAM 2's load does. They do not:

| | pool after a run | `reclaim_empty_pool_blocks!` frees |
|---|---|---|
| neural LUT | 1 block (64 MiB) | **0** |
| Depth Anything | 3 blocks (192 MiB) | **0** |
| RIFE | 7 blocks (448 MiB) | **0** |

So the one-line fix would recover nothing here and has **not** been added to
`neurallut`, `rife` or `depthanything` — a call that always frees zero is a
reader's puzzle, not a safeguard.

That also bounds the bug usefully. These three upload 21–239 weight entries
totalling 2–95 MiB and strand nothing; SAM 2 uploads 734 entries totalling
941 MiB and strands 3.2 GB. Whatever the threshold is, it is reached at SAM 2's
scale and not at these — which means the models to check next are the other large
ones (MatAnyone, and Whisper's encoder at 2.55 GB fp32) rather than any of the
small ports.

Soak, meanwhile: **221 450 trials, zero hangs** since the restart, on top of the
508 740 recorded yesterday.

### A standalone reproducer: it is lifetime mixing, and reclaim is only a mitigation

`tools/pool_fragmentation_probe.jl` — no model, no weights, no export, ~90 lines
of `KA.allocate`. Three patterns over SAM 2's exact size distribution:

    909 buffers, 941 MiB total, median 2.2 KiB — minimum 15 blocks

    1. hold all live (control)             grew 16 blocks (1024 MiB),  0 empty,  16 PINNED
    2. allocate and drop each immediately  grew 16 blocks (1024 MiB), 16 empty,   0 PINNED
    3. interleave transient + resident     grew 32 blocks (2048 MiB),  1 empty,  31 PINNED

Reading them together:

- **(1) the allocator is fine.** Same sizes, every buffer live, 16 blocks against
  a 15-block minimum — 1.1x on a 2 KiB median.
- **(2) transient churn strands everything.** The pool grows to the whole working
  set and every block ends up *empty*. This is the part `reclaim_empty_pool_blocks!`
  can fix, and it is the 51 blocks / 3.2 GB it recovered on SAM 2.
- **(3) a weight upload's actual shape doubles the pool, and reclaim cannot
  touch it.** 31 of 32 blocks are **pinned** — each holds at least one live
  tensor, so no scan can return them however empty they are.

**So the one-line mitigation is not the fix.** It recovers the wholly-transient
blocks and is worth doing, but pattern 3 says the pool will still sit at ~2x the
resident set on any load that allocates a transient beside each keeper. The
allocator has no notion of separating a short-lived allocation from one that
lives as long as the model, and a bump allocator cannot recover from that after
the fact — the placement decision is where it is lost.

Worth stating precisely what this is **not**, because two of them were my own
earlier guesses: not fragmentation by size (pattern 1 has the identical
distribution and wastes 6%), not a leak (every buffer is freed, with `GC.gc(true)`
and `drain_deferred_frees!` before each count), and not the per-device pool split
(`POOL_BLOCKS` → `pool(ctx).blocks` is equivalent with one device).

The probe exits non-zero when pattern 3 pins more than the minimum, so it can go
straight into a test suite once the behaviour is decided on. It is filed from
here rather than fixed because the allocator is `lava-core`'s.

### The trigger, named: 186 constant-subgraph folds at load

MatAnyone was the control worth running, because it is the other large model with
an artifact on this machine and it goes through the same `DNNKernels.Model`:

    MatAnyone   2 blocks (128 MiB), reclaim frees 0  — does NOT reproduce
    SAM 2      63 blocks (4032 MiB), reclaim frees 51 (3.2 GB)

Same loader, opposite outcome. The `@debug` line `Model` already prints says why:

| | constant-subgraph ops folded | batch-norms | blocks stranded |
|---|---|---|---|
| MatAnyone | **0** | 75 | 0 |
| SAM 2 | **186** | 0 | 51 |

`hoistconstants(graphs, weights, backend)` is described in `driver.jl` as "the one
pass that has to **run** the ops it folds, so it comes after the upload and works
on the device weights". For SAM 2 that is 186 computations executed at load time,
each allocating transients *after* the resident weights are already placed —
which is pattern 3 of the probe exactly, and it is why the blocks come out pinned
rather than empty. MatAnyone folds none, allocates no transients after its
upload, and lands at 128 MiB for ~130 MiB of weights.

So the causal chain is complete: **186 device-side constant folds → transients
interleaved with resident weights → 4 GB pool for 941 MiB → 3.2 GB unreclaimable
by anything but the empty-block scan, and 12 blocks not even by that.**

Two fixes follow from it, and both are cheaper than changing the allocator:

1. **Fold constants before uploading the resident weights**, so the transients
   are allocated and freed while the pool is still empty and nothing pins the
   blocks. This is an ordering change in `Model`, not an allocator change.
2. **Or reclaim once after `hoistconstants`**, which recovers the wholly-transient
   blocks at a known-quiet point rather than waiting for an OOM.

Neither is mine to make — `Model` is `kernels-refactor`'s and the allocator is
`lava-core`'s — but the measurement now names the pass, the count, and a control
that does not reproduce.

### Correcting my own two numbers, and how much the one-line fix actually buys

Measuring the resident set rather than the file size, which is what I had been
comparing against:

    resident weights : 734 arrays, 505 MiB  -> minimum 8 blocks
    pool at load     : 63 blocks (4032 MiB) = 7.9x
    after reclaim    : 12 blocks ( 768 MiB) = 1.50x minimum

Two corrections to what I published earlier today:

- **The resident set is 505 MiB, not 941.** The checkpoint on disk is 941 MiB,
  but `Model` drops 824 orphaned weights and `hoistcasts` replaces fp32 masters
  with fp16 — the same pass `STATUS.md` credits with 849 MB on SAM 2. Comparing
  the pool against the *file* was the wrong denominator, which is precisely
  `GUARDRAILS.md` §5's rule and I broke it while quoting it.
- **So the overhead is 7.9x, not 4.3x** — worse than I reported, not better.

And a fairer verdict on the mitigation than "not the fix": **one call to
`reclaim_empty_pool_blocks!` takes SAM 2 from 7.9x to 1.50x.** That is most of
3.3 GB on an 8 GB card for one line at a known-quiet point. The residual 1.50x —
about 256 MiB of blocks pinned by a live tensor apiece — is the genuine pattern-3
part that ordering (fold constants before the resident upload) would be needed to
remove.

So the two fixes are not alternatives of equal weight. The reclaim is the cheap
one and it recovers ~81% of the excess; the ordering change is what would take
the last 50%, and only it addresses the mechanism.

### The ordering fix as I filed it is impossible — and the mechanism is sharper than "transients"

I recommended "fold constants before uploading the resident weights" without
checking it could be done. It cannot. `constops` seeds its known-constant set
with `kind === :weight` and grows it to fixpoint, so a constant subgraph is by
definition one that **reads only weights** — folding consumes weights, and cannot
precede the upload that makes them readable. That is why `driver.jl` puts the
pass after the upload, and the comment there says so.

What that clarifies is the mechanism, which I had been describing as "transients"
without knowing what they were. They are not staging buffers. They are **the
weights the fold consumes and `dropdead` then discards** — uploaded into the same
blocks as the weights that stay, then dropped, leaving those blocks partly empty.
The debug line names the quantity: 824 orphaned weights dropped, 186 subgraphs
folded, against 734 that remain. The pool cannot tell the two populations apart
because nothing told it they had different lifetimes.

So the feasible form of the ordering fix is a **two-phase upload**: upload only
the weights the constant subgraphs consume, fold, reclaim, then upload the 734
survivors into a pool that is empty again. That is a real change to `Model` and a
bigger one than I implied — it needs `constops` run before the upload to know
which weights are in phase one, which is possible since that analysis is
host-side and reads only the graph.

### The cheap fix, costed

    reclaim scanned 63 blocks, freed 51 (3264 MiB) in 52.4 ms
    second call (nothing to free)                  in  5.12 ms

**52 ms, once, at load, to recover 3.2 GB** — against a load that takes seconds.
The scan alone is 5 ms when there is nothing to free, so it is cheap to call
unconditionally. Against 7.9x → 1.50x, that is the whole recommendation: do this
one, and treat the two-phase upload as the follow-up that removes the last 50%
rather than as an alternative to it.

Both numbers are from this machine, and the reclaim cost will scale with block
count rather than with model size.

### …and the two-phase upload does not work either. Both structural fixes are dead

I claimed the two-phase upload was feasible because `constops` is host-side, then
made the same mistake twice in a row by not testing it. Running the analysis on
SAM 2's graphs without a device:

    sam2_encoder   569/1353 ops foldable, phase-1 weights 410/909 = 896 of 941 MiB (95%)
    sam2_decoder    25/ 174 ops foldable, phase-1 weights   4/903 =   0 of 856 MiB ( 0%)

The analysis *is* host-computable, so that half of the claim holds. The fix built
on it does not: **"upload only what the constant subgraphs consume" is 95% of the
weight bytes.** Phase one would upload nearly everything, fold, reclaim, and then
upload the remaining 5% — the interleaving it was supposed to avoid happens
inside phase one, because the weights that feed foldable ops and the weights that
survive are overwhelmingly the same weights.

So both structural proposals are disproved, by measurement, and neither should
reach anyone's worklist:

1. ~~Fold constants before uploading the resident weights~~ — impossible;
   `constops` seeds from `:weight`, so folding consumes what it would precede.
2. ~~Two-phase upload~~ — possible to compute, pointless to do; phase one is 95%
   of the bytes.

**What stands is the reclaim: 7.9x → 1.50x for 52 ms, once, at load.** The
residual 1.50x is ~256 MiB of blocks each pinned by one surviving tensor, and
nothing at the `Model` level can separate those lifetimes — the two populations
are genuinely interleaved in the graph, not merely in the upload order. Removing
the last 50% therefore needs the *allocator* to learn about lifetimes (a separate
pool or an arena for load-time transients), which is `lava-core`'s call and a
real design change rather than an ordering tweak.

Recorded in full because the negative result is the useful part: without it, two
plausible-sounding reorderings would have been attempted and both would have
failed for reasons that cost an afternoon each to discover.


---

## 2026-08-03 (fourth entry) — a REPRODUCIBLE vkWaitSemaphores hang (ATTRIBUTED: non-square attention)

Adding a test target to the remaining runner packages made two suites runnable
for the first time, and both fail the same way:

    LavaError during vkWaitSemaphores:
      timed out after 120.0 s waiting for timeline value 405 on 50 in-flight batch(es)
      timeline counter = 355, next_timeline = 405, replay watermark = 0
      batch 1: signals 356, waits on nothing ...

`Pkg.test("MatAnyoneRunner")`, on this machine, **every time**.

**Correction: SAM 2 does not fail this way.** I wrote that both suites failed
identically, having read MatAnyone's error and only a summary line for SAM 2 —
and SAM 2's fixture is 1024 x 1024, square, which the characterisation below says
should *not* hang. It does not. Its suite failed on **missing test deps**
(`Random`, then `Printf`) in the target I had just added, and then on **a
regression of mine**: its subprocess reads `refs.safetensors` from `assetdir()`,
but the references are their own artifact (`sam2-large-refs`) and `assetdir()` is
the weights-only one. That read worked before only because `assetdir()` fell
through to a generated tree where both happened to sit side by side — exactly the
fallback I deleted this morning. Pointed at `refsdir()`, **SAM 2's suite passes:
6 + 24 assertions, first call compiles nothing.**

Its own test could not say any of this, because it used `read(cmd)` without
capturing stderr — the identical defect I had fixed in my own latency test an
hour earlier and then failed to recognise when it was hiding this from me. Fixed
there too. Reproduced with the soak running and again with the GPU fully
idle, so it is not contention — checked specifically, having made exactly that
mistake this morning.

### ATTRIBUTED, and this corrects what the rest of this entry first claimed

One A/B settles it. Same model, same code, same machine — only the aspect ratio
of the input changes:

    128 x 96   timed out after 120.0 s at timeline 405, 50 in-flight batches
    128 x 128  OK in 23.9 s

That is `STATUS.md`'s **"`blockfor` refuses non-square attention"** — it is **not**
the flush hang and not a new bug.

**But "non-square" is the wrong rule, and the tracker should say so.** Five
shapes through the same call:

| W x H | result |
|---|---|
| 128 x 96 | **hangs** |
| 96 x 128 | **hangs** (so it is not orientation) |
| 96 x 96 | OK, 17.2 s (so it is not the dimension 96) |
| 128 x 128 | OK, 23.9 s |
| **160 x 128** | **OK, 23.4 s — and this one is non-square** |

| 160 x 96 | OK, 20.5 s |
| 112 x 128 | OK, 21.2 s |
| 96 x 160 | OK, 17.3 s |

**And that last row disproves the rule I proposed in the row above it.** I wrote
that the hanging cases were "unequal dims where the smaller is below 128", then
ran the shape that predicts — 160 x 96, unequal, smaller side 96 — and it passes.

So of six shapes, **only the {96, 128} pair reproduces**, in either orientation.
Non-squareness does not explain it (160 x 128 and 160 x 96 are both lopsided and
fine), the dimension 96 does not (96 x 96 is fine), and neither does my
smaller-side rule. As a token grid at /16 the hang is 6 x 8 or 8 x 6, while
6 x 6, 8 x 8, 10 x 8 and 10 x 6 all pass — no invariant I can see fits six
points.

**It is one pair, not a range and not an axis.** Eight shapes, expressed as the
/16 token grid the attention actually sees:

    hangs:  6 x 8,  8 x 6
    passes: 5 x 8,  6 x 6,  6 x 7,  6 x 9,  6 x 10,
            7 x 7,  7 x 8,  8 x 8,  10 x 6,  10 x 8

### What the validation layers did and did not give

Asked directly whether the layers nailed this down: **no.** They eliminated, they
did not localise. Recorded because a negative instrument result is worth as much
as a positive one and is easier to forget.

| instrument | outcome |
|---|---|
| GPU-AV (`activate_all_debugging`, `verify_gpu_av` proving it fires) | **no OOB reported** during the run |
| Synchronization validation (no GPU-AV) | **zero `SYNC-HAZARD`s**, ran to the fault and reported nothing |
| Core validation + best practices | nothing before the fault |
| GPU-AV enabled | the layer itself **segfaults** in `vkQueueSubmit2` |

What actually localised it was **Lava's own** `set_dispatch_logging!(true)`, which
named `gpu_strided_gemm_kernel!`, and the `AUTO_SUBMIT_THRESHOLD` sweep. Neither
is a validation layer.

**Two caveats that matter for what to conclude.**

1. **"No OOB" is narrower than it sounds.** GPU-AV's own verification probe
   reports *"1 bytes **written** at buffer device address"* — it instruments
   **stores**. An out-of-bounds **read** of a garbage address would not be
   flagged, so address corruption is *not* excluded; only bad writes are.
2. The zero sync-hazard result is consistent with the barrier hypothesis being
   disproved by experiment, so two independent lines now agree this is not a
   missing barrier.

Where that leaves the search: not a bad store, not a race, not the GEMM kernel or
its shape — but plausibly a **bad address handed to the kernel**, which fits both
the surviving evidence and the layer crashing while instrumenting that submit.

### Root cause is in Lava's auto-submit path — but the fix is not found yet

`AUTO_SUBMIT_THRESHOLD` (`command.jl:149`) makes Lava submit mid-recording every
64 dispatches so host recording and GPU execution overlap. Varying **only** that
constant, everything else identical, MatAnyone at 96 x 128:

| threshold | result |
|---|---|
| 1 | hangs — **2 911 in-flight batches**, timeline wait times out |
| 32 | **OK**, 45.9 s |
| **64** (default) | **`ERROR_DEVICE_LOST`** |
| 63 | **hangs** — timeline **405**, 47 in-flight |
| 65 | **OK**, 45 s |
| 72 | **OK**, 45 s |
| 128 | **OK**, 45.3 s |
| 256 | **OK**, 45.8 s |
| disabled | **OK**, 46.1 s |

**It is not monotonic.** 32 and 128 are both clean, so "submitting mid-stream is
broken" is wrong — the threshold only decides *where* the batch boundary lands in
the dispatch sequence, and 64 puts it somewhere unsafe for this workload. That is
also why the bug looked input-size dependent: the input decides the sequence, the
threshold decides where it is cut.

The crashing dispatch is named by `set_dispatch_logging!(true)`:
`gpu_strided_gemm_kernel!` at `groups=(192,1,1)` — `ndrange = 12288` at workgroup
64 — which is the `addmm` GEMM `(256,256) x (256,48) -> (256,48)`. That GEMM run
standalone is clean, so it is the context, not the kernel.

**A hypothesis I tested and disproved.** `record_dispatch!` gates its barrier as
`(intra || boundary) && !effective_skip`, so a dispatch marked skip-barrier —
elidable against the *previous dispatch in its batch* — also suppresses the
barrier against a previously **submitted** batch, which `submit!` does not wait
on. Rewriting it as `(intra && !skip) || boundary` is defensible on its own terms
and **does not fix this**: still `DEVICE_LOST` at 64. Reverted rather than shipped,
because an unverified "improvement" to shared code is exactly what has gone wrong
elsewhere today.

**A second, distinct defect found on the way:** at `threshold = 1` the queue
accumulates **2 911 in-flight batches** before timing out. Nothing caps in-flight
batch count; that is its own bug and not the one being chased here.

**The threshold is not the variable — a specific point in the workload is.**
48, 56, 65 and 72 all pass; **63 and 64 both fail**, and 63 fails waiting on
**timeline value 405**, the same value the original 64 failure reports. Two
adjacent thresholds sharing a failure point, with neighbours clean on both sides,
means there is one place in MatAnyone's step where cutting the batch is fatal —
63 and 64 put a boundary there and the others do not. (63 and 64 are coprime, so
it cannot be one dispatch index divisible by both; the fatal region is a short
*window* that both cut into.)

Also disproved by experiment, not argument:

- **Not a use-after-free.** Forcing *every* free to defer — never destroying a
  buffer inline — still gives `DEVICE_LOST`. The narrower fix (also defer when
  `!isempty(bq.in_flight)`, the second window `memory.jl`'s own note predicts)
  likewise does not fix it. Both reverted.
- **Not a missing barrier**, per the earlier test.
- **Sync validation cannot speak to either.** Lava passes buffers as device
  addresses in push constants (`PhysicalStorageBuffer`), and sync-val tracks
  *bound resources* — it cannot see BDA traffic. Its "zero hazards" is a dead
  instrument, not evidence, and I wrongly cited it as a second independent line.

Next: log the dispatch index and kernel at each auto-submit boundary for 64
(fatal) against 65 (clean) and diff — that names the dispatch it is fatal to cut
before.

### Localised: the faulting submit is an `addmm` GEMM, and GPU-AV says it is not an OOB

*Re-verified against current upstream on 2026-08-03: Lava `328827f`, JuliaVision
`main` merged in. Still `ERROR_DEVICE_LOST` at 96 x 128 — the localisation below
was done on Lava `0903c6f` and survives the update.*

Ran the Rule-0 instruments properly. **`activate_all_debugging()`**, not
`enable_gpu_av()` — the latter defaults to `pool_disabled=false`, which its own
docstring says is *blind to sub-pool overruns*, and my first attempt used it and
got a false "GPU-AV is non-functional on this driver". The official entry point
exists to remove exactly that trap. With it, `verify_gpu_av` reports **GPU-AV
verified working**, so the instrument is proven to fire before anything is
concluded from it.

Under full validation the run does not report a bad access. It **segfaults inside
`libVkLayer_khronos_validation.so` during `vkQueueSubmit2`**, and the Julia stack
names the dispatch:

    MatAnyoneRunner.runmatanyone -> step! -> execute! -> timeop!
      -> runop!(::Val{Symbol("addmm.default")})        ops.jl:745
      -> DNNKernels.matmul!                            matmul.jl:80
      -> Lava mul! -> gemmlaunch!                      gemm.jl:1185
      -> ka_launch! -> vk_dispatch! -> submit! -> queue_submit_2

Instrumenting `addmm` gives the exact GEMM: it is the **149th** `addmm` of the
call and its shape is

    (16, 256) x (256, 48) -> (16, 48)      op id `addmm_3`

**But that GEMM alone does not fault.** `mul!` on freshly allocated
`(16,256) x (256,48)` completes in 5.0 s, as do N = 44 and 49. So the shape is
not sufficient — the fault needs the surrounding state. The non-validation
signature is consistent with that: `ERROR_DEVICE_LOST` **after 64 dispatches in a
256-dispatch batch**, i.e. mid-batch rather than on a cold submit.

What this rules out, all by measurement:

- **not an out-of-bounds access** — GPU-AV, proven firing, reports none during
  the run (the two OOB messages in the log are its own verification probe,
  before `RUN START`)
- **not attention** — zero attention calls happen before the fault
- **not the GEMM shape** — reproduced standalone, clean
- **not `blockfor`** — those shapes never take the blocked path

The validation layer *itself* crashing on `vkQueueSubmit2` is a signal worth
following: it suggests the submit is malformed rather than the shader misbehaving,
which points at Lava's command/batch construction around dispatch 64, not at the
GEMM kernel.

### SUPERSEDED AGAIN — it is `ERROR_DEVICE_LOST`, and attention is not involved

Driving the attention path directly, and then instrumenting the model, disproved
the mechanism in this entry and the one below it. What is actually happening:

    ERROR: LavaError during vkQueueSubmit2:
      ... Vulkan.ERROR_DEVICE_LOST) after 64 dispatches in batch (256 total)
    Crashed batch dispatch: 64 dispatches (compute)

**The GPU faults.** `ERROR_DEVICE_LOST` is a driver-level device loss, not queue
bookkeeping — which is what `vkWaitSemaphores timed out ... 50 in-flight batches`
looks like from the other side when the device dies mid-batch and nothing will
ever signal. Both signatures come from the same input sizes; the submit is
sometimes the call that notices and sometimes the wait is.

Three mechanisms of mine, all disproved by measurement:

1. **Not `blockfor`'s non-square bug.** `blockfor` returns 1 (no blocking) when
   `n != other` **or** `n < 64`, so these shapes never take the blocked path the
   tracker's workaround is about.
2. **Not "sequence length 48".** Instrumenting `sdpa!` shows 96 x 96 — which
   **passes** — already runs `Lq = 16, Lk = 48`. And a direct `sdpa!` at
   `L = 48` (E = 64, H = 8, B = 1) completes in 11.3 s.
3. **Attention is not reached at all.** With the probe in place, the hanging run
   at 96 x 128 emits **zero** `SDPA_PROBE` lines before dying. The fault is
   upstream of the first attention call.

The device recovers between processes — 96 x 96 re-verified passing afterwards —
so the shape matrix below stands as data. What does not stand is every
explanation I attached to it.

**What is established:** specific input sizes reproducibly cause a device loss in
`MatAnyoneRunner`, others reproducibly do not, and the boundary is sharp (44 and
49 clean either side of 48 in the original 2D sweep). The mechanism is
**unidentified**, and it is a GPU fault rather than a synchronisation bug — which
points at a bad dispatch, not at batch accounting. `Lava.enable_gpu_av` and
`spirv-val` are the Rule-0 instruments that should go at it next, on a machine
that can afford the device resets.

### SUPERSEDED — it is the sequence LENGTH 48, and it is not `blockfor`'s bug

The 2D framing above is wrong, and so is my attribution. Three geometrically
unrelated grids that all come to **48 tokens** hang identically:

    6 x 8   (96 x 128)   HANGS
    8 x 6   (128 x 96)   HANGS
    4 x 12  (64 x 192)   HANGS
    3 x 16  (48 x 256)   HANGS

and the neighbouring *lengths* pass, whatever grid produces them:

    32, 36, 40, 42, 44, 49, 54, 56, 60 tokens — all OK
    48 tokens — hangs, from every grid tried

So it is one **attention sequence length**, not a shape, a ratio or an axis.

**And it is not the bug the tracker names.** `blockfor(n, other, minl = 64)`
returns 1 — no blocking — whenever `n != other` *or* `n < minl`. At 48 tokens the
second guard fires regardless of squareness, so every one of these shapes takes
the **plain, unblocked** path. The documented workaround is not even engaged.
`STATUS.md`'s "`blockfor` refuses non-square attention … blocking the decoder's
lopsided shapes reproducibly hangs" describes the *blocked* path; this hangs on
the other one.

Note also that 64 and 80 tokens pass **while blocked** (`n == other`, `n >= 64`),
so the blocked path is fine at the lengths that reach it here.

That makes this a **distinct, undocumented bug**: plain attention at sequence
length 48 wedges `vkWaitSemaphores`, reproducibly, from four different input
shapes, with 44 and 49 clean either side.

### The 2D sweep that led there (kept — it is how the length was found)

**Every immediate neighbour of the hanging point passes.** Twelve shapes tested;
holding the grid at 6 and stepping the other axis gives 6x6, 6x7, **6x8 HANGS**,
6x9, 6x10 — the hang is a single value with clean neighbours on both sides. Same
on the other axis: 5x8, **6x8 HANGS**, 7x8, 8x8, 10x8.

It is an isolated point, not a region or a ratio. Whatever decides this is
picking something different for exactly one shape.

Sweeping W against H = 128 (grid height 8): 6 hangs, 7 passes, 8 passes, 10
passes — so it is not a magnitude threshold. Sweeping H against W = 96 (grid
width 6): 6 passes, 8 hangs, 10 passes — so neither axis is broken on its own.
**Only the {6, 8} combination reproduces, in either orientation**, with its
immediate neighbours on both axes clean.

What is solid is the reproducer and the counter-examples: `runmatanyone` at
128 x 96 wedges `vkWaitSemaphores` every time, at 160 x 96 it does not, and the
difference is one dimension. Whoever fixes this should start from that pair
rather than from any predicate — including the tracker's "non-square", which the
table falsifies.

`MatAnyoneRunner` and `SAM2Runner` hit it because their test fixtures are
128 x 96. The three small ports never do: two have no attention at all, and Depth
Anything's is square by construction (518 x 518).

**So the paragraph below overstates the case, and I am leaving it visible rather
than deleting it.** The soak's 2 226 980 clean trials remain valid evidence about
the *flush hang*, which is what it was built to provoke. What is true is narrower:
this project ran a soak for two days and the hang it eventually hit was the other
open bug, reachable by one command the whole time.

### What I first wrote, and why it was too strong

`STATUS.md` lists the flush hang as "dominant path fixed, one recurrence after
~90 clean trials", and throwing trials at it was this project's standing job. It
has now thrown **2 226 980** across two days without a single hang — and a plain
`Pkg.test` wedges the queue deterministically.

The honest conclusion is that **the soak does not exercise the path that hangs.**
`soak_flush_hang.jl` drives buffer lifetime — allocate, drop mid-recording,
collect, flush — which is the mechanism the *already-fixed* path had, and it
synchronises every trial, so it never accumulates the 50 in-flight batches this
wedges on. 2.2 million repetitions of the wrong shape. The number should not be
read as evidence the bug is gone; it is evidence that one known reproduction no
longer fires, which was already known.

### Which bug, and where it goes

Attributed above, by the aspect-ratio A/B. I had written that this machine could
not separate the two candidates; it can, and the experiment was one script.

Nor can I say whether it predates today: **no runner package has ever declared a
test target**, so there is no "it used to pass" to compare against. It may be
long-standing and simply unobserved.

A deterministic one-command reproducer is worth more than any number of clean
trials, so this goes to `lava-core` with the timeline dump rather than being
chased here.
