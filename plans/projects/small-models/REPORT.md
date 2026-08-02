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
is ~10 ms — so grading a shot at 60 fps costs 0.79 ms/frame and only re-prediction
is expensive. That is the right shape for the feature: predict per shot or per
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
