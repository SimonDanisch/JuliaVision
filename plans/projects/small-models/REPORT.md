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
| Neural 3D LUT | yes, 26 ops | yes | **3.6e-7** on the LUT | apply **0.88 ms** at 4K | < 2 ms — **met** |
| RIFE 4.26 | yes, 366 ops | yes | **3.3e-4** max, 4.9e-7 mean | **~500 ms** at 1080p | 16.67 ms — **missed, ~30x** |
| Depth Anything V2 S | yes, 311 ops | yes | **4.2e-5** (0.0025% of range) | **421 ms** at 518² | ≥ PyTorch (27.6 ms) — **missed, ~15x** |

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

- Depth Anything **763.56 → 421 ms**, 311 → 290 ops. Nearly 2x.
- The neural LUT classifier **14.08 → ~8 ms**, almost all of it from passing a
  planned slab rather than letting every intermediate allocate.

Both benchmarks originally did the wrong thing and both numbers were wrong in the
pessimistic direction. All three `tools/bench_*.jl` now build a `Model` and a
planned slab.

**RIFE is the exception, and it goes the wrong way.** Through `Model` it is
478/500/516 ms across three runs; through plain `loadgraph` + a slab it was
327 ms. The preparation passes appear to **cost** this graph ~50%, which is not
what they do to the other two. Not chased further — it does not change the
verdict (30x either way) and the desktop is the machine that should confirm it.
Flagged rather than explained.

### Why RIFE and Depth Anything miss

Measured, not guessed (`GUARDRAILS.md` §5 — no unmeasured denominators).

RIFE's 63 convolutions are **149.0 GFLOP** per interpolated frame at 1080p, of
which one transposed convolution is 47.9 GF. At the measured ~500 ms that is
**~0.3 TFLOP/s**, a low single-digit percentage of this card's fp32 peak. Even a
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
4K is **0.88 ms**, the 4K→256 reduce is **0.032 ms**, and predicting a new look
is ~8 ms — so grading a shot at 60 fps costs 0.88 ms/frame and only re-prediction
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

### What is not done

- **`RIFERunner` and `DepthAnythingRunner` are still scaffolds.** Both models
  export, run and match through `tools/verify_*.jl`, but neither package has a
  model struct or a workload yet — only `NeuralLUTRunner` does. Whoever picks
  these up should copy `NeuralLUTRunner`'s shape; the missing piece for RIFE is a
  pad-to-128 planar input kernel (the frame is padded, not resized, so
  `GPUFiltering.resizeplanar!` does not fit), and for Depth Anything it is the
  ImageNet normalisation plus the square resize.
- **No `Artifacts.toml` for any of the three.** RIFE's `train_log/` still needs
  re-hosting on the assets release before it can be one (`models-to-port.md`).
- **The conv findings are not fixed**, per the brief — these three are bring-up,
  not kernel work. They are handed to whoever owns `kernels-to-port.md` item 1.

### New in this branch

`GPUFiltering`: `lut3d.jl` (trilinear apply, matches upstream's own
`trilinear_kernel.cu` to 3.0e-7 with full coverage asserted) and `resizeplanar.jl`
(RGB frame → planar NCHW, `align_corners=false` to match `nn.Upsample`).
`DNNKernels`: the InstanceNorm op. `tools/`: three exporters, three verifiers,
three benchmarks, one PyTorch baseline, and the soak.
