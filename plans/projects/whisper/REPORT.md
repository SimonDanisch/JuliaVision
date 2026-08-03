# whisper — report

Append, newest last. One entry per working session: what was done, what was
measured, what was **dis**proved. Negative results in full — roughly half the
value in `plans/perf-plan.md` is knowing which things were tried and lost.

Numbers only from the machine this project is assigned to, and engine
comparisons only from the desktop (`plans/GUARDRAILS.md` §6).

---

## 2026-08-02 — the encoder runs and matches; fp16 exports but does not

Desktop, RTX 4000 Ada. Steps 1 and 2 of the BRIEF. No benchmark was run and no
timing is recorded anywhere — everything below is correctness.

### Step 1: the fp32 encoder matches PyTorch. Done.

**It was a bring-up, exactly as the BRIEF said.** All 617 ops ran on Lava on the
first attempt with no new kernel, no new op and no change to any kernel file.
The only code that had to change was the *verifier*, which had never been asked
to run on a device backend (below).

| check | result |
|---|---|
| ops covered | 617 ops, 11 ATen types, `coverage` reports **0 missing** |
| prefix, blocks 0–1, node by node | 44/46 within tolerance, worst accumulated max\|Δ\| 2.58e-5 |
| all 617 ops, node by node (104 recorded) | **all within tolerance**, worst accumulated max\|Δ\| 0.0155 at `add_64` |
| output, end to end through `Model` | max\|Δ\| **2.80e-3** on scale 9.45 → rel 2.96e-4; **rel rms 6.30e-5**; **cosine 0.9999999980** |
| output coverage | slab poisoned with `0xff`: **0 NaN** of 1 920 000, result bit-identical |
| determinism | two consecutive calls bit-identical |

The residual stream drifts by a *constant relative* amount rather than a growing
one, which is the signature of reassociation and not of a bug — per block:

```
blk  node        max|Δ|      scale      rel
  0  add         3.338e-06      4.618   7.23e-07
  8  add_16      2.983e-05      1.825   1.63e-05
 16  add_32      1.651e-05      1.516   1.09e-05
 24  add_48      0.01543        288.5   5.35e-05
 32  add_64      0.0155         289.7   5.35e-05
```

The absolute jump at block ~21 is **not** an error event: the residual stream
itself grows an outlier feature there, from scale ~2 to scale ~289. Relative
error is flat at 5.35e-5 across all 32 blocks. Worth knowing about because it is
what makes fp16 hard (below), and because anyone reading only `max|Δ|` will
misread it.

Resident cost of the fp32 encoder on the device: **2.373 GiB** of weights and a
**65.9 MB** scratch slab for the whole 617-op graph.

### Step 2: fp16 exports cleanly and halves the artifact. Its numerics are not acceptable yet.

`tools/export_whisper.py --precision fp16` now loads the checkpoint in its own
dtype instead of widening it. `gen/graphs/whisper-fp16`:

- **1.186 GiB against 2.373 GiB**, 637.0M parameters — an exact halving, as
  expected. (The 1.6 GB the BRIEF quotes is the *whole* model; the encoder's
  share of it is 1.27 GB decimal.)
- **681 ops, not 617**, over **12** ATen types, not 11. Two differences, both
  already implemented, so coverage is still 0 missing:
  - `clamp.default` ×32 — `WhisperEncoderLayer.forward` clamps the residual to
    `finfo(fp16).max - 1000`; it is dead code in fp32 and traced in fp16.
  - `_scaled_dot_product_flash_attention.default` ×32 **replaces**
    `_scaled_dot_product_efficient_attention.default`. PyTorch picks a different
    sdpa backend for fp16, so the export is not just the fp32 graph with narrower
    buffers.

And it does not match:

| | rel rms at the output | cosine |
|---|---|---|
| ours fp32 vs torch fp32 | 6.30e-5 | 0.9999999980 |
| **torch fp16** vs torch fp32 | **0.0254** | 0.9996778883 |
| **ours fp16** vs torch fp16 | **0.1368** | 0.9906393126 |

The denominator is the middle row and it is measured, not assumed
(GUARDRAILS §5): fp16 costs PyTorch itself 2.5% rel rms on this encoder. **We
cost 13.7%, i.e. 5.4x more than fp16 is inherently worth here.** So this is our
gap, not fp16's.

Localised to **block 21's MLP**, node by node against the fp16 reference
(`--blocks 20-22`):

```
add_41   (block 20 out)   rel rms 0.0057    scale   2.28   <- torch's own fp16 error here is 0.00085
ln_41                     rel rms 0.041     scale   3.65
gelu(fc1)                 rel rms 0.109     scale  22.45
addmm_104 (fc2)           rel rms 0.205     scale 288.8    <- the outlier feature is created here
add_42                    rel rms 0.168     scale 288.5
```

**No overflow anywhere**: the largest magnitude over every fp16 intermediate is
290.8 against fp16's 65504, and no intermediate holds a non-finite value. This
was the first hypothesis (288² = 82944 does overflow fp16, so a layer norm that
squared in fp16 would go infinite) and it is **disproved** — `DNNKernels`' layer
norm already reduces in fp32 (`kernels/layernorm.jl:58`).

We are ~6.7x worse than torch's fp16 *before* the outlier (0.0057 vs 0.00085 at
block 20) and the outlier then converts that into a large absolute error. So the
fix is not at block 21; block 21 is where an existing per-block gap becomes
visible.

### The bug behind most of it: Lava's scalar GEMM accumulates in the destination type

`dev/Lava/src/array/gemm.jl:863`, `strided_gemm_kernel!`:

```julia
T = eltype(C)
acc = zero(T)
```

With an fp16 destination that is an **fp16 accumulator over K = 1280 or 5120**.
`DNNKernels`' own scalar kernel gets this right — `mm2` uses `accum(eltype(A))`,
which is `Float32` for half operands — so the two implementations of the same
operation disagree.

Isolated on one op (`addmm_9`, block 1's fc2, K = 5120), identical fp16 operands,
against an fp64 reference computed over those same fp16 inputs:

| path | rel rms |
|---|---|
| fp16 destination → fp16 accumulator | **0.0483** |
| ...predicted by `sqrt(K)·eps16/2` | 0.0349 |
| same kernel, fp32 destination, then rounded to fp16 | **2.06e-4** |
| Lava's cooperative-matrix path (fp32 accumulator) | 2.13e-4 |
| PyTorch's own fp16 `addmm` | 3.96e-4 |

**234x, from one line.** The fix belongs to `lava-core` — accumulate in a type
widened for half operands (`Float32` when the operands are `Float16`) and convert
on the store, which is what the coopmat path and `mm2` already do. It is not
whisper-specific: every fp16 matmul in every model that misses the
cooperative-matrix path is affected.

**Where it bites, and why SAM 2 never saw it.** `mm_coopmat_applicable`
(`DNNKernels/src/kernels/extern/matmul.jl:87`) requires both operands to be bare
`LavaArray{Float16,2}`. In the **raw exported graph** every `addmm` weight
arrives as a permuted *view*, so the test fails and every fp16 matmul takes the
scalar path. In the **`Model` path** `hoistpermutes` has already materialised that
weight, the test passes, and the accurate path is taken. The two therefore
disagree by two orders of magnitude in fp16 — which is the next finding.

### `verifygraph` was not verifying the path that ships

Three things, all now fixed in `DNNKernels/src/verify.jl`:

1. **It was CPU-only.** `maxerr` walks both tensors element by element, which on
   a device array is one transfer per element. That is why SAM 2's encoder had to
   be checked by a separate end-to-end script instead. Added `tohost`, so each
   result is downloaded once and the layer-by-layer criterion is available on
   whichever backend actually runs the model. For Whisper's encoder — ~2.3 TFLOP
   for one 30 s window — that is the only backend there is.
2. **It never uploaded the weights.** `execute!` does not; `Model` was the only
   thing that did. On a device backend the first convolution got a host `Array`
   and failed to compile. Added `weightkeys`, which follows views to their parent
   (a transposed weight reaches a `mm` as a `:view`, so scanning `op.ins` for
   `:weight` finds none of a transformer's projections) and uploads only what the
   graph names.
3. **It only took a path.** Added a `Graph` method so a *prefix* can be checked on
   its own — two of Whisper's 32 identical blocks cover all eleven op types for a
   sixteenth of the arithmetic.

The deeper point, and the reason `tools/verify_whisper.jl` runs both paths:
**`verifygraph` runs the graph as exported, and `Model` runs it rewritten.**
`hoistpermutes`, `hoistcasts`, `hoistconstants` and `dropdead` take Whisper's
encoder from 617 ops to 583 (fp32) and 681 to 647 (fp16), and — as above — that
rewrite is what decides which GEMM runs. A graph that passes `verifygraph` has
not been verified.

### Two things found in passing, both someone else's

- **SAM 2's decoder does not verify, and has not for a while.**
  `verifygraph(sam2_decoder.json, …; dims=(res=1024,), backend=CPU())` stops at
  op 230, `_to_copy_95` (`_to_copy.default`), on a **shape** mismatch —
  `(128, 128, 64, 1)` against the reference. Checked against a stashed, pristine
  `verify.jl`: **identical failure**, so it is not a regression from the changes
  above. A `sam2_decoder.fp32.json.bak` sits next to `sam2_decoder.json` in
  `gen/graphs/sam2-large/`, so the likeliest cause is a graph that was swapped
  without regenerating `refs.safetensors`. Not investigated further — not this
  project's file.
- **Eight Julia tools still derive their root as `@__DIR__/".."`** —
  `bench_sam2.jl`, `demopage.jl`, `lavadnn_bench.jl`, `lavadnn_e2e_cpu.jl`,
  `memcheck.jl`, `publish_artifacts.jl`, `verify_sam2.jl`, `wan_generate.jl`.
  That is the same breakage `common.find_root()` fixed on the Python side
  (0753cf0): `tools/` now lives in the repo and `gen/` does not, so every one of
  those points at a `gen/` that does not exist. `tools/verify_whisper.jl` uses an
  absolute path to sidestep it rather than inventing a second convention.

### What the plan files got wrong

- `models-to-port.md` and `BRIEF.md`: "617 nodes over 11 distinct ATen ops" is
  true **of the fp32 export only**. The fp16 export is 681 nodes over 12.
- "fp32 encoder weights are 2.55 GB" — 2.373 GiB, i.e. 2.548 GB decimal, so the
  number is right in decimal units; fp16 is 1.186 GiB / 1.274 GB.
- "the encoder is 635M of 809M parameters" — measured **637.0M**.
- Everything else held. In particular the CUDA-export trap was already paid and
  did not recur, and the encoder genuinely needed no new ops.

### Operational notes for whoever is next

- `GC.gc()` does **not** return VRAM. `Lava.trim_gpu_pool!()` followed by
  `Lava.reclaim_empty_pool_blocks!(Lava.vk_context().default_bq)` does — 18 239 →
  6 211 MiB in one call after a verification run. Worth knowing on a card three
  projects share; a PyTorch export OOM'd against this session's pool.
- The node-by-node check keeps every intermediate (no slab), which is ~10 GB of
  VRAM for this graph against 65.9 MB planned. Use the prefix while iterating.
- `tools/dump_whisper_refs.py` synthesises its own 30 s speech-shaped audio from
  a fixed seed, so it needs none of the editor's media, and it saves
  `whisper/audio` and `whisper/mel` alongside the activations. **Step 3's
  reference already exists**: the mel front end has to reproduce `whisper/mel`
  from `whisper/audio`.
- The reference input is a real log-mel, not `randn`. A Gaussian input puts the
  whole residual stream in a range the model never sees and makes every tolerance
  meaningless.

### What the next step needs

1. **fp16 is blocked on the `lava-core` accumulator fix.** Re-measure after it
   lands; the isolated 234x says most of the gap goes with it, but that is a
   prediction and the end-to-end number has to be taken again.
2. **The third option has not been measured**: an `autocast` export — fp32
   masters on disk, which `hoistcasts` folds to fp16 weights while leaving the
   norms in fp32. Same resident bytes as fp16, better numerics, twice the
   artifact. `export_whisper.py --precision autocast` already produces it.
3. **fp32 is shippable now** and is what the mel front end and the decoder should
   be built against. Deciding fp16 is not on their critical path.
4. Nothing here says anything about speed. The encoder has never been timed and
   must not be until the GPU is free and `main` has been asked.

### Artifacts

- `tools/dump_whisper_refs.py` — new. Per-node PyTorch references, any precision,
  a leading block count or an inclusive block range (`--blocks 20-22`).
- `tools/verify_whisper.jl` — new. Prefix / all-ops / end-to-end, `gpu|cpu`,
  `fp32|fp16`. Poisons the scratch slab before the end-to-end call so a partial
  write reads as NaN rather than as a plausible number (GUARDRAILS §3).
- `tools/export_whisper.py` — `--precision fp16` now means what it says, and the
  output directory follows the precision.
- `DNNKernels/src/verify.jl` — `tohost`, `weightkeys`, weight upload, and the
  `Graph` method.
- `gen/graphs/whisper-fp16/` — the fp16 export and its references (scratch, not
  committed).
