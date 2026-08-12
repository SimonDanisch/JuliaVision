# JuliaVision

GPU vision on [Lava](https://github.com/SimonDanisch/Lava.jl): a graph runtime
for exported PyTorch models, the image kernels around them, and one package per
model whose job is to have no cold start.

| package | what it is |
|---|---|
| `DNNKernels` | ATen graph runtime — loads a `torch.export` graph, plans its memory, runs it on Lava (see [What DNNKernels is](#what-dnnkernels-is)) |
| `GPUFiltering` | image kernels: colour, blur, warp, optical flow, patch tracking |
| `SAM2Runner` | SAM 2.1, precompiled |
| `MatAnyoneRunner` | MatAnyone2 video matting, precompiled |

Scaffolded, not ported — one editor feature each, see `models-to-port.md`:

| package | feature | licence |
|---|---|---|
| `WhisperRunner` | speech → text | MIT |
| `DeepFilterRunner` | voice denoising | MIT / Apache-2.0 |
| `DemucsRunner` | stem separation | MIT |
| `KokoroRunner` | text → speech | Apache-2.0 |
| `NeuralLUTRunner` | style / mood grading | Apache-2.0 |
| `RIFERunner` | frame interpolation | MIT |
| `DepthAnythingRunner` | monocular depth | Apache-2.0 |
| `BasicVSRRunner` | video upscaling | Apache-2.0 |
| `ProPainterRunner` | object removal | **S-Lab 1.0, non-commercial** |

Each of those loads and precompiles with no assets installed, so they cost
nothing until their port lands. Only the two working runners are dev'd into the
editor's own environment; the rest live here alone until they run.

A monorepo because these change together: an op added to `DNNKernels` is usually
a model that needed it, and a kernel frozen by one model is a cache hit for the
rest.

## What DNNKernels is

**A Lava kernel library that uses KernelAbstractions as its kernel-authoring
syntax, not as a portability layer.** This is decision (a) of
`plans/kernel-library-review.md` finding 8, settled 2026-08-02.

The package `using`s KernelAbstractions and writes `@kernel` bodies, which reads
as backend-portable. It is not, and the deciding evidence is not the raw count of
`Lava.*` references (181 occurrences, 33 distinct symbols) but *where* they sit:

- The fast paths gate on `A isa Lava.LavaArray{Float16,2}`, a type assertion on a
  Lava-specific array type, so no other KernelAbstractions backend can reach them
  by construction. That is not "a portable library with a Lava fast path"; the
  fast path exists only on Lava.
- `Lava.GEMM_TILE` is used as the literal 16 in 75 places, including pure shape
  arithmetic, and `Lava.splitidx` / `Lava.FastDiv32` / `Lava.cart32` /
  `Lava.staticgroup` appear inside the *generic* elementwise launcher.
- Five `@kernel cpu=false` sites (`layernorm_kernel!`, `attn_softmax_rows!`,
  `attn_flash!`, `attn_flash_cm!`, and the generated `toLE_tiled_*` layout
  kernels in `attention.jl`) have no CPU form at all, in a package whose
  verification story depends on running the same source on the CPU.
- `Lava.VK_CONTEXT_REF`, `Lava.capture` / `replay!` and `Lava.with_dispatch_timing`
  are Vulkan runtime concepts with no KernelAbstractions analogue at all.

Some of that is unavoidable: cooperative-matrix intrinsics have no KA equivalent,
and that is a legitimate reason for a backend-specific kernel. Some is
incidental. Either way the facade was promising something the package does not
deliver.

**Saying (a) plainly costs nothing real, because the portability that matters
here is a different axis and DNNKernels genuinely has it.** KA would buy
portability across *Julia GPU backends* (CUDA, ROCm, oneAPI, CPU). Lava buys
portability across *Vulkan devices* from one SPIR-V module. The second is the one
in use, and it is verified rather than assumed: the same kernel sources run on an
RTX 4000 Ada (subgroup 32, `VK_NV_cooperative_matrix2`) and on a Radeon 8060S
(RDNA 3.5, subgroup 64, KHR cooperative matrix only, no coopmat2 at all), with
the tile size and cooperative-matrix availability queried per device. See
`plans/projects/portability/REPORT.md`.

CPU execution through KA remains available for the kernels that have a CPU form,
and it is a verification tool (`verify.jl` compares against it), not a supported
deployment backend.

## Why the `*Runner` packages exist

They contain almost no code. Each one owns a `@compile_workload` that runs its
model for real during precompilation, so both halves of the cold start —
Julia's inference and the SPIR-V compile — are paid once at `Pkg.precompile`
instead of by whoever clicks first. Measured on SAM 2.1: 48.2 s to 3.3 s.

The split is not tidiness. A workload has to live *downstream* of everything it
uses, because loading a dependent package invalidates the caller's precompiled
code; putting the workload inside `DNNKernels` would make it depend on the
weights and cache nothing useful.

## Layout

Packages are plain subdirectories, [Makie](https://github.com/MakieOrg/Makie.jl)-style.
History for `DNNKernels` and `GPUFiltering` was carried over with `git subtree`,
so `git log --follow` still works through the move.

## Speed, against PyTorch on the same card

Measured 2026-08-05 on an **NVIDIA RTX 4000 Ada**, all in one sitting. Ours
through `tools/bench_all.jl` (SAM 2's decode through `tools/bench_sam2.jl`),
PyTorch through `tools/baseline_*.py`. `±` is the sample spread the harness
reports; a row is only worth quoting when it is small.

| model | shape | Lava | PyTorch | ratio |
|---|---|---:|---:|---:|
| SAM 2.1 encode | 1024² | **102.9 ms** ±0.7% | 79.3 ms | 0.77x |
| SAM 2.1 decode | one click | **6.0 ms** | 1.76 ms | 0.29x |
| Whisper large-v3-turbo encode | 30 s window, fp16 | **120.5 ms** (see GC note) | 71.9 ms | 0.60x |
| Kokoro-82M | one sentence | **~390 ms** core (see GC note) | — | — |
| RIFE 4.26 | 1920×1152 | *213.6 ms* ±81% | — | *spread too wide to quote* |
| Depth Anything V2 S | 518² | *51.7 ms* ±292% | 21.9 ms | *spread too wide to quote* |
| MatAnyone2 step | 512×288 | *48.8 ms* ±322%, clock 11% | — | *not trustworthy* |
| Neural 3D LUT | 256² classifier | *no sample survived the clock gate* | — | — |

**An earlier version of this table was inflated by a bug in Lava, and by a lot.**
Every number in it was taken while `submit!` scanned the whole argument slab on
every submit — a debug facility that was on because its field was typed `Any` and
defaulted to `nothing`, against a `!= UInt64(0)` guard (Lava `963f633`). That is
~41 µs of host CPU **per dispatch**, so it inflated every model in proportion to
how many dispatches it issues, and it was silent. What it cost, old → new:
SAM 2 encode 143.6 → 102.9, decode 25.2 → 6.0, Whisper 457.7 → 394.6, Kokoro
645.4 → 535.1, RIFE 274.7 → 213.6, Depth Anything 92.4 → 51.7. If you are holding
a copy of the old table, none of its rows were right.

Neural 3D LUT is not broken — at ~7 ms it never spins this card past the clock
plateau while ten desktop processes share it, so every sample is discarded. It
needs a longer-running shape or a quiet machine, not a fix.

**Whisper's row was 394.6 ms until 2026-08-06, and the 3.2x came off one flag.**
The audio context is 1500 frames; the flash kernel's extent gate is
`clamp || (Lq % BR == 0 && Lk % BC == 0)`; and 1500 = 2²·3·5³ divides no block
any tiling uses. So all 32 attentions declined *both* flash and the
cooperative-matrix SDPA and ran the three-pass fallback — ~75% of the model,
priced by doubling the op family and differencing. `clampattn` exists for exactly
this case and the encoder never set it. It is not an accuracy trade: against the
PyTorch reference in Float64 the clamped path is slightly *more* accurate than
the fallback it replaces. This is the same alignment cliff `Lava.gemm_padn`
already fixed for the GEMM's N=1500, one kernel over — worth checking wherever a
sequence length is not a round number.

**Kokoro's row was 535.1 ms and that number was mostly garbage collection.**
Forty consecutive runs: `min 387.1 · p25 390.1 · median 398.9 · p75 488.3 · p90
501.0 · max 1824.7 ms`. GC is **18.9% of wall**, with single pauses up to 316 ms
on a ~390 ms call. `bench_all` takes 11 samples, and with p75 already at 488 an
11-sample median lands anywhere between 400 and 600 — it read 535.1 once and
611.7 another time, from the same code. **A model whose GC is a fifth of its wall
time cannot be measured by a short median.** The core is ~390 ms; chasing the
611.7 as a regression cost an hour and found nothing.

**Whisper's spread is Julia's GC, and it is worth knowing before you quote any
row.** Sixty consecutive encodes: `min 119.94 · p25 120.31 · median 120.54 · p75
126.40 · p90 168.93 · max 286.03 ms`. The core is **±0.5%** — the tail is entirely
garbage collection. Fifteen of the sixty ran slow, and those carry **68.9 ms of GC
against 2.96 ms** for the fast ones, at an identical 2265 MHz clock, with
device-side allocation flat at 3.85 MB. GC is 5% of wall on a quiet machine and
17% with another Julia process running.

So a spread quoted as "±42%" here is not the GPU being erratic; it is host pauses
of up to 73 ms landing inside a 120 ms call. That also means **the median is
sensitive to how many pauses a short sample happens to catch** — an earlier
15-sample run of this same build read 124.5 ms for that reason alone.

**How to read this, and how not to.** These are honest numbers and mostly not
flattering ones — the engine is between 1.3x and 3.4x off PyTorch wherever both
sides are measured, and worst on SAM 2's decode. That is the gap
`plans/perf-plan.md` argues about.

Every rule below exists because breaking it produced a confidently wrong number
here at least once:

  * **Warm the clock.** This card idles at **210 MHz of 3105** and plateaus near
    73% under sustained load. A cold sample reads several times slow and looks
    exactly like a regression. `tools/measure.jl` warms to a measured plateau,
    brackets every sample with a clock reading and discards any that dipped.
  * **Report what was discarded, and the spread.** Neural 3D LUT kept **0 of 11**
    and is quoted as nothing at all rather than as a number. MatAnyone kept 11 but
    at **11% of clock with ±322% spread**, which is a median of noise; Depth
    Anything and RIFE are the same story more mildly. Only SAM 2's encode (±0.7%)
    is tight enough to argue about.
  * **A silent debug switch will not show up as a failure.** The table above was
    wrong for three days because a scan nobody asked for ran on every submit and
    logged nothing when it found nothing. Nothing failed; everything was slower.
    Prefer a guard that can only be satisfied by the type it compares against.
  * **TF32 off on the PyTorch side** for fp32 models. Leaving it on hands
    PyTorch the tensor cores for every matmul while Lava runs true fp32, which
    measures a dtype choice and calls it an engine gap.
  * **No `torch.compile`.** The graphs here came out of `torch.export`, which is
    eager PyTorch's own decomposition. Comparing against a fused Inductor build
    would compare a compiler we do not have against a runtime we do.
  * **Never rank against an unmeasured denominator.** SAM 2's "85% of PyTorch"
    was computed against 87.6 ms; PyTorch measures **79.3 ms** on this card now.
    The denominator moved and the claim went stale on its own.

**These are still pessimistic, but less than the previous version claimed.** The
card plateaus at **69-73% of its 3105 MHz** with ten desktop processes on it
holding ~3.6 GB, so a quiet machine would move the Lava column down further.

An earlier revision of this section blamed exactly that contention for SAM 2
encode reading 143.6 ms where `perf-plan.md` records 100.4, and offered a
pre-merge control (151.3 ms) as proof that the code was innocent. **The control
was sound and the conclusion was wrong.** Both of its rows were taken with the
slab scan running, so it compared one bugged tree against another and concluded
"not the code" from two equally bugged numbers. It was the code. With the scan
gone, encode reads **102.9 ms ±0.7%** at the same 72% clock — the record, on the
same contended machine, with no clock-scaling argument needed.

The lesson is not "measure more carefully"; it is that **a control between two
builds only exonerates the code if the defect postdates both of them.** Reach
past the suspected change, not just before your own commits.

MatAnyone still needs a quiet machine: ±322% spread at 11% clock is noise, not a
measurement, and that one is unchanged.

Blanks are PyTorch baselines not yet written, not models that failed.

## Not here yet

The nine packages above are skeletons: asset lookup, graph loading and a guarded
workload, with the workload body and the missing ops still to write.
`BasicVSRRunner` is the furthest along — its graph is already exported to
`gen/graphs/basicvsrpp-fp32` by `tools/export_basicvsrpp.py`, so it is the only
one whose `ready()` is already true.

The name is now wrong: two of these are audio models and `JuliaVision` is not
where a speech recogniser belongs. Renaming is a separate job from porting.

## Getting started

```julia
using Pkg
Pkg.develop(url = "https://github.com/SimonDanisch/JuliaVision")   # or clone + activate
using SAM2Runner

mask = segment(img, [(0.5, 0.5)])                       # one click, normalized
mask = segment(img, [(0.3, 0.4), (0.8, 0.9, false)])    # include, and exclude
```

The weights download on first use — `sam2-large` is 943 MB — and are shared
between every environment on the machine. Nothing is fetched at install time.

`Lava` is not in the General registry, so `[sources]` in each `Project.toml`
says where to find it; `Pkg.instantiate` handles the rest.

## Assets

Graphs and weights are Julia artifacts, not files in this repository:

| artifact | size | what for |
|---|---|---|
| `sam2-large` | 943 MB | SAM 2.1 graphs + weights — needed to run the model |
| `sam2-large-refs` | 1.2 GB | PyTorch reference activations, **tests only** |
| `matanyone` | 142 MB | MatAnyone2 graphs + weights |

The refs are separate on purpose: they are what the layer-by-layer verification
compares against, and nobody segmenting a picture should download them.

**There is no `DNNKernels.assetpath`, and this paragraph used to claim there
was.** It described an env-var → local-`gen/`-tree → artifact resolution order
that was deliberately DELETED (see the docstring at the top of
`DNNKernels/src/assets.jl`): nothing ever set the environment variables, and the
generated-tree branch made a broken download invisible, because on any machine
with a `gen/` tree the fallback always answered. Tests had the same fallback and
it meant the PyTorch parity gate ran only on the machine that produced the export.

Each package writes `assetdir() = @artifact_str("<name>")` and reads its assets
from that artifact and nowhere else. Changing a model's assets means re-binding
its artifact: re-export, then `tools/make_artifacts.jl <name>` hashes the new
tree and rewrites the `Artifacts.toml`; `tools/publish_artifacts.jl` uploads it
so it reaches anyone else.
