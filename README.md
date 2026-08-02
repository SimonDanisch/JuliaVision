# JuliaVision

GPU vision on [Lava](https://github.com/SimonDanisch/Lava.jl): a graph runtime
for exported PyTorch models, the image kernels around them, and one package per
model whose job is to have no cold start.

| package | what it is |
|---|---|
| `DNNKernels` | ATen graph runtime — loads a `torch.export` graph, plans its memory, runs it on any KernelAbstractions backend |
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

`DNNKernels.assetpath` resolves an environment variable, then a locally
generated tree found by walking up from the package, then the artifact — in that
order, so a checkout that *produces* these files with `tools/export_sam2.py`
keeps using its own output and a published copy never lands on top of work in
progress. `tools/publish_artifacts.jl` (in the project that generates them) is
what packages, uploads and binds a new set.
