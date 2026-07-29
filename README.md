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

`BasicVSRRunner` — the graphs are exported (`gen/graphs/basicvsrpp-fp32`) and
`tools/export_basicvsrpp.py` produces them, but the runner package is not
written.
