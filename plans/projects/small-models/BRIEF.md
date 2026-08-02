# small-models

**Machine** NVIDIA laptop (RTX 3070, 8 GB) · **Repo** `dev/JuliaVision` @ `sd/small-models`
**Read first** `plans/models-to-port.md` for each model's section, then `plans/GUARDRAILS.md`.

Three ports that need **no new runtime ops**, so they are bring-up rather than
kernel work. That is what makes them right for this machine: the CPU is weak and
this stack is compile-bound, so send it runs, not iterative kernel development.

Order is the review's own, cheapest first.

## 1. Neural 3D LUT (`NeuralLUTRunner`)

Days, not weeks, and immediately visible in the editor. Nothing new for the
runtime; the op is a trilinear LUT apply. Target **< 2 ms at 4K**.

One snag recorded in `models-to-port.md`: it is the only model whose weights are
*not* at a fetchable URL — they live in the upstream checkout, so
`tools/models.py fetch` does not cover it. Sort that first or the rest is blocked.

## 2. RIFE 4.26 (`RIFERunner`)

Frame interpolation. Cheap because the warp is `grid_sampler_2d`, which is
already implemented. Blocked only on mirroring the weights. Target **1080p at
60 fps**.

## 3. Depth Anything V2 Small (`DepthAnythingRunner`)

**No new ops at all** — pure editor value (depth-keyed grading, depth-aware
effects). Target **≥ PyTorch**.

## The pattern all three follow

`tools/export_whisper.py` is the working template: load the module, name the
inputs, **export from CUDA**, hand the `ExportedProgram` to
`export_graphs.convert`. Then load the graph from Julia, run it on Lava, diff
against PyTorch.

**Export from CUDA is not optional** — a CPU export decomposes attention and the
exporter now refuses one. This machine has CUDA, so that is fine; the AMD laptop
could not do this work at all.

Watch VRAM: 8 GB. These three are small, but check before assuming — Whisper's
fp32 encoder is 2.55 GB and the big models will not fit here.

## Two targets, on purpose

**vs PyTorch** is the engine goal. **The editor budget** is what decides whether
the port was worth doing — a depth model at 3x PyTorch that still takes 400 ms a
frame does not buy a depth-keyed grade.

## Numbers

Measure the editor budget here (is it usable?), but **do not put engine
comparisons in `perf-plan.md`** — that file's numbers are all desktop
measurements and cross-machine numbers are not comparable
(`GUARDRAILS.md` §6). Report a verdict; the desktop confirms a number if one is
needed.

## Also on this machine

The **flush-hang soak** (`STATUS.md`, open bugs): the dominant path is fixed but
one recurrence followed ~90 clean trials, so it needs trials rather than
attention. Start it and let it run in the background across everything above.
Report the trial count and any recurrence with its full log.

## Report

Append to `REPORT.md`: per model, whether it exports, whether it runs, the diff
against PyTorch, and the editor-budget timing with the caveat that it is this
machine's number.
