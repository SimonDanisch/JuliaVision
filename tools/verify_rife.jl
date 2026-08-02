"""
RIFE 4.26 on Lava: does the interpolated frame agree with PyTorch?

    julia --project=. tools/verify_rife.jl [gpu|cpu]

The reference is `reference.safetensors`, written by `export_rife.py` from the
same process that exported the graph, so a mismatch here cannot be a different
checkpoint or a different input.

**The reference frames are structured, not noise.** RIFE estimates motion, and
between two random fields there is none to estimate — a warp that ignored its
flow entirely would still land close, because every candidate source pixel is
equally wrong. The pair the exporter saves is a moving pattern with a known
offset, so the flow has something to find and a broken `grid_sampler_2d` shows up
as a large difference rather than a small one.

Three numbers, because they fail differently:

  * **max|Δ|** — the worst pixel. Sensitive to a single misplaced sample, which
    is exactly what a half-pixel error in the warp grid looks like.
  * **mean|Δ|** — the bulk. A large mean with a small max means something
    systematic (a normalisation, a padding mode); a large max with a small mean
    means a few pixels, usually at a border.
  * **the fraction of pixels over 1e-3** — how much of the frame is involved.

No `permutedims` anywhere on purpose: `readsafetensors` reshapes the raw
row-major buffer with reversed dims, which *is* the Julia column-major view of
the same bytes. A torch `(1, 3, H, W)` is a Julia `(W, H, 3, 1)` already.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics, Lava
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  planslab, fusableset, Workspace
const KA = KernelAbstractions

# `tools/` is symlinked from the workspace root, and `@__DIR__` does not resolve
# the symlink, so one level up is the root. Same convention as `bench_sam2.jl`.
const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "rife"))

mode = isempty(ARGS) ? "gpu" : lowercase(ARGS[1])
backend = mode == "gpu" ? LavaBackend() : KA.CPU()

isdir(DIR) || error("no export at $DIR — run `uv run tools/export_rife.py`")

graph = loadgraph(joinpath(DIR, "rife.json"))
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
ref = readsafetensors(joinpath(DIR, "reference.safetensors"))

println("graph: $(length(graph.ops)) ops, $(length(graph.buffers)) buffers  [$mode]")

imgs = toback(backend, ref["imgs"])
tstep = toback(backend, ref["timestep"])
w = Dict{String,Any}(k => toback(backend, v) for (k, v) in weights)

inputs = Dict{String,Any}("imgs" => imgs, "timestep" => tstep)

# The slab is not an optimisation here, it is what makes the graph run at all.
# Unplanned, every one of the 645 buffers stays live for the whole graph and
# 1920x1152 fp32 intermediates sum to more than this card has — the first attempt
# died at "4973 MiB live across 18 buffers" with 5135 MiB of budget. `planslab`
# places them by lifetime so the transients reuse each other's bytes.
plan = planslab(graph, (;))
slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
@printf("slab: %.1f MiB planned\n", plan.bytes / 2^20)

out = execute!(graph, inputs, w; dims = (;), backend,
               slab, plan, ws = Workspace(backend), lazy = fusableset(graph))
got = Array(out[only(graph.outputs)])

want = ref["out"]
size(got) == size(want) || error("shape $(size(got)) vs reference $(size(want))")

d = abs.(got .- want)
Δ = maximum(d)
μ = mean(d)
frac = count(>(1.0f-3), d) / length(d)

@printf("frame    %d x %d, timestep %.3f\n", size(got, 1), size(got, 2), Array(tstep)[1])
@printf("         max|Δ|  = %.3e\n", Δ)
@printf("         mean|Δ| = %.3e\n", μ)
@printf("         %.4f%% of samples over 1e-3   (ref range %.4f .. %.4f)\n",
        100 * frac, minimum(want), maximum(want))

# 1e-3 on a 0..1 frame is a quarter of an 8-bit code value, and this graph is 63
# convolutions and 18 warps deep in fp32 — tighter than that measures the
# accumulation order, not correctness. The mean is the one that would move if a
# kernel were actually wrong.
ok = Δ < 1e-3
println(ok ? "PARITY OK" : "PARITY FAILED")
exit(ok ? 0 : 1)
