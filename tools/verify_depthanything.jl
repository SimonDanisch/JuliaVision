"""
Depth Anything V2 Small on Lava: does the depth map agree with PyTorch?

    julia --project=. tools/verify_depthanything.jl [gpu|cpu]

The reference is `reference.safetensors`, written by `export_depthanything.py`
from the same process that exported the graph, so a mismatch here cannot be a
different checkpoint or a different input.

**Relative error, not absolute.** This model's output is inverse relative depth
with no fixed scale — the reference lands around 1.3 .. 3.0, and a 1e-3 absolute
difference means something different at the near plane than at the far one. The
verdict is on the difference relative to the reference's own range, which is what
a depth-keyed grade actually cares about: a mask cut at 50% moves if the *shape*
of the map moves, not if everything shifts by a constant.

**Attention is decomposed here, unlike Whisper.** DINOv2 falls back to a manual
`bmm` + `softmax` attention when xFormers is absent, so this graph carries 24
`bmm` and 12 `_softmax` rather than a fused `_scaled_dot_product_*` — and the
export device does not change that. It means the tolerance sits on a longer
accumulation chain than the op count alone suggests.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics, Lava
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  planslab, fusableset, Workspace, Ctx, value
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "depthanything"))

mode = isempty(ARGS) ? "gpu" : lowercase(ARGS[1])
backend = mode == "gpu" ? LavaBackend() : KA.CPU()

isdir(DIR) || error("no export at $DIR — run `uv run tools/export_depthanything.py`")

graph = loadgraph(joinpath(DIR, "depthanything.json"))
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
ref = readsafetensors(joinpath(DIR, "reference.safetensors"))

println("graph: $(length(graph.ops)) ops, $(length(graph.buffers)) buffers  [$mode]")

img = toback(backend, ref["input"])
w = Dict{String,Any}(k => toback(backend, v) for (k, v) in weights)

plan = planslab(graph, (;))
slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
@printf("slab: %.1f MiB planned\n", plan.bytes / 2^20)

out = execute!(graph, Dict{String,Any}("x" => img), w; dims = (;), backend,
               slab, plan, ws = Workspace(backend), lazy = fusableset(graph))
# The graph's output is `unsqueeze`, a *view* — `Depth.forward` ends in
# `unsqueeze(1)` — so it has no entry of its own in the value table. `value`
# resolves views against the buffer they are of, which is what `wan.jl`'s
# `rungraph` does for the same reason.
got = Array(value(Ctx(out, graph, (;), backend), only(graph.outputs)))

want = ref["depth"]
size(got) == size(want) || error("shape $(size(got)) vs reference $(size(want))")

lo, hi = extrema(want)
span = hi - lo
d = abs.(got .- want)
Δ = maximum(d)
μ = mean(d)

@printf("depth    %d x %d, reference range %.4f .. %.4f\n", size(got, 1), size(got, 2), lo, hi)
@printf("         max|Δ|  = %.3e   (%.4f%% of range)\n", Δ, 100Δ / span)
@printf("         mean|Δ| = %.3e   (%.4f%% of range)\n", μ, 100μ / span)

# 0.1% of the map's own range. A depth-keyed grade or a fake defocus samples this
# with a soft threshold, so what matters is that the surface keeps its shape;
# tighter than this measures fp32 accumulation order over 24 `bmm`, not
# correctness.
ok = Δ < 1e-3 * span
println(ok ? "PARITY OK" : "PARITY FAILED")
exit(ok ? 0 : 1)
