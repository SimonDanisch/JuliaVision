"""
Image-Adaptive 3D LUT on Lava: does the predicted LUT agree with PyTorch?

    julia --project=. tools/verify_neurallut.jl [gpu|cpu]

The graph's output is the blended 33³ LUT, not an image — the trilinear apply is
a `GPUFiltering` kernel and is checked by `tools/bench_neurallut.jl` instead.
That split is the whole point of the port: what the network produces is a
grading object, and this file asks only whether that object is right.

The reference is `reference.safetensors`, written by `export_neurallut.py` from
the same process that exported the graph, so a mismatch here cannot be a
different checkpoint or a different input.

Two numbers, because they fail differently:

  * **the LUT** — max absolute difference over all 3x33x33x33 entries. This is
    what gets applied, so it is the one that matters.
  * **the blend weights** — the classifier's three outputs, recovered from the
    LUT by least squares against the basis tables. The LUT is linear in them, so
    a weight that is off says the CNN is wrong, while right weights with a wrong
    LUT say the blend or a basis table is wrong. Reported separately so the next
    person does not have to bisect that.

No `permutedims` anywhere on purpose: `readsafetensors` reshapes the raw
row-major buffer with reversed dims, which *is* the Julia column-major view of
the same bytes. A torch `(3, 33, 33, 33)` is a Julia `(33, 33, 33, 3)` already,
and permuting it here would move the data twice and compare the wrong axes.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics, Lava
using DNNKernels: loadgraph, execute!, readsafetensors, toback
const KA = KernelAbstractions

# `tools/` is symlinked from the workspace root, and `@__DIR__` does not resolve
# the symlink, so one level up is the root. Same convention as `bench_sam2.jl`.
const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "neurallut"))

mode = isempty(ARGS) ? "gpu" : lowercase(ARGS[1])
backend = mode == "gpu" ? LavaBackend() : KA.CPU()

isdir(DIR) || error("no export at $DIR — run `uv run tools/export_neurallut.py`")

graph = loadgraph(joinpath(DIR, "neurallut.json"))
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
ref = readsafetensors(joinpath(DIR, "reference.safetensors"))

println("graph: $(length(graph.ops)) ops, $(length(graph.buffers)) buffers  [$mode]")

img = toback(backend, ref["input"])
w = Dict{String,Any}(k => toback(backend, v) for (k, v) in weights)

# Empty NamedTuple: every dim in this graph is static, so there is nothing to
# bind. `Ctx` takes a NamedTuple, not a Dict.
out = execute!(graph, Dict{String,Any}("img" => img), w; dims = (;), backend)
lut = Array(out["sum_1"])                # Julia (33, 33, 33, 3)

want = ref["lut"]
size(lut) == size(want) || error("shape $(size(lut)) vs reference $(size(want))")
Δ = maximum(abs.(lut .- want))
@printf("LUT      max|Δ| = %.3e   (ref range %.4f .. %.4f)\n",
        Δ, minimum(want), maximum(want))

# The blend weights are not a graph output, but the LUT is linear in them and
# the basis tables are weights we hold, so they come back by least squares
# without re-exporting anything. `luts` is torch (3 basis, 3, D, D, D), so in
# Julia the basis index is the last dim and `reshape` already columnises it.
basis = weights["luts"]
A = reshape(basis, :, size(basis, ndims(basis)))
got_pred = A \ vec(lut)
want_pred = ref["pred"]
@printf("weights  got  %s\n         want %s\n         max|Δ| = %.3e\n",
        string(round.(got_pred; digits = 6)), string(round.(want_pred; digits = 6)),
        maximum(abs.(got_pred .- want_pred)))

# fp32 accumulation over a 6-conv CNN plus four instance norms; 1e-4 is loose
# enough not to be flaky and tight enough that a wrong kernel cannot pass.
ok = Δ < 1e-4
println(ok ? "PARITY OK" : "PARITY FAILED")
exit(ok ? 0 : 1)
