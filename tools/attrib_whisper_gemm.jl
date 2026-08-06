# Which of Whisper's three GEMM shapes is the slow one?
#
# The family attribution puts `addmm` at 64.78 ms of a 124.66 ms encode and
# 1.73 TFLOP, i.e. 26.7 TF/s — against the 37.6 TF/s (35% of a measured 107.3
# ceiling) that task #36 measured for the GEMM standalone. A 1.41x gap between
# the same kernel in a microbenchmark and in the model is worth a name.
#
# `opdoublefilter` narrows `opdouble` to a subset of a family, which is the only
# way to get per-shape cost here: standalone GEMM timings on this machine run the
# card at ~39% of its clock (measured, same session — the flash A/B reported
# `clock 0.39` and 4.6 TF/s for a kernel the in-situ attribution prices at 8.3),
# so a microbenchmark cannot answer a 1.4x question.
#
# Whisper's three shapes, in OUR column-major orientation (M, N, K) with N padded
# to 1536 by `gemm_padn`:
#
#     (1280, 1536, 1280)  x96   q/k/v/out projections   0.472 TFLOP total
#     (5120, 1536, 1280)  x32   ffn fc1                 0.629
#     (1280, 1536, 5120)  x32   ffn fc2                 0.629
#
# The weight is `op.ins[3]`; its PyTorch shape (K, N) distinguishes all three.

using WhisperRunner, DNNKernels, Lava, KernelAbstractions, Statistics, Printf
const DK = DNNKernels
const KA = KernelAbstractions

backend = LavaBackend()
w = WhisperRunner.whispermodel(; backend)
mel = rand(Float32, 3000, 128, 1)

encode1() = (WhisperRunner.encode(w, mel); KA.synchronize(backend))
encode1(); encode1(); encode1()

graph = w.model.graphs["whisper"]
"""PyTorch (K, N) of this op's weight operand, or `nothing`."""
function wshape(op)
    length(op.ins) >= 3 || return nothing
    b = get(graph.buffers, op.ins[3], nothing)
    b === nothing && return nothing
    length(b.shape) == 2 ? (Int(b.shape[1]), Int(b.shape[2])) : nothing
end

# what is actually in the graph, so the filters below cannot silently match nothing
counts = Dict{Any,Int}()
for op in graph.ops
    op.aten == "addmm.default" || continue
    counts[wshape(op)] = get(counts, wshape(op), 0) + 1
end
println("addmm weight shapes (pytorch K x N):")
for (s, n) in sort(collect(counts); by = x -> -x[2])
    println("   ", s, "  x", n)
end
println()

function arm(label, setup!; n = 9)
    setup!(w.model.diag)
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); encode1(); push!(ts, (time_ns() - t0) / 1e6)
    end
    w.model.diag.opdouble = ""
    w.model.diag.opdoublefilter = nothing
    sort!(ts)
    med = ts[cld(length(ts), 2)]
    spread = (ts[max(1, round(Int, 0.9 * end))] - ts[max(1, round(Int, 0.1 * end))]) / med
    (; label, med, spread)
end

# (label, weight shape, call count, useful FLOPs for that subset)
SHAPES = [("qkv/out  1280x1280", (1280, 1280), 96, 96 * 2 * 1500 * 1280 * 1280),
          ("ffn fc1  1280x5120", (1280, 5120), 32, 32 * 2 * 1500 * 1280 * 5120),
          ("ffn fc2  5120x1280", (5120, 1280), 32, 32 * 2 * 1500 * 5120 * 1280)]

results = Any[]; baselines = Any[]
for (label, shp, _, _) in SHAPES
    push!(baselines, arm("base", d -> (d.opdouble = ""; d.opdoublefilter = nothing)))
    push!(results, arm(label, d -> begin
        d.opdouble = "addmm.default"
        d.opdoublefilter = (ctx, op) -> wshape(op) === shp
    end))
end
push!(baselines, arm("base", d -> (d.opdouble = ""; d.opdoublefilter = nothing)))
# the null: the whole family doubled, so the parts must sum to it
push!(results, arm("ALL addmm", d -> (d.opdouble = "addmm.default")))

bmed = median(b.med for b in baselines)
@printf("baseline %.2f ms  (%d arms, worst spread %.1f%%)\n\n",
        bmed, length(baselines), 100maximum(b.spread for b in baselines))
@printf("%-22s %8s %7s %9s %8s\n", "shape", "cost ms", "calls", "TF/s", "spread")
for (r, (label, _, n, fl)) in zip(results, SHAPES)
    c = r.med - bmed
    @printf("%-22s %8.2f %7d %9.1f %7.1f%%\n", label, c, n, fl / (c / 1e3) / 1e12, 100r.spread)
end
let r = results[end], c = r.med - bmed
    @printf("%-22s %8.2f %7d %9.1f %7.1f%%\n", "ALL addmm (check)", c, 160,
            sum(x[4] for x in SHAPES) / (c / 1e3) / 1e12, 100r.spread)
    @printf("\nparts sum to %.2f ms vs %.2f ms for the whole family\n",
            sum(x.med - bmed for x in results[1:end-1]), c)
end
