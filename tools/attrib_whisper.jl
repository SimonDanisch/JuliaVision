# Where does Whisper's encode go now?
#
# The `clampattn` fix (task #79) took encode 352.6 -> 124.5 ms, which invalidated
# every attribution taken before it: the 32 attentions used to be ~75% of the
# step and are now on the flash path. This re-attributes from scratch.
#
# METHOD: `Diagnostics.opdouble` — run one aten family twice per step and read
# the wall-time difference as its cost. Per-op synchronisation is not usable here
# (it buries a 681-op step under its own barriers, and per-dispatch timestamps
# serialise just as badly).
#
# The controls matter more than the arms. `lavadnn-perf-attribution` records that
# opdouble's null control once read +60 ms on a 360 ms baseline, and that the
# machine throws occasional ~55 ms excursions which announce themselves through
# SPREAD, not through the median. So:
#
#   * baselines are interleaved BETWEEN the arms, not run once at the start
#   * every arm reports its own spread; a wide arm is discounted, not subtracted
#   * the null arm (`opdouble` naming no op in the graph) is a real arm
#
# Usage (VS Code eval blocks, per Simon's workflow — plain re-assignable vars):
#
#     include("tools/attrib_whisper.jl")

using WhisperRunner, DNNKernels, Lava, KernelAbstractions, Statistics
const KA = KernelAbstractions

backend = LavaBackend()
w = WhisperRunner.whispermodel(; backend)
# the export's own layout: 3000 frames of 128 mel bins
mel = rand(Float32, 3000, 128, 1)

encode1() = (WhisperRunner.encode(w, mel); KA.synchronize(backend))

# warm: compile, build plans, fill the frozen-kernel cache
encode1(); encode1(); encode1()

"""Median and spread of `n` timed steps under a given diagnostics setting."""
function arm(label, setup!; n = 9)
    setup!(w.model.diag)
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); encode1(); push!(ts, (time_ns() - t0) / 1e6)
    end
    w.model.diag.opdouble = ""          # always leave it off
    sort!(ts)
    med = ts[cld(length(ts), 2)]
    spread = (ts[max(1, round(Int, 0.9 * end))] - ts[max(1, round(Int, 0.1 * end))]) / med
    (; label, med, spread, min = first(ts))
end

FAMILIES = ["addmm.default", "mm.default",
            "_scaled_dot_product_flash_attention.default",
            "native_layer_norm.default", "gelu.default", "add.Tensor",
            "clone.default", "mul.Tensor", "clamp.default",
            "convolution.default",
            "no_such_aten_zzz"]          # <- the null control, a real arm

results = Any[]
baselines = Any[]
for (i, fam) in enumerate(FAMILIES)
    # a baseline BEFORE each arm, so drift lands on both
    push!(baselines, arm("baseline$i", d -> (d.opdouble = "")))
    push!(results, arm(fam, d -> (d.opdouble = fam)))
end
push!(baselines, arm("baseline_end", d -> (d.opdouble = "")))

bmed = median(b.med for b in baselines)
bspread = maximum(b.spread for b in baselines)
println("\nbaseline median ", round(bmed, digits = 2), " ms   ",
        length(baselines), " arms, worst spread ", round(100bspread, digits = 1), "%")
for b in baselines
    println("   ", rpad(b.label, 14), lpad(round(b.med, digits = 2), 8), " ms  ±",
            lpad(round(100b.spread, digits = 1), 5), "%")
end

println("\n", rpad("aten", 46), lpad("cost", 9), lpad("%", 7), lpad("spread", 8))
for r in sort(results; by = x -> -(x.med - bmed))
    cost = r.med - bmed
    println(rpad(r.label, 46), lpad(round(cost, digits = 2), 9),
            lpad(round(100cost / bmed, digits = 1), 7), lpad(round(100r.spread, digits = 1), 8))
end
println("\nRead the null control first: whatever it reports is this run's noise ",
        "floor, and no arm below it is a finding.")
