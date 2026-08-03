"""
RIFE 4.26: does an interpolated frame fit the editor's budget?

    julia --project=. tools/bench_rife.jl

`models-to-port.md` sets the target at **1080p at 60 fps** — 16.7 ms per output
frame. That is the number this prints, plus the two things that decide whether it
is reachable at all on an 8 GB laptop: the planned slab and the resident weights.

**No engine comparison here** — `GUARDRAILS.md` §6: `perf-plan.md`'s numbers are
all desktop measurements and cross-machine numbers do not compare. This prints
this machine's wall clock and says so.

Correctness lives in `tools/verify_rife.jl`; this file assumes it passed and only
asks how long the same call takes.

**Interpolation is not framerate.** One graph call produces one intermediate
frame, so 2x slow motion at 1080p60 needs one call per source frame pair and the
budget is the whole 16.7 ms. 4x needs three calls at three different `t` and the
budget is a third of that each, which is why `timestep` is a graph input rather
than a baked constant — the three calls differ only in one scalar buffer.
"""

using DNNKernels, KernelAbstractions, Lava, Printf, Statistics
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  planslab, fusableset, Workspace
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "rife"))

smclock() = try
    parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                String))))
catch ex
    # 9999 is "clock is fine, do not gate" — the right answer on a machine with
    # no `nvidia-smi` at all. It is the WRONG answer for an nvidia-smi that is
    # present and replies with something unparseable, because the gate this feeds
    # (GUARDRAILS §6, warm-up on clock) would then pass on every run and quietly
    # stop protecting the measurement. Narrowed to the absent-binary case.
    ex isa Union{Base.IOError, SystemError, Base.ProcessFailedException} || rethrow()
    9999
end

"""
Median of `n` samples, each `reps` launches with one sync. Same shape as
`gemm_bench.jl`'s `timedall`: per-launch syncs measure the sync, not the graph.
"""
function timed(f, backend; n = 9, reps = 3)
    # 30 warm-up calls, not 3. Measured on the neural LUT classifier: the first
    # call after the clock ramp took 2 834 ms and calls 2-24 ran 7-20 ms before
    # settling at ~2 ms from call 25 on. A short warm-up therefore measures the
    # ramp, and a median over it lands anywhere in a 5x band — which is exactly
    # the run-to-run "noise" this file used to report on the small graph. Costly
    # only where a call is cheap, which is where it matters.
    for _ in 1:30; f(); end; KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:n
        KA.synchronize(backend)
        t0 = time_ns()
        for _ in 1:reps; f(); end
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) / 1e6 / reps)
    end
    median(ts)
end

backend = LavaBackend()
isdir(DIR) || error("no export at $DIR — run `uv run tools/export_rife.py`")

# Through `Model`, not `loadgraph`: it runs the host-side preparation passes
# (fold, hoist, dropdead) that the editor's own path gets, so a benchmark that
# skips them measures a graph nothing ships. Worth 366 -> 345 ops here, and
# ~1.8x on Depth Anything; on this graph it is within noise, which is itself
# worth knowing — the passes have nothing to fold in an fp32 leaky-ReLU network.
model = Model(DIR, joinpath(DIR, "weights.safetensors"); names = ["rife"], backend)
graph = model.graphs["rife"]
weights = model.weights
hw = readsafetensors(joinpath(DIR, "weights.safetensors"))
ref = readsafetensors(joinpath(DIR, "reference.safetensors"))

imgs = toback(backend, ref["imgs"])
tstep = toback(backend, ref["timestep"])
inputs = Dict{String,Any}("imgs" => imgs, "timestep" => tstep)

plan = planslab(graph, (;))
slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
ws = Workspace(backend)
lazy = fusableset(graph)

wbytes = sum(length(v) * sizeof(eltype(v)) for v in values(hw))
W, H = size(ref["imgs"], 1), size(ref["imgs"], 2)

# Heat the card: this one idles at 210 MHz of 2265 and a cold clock is worth
# several x. GUARDRAILS §6.
heat = KA.allocate(backend, Float32, 1 << 22)
heat2 = KA.allocate(backend, Float32, 1 << 22)
for _ in 1:25
    for _ in 1:200; heat .= heat2 .* 1.0001f0 .+ 0.5f0; end
    KA.synchronize(backend)
    smclock() >= 1800 && break
end
clk = smclock()

t = timed(backend) do
    execute!(graph, inputs, weights; dims = (;), backend, slab, plan, ws, lazy)
end

println()
println("RIFE 4.26 at $(W)x$(H) — THIS MACHINE (RTX 3070 laptop, 8 GB), SM clock $clk MHz")
println("cross-machine numbers do not compare; perf-plan.md keeps desktop numbers only")
@printf("  one interpolated frame   %8.2f ms   [target 16.67 = 1080p60]\n", t)
@printf("  implied rate             %8.2f fps\n", 1000 / t)
@printf("  planned slab             %8.1f MiB\n", plan.bytes / 2^20)
@printf("  resident weights         %8.1f MiB\n", wbytes / 2^20)
println(t < 16.67 ? "WITHIN BUDGET" : "OVER BUDGET")
