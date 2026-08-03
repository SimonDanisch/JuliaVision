"""
Depth Anything V2 Small: how long does one depth map take?

    julia --project=. tools/bench_depthanything.jl

`models-to-port.md` sets this one's target at **≥ PyTorch**, so the number here
is half of a comparison — `tools/baseline_depthanything.py` prints the other
half, on the same machine and the same input size, and the two are read together.

**No engine comparison goes in `perf-plan.md`** — GUARDRAILS §6: that file's
numbers are all desktop measurements and cross-machine numbers do not compare.
This machine reports a verdict; the desktop confirms a number if one is needed.

The editor budget is the other question and it is a different one: depth buys
depth-keyed grading, fake defocus and parallax push-ins, all of which are
per-frame effects. A depth map that takes 400 ms does not buy them at 3x PyTorch
any more than it does at 1x.
"""

using DNNKernels, KernelAbstractions, Lava, Printf, Statistics
using DNNKernels: loadgraph, execute!, readsafetensors, toback,
                  planslab, fusableset, Workspace, Model
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "depthanything"))

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

function timed(f, backend; n = 9, reps = 3)
    f(); KA.synchronize(backend)
    for _ in 1:2; f(); end; KA.synchronize(backend)
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
isdir(DIR) || error("no export at $DIR — run `uv run tools/export_depthanything.py`")

# Through `Model`, not `loadgraph`: it runs the host-side preparation passes
# (fold, hoist, dropdead) that the editor's own path gets. Not a detail on this
# model — 311 ops become 290 and the frame goes from 763.56 ms to 413.46 ms, so a
# benchmark that skips them overstates the engine's cost by 1.8x.
model = Model(DIR, joinpath(DIR, "weights.safetensors");
              names = ["depthanything"], backend)
graph = model.graphs["depthanything"]
weights = model.weights
hw = readsafetensors(joinpath(DIR, "weights.safetensors"))
ref = readsafetensors(joinpath(DIR, "reference.safetensors"))
img = toback(backend, ref["input"])

plan = planslab(graph, (;))
slab = KA.allocate(backend, UInt8, max(plan.bytes, 1))
ws = Workspace(backend)
lazy = fusableset(graph)
inputs = Dict{String,Any}("x" => img)

wbytes = sum(length(v) * sizeof(eltype(v)) for v in values(hw))
S = size(ref["input"], 1)

# Heat the card: this one idles at 210 MHz of 2265. GUARDRAILS §6.
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
println("Depth Anything V2 Small at $(S)x$(S) — THIS MACHINE (RTX 3070 laptop, 8 GB), ",
        "SM clock $clk MHz")
println("cross-machine numbers do not compare; perf-plan.md keeps desktop numbers only")
@printf("  one depth map            %8.2f ms\n", t)
@printf("  implied rate             %8.2f fps\n", 1000 / t)
@printf("  planned slab             %8.1f MiB\n", plan.bytes / 2^20)
@printf("  resident weights         %8.1f MiB\n", wbytes / 2^20)
println("compare against `uv run tools/baseline_depthanything.py` on this machine")
