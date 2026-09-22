# The whole suite: every model, this tree against PyTorch, tracked over time.
#
#     julia --project=<repo-root> benchmark/suite.jl                # everything
#     julia --project=<repo-root> benchmark/suite.jl sam2-encode rife
#     julia --project=<repo-root> benchmark/suite.jl --no-torch      # this tree only
#     julia --project=<repo-root> benchmark/suite.jl --compare results/latest.json
#
# The project is the REPO ROOT and not `JuliaVision`: the runners and
# `BenchmarkTools` are both resolvable there, and adding BenchmarkTools to
# JuliaVision's own Project would be a dependency for a benchmark.
#
# ## What is stored, and why in BenchmarkTools' format
#
# Every timing lands in a `BenchmarkGroup` as a `Trial` built from the real
# per-sample times, so the saved file is the one `BenchmarkTools.judge` already
# knows how to read. That buys regression detection for free and without a
# bespoke comparator:
#
#     julia> using BenchmarkTools
#     julia> old, new = load("results/a.json")[1], load("results/b.json")[1]
#     julia> judge(median(new), median(old))
#
# The leaves are `"julia"`, `"torch-eager"` and `"torch-compiled"` under each
# model, so `judge` also compares the two runtimes against each other inside one
# run, not only one run against another.
#
# What BenchmarkTools cannot hold — the shape, the dtype, the clock a sample ran
# at, how many samples the clock gate rejected — goes in the markdown written
# beside it. A number without those is not comparable with another number, and
# the markdown is where a human reads them.

using Printf, Statistics, Dates
using BenchmarkTools
import JSON3
import Mantle
using Mantle: LavaBackend
using KernelAbstractions

const HERE = @__DIR__
const RESULTS = joinpath(HERE, "results")
include(joinpath(HERE, "..", "tools", "measure.jl"))   # plateau/bench/report
include(joinpath(HERE, "julia_models.jl"))             # SPEC, BUILDERS, ORDER

const BACKEND = LavaBackend()
sync() = KernelAbstractions.synchronize(BACKEND)

# ── argument handling ────────────────────────────────────────────────────────

args = copy(ARGS)
takeflag!(f) = (i = findfirst(==(f), args); i === nothing ? false : (deleteat!(args, i); true))
function takeopt!(f)
    i = findfirst(==(f), args)
    i === nothing && return nothing
    v = args[i + 1]; deleteat!(args, i:(i + 1)); v
end

const NOTORCH  = takeflag!("--no-torch")
const TORCHONLY = takeflag!("--torch-only")
const COMPARE  = takeopt!("--compare")
const MODES    = something(takeopt!("--modes"), "eager,compiled")
const WANT     = isempty(args) ? ORDER : args

for m in WANT
    haskey(SPEC["models"], m) || error("no model `$m` in models.toml; have $(ORDER)")
end

# ── the Julia side ───────────────────────────────────────────────────────────

"""A `Trial` from real per-sample times, in nanoseconds.

`BenchmarkTools` is being used as a container and a comparator here, not as the
timer: its own `@benchmark` tunes an evaluation count against a fast, repeatable
function, and these are GPU forwards where the card's clock has to be held up
and individual samples have to be thrown away when it is not. `tools/measure.jl`
does that; this turns what it measured into the shape `judge` reads.
"""
trial(times_s) = BenchmarkTools.Trial(
    BenchmarkTools.Parameters(; samples = length(times_s), evals = 1),
    collect(Float64.(times_s) .* 1e9),        # seconds -> ns
    zeros(Float64, length(times_s)), 0, 0)

function runjulia(want)
    out = Dict{String,Any}()
    report()
    for name in want
        print(rpad(name, 18), "julia     "); flush(stdout)
        try
            call, md = Base.invokelatest(BUILDERS[name], BACKEND)
            f = () -> (call(); sync())
            plat, _ = plateau(f; seconds = SPEC["suite"]["ramp_seconds"])
            r = bench(f; sync = nothing, plat, samples = SPEC["suite"]["samples"],
                      floor = SPEC["suite"]["clock_floor"], label = name)
            # A row where the clock gate rejects everything is not a broken
            # model — `neurallut` is host-bound, so the card idles inside its own
            # sample and never holds the clock however long the sample is made.
            # Keep the ungated samples and mark the row; hiding which rows those
            # are is how a summary lies.
            kept = [s.seconds for s in r.samples if s.ok]
            times = isempty(kept) ? [s.seconds for s in r.samples] : kept
            out[name] = (; times, gated = !isempty(kept), md,
                         clock = r.clock, kept = r.kept,
                         total = r.kept + r.rejected)
            @printf("%9.2f ms%s ±%4.1f%%  clock %3.0f%%  %d/%d\n",
                    1000median(times), isempty(kept) ? "!" : " ", 100r.spread,
                    100r.clock, r.kept, r.kept + r.rejected)
        catch e
            println("FAILED: ", first(split(sprint(showerror, e), '\n'))[1:min(end, 80)])
        end
        GC.gc(true); Mantle.trim_gpu_pool!()
    end
    out
end

# ── the PyTorch side ─────────────────────────────────────────────────────────

"""Run `torch_models.py` under the ROCm interpreter and read back its JSON.

A separate process because it is a separate runtime, and after the Julia models
are gone because both want the card's memory. `nothing` when it could not run,
with the reason printed — a suite that silently drops the denominator reports
our own times as though they were a comparison.
"""
function runtorch(want, modes)
    json = joinpath(RESULTS, "torch-raw.json")
    mkpath(RESULTS)
    cmd = `$(joinpath(HERE, "torch.sh")) $(joinpath(HERE, "torch_models.py")) $want --modes $modes --out $json`
    println("\n── PyTorch ──")
    try
        run(cmd)
    catch e
        @warn "the PyTorch side did not run; the table will have our times and no denominator" exception = e
        return nothing
    end
    isfile(json) || return nothing
    JSON3.read(read(json, String))
end

# ── assembling and reporting ─────────────────────────────────────────────────

function group(jl, torch, want)
    g = BenchmarkGroup()
    for name in want
        gm = BenchmarkGroup()
        haskey(jl, name) && (gm["julia"] = trial(jl[name].times))
        if torch !== nothing && haskey(torch.models, name)
            for (mode, key) in ("eager" => "torch-eager", "compiled" => "torch-compiled")
                e = torch.models[name]
                haskey(e, mode) || continue
                haskey(e[mode], :times_ms) || continue      # unsupported/error entries
                gm[key] = trial(Float64.(e[mode].times_ms) ./ 1e3)
            end
        end
        isempty(gm) || (g[name] = gm)
    end
    g
end

ms(t) = BenchmarkTools.median(t).time / 1e6

function table(io, jl, torch, want, g)
    @printf(io, "%-18s %10s %10s %10s %9s %9s  %-16s %s\n", "model", "ours", "eager",
            "compiled", "vs eager", "vs comp", "dtype", "shape")
    for name in want
        haskey(g, name) || continue
        gm = g[name]
        o = haskey(gm, "julia") ? ms(gm["julia"]) : NaN
        e = haskey(gm, "torch-eager") ? ms(gm["torch-eager"]) : NaN
        c = haskey(gm, "torch-compiled") ? ms(gm["torch-compiled"]) : NaN
        md = haskey(jl, name) ? jl[name].md : (; dtype = "?", shape = "?", note = "")
        gate = haskey(jl, name) && !jl[name].gated ? "!" : " "
        f(x) = isnan(x) ? "     -    " : @sprintf("%9.2f ", x)
        rr(a, b) = isnan(a) || isnan(b) ? "     -   " : @sprintf("%8.2fx ", a / b)
        @printf(io, "%-18s %s%s%s%s%s%s  %-16s %s%s\n", name, f(o), gate, f(e), f(c),
                rr(o, e), rr(o, c), md.dtype, md.shape,
                isempty(md.note) ? "" : "  ($(md.note))")
    end
    # Everything the spec or the runtime says about a model that has no number.
    for name in want
        torch === nothing && break
        haskey(torch.models, name) || continue
        e = torch.models[name]
        for mode in ("eager", "compiled")
            haskey(e, mode) || continue
            haskey(e[mode], :unsupported) && @printf(io, "  %s torch-%s: %s\n",
                                                     name, mode, e[mode].unsupported)
            haskey(e[mode], :error) && @printf(io, "  %s torch-%s FAILED: %s\n",
                                               name, mode, first(e[mode].error, 160))
        end
        haskey(e, :error) && @printf(io, "  %s torch setup FAILED: %s\n",
                                     name, first(e.error, 160))
    end
    println(io, "\n`!` marks a row whose samples the clock gate rejected: the card idled")
    println(io, "inside the sample, so the number is worth less than a gated one and more")
    println(io, "than nothing. `ours/eager` is the like-for-like ratio — same graph")
    println(io, "decomposition, different runtime. `ours/compiled` is against Inductor")
    println(io, "fusing that graph into kernels this tree does not generate.")
end

# ── main ─────────────────────────────────────────────────────────────────────

jl = TORCHONLY ? Dict{String,Any}() : runjulia(WANT)
GC.gc(true); Mantle.trim_gpu_pool!()
torch = NOTORCH ? nothing : runtorch(WANT, MODES)

g = group(jl, torch, WANT)
println("\n── results ──")
table(stdout, jl, torch, WANT, g)

mkpath(RESULTS)
stamp = Dates.format(now(), "yyyy-mm-dd-HHMM")
host = gethostname()
base = joinpath(RESULTS, "$(stamp)-$(host)")
BenchmarkTools.save(base * ".json", g)
cp(base * ".json", joinpath(RESULTS, "latest.json"); force = true)
open(base * ".md", "w") do io
    println(io, "# ", stamp, " — ", host)
    println(io)
    st = gpustate()
    @printf(io, "GPU %.0f/%.0f MHz, %.0f MiB of %.0f, %.0f C\n",
            st.sm, st.smmax, st.mem, st.memtotal, st.temp)
    if torch !== nothing
        println(io, "PyTorch ", torch.torch, " on ", torch.device)
    else
        println(io, "PyTorch side did not run.")
    end
    println(io, "\n```")
    table(io, jl, torch, WANT, g)
    println(io, "```")
end
println("\nwrote $(base).json, $(base).md and results/latest.json")

if COMPARE !== nothing
    old = BenchmarkTools.load(COMPARE)[1]
    println("\n── judge against $(COMPARE) ──")
    show(stdout, MIME"text/plain"(), judge(median(g), median(old)))
    println()
end
