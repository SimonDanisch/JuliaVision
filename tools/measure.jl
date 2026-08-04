# A GPU measurement harness that refuses to report a number it cannot trust.
#
# ## Why this exists
#
# This card idles at **720 MHz of 3105** — 23% of its clock — and ramps only
# under sustained load. A kernel timed cold and the same kernel timed warm differ
# by more than four times, which is larger than almost any optimisation worth
# making. The machine is also a desktop: the compositor, a browser, Slack,
# Discord and VS Code all hold GPU contexts, and a Julia REPL in another window
# holds several GB.
#
# So a bare `@elapsed` here does not measure the code. It measures the code, the
# clock state, and whatever else happened to be drawing. Three attributions in
# this project were wrong for exactly that reason:
#
#   * a per-op serialised profile blamed `pow` for 157 ms; timed directly, the
#     two forms were 1.0x apart — the instrument was measuring itself;
#   * an ablation run against a single baseline taken at the start reported two
#     op families as NEGATIVE, because the card warmed across the sequence;
#   * a cross-session A/B inflated a 1.2% effect to 3.5-9%, because the baseline
#     was a different process on a differently-warmed card.
#
# ## What it does
#
# 1. **Warms the clock** before anything is recorded, and reports what it reached.
# 2. **Brackets every sample with a clock reading** and discards any sample whose
#    clock dipped below the floor. A discarded sample is reported, not hidden —
#    if most of them go, the answer is "the machine was busy", not a number.
# 3. Reports the **median and the spread**, never a bare minimum. A minimum
#    hides exactly the variance this file exists to expose.
# 4. Interleaves the A/B arms rather than running one then the other, so a drift
#    in either direction hits both equally.
#
# The `nvidia-smi` calls sit outside the timed region and cost ~20 ms each, which
# is why samples shorter than a few ms should be batched by the caller into a
# `for` loop inside `f` rather than timed individually.

using Printf, Statistics

"""
    gpustate() -> NamedTuple

`(; sm, smmax, util, mem, memtotal, temp, power, load1)`, clocks in MHz and
memory in MiB. One `nvidia-smi` call, ~20 ms.
"""
function gpustate()
    q = "clocks.sm,clocks.max.sm,utilization.gpu,memory.used,memory.total," *
        "temperature.gpu,power.draw"
    out = read(`nvidia-smi --query-gpu=$q --format=csv,noheader,nounits`, String)
    v = parse.(Float64, strip.(split(strip(out), ',')))
    load1 = parse(Float64, first(split(read("/proc/loadavg", String))))
    (; sm = v[1], smmax = v[2], util = v[3], mem = v[4], memtotal = v[5],
       temp = v[6], power = v[7], load1)
end

"""
    otherprocs() -> Vector{Tuple{Int,String,Int}}

Compute processes on the GPU that are **not** this one: `(pid, name, MiB)`.

A desktop always has several — the compositor and every Electron app hold a
context. They matter here because they are also *scheduling* work: a browser
repainting during a sample is contention this harness cannot subtract, only
notice.
"""
function otherprocs()
    out = read(`nvidia-smi --query-compute-apps=pid,process_name,used_memory
                --format=csv,noheader,nounits`, String)
    me = getpid()
    rows = Tuple{Int,String,Int}[]
    for line in split(strip(out), '\n')
        isempty(strip(line)) && continue
        f = split(line, ", "; limit = 3)
        length(f) == 3 || continue
        pid = parse(Int, strip(f[1]))
        pid == me && continue
        push!(rows, (pid, first(split(strip(f[2]), ' ')), parse(Int, strip(f[3]))))
    end
    rows
end

"""
    plateau(f; seconds = 5) -> (clock, smmax)

The SM clock this workload actually **sustains**, in MHz.

Not `clocks.max.sm`, and the difference is the whole point. Measured here: this
card reports a 3105 MHz maximum and holds **2070-2265 MHz** running Kokoro's
vocoder, with `clocks_throttle_reasons.active = 0x4` — the software power cap, at
98 W. 3105 is a boost ceiling it will not hold under compute for a moment longer
than the sample it is measured in.

So gating a benchmark at "80% of maximum" rejects every sample on a perfectly
healthy machine, which is what happened the first time this file was used. The
floor has to be a fraction of what the workload *reaches*, and the only way to
know that is to run it and look.

The first second is discarded: that is the ramp, and including it drags the
plateau down by whatever fraction of the window it occupies.
"""
function plateau(f; seconds::Real = 5)
    t0 = time()
    clocks = Float64[]
    st = gpustate()
    while time() - t0 < seconds
        for _ in 1:5; f(); end
        st = gpustate()
        time() - t0 > 1 && push!(clocks, st.sm)
    end
    (isempty(clocks) ? st.sm : median(clocks), st.smmax)
end

"""
    Sample

One timing and the machine state around it. `ok` is false when the clock dipped
below the floor at either end, which is the only automatic rejection: everything
else is reported and left to the reader.
"""
struct Sample
    seconds::Float64
    smbefore::Float64
    smafter::Float64
    ok::Bool
end

"""
    bench(f; samples = 15, floor = 0.80, warm = true, label = "") -> Result

Time `f` with the clock gated.

`floor` is a fraction of the clock this workload **sustains** ([`plateau`](@ref)),
not of the card's advertised maximum — see there for why the distinction is not
pedantic. 0.90 of the plateau is tight enough to catch another process stealing
the card and loose enough to survive the few-percent wobble a desktop always has.

`sync` runs **inside** the timed region, after `f`. It is mandatory for GPU work
and there is no useful default: a Lava/KA launch is asynchronous, so timing `f`
alone times the *queue submission* and nothing else. That failure is not subtle
once you look — it reported this card at **256 TFLOP/s**, an order of magnitude
past its fp32 peak, with a ±211% spread — but it is completely silent if you only
look at the ratio between two arms, because both are equally meaningless.

[`implausible`](@ref) catches the shape of it after the fact; passing `sync` is
what prevents it.

Returns a [`Result`](@ref). **Read `spread` before `median`**: if the spread is
larger than the effect being chased, there is no effect to report yet.
"""
struct Result
    label::String
    median::Float64          # seconds
    spread::Float64          # (p90 - p10) / median, dimensionless
    kept::Int
    rejected::Int
    clock::Float64           # mean SM clock over the kept samples, fraction of max
    samples::Vector{Sample}
end

function bench(f; samples::Int = 15, floor::Real = 0.90, warm::Bool = true,
               label::AbstractString = "", sync = nothing)
    g = sync === nothing ? f : (() -> (f(); sync()))
    g()                                     # compile, allocate, page in
    plat, smmax = warm ? plateau(g) : (gpustate().sm, gpustate().smmax)
    lo = floor * plat
    out = Sample[]
    for _ in 1:samples
        a = gpustate().sm
        t = @elapsed g()
        b = gpustate().sm
        push!(out, Sample(t, a, b, a >= lo && b >= lo))
    end
    keep = [s for s in out if s.ok]
    isempty(keep) && return Result(label, NaN, NaN, 0, length(out), plat / smmax, out)
    ts = sort([s.seconds for s in keep])
    med = median(ts)
    p(q) = ts[clamp(round(Int, q * length(ts)), 1, length(ts))]
    Result(label, med, (p(0.9) - p(0.1)) / med, length(keep), length(out) - length(keep),
           mean(s -> (s.smbefore + s.smafter) / 2, keep) / smmax, out)
end

function Base.show(io::IO, r::Result)
    if r.kept == 0
        @printf(io, "%-28s  NO TRUSTWORTHY SAMPLE (%d rejected on clock)",
                r.label, r.rejected)
        return
    end
    @printf(io, "%-28s %8.3f ms  ±%4.1f%%  clock %3.0f%%  %d/%d kept",
            r.label, r.median * 1000, 100 * r.spread, 100 * r.clock,
            r.kept, r.kept + r.rejected)
end

"""
    compare(a, b; samples = 15, ...) -> (ra, rb, ratio, trustworthy)

Two arms, **interleaved**, one sample of each per round.

This is the only comparison shape worth making on this machine. Running arm A to
completion and then arm B lets any drift — the clock ramping, another process
starting, the card heating — land entirely on one of them; that is how an
earlier ablation here reported that doubling an op made the run *faster*.

`trustworthy` is false when the measured difference is inside the noise of either
arm, which is the question to ask before quoting a ratio.
"""
function compare(fa, fb; samples::Int = 15, floor::Real = 0.90,
                 labels = ("A", "B"), sync = nothing)
    # NEW names. `fa = () -> (fa(); sync())` captures the *variable*, so the
    # closure calls itself — a StackOverflowError from a benchmark harness, which
    # is a confusing place to get one.
    ga = sync === nothing ? fa : (() -> (fa(); sync()))
    gb = sync === nothing ? fb : (() -> (fb(); sync()))
    ga(); gb()
    plat, smmax = plateau(() -> (ga(); gb()))
    lo = floor * plat
    sa, sb = Sample[], Sample[]
    for _ in 1:samples
        for (f, acc) in ((ga, sa), (gb, sb))
            x = gpustate().sm
            t = @elapsed f()
            y = gpustate().sm
            push!(acc, Sample(t, x, y, x >= lo && y >= lo))
        end
    end
    mk(s, l) = begin
        keep = [z for z in s if z.ok]
        isempty(keep) && return Result(l, NaN, NaN, 0, length(s), 0.0, s)
        ts = sort([z.seconds for z in keep]); med = median(ts)
        p(q) = ts[clamp(round(Int, q * length(ts)), 1, length(ts))]
        Result(l, med, (p(0.9) - p(0.1)) / med, length(keep), length(s) - length(keep),
               mean(z -> (z.smbefore + z.smafter) / 2, keep) / smmax, s)
    end
    ra, rb = mk(sa, labels[1]), mk(sb, labels[2])
    ratio = ra.median / rb.median
    # The effect has to be bigger than the noise of BOTH arms to be an effect.
    trustworthy = ra.kept > 0 && rb.kept > 0 &&
                  abs(1 - ratio) > max(ra.spread, rb.spread)
    (ra, rb, ratio, trustworthy)
end

"""
    report()

Print what the machine looks like right now: clock, other GPU processes, load.
Worth doing before a measurement session and quoting alongside the numbers.
"""
function report()
    st = gpustate()
    @printf("GPU  %.0f/%.0f MHz (%.0f%%)  util %.0f%%  mem %.0f/%.0f MiB  %.0f C  %.0f W\n",
            st.sm, st.smmax, 100 * st.sm / st.smmax, st.util, st.mem, st.memtotal,
            st.temp, st.power)
    others = otherprocs()
    @printf("load %.2f on %d cores; %d other GPU process%s",
            st.load1, Sys.CPU_THREADS, length(others), length(others) == 1 ? "" : "es")
    isempty(others) || @printf(" holding %d MiB", sum(o -> o[3], others))
    println()
    for (pid, name, mib) in sort(others; by = o -> -o[3])[1:min(5, end)]
        @printf("    %-28s %6d MiB  pid %d\n", basename(name), mib, pid)
    end
end


"""
    implausible(r, flops, peak) -> Union{Nothing,String}

Why this result should not be believed, or `nothing`.

Two checks, and both have caught a real mistake here:

  * **Above the hardware.** A rate past the device's peak means the work did not
    happen inside the timed region — almost always a missing `sync` on an
    asynchronous launch. Reported as 256 TFLOP/s on a 26.7 TFLOP/s card.
  * **Spread wider than the signal.** Past about 30% the median is not describing
    a distribution anyone should quote a ratio from.

`peak` is the device's arithmetic ceiling for the dtype in question, which is a
number the caller has to supply because this file cannot know what precision or
what units the work was in.
"""
function implausible(r::Result, flops::Real, peak::Real)
    r.kept == 0 && return "no sample survived the clock gate"
    rate = flops / r.median
    rate > peak && return @sprintf("%.1f > device peak %.1f — work is outside the timed region (missing sync?)",
                                   rate, peak)
    r.spread > 0.30 && return @sprintf("spread ±%.0f%% is too wide to quote", 100 * r.spread)
    nothing
end
