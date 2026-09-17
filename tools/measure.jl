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

# ── where the machine's state comes from ─────────────────────────────────────
#
# This file was written against `nvidia-smi` and the card it runs on now is a
# `Radeon 8060S (RADV STRIX_HALO)`, so every measurement in it threw ENOENT at
# the first clock read: `bench_all.jl`, the one harness for every model's forward
# pass, could not take a single sample.
#
# The source is chosen ONCE, below, and answered by dispatch — not asked per call
# site. Which machine this is cannot be a branch scattered through the
# measurement code, and it is a fact about the machine rather than about any
# shader, so it is nothing like the vendor-conditional code the project bans.
#
# **The amdgpu path is the better instrument, not a fallback.** Everything here
# is one `read` of a sysfs file, tens of microseconds against `nvidia-smi`'s 26
# ms — and that 26 ms is the defect three docstrings below this one describe: a
# probe that takes long enough for the queue to drain reports the hole it dug.
# The overlap trick those comments explain exists to work around a cost this path
# does not have.

"""The seven fields every source answers, plus the host's load."""
loadavg1() = parse(Float64, first(split(read("/proc/loadavg", String))))

"""`nvidia-smi`, one CSV row."""
struct NvidiaSmi end

const SMIQUERY = "clocks.sm,clocks.max.sm,utilization.gpu,memory.used,memory.total," *
                 "temperature.gpu,power.draw"

smiproc() = open(`nvidia-smi --query-gpu=$SMIQUERY --format=csv,noheader,nounits`, "r")

function smiparse(out::AbstractString)
    v = parse.(Float64, strip.(split(strip(out), ',')))
    (; sm = v[1], smmax = v[2], util = v[3], mem = v[4], memtotal = v[5],
       temp = v[6], power = v[7], load1 = loadavg1())
end

sample(::NvidiaSmi) = smiparse(read(smiproc(), String))

"""
The amdgpu kernel driver, through sysfs.

`dev` is the card's device directory and `hwmon` the sensor directory under it.
`smax` is read once at construction: the DPM table does not change while the
machine is up, and it is the only field here that needs parsing rather than a
single number.
"""
struct AmdSysfs
    dev::String
    hwmon::String
    smax::Float64
end

"""The top entry of `pp_dpm_sclk`, in MHz. The file is one level per line,
`\"2: 2900Mhz\"`, with a `*` on whichever is current."""
function dpmmax(dev::AbstractString)
    best = 0.0
    for line in eachline(joinpath(dev, "pp_dpm_sclk"))
        m = match(r"([0-9]+)\s*[MmGg][Hh]z", line)
        m === nothing && continue
        best = max(best, parse(Float64, m.captures[1]))
    end
    best > 0 || error("measure.jl: $(dev)/pp_dpm_sclk named no clock level.")
    return best
end

num(path) = parse(Float64, strip(read(path, String)))

function sample(a::AmdSysfs)
    (; sm = num(joinpath(a.hwmon, "freq1_input")) / 1e6,      # Hz -> MHz
       smmax = a.smax,
       util = num(joinpath(a.dev, "gpu_busy_percent")),
       mem = num(joinpath(a.dev, "mem_info_vram_used")) / 2^20,
       memtotal = num(joinpath(a.dev, "mem_info_vram_total")) / 2^20,
       temp = num(joinpath(a.hwmon, "temp1_input")) / 1000,   # milli-C
       power = num(joinpath(a.hwmon, "power1_average")) / 1e6, # uW -> W
       load1 = loadavg1())
end

"""
    telemetry() -> NvidiaSmi | AmdSysfs

The source this machine has, refused rather than guessed when it has neither.

A card is identified by the files it exposes and not by a name: `freq1_input`
under the device's `hwmon` is what this needs, and a card whose driver does not
publish it cannot be gated on however it is labelled.
"""
function telemetry()
    for dev in sort(Base.Filesystem.readdir("/sys/class/drm"; join = true))
        d = joinpath(dev, "device")
        isfile(joinpath(d, "gpu_busy_percent")) || continue
        hw = joinpath(d, "hwmon")
        isdir(hw) || continue
        for h in readdir(hw; join = true)
            isfile(joinpath(h, "freq1_input")) || continue
            return AmdSysfs(d, h, dpmmax(d))
        end
    end
    Sys.which("nvidia-smi") === nothing && error(
        "measure.jl: this machine publishes neither an amdgpu `freq1_input` " *
        "under /sys/class/drm/*/device/hwmon nor an `nvidia-smi`, so the SM " *
        "clock cannot be read. Every number here is gated on that clock — see " *
        "`plateau` — so there is nothing to fall back to that would still mean " *
        "what these rows claim to mean.")
    return NvidiaSmi()
end

const TELEMETRY = telemetry()

"""
    gpustate() -> NamedTuple

`(; sm, smmax, util, mem, memtotal, temp, power, load1)`, clocks in MHz and
memory in MiB. One `nvidia-smi` call, ~26 ms.

**This reports an IDLE card whenever it is called between synchronised samples,
and that is not a quirk — it is the reason the whole clock gate read 7% while the
card was at 2070 MHz.** `nvidia-smi` takes 26 ms; if the workload ended in a
`synchronize`, the queue is empty for every one of them, and the SM clock
collapses to 210 MHz and comes back the moment work resumes. Nothing survives
that: `plateau` measured 210, `bench` gated 210 against 210 and kept everything,
and a 15 us copy was reported at 164 us.

The check that misled me first: a busy loop of *unsynchronised* launches shows no
decay at all, even after 500 ms of "idle", because thousands of queued copies are
still draining through the read. Both arms have to end in a sync before the
question means anything.

Use [`gpustate(f)`](@ref) for a reading that means something under load.
"""
gpustate() = sample(TELEMETRY)

"""
    gpustate(f; window = 0.2) -> NamedTuple

Machine state sampled **while `f` runs**, which is the only kind worth gating on.

`nvidia-smi` is started first and `f` is run back to back until it has answered,
so the sample it takes internally lands on a card that is doing the work being
measured, not on the hole its own 26 ms dug. `f` runs untimed here — this changes
how often the workload runs, not what is measured.
"""
gpustate(f; window::Real = 0.2) = clockunder(f, window)

"""
    clockunder(f, seconds) -> NamedTuple

The machine's state at the HIGHEST clock `f` drove it to over `seconds`.

One reading per call of `f`, taken between calls, which is what "sampled while
`f` runs" can mean when the probe costs microseconds. The `nvidia-smi` path
could not do this — its 26 ms probe had to be overlapped with the workload and
answered with whatever moment the driver happened to pick inside it — so this is
the same intent measured directly rather than inferred.

The MAXIMUM and not the mean: the question every caller asks is what clock this
workload reaches, and a mean over a window that begins before the card has ramped
answers a different one. `plateau` then medians these across windows, so a single
spike cannot carry a run.
"""
function clockunder(f, seconds::Real)
    best = gpustate()
    t0 = time()
    while time() - t0 < seconds
        f()
        st = gpustate()
        st.sm > best.sm && (best = st)
    end
    return best
end

"""
    otherprocs() -> Vector{Tuple{Int,String,Int}}

Compute processes on the GPU that are **not** this one: `(pid, name, MiB)`.

A desktop always has several — the compositor and every Electron app hold a
context. They matter here because they are also *scheduling* work: a browser
repainting during a sample is contention this harness cannot subtract, only
notice.
"""
otherprocs() = otherprocs(TELEMETRY)

function otherprocs(::NvidiaSmi)
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
Whoever else holds the render node.

amdgpu publishes no per-process memory the way `nvidia-smi` does, so the third
field is **0 here and that is a gap, not a zero**. `report` prints the count,
which is the part that matters: a compositor repainting during a sample is
contention this harness can notice and cannot subtract.

`fuser` and `ps` rather than a walk over `/proc/*/fd`, because a process list
read entry by entry changes underneath the reader — a pid that exits between the
`isdir` and the `readdir` throws, and "a process exited" is not a condition worth
a `catch` that would also hide a real one. Each of these is one call that
answers about the set as it was.
"""
function otherprocs(::AmdSysfs)
    nodes = filter(startswith("render"), readdir("/dev/dri"))
    isempty(nodes) && return Tuple{Int,String,Int}[]
    out = read(pipeline(`fuser $(["/dev/dri/" * n for n in nodes])`;
                        stderr = devnull), String)
    me = getpid()
    pids = [p for p in parse.(Int, split(strip(out))) if p != me]
    isempty(pids) && return Tuple{Int,String,Int}[]
    # `ps` omits a pid that has since exited, which is the answer and not a
    # failure; `-o comm=` gives the name without a header to strip.
    names = read(pipeline(`ps -o pid=,comm= -p $(join(pids, ','))`;
                          stderr = devnull), String)
    rows = Tuple{Int,String,Int}[]
    for line in split(strip(names), '\n')
        f = split(strip(line); limit = 2)
        length(f) == 2 || continue
        push!(rows, (parse(Int, f[1]), f[2], 0))
    end
    rows
end

"""Run `f` back to back for `seconds`, untimed, to hold the card at its clock."""
function busy(f, seconds::Real)
    t0 = time()
    while time() - t0 < seconds
        f()
    end
end

"""
    plateau(f; seconds = 5, window = 0.25) -> (clock, smmax)

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

**`window` is what makes this work at all, and it was missing.** The loop used to
read the clock after every 5 calls of `f`; [`gpustate`](@ref) is an `nvidia-smi`
*subprocess* costing 26 ms, so for a kernel of tens of microseconds the card was
idle 99.7% of a "warm-up" whose entire job is to make it busy. It never left
210 MHz, the plateau came back as the idle clock, and [`bench`](@ref)'s gate then
compared idle against idle and kept every sample — a 15 us copy reported as
164 us, and layer-norm against a copy of the same tensor reported at 1.17x when
it is 2.68x. Measured, not reasoned: the clock does **not** decay for at least
500 ms after a load stops, so the failure is entirely that the ramp never
happens, and one busy window per clock reading is the whole fix.

`f` runs uninstrumented inside the window. That is safe for a *stateful* model in
the way that repeating it inside a timed sample is not — see the note above
`bench` — because it changes how often `f` is called, not what is measured.
"""
function plateau(f; seconds::Real = 5, window::Real = 0.25)
    t0 = time()
    clocks = Float64[]
    st = gpustate(f; window)
    while time() - t0 < seconds
        st = gpustate(f; window)
        time() - t0 > 1 && push!(clocks, st.sm)
    end
    (isempty(clocks) ? st.sm : median(clocks), st.smmax)
end

"""
    Sample

One timing and the machine state around it. `ok` is false when the clock dipped
below the floor at either end, which is the only automatic rejection: everything
else is reported and left to the reader.

`gc` is host garbage collection **inside** the timed region, and it is recorded
because on this workload it is not a rounding error: `depthanything` read
73.20 ms ±216% and is 47.17 ms ±3% — a 36% error and a spread that reads as a
broken measurement, all of it one collection landing in some samples and not
others. A median does not save you when most samples are hit. See
[[gpu-spread-is-host-gc]] for the same trap costing a whole afternoon.
"""
struct Sample
    seconds::Float64
    gc::Float64
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

**Two `bench` calls are NOT a comparison.** Each measures its own
[`plateau`](@ref) and gates against it, so a kernel too short to hold the card
awake is accepted at a low clock while a longer one runs at a high clock — and
the ratio between them is then mostly the clock. That produced two wrong
conclusions here in one session: "the lifted convolution is only 1.21x over the
direct kernel" (it is **6.61x**; the arms ran at 7% and 73%) and a cap sweep that
read as 1.04x. Use [`compare`](@ref), which shares one plateau and interleaves,
or pass the same `plat` to both.

The `clock` column exists to make this visible — if two results you are about to
divide report different clocks, the division is meaningless.

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

**One number only.** Two `bench` calls are not a comparison, and neither are
three: each runs its arm to completion, so the clock ramp lands on whichever went
first. Passing a shared `plat` does not fix that — it makes the *gate* common,
not the conditions. Sharing a plateau and then benching three arms in sequence
produced a 1.75x here with the arms at 7%, 63% and 70% of the clock, on this
file. Use [`compare`](@ref), which interleaves; it takes a tuple for more than
two.
"""
struct Result
    label::String
    median::Float64          # seconds
    spread::Float64          # (p90 - p10) / median, dimensionless
    kept::Int
    rejected::Int
    clock::Float64           # mean SM clock over the kept samples, fraction of max
    gc::Float64              # median host GC inside the timed region, fraction of median
    # ── Which PROCESS produced this. Two Results with different `session`s were
    # measured on differently-warmed cards and cannot be divided; `ratio` refuses
    # to. Cross-session variance here is ~13% against a ~5% within-session figure
    # (see `lavadnn-benchmark-variance`), so a 1.4% effect is invisible across a
    # restart — and reading one as a null is how working code got reverted.
    session::UInt64
    samples::Vector{Sample}
end

"""A value unique to this process, stamped into every `Result`."""
const SESSION = rand(UInt64)

"""
    ratio(a::Result, b::Result) -> Float64

`a.median / b.median`, **or an error naming why the division is meaningless.**

The three ways a ratio here has been wrong, each of which this refuses:

  * **different processes.** 91.85 vs 91.05 ms across a restart was reported as
    "1.01x, nothing"; the same pair measured in one session is 0.986x and
    trustworthy. Cross-session noise is ~13%.
  * **different clocks.** Two `bench` calls whose arms ran at 14% and 73% of the
    clock were read as equal. The `clock` field existed the whole time.
  * **no trustworthy sample.** A `NaN` median divides to `NaN` and prints as a
    number-shaped thing.

`compare` produces Results that satisfy all three by construction, which is why
it is the only supported way to get two numbers to divide.
"""
function ratio(a::Result, b::Result)
    a.session == b.session || error(
        "ratio: these Results are from DIFFERENT PROCESSES ($(a.label), $(b.label)). " *
        "Cross-session variance is ~13% here, so anything under that is noise. " *
        "Measure both arms in one session with `compare`.")
    (a.kept > 0 && b.kept > 0) || error(
        "ratio: $(a.kept == 0 ? a.label : b.label) has no sample the clock gate kept.")
    # 0.30, and both bounds are observations rather than a guess. The failure to
    # catch is two separate `bench` calls at **14% and 73%** (relative difference
    # 0.81) read as equal. The false positive to avoid is that `compare`'s own
    # interleaved arms routinely differ — 35% vs 45% here, relative 0.22 — because
    # `clock` is the mean of single `nvidia-smi` readings of a noisy quantity, not
    # a measurement of the clock the kernel actually ran at. Interleaving makes
    # the true clocks equal in expectation; it cannot make the READINGS equal.
    rel = abs(a.clock - b.clock) / max(a.clock, b.clock)
    rel <= 0.30 || error(
        "ratio: the arms ran at different clocks — $(a.label) at " *
        "$(round(100a.clock))% and $(b.label) at $(round(100b.clock))%, " *
        "a relative difference of $(round(rel; digits = 2)). That difference IS " *
        "the ratio. Use `compare`, which interleaves them.")
    a.median / b.median
end

"""
    L2_BYTES

48 MB — this card's L2, and the line a bandwidth number has to declare which side
of it lives on.

A staging benchmark on a 16.8 MB working set measured 693 GB/s and that was
written down as "the card's ceiling"; the same kernels on 134 MB measure
**278 GB/s**, and the second is the one that describes a GEMM streaming weights.
`bandwidth` refuses to report a figure without saying which regime it is.
"""
const L2_BYTES = 48 * 1024 * 1024

"""
    bandwidth(r::Result, bytes::Integer, workingset::Integer) -> NamedTuple

`(; gbps, resident)` for a `Result` that moved `bytes` per timed call over a
`workingset` of distinct memory.

**Both, because they are different numbers and conflating them is how the 693
GB/s mistake was made.** A sample that repeats a kernel 40 times moves 40x the
traffic over the SAME working set: divide by the traffic and you get a rate,
judge residency by the traffic and a 25 MB array "does not fit in L2" because it
was touched forty times. That combination reported 1447 GB/s — four times this
card's DRAM peak — labelled as a streaming figure.

`resident = workingset <= L2_BYTES`, and a resident figure is L2 bandwidth. Say
which one a number is whenever you write it down.
"""
function bandwidth(r::Result, bytes::Integer, workingset::Integer)
    r.kept > 0 || error("bandwidth: no trustworthy sample for $(r.label)")
    gbps = bytes / 1e9 / r.median
    # ~360 GB/s is this card's spec. Anything well past it is either an L2 figure
    # or an arithmetic slip, and both deserve to be said out loud.
    resident = workingset <= L2_BYTES
    (!resident && gbps > 400) && @warn "bandwidth: $(round(Int, gbps)) GB/s exceeds this " *
        "card's ~360 GB/s DRAM peak but the working set is $(round(workingset/1e6)) MB — " *
        "check whether `bytes` is per-call and `workingset` is distinct memory" r.label
    (; gbps, resident)
end

# ── repeating a short call within a sample: TRIED, and REMOVED ──────────────
#
# A call too short to hold the clock up gets every sample rejected — `neurallut`
# (a 256x256 classifier) reported `NaN ms, 0/11 kept`, which reads as a broken
# model. The obvious fix is to repeat the call within the sample until it is long
# enough, and divide.
#
# **It is not safe, because a model may be STATEFUL.** `matanyone-step` carries a
# five-frame memory bank; timing two steps per sample is not timing one step
# twice, and the row moved 36.57 -> 42.00 ms the moment repetition kicked in at a
# 50 ms target. A harness that silently changes the workload to make its own gate
# happy is worse than one that reports nothing.
#
# It did not even fix the case it was for: with repetition `neurallut` passes
# alone (4.25 ms, 11/11) and still returns `NaN` in the full sweep, at 5 ms and at
# 50 — `predictlut` takes a host `RGB{N0f8}` frame and does enough host work that
# the GPU idles *inside* the sample. **The gate is right and that model is
# host-bound.** `bench_all` prints the ungated median with a `!` instead of
# `NaN`, which says exactly that and changes no measurement.

function bench(f; samples::Int = 15, floor::Real = 0.90, warm::Bool = true,
               label::AbstractString = "", sync = nothing, plat = nothing,
               hold::Real = 0.05)
    g = sync === nothing ? f : (() -> (f(); sync()))
    g()                                     # compile, allocate, page in
    plat, smmax = plat !== nothing ? (plat, gpustate().smmax) :
                  warm ? plateau(g) : (gpustate().sm, gpustate().smmax)
    lo = floor * plat
    out = Sample[]
    for _ in 1:samples
        # ── hold, read, hold, time, read. Every step of that order is load-bearing.
        #
        # `gpustate` is a 26 ms `nvidia-smi` subprocess and the card is idle for
        # all of it, so a clock read placed anywhere near the timed region digs
        # the hole it then reports. The workload runs untimed on both sides of the
        # read: the first hold makes `a` a genuine under-load reading, the second
        # makes the timed region start on a busy card. With the read before the
        # first hold instead, every `depthanything` sample was rejected (0/11) by
        # a gate comparing an idle reading against a loaded plateau.
        #
        # `g` in a hold is outside `@elapsed`, so this changes how OFTEN the
        # workload runs, not what is measured — the distinction that makes it safe
        # for a stateful model where repeating inside the sample was not.
        a = gpustate(g; window = hold).sm
        # Take the collection the hold just earned, HERE, where it is not being
        # timed. Without it, whether a sample contains a GC is a coin flip:
        # `depthanything` read 73.20 ms ±216% and is 47 ms ±3%.
        GC.gc(false)
        # `b` covers the timed region itself. A sysfs read costs microseconds, so
        # it goes IMMEDIATELY after the timed region — before the card can drop —
        # and is then maxed against the following hold. The `nvidia-smi` form of
        # this had to start the probe before `@elapsed` and keep the card busy
        # until it answered, because the probe was slower than the thing being
        # measured; reading straight after is the same statement without the
        # inference.
        g0 = Base.gc_num().total_time
        t = @elapsed g()
        gc = (Base.gc_num().total_time - g0) / 1e9
        b = max(gpustate().sm, clockunder(g, hold).sm)
        push!(out, Sample(t, gc, a, b, a >= lo && b >= lo))
    end
    keep = [s for s in out if s.ok]
    isempty(keep) && return Result(label, NaN, NaN, 0, length(out), plat / smmax, NaN, SESSION, out)
    ts = sort([s.seconds for s in keep])
    med = median(ts)
    p(q) = ts[clamp(round(Int, q * length(ts)), 1, length(ts))]
    Result(label, med, (p(0.9) - p(0.1)) / med, length(keep), length(out) - length(keep),
           mean(s -> (s.smbefore + s.smafter) / 2, keep) / smmax,
           median([s.gc for s in keep]) / med, SESSION, out)
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
    # Only when it is worth acting on. A GC share above a few percent means the
    # number is partly host memory pressure and moves with the rest of the
    # process, not with the kernel under test.
    r.gc > 0.01 && @printf(io, "  GC %2.0f%%", 100 * r.gc)
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
    ra, rb = compare((fa, fb); samples, floor, labels, sync)
    ratio = ra.median / rb.median
    # The effect has to be bigger than the noise of BOTH arms to be an effect.
    trustworthy = ra.kept > 0 && rb.kept > 0 &&
                  abs(1 - ratio) > max(ra.spread, rb.spread)
    (ra, rb, ratio, trustworthy)
end

"""
    compare(fs::Tuple; samples = 15, labels, sync) -> Vector{Result}

Any number of arms, interleaved one sample of each per round.

Three arms is not "two comparisons": running `bench` once per arm gives each arm
its own stretch of the ramp, and on a card that idles at 7% of its clock the
first arm can be measured at 217 MHz and the third at 2170. That is not a
hypothetical — it produced a 1.75x here that was mostly the ramp, on a harness
written to prevent exactly this. `bench` is for **one** number; anything being
compared goes through this.
"""
function compare(fs::Tuple; samples::Int = 15, floor::Real = 0.90,
                 labels = ntuple(i -> "arm $i", length(fs)), sync = nothing,
                 hold::Real = 0.05)
    # NEW names. `fa = () -> (fa(); sync())` captures the *variable*, so the
    # closure calls itself — a StackOverflowError from a benchmark harness, which
    # is a confusing place to get one.
    gs = sync === nothing ? collect(fs) : [(() -> (f(); sync())) for f in fs]
    for g in gs; g(); end
    plat, smmax = plateau(() -> for g in gs; g(); end)
    lo = floor * plat
    acc = [Sample[] for _ in gs]
    # Same hold/GC discipline as `bench` — see there. An interleaved comparison is
    # not protected from either defect: both arms get measured inside the same
    # instrument hole, which compresses the ratio toward 1 rather than biasing it
    # one way, and that is the failure mode hardest to notice.
    for _ in 1:samples, (i, g) in enumerate(gs)
        x = gpustate(g; window = hold).sm
        GC.gc(false)
        g0 = Base.gc_num().total_time
        t = @elapsed g()
        gc = (Base.gc_num().total_time - g0) / 1e9
        y = max(gpustate().sm, clockunder(g, hold).sm)
        push!(acc[i], Sample(t, gc, x, y, x >= lo && y >= lo))
    end
    map(zip(acc, labels)) do (s, l)
        keep = [z for z in s if z.ok]
        isempty(keep) && return Result(l, NaN, NaN, 0, length(s), 0.0, NaN, SESSION, s)
        ts = sort([z.seconds for z in keep]); med = median(ts)
        p(q) = ts[clamp(round(Int, q * length(ts)), 1, length(ts))]
        Result(l, med, (p(0.9) - p(0.1)) / med, length(keep), length(s) - length(keep),
               mean(z -> (z.smbefore + z.smafter) / 2, keep) / smmax,
               median([z.gc for z in keep]) / med, SESSION, s)
    end
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
