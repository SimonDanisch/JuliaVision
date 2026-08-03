"""
Soak for the intermittent flush hang (`STATUS.md`, open bugs).

    julia -t 4,1 --project=. tools/soak_flush_hang.jl [trials]

**What it is trying to provoke.** `Lava/src/runtime/memory.jl` records the
dominant path as fixed — a buffer freed while a batch is *recording* but never
submitted read `last_write === nothing` and fell through to immediate
destruction with an open command buffer still naming it. The queue then wedged
on `vkWaitSemaphores` with batches holding timeline values nothing signals.

The same note qualifies the fix: **the hang was seen once more afterwards**,
under `with_dispatch_timing`, against roughly 90 clean trials across every
reproduction that used to fail in ten or fewer. So this is a trials problem, not
an attention problem, and the only useful thing to do with it is run it a lot.

**Why not SAM 2 decode.** That is the recorded reproduction (60 probe-decodes
with the collector live hung within 15), but SAM 2 is not in `tools/models.py`'s
registry and its graph is not fetchable, so this machine has no decode to run.
What the bug actually turns on is a *buffer lifetime* — a GC landing inside a
recording — and that is reachable directly. This drives the same state machine
from `test_free_during_recording.jl`, but as a race rather than an assertion:
allocations dropped mid-recording, the collector left live to land where it
lands, and the dispatch timing that the one recurrence carried.

**How a hang is reported.** The main thread blocks inside `vkWaitSemaphores`, so
it cannot report on itself. A watchdog on the interactive thread pool holds the
trial's start time; if a trial outlives `TRIAL_TIMEOUT` it prints
`Lava.flush_stall_report` — which is the diagnostic that names a wait on a value
nothing will signal — plus the deferred-free and batch state, and exits non-zero.
A clean run just prints a heartbeat per trial, so a stalled log is itself signal.
"""

using Lava, KernelAbstractions, Printf, Dates
const KA = KernelAbstractions

const TRIALS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)
const TRIAL_TIMEOUT = Second(90)
const BUFS_PER_TRIAL = 12       # ~matches the VolPathState count the MWEs used
const N = 4096

@kernel function _fill!(out, val)
    i = @index(Global)
    out[i] = val
end

@kernel function _axpy!(out, a, x)
    i = @index(Global)
    out[i] = out[i] + a * x[i]
end

backend = LavaBackend()
ctx = Lava.vk_context()
bq = ctx.default_bq

# Watchdog state. `trial_started` is the wall clock at the top of the current
# trial; the watchdog compares against it and never touches Vulkan itself.
const trial_no = Threads.Atomic{Int}(0)
const trial_started = Ref(now())
const finished = Threads.Atomic{Bool}(false)

function dump_hang(n)
    println()
    println("=== HANG DETECTED on trial $n after $(TRIAL_TIMEOUT) ===")
    println("time: ", now())
    try
        # The target is what the flush is waiting for; `next_timeline - 1` is the
        # last value handed out, which is what a stalled flush is blocked on.
        target = UInt64(max(bq.next_timeline - 1, 0))
        print(Lava.flush_stall_report(bq, target))
    catch e
        println("  flush_stall_report threw: ", e)
    end
    try
        println("  deferred_frees = ", length(bq.deferred_frees))
        ab = bq.active_batch
        println("  active_batch = ", ab === nothing ? "nothing" :
                "recording=$(ab.recording) signal=$(ab.signal_value)")
        println("  in_flight = ", length(bq.in_flight))
    catch e
        println("  batch state unreadable: ", e)
    end
    # `stacktrace_from_all_threads` is not in this Julia (1.12), so the blocked
    # main thread cannot be walked from here. SIGQUIT makes the runtime print
    # every thread's backtrace itself, which is the same information and is what
    # names the `vkWaitSemaphores` frame.
    println("=== all-thread backtrace via SIGQUIT ===")
    flush(stdout)
    ccall(:raise, Cint, (Cint,), 3)
    sleep(2)
    flush(stdout)
    exit(2)
end

Threads.@spawn :interactive begin
    while !finished[]
        sleep(5)
        n = trial_no[]
        if n > 0 && (now() - trial_started[]) > TRIAL_TIMEOUT
            dump_hang(n)
        end
    end
end

"""
One trial: allocate, dispatch, drop mid-recording, collect, flush.

The order matters and each step is one of the conditions in the bug note.
`ensure_active_batch!` leaves a batch open so every free below lands *during a
recording*; the arrays that are never dispatched against are the ones whose
`last_write` is still `nothing`, which is the case that fell through.
"""
function trial!(backend, bq, timed::Bool)
    body = function ()
        Lava.ensure_active_batch!(bq)

        live = Lava.LavaArray{Float32,1}[]
        for k in 1:BUFS_PER_TRIAL
            a = KA.allocate(backend, Float32, N)
            fill!(a, Float32(k))
            # Half get a dispatch (so `last_write` is set at submit), half never
            # do (so `last_write` stays `nothing` — the case the fix added).
            if iseven(k)
                _fill!(backend)(a, Float32(k); ndrange=N)
            end
            push!(live, a)
        end

        # Drop half the references without freeing them, then collect *inside*
        # the recording. This is the GC-lands-in-a-recording shape; the
        # finalizer runs `unsafe_free!` from the finalizer thread with the batch
        # still open.
        for _ in 1:(BUFS_PER_TRIAL ÷ 2)
            popfirst!(live)
        end
        GC.gc(false)

        # Keep recording against what is left, so the open batch is still naming
        # buffers while those finalizers run.
        for a in live
            _axpy!(backend)(a, 2.0f0, a; ndrange=N)
        end

        # Explicit frees under the same open batch.
        for a in live
            Lava.unsafe_free!(a)
        end
        empty!(live)
        GC.gc(false)

        KA.synchronize(backend)
        Lava.drain_deferred_frees!(bq)
    end
    timed ? Lava.with_dispatch_timing(body) : body()
    return nothing
end

println("soak: flush hang — ", TRIALS == typemax(Int) ? "unbounded" : "$TRIALS trials",
        ", timeout $(TRIAL_TIMEOUT)/trial, pid $(getpid())")
println("device: ", ctx.device_name)
flush(stdout)

function run_soak(backend, bq)
    t0 = now()
    n = 0
    try
        while n < TRIALS
            n += 1
            trial_no[] = n
            trial_started[] = now()
            # Every third trial carries dispatch timing: that is what the one
            # post-fix recurrence was running under, and it is the only
            # condition the fix note singles out as still open.
            trial!(backend, bq, n % 3 == 0)
            if n % 10 == 0
                @printf("trial %d ok  (%s elapsed, deferred=%d)\n",
                        n, Dates.canonicalize(now() - t0), length(bq.deferred_frees))
                flush(stdout)
            end
        end
    finally
        finished[] = true
    end
    @printf("soak clean: %d trials in %s\n", n, Dates.canonicalize(now() - t0))
    flush(stdout)
end

run_soak(backend, bq)
