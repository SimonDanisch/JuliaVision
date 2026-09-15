"""
Bench harness for the GEMM port, as an `include`-able file rather than a script.

    julia> include("tools/gemm_lab.jl")
    julia> warmclock(); bench(["direct" => () -> (Lava.GEMM_STAGED[] = false), ...])

`tools/gemm_bench.jl` runs one fixed comparison and exits; iterating on a tiling
means redefining `@kernel`s, which Revise cannot hot-load, so the session gets
restarted often and everything below has to come back in one call.

Two things here that a naive harness gets wrong, both learned the hard way:

  * **the clock**. This card idles at 210 MHz of 2265, cannot be locked without
    root, and takes several seconds of sustained load to boost — `warmclock()`
    below needs six rounds of 4000 broadcast launches. A measurement taken at
    825 MHz reads as a 2.7x regression.
  * **the workspace**. `DNNKernels.Workspace` is a bump allocator and only
    `reset!` frees it; without a reset per call a benchmark loop takes fresh
    split-K planes every launch and OOMs at 14 GB.

`enable_pipeline_executable_properties!` must run before the `VkContext` exists,
which is why it is the first line and why this file is included into a fresh
session rather than evaluated into a running one.
"""

using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf, Statistics
const KA = KernelAbstractions
Mantle.enable_pipeline_executable_properties!()

# (M, N, K, share of the encoder's GEMM arithmetic). Four of the six are 72.7%
# of it; all are tile-aligned so none falls off the cooperative-matrix path.
const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]
"""cuBLAS on THIS card at the shape mix above, **measured** — 2026-08-12,
`gemm_vs_cublas.jl`: CUDA.jl `mul!` on `CuArray{Float16}` interleaved with our
kernel in one process, correctness checked between the arms, clock pinned at
2175 MHz for every row.

    M x N x K              ours    cuBLAS   of cuBLAS
    2304 x 4096 x  576    41.91     67.12     62%
     576 x 4096 x 2304    40.89     65.27     63%
    1728 x 4096 x  576    41.41     62.96     66%
     576 x 4096 x  576    33.88     46.05     74%
     288 x 16384 x 1152   39.76     52.28     76%
    1152 x 16384 x  288   33.76     55.03     61%
    share-weighted        40.4      62.7      64%

**This constant used to be 44.6 with no provenance and it understated cuBLAS by
40%**, which turned a 1.55x gap into a 1.18x one and made the GEMM look nearly
finished. It is a single number for six shapes, so it is kept only as the
headline; the per-shape table above is the baseline to compare against, and
`gemm_vs_cublas.jl` re-measures it rather than trusting either."""
const CUBLAS_TFLOPS = 62.7

"""SM clock below which a row is not a measurement of the kernel. The card
boosts to ~2175 of a nominal 2265 and idles at 210; anything between is the
clock ramping, and time measured there is wrong in proportion. Sampled with the
queue full — see `bench` — because an idle card reads 210 within a second of the
last kernel and would condemn every row."""
const CLOCK_FLOOR = 2000
const NBUF = 3          # A+B+C for the biggest shape is ~60 MB; 3 sets clears L2

tflops(M, N, K, secs) = 2.0 * M * N * K / secs / 1e12
smclock() = parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                        String))))

const BACKEND = LavaBackend()
const HEATW = KA.allocate(BACKEND, Float32, 1 << 22)
const HEATV = KA.allocate(BACKEND, Float32, 1 << 22)
const WS = DNNKernels.Workspace(BACKEND)
# The kernel entry points take a context; `Ctx(backend; ws)` is the no-graph form
# a direct caller uses, and it carries the workspace these benchmarks reset.
const CTX = DNNKernels.Ctx(BACKEND; ws = WS)

heat(k = 200) = (for _ in 1:k; HEATW .= HEATV .* 1.0001f0 .+ 0.5f0; end)

"Boost the SM clock and report where it landed; anything under ~2000 invalidates
the absolute TFLOP/s (the ratios survive, which is why variants interleave)."
function warmclock(rounds = 8)
    c = 0
    for _ in 1:rounds
        for _ in 1:10; heat(400); end
        KA.synchronize(BACKEND)
        (c = smclock()) >= 2200 && break
    end
    c
end

"Median of `n` interleaved samples per variant; a sample is `reps` launches
with one sync, so the per-launch overhead is amortised the way the encoder's is."
function timedall(fs; n = 11, reps = 8)
    for f in fs; f(1); end; KA.synchronize(BACKEND)
    for r in 1:3, f in fs; f(r); end; KA.synchronize(BACKEND)
    ts = [Float64[] for _ in fs]
    for _ in 1:n
        heat(60)
        for (i, f) in enumerate(fs)
            KA.synchronize(BACKEND)
            t0 = time_ns()
            for r in 1:reps; f(r); end
            KA.synchronize(BACKEND)
            push!(ts[i], (time_ns() - t0) / 1e9 / reps)
        end
    end
    map(median, ts)
end

"""
    bench(["name" => setup, ...]) -> weighted TFLOP/s per variant

Each `setup` is a nullary function run immediately before every launch — it sets
the `Ref`s that pick a kernel. Correctness is checked against a Float32 CPU
reference on the first 64 rows; a variant that reads `ERR` computed something
else and its timing means nothing.
"""
function bench(variants::Vector{<:Pair}; shapes = SHAPES, n = 11, reps = 8, check = true)
    @printf("%-20s %6s", "M x N x K", "share")
    for (name, _) in variants; @printf(" %9s", name); end
    println("   MHz")
    tot = zeros(length(variants))
    wsum = sum(s for (_, _, _, s) in shapes)
    wok = 0.0                       # share whose row was actually on the plateau
    slow = Tuple{String,Int}[]
    for (M, N, K, share) in shapes
        hA = rand(Float16, M, K) .- Float16(0.5)
        hB = rand(Float16, K, N) .- Float16(0.5)
        As = [KA.allocate(BACKEND, Float16, M, K) for _ in 1:NBUF]
        Bs = [KA.allocate(BACKEND, Float16, K, N) for _ in 1:NBUF]
        Cs = [KA.allocate(BACKEND, Float16, M, N) for _ in 1:NBUF]
        foreach(a -> copyto!(a, hA), As); foreach(b -> copyto!(b, hB), Bs)
        pick(x, r) = @inbounds x[mod1(r, NBUF)]
        # `matmul!(ctx, out, A, B, bias)` — it grew a context argument in the
        # `Ctx` refactor and this file was never updated, so both call sites here
        # had been `MethodError`s. Found 2026-08-11; see `kernelstats` below.
        fs = [r -> (set(); DNNKernels.reset!(WS);
                    DNNKernels.matmul!(CTX, pick(Cs, r), pick(As, r), pick(Bs, r), nothing))
              for (_, set) in variants]
        errs = Float64[]
        if check
            rows = 1:min(M, 64)
            ref = Float32.(hA[rows, :]) * Float32.(hB)
            for f in fs
                fill!(Cs[1], zero(Float16)); f(1); KA.synchronize(BACKEND)
                got = Float32.(Array(Cs[1])[rows, :])
                push!(errs, maximum(abs.(got .- ref)) / max(1f-6, maximum(abs.(ref))))
            end
        end
        # RE-BOOST PER SHAPE. `warmclock()` once at the top is not enough: the
        # host-side setup above (allocate, `rand`, `copyto!`, the CPU reference)
        # leaves the card idle for seconds and it falls back toward 210 MHz. A
        # single-warm run of this table read 3.9 TF/s at 390 MHz for the largest
        # shape and 19.9 at 1245 for another, then published a weighted mean of
        # 27.6 against a true ~39 — the two contaminated rows carried it.
        # Boost, and read the clock WHILE THE CARD IS STILL BUSY. `heat` launches
        # asynchronously, so the CPU reaches `smclock()` with the queue full and
        # nvidia-smi samples the clock the kernels are actually running at.
        # Reading it after `timedall` instead samples an idle card: the first
        # version of this gate reported 210 MHz for a row that measured 38.9
        # TF/s and excluded every row in the table.
        # Three rounds is ~90 ms of traffic and `nvidia-smi` takes longer than
        # that to spawn, so the FIRST shape of a table still reads 210 — the
        # burst is over before the sample lands. Later shapes read 2175 because
        # the card has not fully come down between them. A row excluded at 210
        # is therefore suspect as a *reading*, not as a measurement: the 210 row
        # has measured 34.4, 38.9 and 3.9 TF/s across three runs of this table,
        # and only the 3.9 was genuinely off the plateau.
        for _ in 1:6; heat(400); end
        mhz = smclock()
        KA.synchronize(BACKEND)
        tf = [tflops(M, N, K, x) for x in timedall(fs; n, reps)]
        ok = mhz >= CLOCK_FLOOR
        ok || push!(slow, ("$(M)x$(N)x$(K)", mhz))
        # A row taken off the plateau is EXCLUDED from the mean rather than
        # printed with a caveat, because a mean is what gets quoted.
        ok && (tot .+= share .* tf; wok += share)
        @printf("%-20s %5.1f%%", "$(M)x$(N)x$(K)", share)
        for (i, x) in enumerate(tf)
            @printf(" %9s", (check && errs[i] > 2e-2) ? "ERR" : @sprintf("%.1f", x))
        end
        @printf("   %d%s\n", mhz, ok ? "" : "  <-- OFF THE PLATEAU, excluded")
        flush(stdout)
        As = Bs = Cs = nothing; GC.gc()
    end
    if wok == 0
        println("EVERY row was taken below $(CLOCK_FLOOR) MHz — no mean to report.")
        return fill(NaN, length(variants))
    end
    @printf("%-20s %6s", "weighted mean", "")
    for x in tot; @printf(" %9.1f", x / wok); end
    @printf("   (cuBLAS %.1f)", CUBLAS_TFLOPS)
    wok < wsum && @printf("  [%.0f%% of arithmetic; %d row(s) dropped]",
                          100 * wok / wsum, length(slow))
    println()
    for (nm, mhz) in slow
        @printf("   dropped %-20s at %d MHz\n", nm, mhz)
    end
    tot ./ wok
end

"""
    kernelstats(setup; M, N, K) -> NamedTuple

Register count, spill/scratch bytes and shared memory for whichever kernel
`setup` selects, from `VK_KHR_pipeline_executable_properties`.

This is the one hardware fact reachable without Nsight, and it is the one that
decides a tiling: a cooperative-matrix accumulator block that does not fit in
registers is spilled to local memory by the driver, silently, and the kernel
then runs at a quarter speed with no other symptom.
"""
function kernelstats(setup; M = 2304, N = 4096, K = 576)
    A = KA.allocate(BACKEND, Float16, M, K); fill!(A, Float16(0.01))
    B = KA.allocate(BACKEND, Float16, K, N); fill!(B, Float16(0.01))
    C = KA.allocate(BACKEND, Float16, M, N)
    # `ctx.caches.pipelines`, not the module-level `Lava.PIPELINE_CACHE`: that was
    # one of the twelve globals that moved onto the context, and this function
    # had been throwing `UndefVarError` ever since. Found 2026-08-11 by calling
    # it. A lab tool nobody calls rots exactly like a test nobody runs.
    pipes() = Mantle.vk_context().caches.pipelines
    before = Set(keys(pipes()))
    setup(); DNNKernels.reset!(WS)
    DNNKernels.matmul!(CTX, C, A, B, nothing)
    KA.synchronize(BACKEND)
    fresh = [k for k in keys(pipes()) if !(k in before)]
    isempty(fresh) && return nothing
    [Mantle.pipeline_exec_stats(pipes()[k]) for k in fresh]
end

"Print the driver's statistics for every pipeline `setup` newly compiles."
function showstats(name, setup; kw...)
    st = kernelstats(setup; kw...)
    st === nothing && return println(name, ": no new pipeline (already compiled)")
    for s in st
        s === nothing && continue
        vals = join([string(r.name, "=", r.value) for r in s.raw_stats
                     if !(r.value isa Bool)], "  ")
        println(name, ": ", vals)
    end
end

println("gemm_lab ready — warmclock(), bench([...]), kernelstats(setup)")
