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
Lava.enable_pipeline_executable_properties!()

# (M, N, K, share of the encoder's GEMM arithmetic). Four of the six are 72.7%
# of it; all are tile-aligned so none falls off the cooperative-matrix path.
const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]
const CUBLAS_TFLOPS = 44.6
const NBUF = 3          # A+B+C for the biggest shape is ~60 MB; 3 sets clears L2

tflops(M, N, K, secs) = 2.0 * M * N * K / secs / 1e12
smclock() = parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                        String))))

const BACKEND = LavaBackend()
const HEATW = KA.allocate(BACKEND, Float32, 1 << 22)
const HEATV = KA.allocate(BACKEND, Float32, 1 << 22)
const WS = DNNKernels.Workspace(BACKEND)

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
    for (M, N, K, share) in shapes
        hA = rand(Float16, M, K) .- Float16(0.5)
        hB = rand(Float16, K, N) .- Float16(0.5)
        As = [KA.allocate(BACKEND, Float16, M, K) for _ in 1:NBUF]
        Bs = [KA.allocate(BACKEND, Float16, K, N) for _ in 1:NBUF]
        Cs = [KA.allocate(BACKEND, Float16, M, N) for _ in 1:NBUF]
        foreach(a -> copyto!(a, hA), As); foreach(b -> copyto!(b, hB), Bs)
        pick(x, r) = @inbounds x[mod1(r, NBUF)]
        fs = [r -> (set(); DNNKernels.reset!(WS);
                    DNNKernels.matmul!(pick(Cs, r), pick(As, r), pick(Bs, r), nothing; ws = WS))
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
        tf = [tflops(M, N, K, x) for x in timedall(fs; n, reps)]
        tot .+= share .* tf
        @printf("%-20s %5.1f%%", "$(M)x$(N)x$(K)", share)
        for (i, x) in enumerate(tf)
            @printf(" %9s", (check && errs[i] > 2e-2) ? "ERR" : @sprintf("%.1f", x))
        end
        @printf("   %d\n", smclock())
        As = Bs = Cs = nothing; GC.gc()
    end
    @printf("%-20s %6s", "weighted mean", "")
    for x in tot; @printf(" %9.1f", x / wsum); end
    @printf("   (cuBLAS %.1f)\n", CUBLAS_TFLOPS)
    tot ./ wsum
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
    before = Set(keys(Lava.PIPELINE_CACHE))
    setup(); DNNKernels.reset!(WS)
    DNNKernels.matmul!(C, A, B, nothing; ws = WS)
    KA.synchronize(BACKEND)
    fresh = [k for k in keys(Lava.PIPELINE_CACHE) if !(k in before)]
    isempty(fresh) && return nothing
    [Lava.pipeline_exec_stats(Lava.PIPELINE_CACHE[k]) for k in fresh]
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
