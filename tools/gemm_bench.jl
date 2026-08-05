"""
The GEMM, in isolation and per shape, so a tiling can be iterated in seconds
instead of one encode at a time.

    julia --project=. tools/gemm_bench.jl

Matmul is **56% of the remaining gap to PyTorch** — 118.9 ms of our 235.1 ms
encode against `addmm`'s 36.0, i.e. cuBLAS at 44.6 TFLOP/s against our 14-22.
`perf-plan.md`'s ledger points at `reference/llama.cpp-vulkan/mul_mm.comp` for
it, and porting a tiling without a per-shape measurement is how the last staged
GEMM got built, measured once end-to-end, and switched off.

The shapes are the encoder's own, from `perf-plan.md`: four of them are 72.7% of
its arithmetic, and all are tile-aligned (`M,N,K ≡ 0 mod 16`) so none falls off
the cooperative-matrix path.

What this measures, per shape, interleaved in one session:

  * **TFLOP/s** for each variant, against cuBLAS's 44.6 on this card.
  * **correctness**, against a Float32 CPU reference at `1e-2` — a faster GEMM
    that multiplies different numbers is not a GEMM, and fp16 accumulation makes
    an exact comparison meaningless.
  * `GEMM_STAGED[]` **on and off in the same session**, because that switch is
    the one existing A/B and its recorded result (1.03x/0.56x) predates every
    change since.

The clock discipline is `permute_bench.jl`'s and is not optional: this card idles
at 210 MHz of 2265, drifts *during* a run, and cannot be locked without root. A
sample is `reps` back-to-back launches with one sync, variants are round-robin,
and `heat` runs between rounds.
"""

using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf, Statistics
const KA = KernelAbstractions

# (M, N, K, calls per encode, share of the encoder's GEMM arithmetic)
const SHAPES = [(2304, 4096,  576, 36, 24.4),
                ( 576, 4096, 2304, 36, 24.4),
                (1728, 4096,  576, 35, 17.8),
                ( 576, 4096,  576, 36,  6.1),
                ( 288, 16384, 1152, 6,  4.1),
                (1152, 16384,  288, 6,  4.1)]

"cuBLAS on this card, for the ratio column. From `tools/sam2_pytorch_kernels.py`."
const CUBLAS_TFLOPS = 44.6

const NBUF = 3          # A+B+C for the biggest shape is ~60 MB; 3 sets clears L2

tflops(M, N, K, secs) = 2.0 * M * N * K / secs / 1e12

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
Median of `n` samples per variant, interleaved; each sample is `reps` launches
with one sync. See `permute_bench.jl` for why both of those matter.
"""
function timedall(fs, backend; n = 15, reps = 8, heat = nothing)
    for f in fs; f(1); end; KA.synchronize(backend)
    for r in 1:3, f in fs; f(r); end; KA.synchronize(backend)
    ts = [Float64[] for _ in fs]
    for _ in 1:n
        heat === nothing || heat()
        for (i, f) in enumerate(fs)
            KA.synchronize(backend)
            t0 = time_ns()
            for r in 1:reps; f(r); end
            KA.synchronize(backend)
            push!(ts[i], (time_ns() - t0) / 1e9 / reps)
        end
    end
    map(median, ts)
end

function main()
    backend = LavaBackend()
    w = KA.allocate(backend, Float32, 1 << 22)
    v = KA.allocate(backend, Float32, 1 << 22)
    heat(k = 200) = (for _ in 1:k; w .= v .* 1.0001f0 .+ 0.5f0; end)
    for _ in 1:20
        heat(); KA.synchronize(backend)
        smclock() >= 1800 && break
    end

    @printf("%-22s %6s %6s %8s %8s %8s %8s %7s  %s\n",
            "M x N x K", "calls", "share", "direct", "staged", "best", "of cuBLAS",
            "MHz", "correct")
    for (M, N, K, calls, share) in SHAPES
        hA = rand(Float16, M, K) .- Float16(0.5)
        hB = rand(Float16, K, N) .- Float16(0.5)
        As = [KA.allocate(backend, Float16, M, K) for _ in 1:NBUF]
        Bs = [KA.allocate(backend, Float16, K, N) for _ in 1:NBUF]
        Cs = [KA.allocate(backend, Float16, M, N) for _ in 1:NBUF]
        foreach(a -> copyto!(a, hA), As)
        foreach(b -> copyto!(b, hB), Bs)
        pick(x, r) = @inbounds x[mod1(r, NBUF)]

        # `DNNKernels.matmul!`, which is the path the encoder takes, and neither
        # of the two obvious alternatives:
        #
        #   * `Lava.coopmat_gemm!` bare computes something else entirely — its
        #     `C` must be **fp32 with `splitk` planes** and it is called with
        #     `partials = C, reduce = false`. Handed an fp16 `M x N` it returned
        #     18946 NaNs and values to 6.55e4 against a reference whose largest
        #     element is 7.87.
        #   * `LinearAlgebra.mul!` takes the **scalar** fallback here: Lava's own
        #     cooperative-matrix path wants an fp32 destination and autocast
        #     writes fp16. That is 1.3 TFLOP/s, and it is why `matmul_coopmat!`
        #     exists.
        #
        # Timing `matmul!` also puts `mm_epilogue_kernel!` inside the
        # measurement, which is right: it is 25.3 ms of the encode, exists only
        # because our GEMM cannot fold bias and split-K itself, and the port is
        # supposed to delete it.
        ws = DNNKernels.Workspace(backend)
        run(staged) = r -> begin
            Lava.GEMM_STAGED[] = staged
            # Per op, as the graph executor does. `scratch!` is a bump allocator
            # over the workspace and only `reset!` frees it; without this each
            # launch takes fresh split-K planes and the loop OOMs at 14 GB.
            DNNKernels.reset!(ws)
            DNNKernels.matmul!(pick(Cs, r), pick(As, r), pick(Bs, r), nothing; ws)
        end
        # A Float32 CPU reference: fp16 accumulation means the GPU result is not
        # bit-comparable to anything, so this is a tolerance check on a slice —
        # the whole product at these sizes is minutes of CPU time.
        rows = 1:min(M, 64)
        ref = Float32.(hA[rows, :]) * Float32.(hB)

        ok = map((false, true)) do staged
            fill!(Cs[1], zero(Float16))
            Lava.GEMM_STAGED[] = staged
            DNNKernels.reset!(ws)
            DNNKernels.matmul!(Cs[1], As[1], Bs[1], nothing; ws)
            KA.synchronize(backend)
            got = Float32.(Array(Cs[1])[rows, :])
            maximum(abs.(got .- ref)) / max(1f-6, maximum(abs.(ref)))
        end

        t = timedall([run(false), run(true)], backend; heat = () -> heat(60))
        Lava.GEMM_STAGED[] = false
        direct, staged = tflops(M, N, K, t[1]), tflops(M, N, K, t[2])
        best = max(direct, staged)
        @printf("%-22s %6d %5.1f%% %8.1f %8.1f %8.1f %7.0f%% %7d  %s\n",
                "$(M) x $(N) x $(K)", calls, share, direct, staged, best,
                100best / CUBLAS_TFLOPS, smclock(),
                all(<(2f-2), ok) ? "ok" : "REL ERR " * string(round.(ok; digits = 4)))
        As = Bs = Cs = nothing
        GC.gc()
    end
    println("\nSM clock: ", smclock(), " MHz   (idles at 210 of 2265 — a low reading ",
            "invalidates the absolute TFLOP/s, not the ratios)")
    println("cuBLAS on this card: ", CUBLAS_TFLOPS, " TFLOP/s")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
