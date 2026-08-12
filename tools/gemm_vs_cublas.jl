# Our GEMM against cuBLAS, PER SHAPE, in ONE process, interleaved.
#
# `gemm_lab.jl` compares against `CUBLAS_TFLOPS = 44.6` — a single constant for
# every shape, of unstated provenance. The in-situ kernel table says PyTorch
# achieves **54.1 TF/s** over the same 195 `addmm` calls. Those cannot both be
# the baseline, and the difference (21%) is larger than the win any tuning round
# has produced, so the comparison has to be measured rather than remembered.
#
# Three rules, all of them earned:
#
#   * **Interleaved.** Vulkan and CUDA in the same process, alternating per rep,
#     so a clock ramp or a background job hits both arms equally. Absolute
#     milliseconds on this machine drift up to 2x between runs; paired ratios do
#     not.
#   * **Correctness first.** Both arms are checked against each other before any
#     time is believed. An arm computing the wrong product is not fast.
#   * **Per shape.** The encoder's GEMM arithmetic is 72.7% four shapes; a
#     weighted mean over them can rank two kernels backwards when one falls off
#     a cliff at a single shape (`gemm-tiling-aliasing-cliff`).
#
# cuBLAS is reached through CUDA.jl's `mul!` on `CuArray{Float16}`, which calls
# `gemmEx` with an fp32 accumulator — the same arrangement our kernel uses
# (fp16 operands, fp32 accumulate), and the same one PyTorch's `addmm` takes for
# an fp16 autocast graph. It is NOT the same as allowing fp16 accumulation,
# which would be faster and is not what either side is doing here.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf, Statistics
using CUDA
const KA = KernelAbstractions
const DK = DNNKernels

# (M, N, K, share of the encoder's GEMM arithmetic). Counted from the graph by
# `gemm_shape_census.jl`, not copied: 22 distinct shapes over 195 calls, and
# these six are 79% of the arithmetic.
const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]

const BACKEND = LavaBackend()
const WS = DK.Workspace(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

smclock() = parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                        String))))

"""Both arms on the same data, timed alternately.

Returns `(ours_ms, cublas_ms, relerr, mhz)`. `n` samples of `reps` launches, and
the arms alternate WITHIN the sample loop rather than being timed in separate
blocks — a first-arm-absorbs-the-ramp error read a 2.305 ms baseline for a 0.235
ms configuration once already."""
function pair(M, N, K; n = 9, reps = 8)
    a = Float16.(0.05f0 .* randn(Float32, M, K))
    b = Float16.(0.05f0 .* randn(Float32, K, N))

    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, a)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, b)
    C = KA.allocate(BACKEND, Float16, M, N)
    cA = CuArray(a); cB = CuArray(b); cC = CUDA.zeros(Float16, M, N)

    ours() = (DK.reset!(WS); DK.matmul!(CTX, C, A, B, nothing))
    theirs() = mul!(cC, cA, cB)

    # Warm both pipelines, then check they agree before timing either.
    for _ in 1:3; ours(); theirs(); end
    KA.synchronize(BACKEND); CUDA.synchronize()
    go = Array(C); gt = Array(cC)
    relerr = maximum(abs.(Float32.(go) .- Float32.(gt))) / max(maximum(abs.(Float32.(gt))), eps())

    to = Float64[]; tt = Float64[]
    for _ in 1:n
        t0 = time_ns(); for _ in 1:reps; ours(); end
        KA.synchronize(BACKEND); push!(to, (time_ns() - t0) / 1e6 / reps)
        t0 = time_ns(); for _ in 1:reps; theirs(); end
        CUDA.synchronize(); push!(tt, (time_ns() - t0) / 1e6 / reps)
    end
    mhz = smclock()
    A = B = C = nothing; cA = cB = cC = nothing; GC.gc()
    (minimum(to), minimum(tt), relerr, mhz)
end

function main()
    @printf("%-22s %6s %9s %9s %9s %9s %7s %8s %5s\n",
            "M x N x K", "share", "ours ms", "cuBLAS", "ourTF/s", "cuTF/s", "of cu", "relerr", "MHz")
    wo = 0.0; wt = 0.0
    for (M, N, K, share) in SHAPES
        g = 2.0 * M * N * K / 1e9
        mo, mt, e, mhz = pair(M, N, K)
        wo += share * g / mo; wt += share * g / mt
        @printf("%-22s %5.1f%% %9.3f %9.3f %9.2f %9.2f %6.0f%% %8.1e %5d %s\n",
                "$(M) x $(N) x $(K)", share, mo, mt, g / mo, g / mt,
                100 * (g / mo) / (g / mt), e, mhz, e < 5e-2 ? "" : "<-- CHECK")
        flush(stdout)
    end
    # Weighted by share, so the summary is the encoder's mix and not the mean of
    # six unrelated shapes.
    s = sum(x[4] for x in SHAPES)
    @printf("\nshare-weighted: ours %.1f TF/s, cuBLAS %.1f TF/s -> %.0f%%\n",
            wo / s, wt / s, 100 * wo / wt)
end

main()
