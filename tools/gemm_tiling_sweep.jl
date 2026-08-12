# Which tiling is fastest AT EACH SHAPE, against cuBLAS measured beside it.
#
# The shipped kernel is one tiling (96 x 128, 8 warps) for all six of SAM 2's
# `addmm` shapes and runs at a flat 61-76% of cuBLAS. `GEMM_TILINGS` now carries
# four candidates past the shipped set; this forces each and reports per shape.
#
# **Per shape, never a weighted mean.** This exact file's own history is the
# reason: a mean over these shapes once ranked 96x128 below 64x128 when 96x128 is
# faster on four of the six — one shape lost by 18% and dragged the average.
#
# Rules, all earned: interleaved arms in one process so a clock ramp hits every
# tiling alike; correctness against cuBLAS checked before any time is believed;
# a tiling whose block does not divide the shape is SKIPPED rather than forced,
# because `gemm_tiling` would otherwise read past the operands and report a time
# for work it did not do.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf
using CUDA
const KA = KernelAbstractions
const DK = DNNKernels

const BACKEND = LavaBackend()
const WS = DK.Workspace(BACKEND)

const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]

label(c) = @sprintf("%dx%d/%dw%s/p%d", Lava.gemm_bm(c), Lava.gemm_bn(c),
                    Lava.gemm_wg(c) ÷ 32, c[5] == 32 ? "" : "/k$(c[5])", c[6])

function sweep(M, N, K, share)
    g = 2.0 * M * N * K / 1e9
    a = Float16.(0.05f0 .* randn(Float32, M, K))
    b = Float16.(0.05f0 .* randn(Float32, K, N))
    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, a)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, b)
    C = KA.allocate(BACKEND, Float16, M, N)
    cA = CuArray(a); cB = CuArray(b); cC = CUDA.zeros(Float16, M, N)

    # Only the tilings whose block divides this shape AND that the aliasing rule
    # does not decline — the same two gates the chooser applies, so a row here is
    # a tiling the shipped path could actually select.
    cands = [c for c in Lava.GEMM_TILINGS
             if Lava.gemm_divides(c, M, N, K) && !Lava.gemm_aliasing(c, K) &&
                haskey(Lava.GEMM_STAGED_KERNELS, c)]

    @printf("\n%d x %d x %d   (%.1f GFLOP, %.1f%% of the encoder's GEMM)\n", M, N, K, g, share)
    isempty(cands) && (println("   no staged tiling divides this shape"); return)

    run(c) = () -> (DK.reset!(WS);
                    Lava.coopmat_gemm!(C, A, B, M, N, K; tiling = c))
    # Annotated: a comprehension fixes the element type to the first closure's,
    # and the cuBLAS arm is a different closure type.
    arms = Tuple{String,Function}[(label(c), run(c)) for c in cands]
    push!(arms, ("cuBLAS", () -> mul!(cC, cA, cB)))

    for _ in 1:3, (_, f) in arms; f(); end
    KA.synchronize(BACKEND); CUDA.synchronize()

    # Correctness against cuBLAS, per tiling, before timing.
    ref = Array(cC)
    scale = max(maximum(abs.(Float32.(ref))), eps())
    errs = Float64[]
    for c in cands
        fill!(C, Float16(NaN)); DK.reset!(WS)
        Lava.coopmat_gemm!(C, A, B, M, N, K; tiling = c)
        KA.synchronize(BACKEND)
        push!(errs, maximum(abs.(Float32.(Array(C)) .- Float32.(ref))) / scale)
    end

    acc = [Float64[] for _ in arms]
    for _ in 1:9, (i, (_, f)) in enumerate(arms)
        t0 = time_ns(); for _ in 1:8; f(); end
        i == length(arms) ? CUDA.synchronize() : KA.synchronize(BACKEND)
        push!(acc[i], (time_ns() - t0) / 1e6 / 8)
    end
    best = map(minimum, acc)
    cub = best[end]
    for (i, (nm, _)) in enumerate(arms)
        e = i <= length(errs) ? errs[i] : 0.0
        ok = i > length(errs) || e < 5e-2
        @printf("   %-16s %8.4f ms  %7.2f TF/s  %5.0f%% of cuBLAS%s\n",
                nm, best[i], g / best[i], 100 * cub / best[i],
                ok ? "" : @sprintf("   relerr %.1e <-- WRONG", e))
    end
    A = B = C = nothing; cA = cB = cC = nothing; GC.gc()
    flush(stdout)
end

for (M, N, K, share) in SHAPES; sweep(M, N, K, share); end
println("\nSWEEPDONE")
