# Is the 64%-of-cuBLAS number the KERNEL, or is some of it our harness?
#
# `gemm_vs_cublas.jl` times `reset!(WS); matmul!(...)` against `mul!`. Two things
# are in our arm and not in theirs, and both have to be priced before the gap is
# attributed to the kernel:
#
#   * `reset!` — the workspace bump allocator, host-side, per call.
#   * per-launch host cost — Lava records a dispatch; CUDA.jl's `mul!` also has
#     host cost, so this is not automatically asymmetric, but it is not
#     automatically equal either.
#
# The test is the same for both: vary how many launches share ONE sync. Host cost
# per launch shows up as a per-launch time that falls as `reps` rises; a kernel
# that is genuinely that slow gives a flat line.
#
# Reading rule: compare the SLOPES, not the absolute values. If both arms are
# flat from reps=4 on, the 64% is the kernel and nothing here is a confound.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf
using CUDA
const KA = KernelAbstractions
const DK = DNNKernels

const BACKEND = LavaBackend()
const WS = DK.Workspace(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

# The dominant shape, and one small one where per-launch cost matters most.
const CASES = [(2304, 4096, 576), (576, 4096, 576)]
const REPS = [1, 2, 4, 8, 16, 32]

function sweep(M, N, K)
    a = Float16.(0.05f0 .* randn(Float32, M, K))
    b = Float16.(0.05f0 .* randn(Float32, K, N))
    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, a)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, b)
    C = KA.allocate(BACKEND, Float16, M, N)
    cA = CuArray(a); cB = CuArray(b); cC = CUDA.zeros(Float16, M, N)

    withreset() = (DK.reset!(WS); DK.matmul!(CTX, C, A, B, nothing))
    noreset()   = DK.matmul!(CTX, C, A, B, nothing)
    cublas()    = mul!(cC, cA, cB)

    best(f, sync, reps) = begin
        for _ in 1:3; f(); end; sync()
        minimum(1:7) do _
            t0 = time_ns(); for _ in 1:reps; f(); end; sync()
            (time_ns() - t0) / 1e6 / reps
        end
    end
    g = 2.0 * M * N * K / 1e9
    @printf("\n%d x %d x %d   (%.1f GFLOP)\n", M, N, K, g)
    @printf("%6s %11s %11s %11s   %s\n", "reps", "ours+reset", "ours", "cuBLAS", "ours/cuBLAS")
    for r in REPS
        # `noreset` still needs the workspace not to fill up; reset once per
        # sample rather than per launch, which is what a real graph does.
        DK.reset!(WS)
        tw = best(withreset, () -> KA.synchronize(BACKEND), r)
        DK.reset!(WS)
        tn = best(noreset, () -> KA.synchronize(BACKEND), r)
        tc = best(cublas, CUDA.synchronize, r)
        @printf("%6d %11.4f %11.4f %11.4f   %10.0f%%\n", r, tw, tn, tc, 100 * tc / tn)
        flush(stdout)
    end
    A = B = C = nothing; cA = cB = cC = nothing; GC.gc()
end

for (M, N, K) in CASES; sweep(M, N, K); end
println("\nDONE")
