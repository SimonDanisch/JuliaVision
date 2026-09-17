# Does double-buffered staging pay? One barrier per k-block instead of two.
#
# The tiling family is exhausted (`gemm_tiling_sweep.jl`: no shape gets past ~64%
# of cuBLAS and the chooser already picks the best block), so the deficit is in
# what the kernel DOES per k-step. The single-buffer loop issues two barriers per
# block; the second exists only because the next block's staging would overwrite
# a tile still being read. Two alternating buffers remove it and let the staging
# of block k+1 overlap the arithmetic of block k — what `mul_mm.comp` does.
#
# Both kernels are checked BIT-IDENTICAL before this runs (`db_check`), so this
# measures one structural difference and nothing else. Interleaved in one
# process with cuBLAS beside it, per shape, never a weighted mean.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava, DNNKernels, KernelAbstractions, LinearAlgebra, Printf
using Mantle: LavaBackend   # Mantle owns it; Lava does not re-export it
using CUDA
const KA = KernelAbstractions
const DK = DNNKernels

const BACKEND = LavaBackend()
const WS = nothing   # `scratch!(nothing, backend, …)` allocates directly
const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]

function ab(M, N, K, share)
    g = 2.0 * M * N * K / 1e9
    a = Float16.(0.05f0 .* randn(Float32, M, K))
    b = Float16.(0.05f0 .* randn(Float32, K, N))
    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, a)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, b)
    C = KA.allocate(BACKEND, Float16, M, N)
    cA = CuArray(a); cB = CuArray(b); cC = CUDA.zeros(Float16, M, N)

    single() = (DK.reset!(WS); Mantle.coopmat_gemm!(C, A, B, M, N, K; doublebuf = false))
    double() = (DK.reset!(WS); Mantle.coopmat_gemm!(C, A, B, M, N, K; doublebuf = true))
    cublas() = mul!(cC, cA, cB)

    for _ in 1:3; single(); double(); cublas(); end
    KA.synchronize(BACKEND); CUDA.synchronize()

    ts = Float64[]; td = Float64[]; tc = Float64[]
    for _ in 1:9
        t0 = time_ns(); for _ in 1:8; single(); end
        KA.synchronize(BACKEND); push!(ts, (time_ns() - t0) / 1e6 / 8)
        t0 = time_ns(); for _ in 1:8; double(); end
        KA.synchronize(BACKEND); push!(td, (time_ns() - t0) / 1e6 / 8)
        t0 = time_ns(); for _ in 1:8; cublas(); end
        CUDA.synchronize(); push!(tc, (time_ns() - t0) / 1e6 / 8)
    end
    s, d, c = minimum(ts), minimum(td), minimum(tc)
    @printf("%-22s %5.1f%% | 1buf %7.4f %6.2f TF/s %3.0f%% | 2buf %7.4f %6.2f TF/s %3.0f%% | %+6.1f%%\n",
            "$M x $N x $K", share, s, g / s, 100 * c / s, d, g / d, 100 * c / d,
            100 * (d - s) / s)
    A = B = C = nothing; cA = cB = cC = nothing; GC.gc()
    flush(stdout)
    (share, g / s, g / d, g / c)
end

println("percentages are of cuBLAS at that shape; last column is 2buf vs 1buf\n")
res = [ab(M, N, K, sh) for (M, N, K, sh) in SHAPES]
tot = sum(r[1] for r in res)
@printf("\nshare-weighted: 1buf %.1f TF/s, 2buf %.1f TF/s, cuBLAS %.1f -> %.0f%% / %.0f%%\n",
        sum(r[1] * r[2] for r in res) / tot, sum(r[1] * r[3] for r in res) / tot,
        sum(r[1] * r[4] for r in res) / tot,
        100 * sum(r[1] * r[2] for r in res) / sum(r[1] * r[4] for r in res),
        100 * sum(r[1] * r[3] for r in res) / sum(r[1] * r[4] for r in res))
println("DBDONE")
