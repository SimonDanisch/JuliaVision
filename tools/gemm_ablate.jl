# Staging or arithmetic? The GEMM's k-loop, one half at a time.
#
# Every external lever is measured and closed — the tiling family in both
# directions, double buffering, staging width — and the kernel still runs at
# ~64% of cuBLAS. This is the instrument that says which half of the k-loop the
# time is in, the same way `flash_cm2_ablate.jl` settled the flash kernel (where
# the answer overturned three paragraphs of confident prose).
#
#   :all       the kernel
#   :nostage   stages the FIRST k-block and no other; every later block's
#              arithmetic runs on that stale tile. Same muladds, same barriers,
#              no staging loads.
#
# It computes a wrong answer on purpose, so only the GAP means anything.
#
# The one thing that would invalidate it: if the compiler noticed the staged tile
# is loop-invariant under `:nostage` and hoisted the cooperative-matrix loads out
# of the loop, the arm would measure a kernel that does no shared reads either.
# The barriers prevent that, and the register/binary statistics are printed so a
# collapse is visible rather than assumed.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava
using Mantle: LavaBackend   # Mantle owns it; Lava does not re-export it
Mantle.enable_pipeline_executable_properties!()
using DNNKernels, KernelAbstractions, LinearAlgebra, Printf
using CUDA
const KA = KernelAbstractions
const DK = DNNKernels

const BACKEND = LavaBackend()
const WS = nothing   # `scratch!(nothing, backend, …)` allocates directly
const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 288, 16384, 1152, 4.1)]

pipes() = Mantle.vk_context().caches.pipelines

function run_abl(C, A, B, M, N, K, c, abl)
    kern = Mantle.GEMM_STAGED_ABL_KERNELS[c]
    wg = Mantle.gemm_wg(c)
    kern(BACKEND, wg)(C, A, B, nothing, identity,
                      Val(M), Val(N), Val(K), Val(abl);
                      ndrange = (M ÷ Mantle.gemm_bm(c)) * (N ÷ Mantle.gemm_bn(c)) * wg)
    C
end

function ablate(M, N, K, share)
    g = 2.0 * M * N * K / 1e9
    c = Mantle.gemm_tiling(M, N, K)
    c === nothing && (println("no tiling for $M x $N x $K"); return)
    a = Float16.(0.05f0 .* randn(Float32, M, K))
    b = Float16.(0.05f0 .* randn(Float32, K, N))
    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, a)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, b)
    C = KA.allocate(BACKEND, Float16, M, N)
    cA = CuArray(a); cB = CuArray(b); cC = CUDA.zeros(Float16, M, N)

    stats = Dict{Symbol,Any}()
    for abl in (:all, :nostage)
        before = Set(keys(pipes()))
        DK.reset!(WS); run_abl(C, A, B, M, N, K, c, abl); KA.synchronize(BACKEND)
        for k in keys(pipes())
            k in before && continue
            s = Mantle.pipeline_exec_stats(pipes()[k]); s === nothing && continue
            d = Dict(String(r.name) => r.value for r in s.raw_stats if !(r.value isa Bool))
            stats[abl] = (regs = get(d, "Register Count", -1),
                          binary = get(d, "Binary Size", -1))
        end
    end

    arms = [(:all, () -> (DK.reset!(WS); run_abl(C, A, B, M, N, K, c, :all))),
            (:nostage, () -> (DK.reset!(WS); run_abl(C, A, B, M, N, K, c, :nostage)))]
    for _ in 1:3, (_, f) in arms; f(); end
    for _ in 1:3; mul!(cC, cA, cB); end
    KA.synchronize(BACKEND); CUDA.synchronize()

    acc = [Float64[] for _ in arms]; tc = Float64[]
    for _ in 1:9
        for (i, (_, f)) in enumerate(arms)
            t0 = time_ns(); for _ in 1:8; f(); end
            KA.synchronize(BACKEND); push!(acc[i], (time_ns() - t0) / 1e6 / 8)
        end
        t0 = time_ns(); for _ in 1:8; mul!(cC, cA, cB); end
        CUDA.synchronize(); push!(tc, (time_ns() - t0) / 1e6 / 8)
    end
    tall, tno, tcu = minimum(acc[1]), minimum(acc[2]), minimum(tc)
    @printf("\n%d x %d x %d  (%.1f%% of the encoder's GEMM, tiling %dx%d/%dw)\n",
            M, N, K, share, Mantle.gemm_bm(c), Mantle.gemm_bn(c), Mantle.gemm_wg(c) ÷ 32)
    @printf("   %-9s %8.4f ms  %6.2f TF/s   %5.0f%% of cuBLAS   regs=%d binary=%d\n",
            "all", tall, g / tall, 100 * tcu / tall,
            get(stats, :all, (regs = -1, binary = -1)).regs,
            get(stats, :all, (regs = -1, binary = -1)).binary)
    @printf("   %-9s %8.4f ms  %6.2f TF/s   %+6.1f%% vs all   regs=%d binary=%d\n",
            "nostage", tno, g / tno, 100 * (tno - tall) / tall,
            get(stats, :nostage, (regs = -1, binary = -1)).regs,
            get(stats, :nostage, (regs = -1, binary = -1)).binary)
    @printf("   %-9s %8.4f ms  %6.2f TF/s\n", "cuBLAS", tcu, g / tcu)
    @printf("   -> staging is %.0f%% of this kernel; the rest is arithmetic + barriers\n",
            100 * (tall - tno) / tall)
    A = B = C = nothing; cA = cB = cC = nothing; GC.gc()
    flush(stdout)
end

for (M, N, K, sh) in SHAPES; ablate(M, N, K, sh); end
println("\nABLDONE")
