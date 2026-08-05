"""
What padding `N` to the staged kernel's block is worth, on Whisper's three GEMM
shapes.

    julia --project=. tools/bench_gemm_padn.jl

Three arms, **interleaved on one clock plateau**, because two `bench` calls are
two measurements of two different clock states and comparing them has produced
wrong numbers here before (`tools/measure.jl`):

  * `tile`  — `NP = cld(N, 16) * 16`, the old rounding, and with it the fp32
              scratch + `mm_epilogue_kernel!` that a padded destination used to
              require. This is what shipped.
  * `block` — `NP = Lava.gemm_padn(...)`, same fp32 route, so the difference
              against `tile` is the KERNEL alone: 1504 divides no tiling's block
              and lands on the register-blocked kernel, 1536 is 12 x 128 and
              reaches the staged one.
  * `now`   — what ships after this change: block padding *and* a destination of
              `out`'s own type, so the bias and the activation stay fused in the
              GEMM's store and the discard is a linear copy.

`tile` and `block` are written out here rather than reached through a switch on
the shipping path, because a knob that exists only for a benchmark is a knob that
outlives the benchmark.

Whisper's encoder runs 128 of the first shape, 32 of the second and 32 of the
third, so the weighted sum of the three predicts what the whole encoder should
move — which is the check that the shape-level numbers mean anything.
"""

using Lava, DNNKernels, KernelAbstractions, Printf, Statistics
using DNNKernels: Ctx, Workspace, scratch!, matmul_coopmat!, MMCoopMatPlan,
                  mm_epilogue_kernel!, padcols_kernel!
const KA = KernelAbstractions
include(joinpath(@__DIR__, "measure.jl"))

backend = LavaBackend()
ctx = Ctx(backend; ws = Workspace(backend))

"The route that shipped: pad `B`, GEMM into fp32 scratch, epilogue back down."
function oldroute!(ctx, out, A, B, bias, M, N, K, NP)
    Bp = B
    if NP != N
        Bp = scratch!(ctx, Float16, K, NP)
        padcols_kernel!(ctx.backend)(Bp, B, Val(K), N; ndrange = (K, NP))
    end
    blk_split = Lava.coopmat_gemm_shape(M, NP, K)
    splitk = blk_split[2]
    C = scratch!(ctx, Float32, M, NP, max(splitk, 1))
    Lava.coopmat_gemm!(C, A, Bp, M, NP, K; blk_split, partials = C, reduce = false)
    mm_epilogue_kernel!(ctx.backend)(out, C, bias, identity, Val(M), Val(splitk),
                                     M * NP, M * N; ndrange = M * N)
    out
end

# (label, M, N, K, how many the encoder runs)
const SHAPES = [("attn proj  1280x1500x1280", 1280, 1500, 1280, 128),
                ("fc1        5120x1500x1280", 5120, 1500, 1280, 32),
                ("fc2        1280x1500x5120", 1280, 1500, 5120, 32)]

report()
total = zeros(3)
for (label, M, N, K, count) in SHAPES
    A = KA.allocate(backend, Float16, M, K); copyto!(A, rand(Float16, M, K) .- Float16(0.5))
    B = KA.allocate(backend, Float16, K, N); copyto!(B, rand(Float16, K, N) .- Float16(0.5))
    bias = KA.allocate(backend, Float16, M); copyto!(bias, rand(Float16, M) .- Float16(0.5))
    out = KA.allocate(backend, Float16, M, N)

    ntile = cld(N, 16) * 16
    nblk  = Lava.gemm_padn(M, N, K)
    @printf("\n%s   tile -> %d, block -> %d\n", label, ntile, nblk)
    println("  tiling at tile-padding : ", Lava.gemm_tiling(M, ntile, K))
    println("  tiling at block-padding: ", Lava.gemm_tiling(M, nblk, K))

    # Ten per sample: one of these is ~0.1-1 ms and `gpustate()` costs ~20 ms.
    f_tile() = for _ in 1:10; DNNKernels.reset!(ctx.ws)
                   oldroute!(ctx, out, A, B, bias, M, N, K, ntile) end
    f_blk()  = for _ in 1:10; DNNKernels.reset!(ctx.ws)
                   oldroute!(ctx, out, A, B, bias, M, N, K, nblk) end
    f_now()  = for _ in 1:10; DNNKernels.reset!(ctx.ws)
                   matmul_coopmat!(ctx, out, MMCoopMatPlan(nblk, 16), A, B, bias, identity) end
    sync() = KA.synchronize(backend)

    # `compare`, not three `bench` calls: `bench` runs one arm to completion
    # before the next starts, so on a card that idles at 7% of its clock the
    # first arm is measured cold and the third warm. Doing it that way here
    # reported 1.75x on this very shape with the arms at 7%, 63% and 70% of the
    # clock — a number that was mostly the ramp.
    rs = compare((f_tile, f_blk, f_now); sync, samples = 9,
                 labels = ("  tile  (shipped)", "  block (kernel only)",
                           "  block + fp16 dst"))
    for (i, r) in enumerate(rs)
        println(r)
        total[i] += r.median / 10 * count
    end
    @printf("  kernel alone %.2fx,  with the destination %.2fx\n",
            rs[1].median / rs[2].median, rs[1].median / rs[3].median)
end

@printf("\nweighted over the encoder's 192 matmuls: tile %.1f ms, block %.1f ms, now %.1f ms (%.2fx)\n",
        1000total[1], 1000total[2], 1000total[3], total[1] / total[3])
