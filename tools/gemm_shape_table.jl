# Where does `addmm`'s 28.5 TF/s come from?
#
# `tools/sam2_kernel_table.jl` puts SAM 2's 195 `addmm` calls at 28.5 TFLOP/s
# against PyTorch's 54.1 — 1.90x, on 1606 of the graph's 1823 GFLOP, which makes
# it the largest single item left in the encode. That is an AVERAGE, and an
# average is not a target: six shapes are 80.9% of the encoder's GEMM arithmetic
# and they differ by a factor of eight in K.
#
# So this prints the rate PER SHAPE, next to the block and split-K the heuristic
# picks for it. Three outcomes are possible and they lead different places:
#
#   uniform ~28      the kernel is slow everywhere; the structure is the target
#   a few bad shapes the heuristic is mispicking; the CHOICE is the target
#   all fast         the average is made by shapes outside this list, and the
#                    list is what needs extending
#
# `bench` from `gemm_lab.jl` does the measuring: it interleaves the variants,
# checks 64 rows against a CPU reference before believing a time, and resets the
# bump allocator per call.
#
# MEASURED, and the first two attempts at this table were both wrong:
#
#     38.6 TF/s weighted, against cuBLAS 44.6 on this card
#
# The first run reported 27.6 because `warmclock()` ran once at the top and the
# host-side setup between shapes let the card fall to 390 MHz for the largest
# one — 3.9 TF/s, a clock reading wearing a kernel's clothes. `bench` now
# re-boosts per shape and drops any row it can show was off the plateau.
#
# The second attempt then excluded EVERY row, because it read the clock after
# the timing loop had synchronised and the card had already dropped to 210. The
# clock has to be sampled with the queue full. It still cannot be sampled for
# the first shape of a table — `nvidia-smi` takes longer to spawn than the boost
# burst lasts — so that row reads 210 whatever its real clock was.
#
# The consequence for the conclusion: `sam2_kernel_table.jl` puts addmm at 28.5
# TF/s, but that is OPTIMES time, which serialises around every op and inflates
# our whole encode by 1.42x (117.1 ms serialised against 82.65 free-running).
# 56.42/1.42 = 39.7 ms is 40.4 TF/s, which is this table. So addmm is 1.40x
# behind PyTorch's 54.1, not the 1.90x the ratio column shows.
#
#     julia --project=. dev/JuliaVision/tools/gemm_shape_table.jl
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
include(joinpath(@__DIR__, "gemm_lab.jl"))

warmclock()

# One variant: whatever ships. The comparison is against the roof, not against
# another setting, so `bench`'s variant machinery is used with a single no-op.
println("\nPer-shape rate for the SHIPPED matmul. `cuBLAS` is this card measured",
        "\nby `gemm_bench.jl`, not a datasheet number.\n")
bench(["shipped" => () -> nothing])

println("\nWhat the heuristic picks, and what it is aiming at:\n")
@printf("%-20s %8s %8s %8s %10s\n", "M x N x K", "blk", "splitk", "tiles", "span")
for (M, N, K, _) in SHAPES
    blk, splitk = Mantle.coopmat_gemm_shape(M, N, K; cores = DNNKernels.caps(BACKEND).cores)
    span = Mantle.GEMM_TILE * blk
    @printf("%-20s %8d %8d %8d %10d\n",
            "$(M)x$(N)x$(K)", blk, splitk, (M ÷ span) * (N ÷ span), span)
end

# The number the flash kernel reaches on the same instruction, for scale: its
# two products alone run at 39.4 TF/s (`flash_cm2_ablate.jl`, `:nosoftmax`),
# which is above every rate this table is likely to print. Whatever the GEMM is
# losing is not the cooperative-matrix instruction.
@printf("\nreference: cuBLAS %.1f TF/s, flash-cm2 products 39.4 TF/s\n", CUBLAS_TFLOPS)
println("GEMMTABLEDONE")
