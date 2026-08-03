"""
One GEMM shape, repeated, for profiling under Nsight Compute:

    /opt/nvidia/nsight-compute/2026.2.1/ncu --target-processes all \\
        --kernel-name regex:coopmat_gemm --launch-count 3 --set detailed \\
        julia --project=. tools/ncu_gemm.jl [staged]

The question this exists to answer: `coopmat_gemm_kernel_4!` runs at **21-23
TFLOP/s against cuBLAS's 44.6** on the shapes SAM 2's encoder actually uses, and
that 2x is now 49% of the whole gap to PyTorch. Everything reachable without a
hardware profiler has been eliminated:

  * **not occupancy** — 255 registers and 17% occupancy, but forcing a smaller
    register block is *worse* on every shape (blk 4/2/1 = 16.8/12.9/7.1);
  * **not the epilogue** — folded into the kernel, `mm_epilogue_kernel!` deleted;
  * **not staging** — the shared-memory kernel is a 0.6% wash end to end;
  * **not the warp tile** — ST=2 already is `mul_mm.comp`'s 4 accumulators per
    warp, and ST=4 hits the 48 KB shared limit because the driver spills
    accumulators there;
  * **not the staging loads** — widening them without widening the shared array
    loses to bank conflicts, monotonically in the width.

What is left is what only a profiler sees: which pipe is saturated, what the
warps are stalled on, and whether the shared accesses conflict. `tools/gemm_bench.jl`
answers "is it faster"; this answers "why is it not".

Shape is the encoder's largest `addmm`, 24.4% of its GEMM arithmetic. `staged`
as an argument profiles the workgroup-staged kernel instead of the register-blocked
one — they are different kernels with different names, so `--kernel-name` picks
either.
"""

using Lava, KernelAbstractions, DNNKernels
const KA = KernelAbstractions

const M, N, K = 2304, 4096, 576

be = LavaBackend()
Lava.GEMM_STAGED[] = length(ARGS) >= 1 && ARGS[1] == "staged"

A = KA.allocate(be, Float16, M, K); copyto!(A, rand(Float16, M, K) .- Float16(0.5))
B = KA.allocate(be, Float16, K, N); copyto!(B, rand(Float16, K, N) .- Float16(0.5))
bias = KA.allocate(be, Float16, M); copyto!(bias, rand(Float16, M) .- Float16(0.5))
out = KA.allocate(be, Float16, M, N)
ws = DNNKernels.Workspace(be)

# Warm up outside the profiled window: the first launch compiles the SPIR-V and
# builds the pipeline, and a profiler replaying *that* measures the compiler.
for _ in 1:3
    DNNKernels.reset!(ws)
    DNNKernels.matmul!(out, A, B, bias; ws)
end
KA.synchronize(be)

for _ in 1:5
    DNNKernels.reset!(ws)
    DNNKernels.matmul!(out, A, B, bias; ws)
    KA.synchronize(be)
end
println("done: ", M, "x", N, "x", K, " staged=", Lava.GEMM_STAGED[])
