# Which matmuls in the shipping models fall off the tensor-core path, and why.
#
# WHY THIS MATTERS. `mm_coopmat_plan` (DNNKernels/src/kernels/extern/matmul.jl)
# pads N internally via `Lava.gemm_padn`, but M and K are taken as-is:
#
#     size(A,1) % dev.tile == 0 && size(A,2) % dev.tile == 0 || return Decline(:extent)
#
# A ragged M or K therefore declines the WHOLE matmul to `Lava.mul!`, which has
# its own `% GEMM_TILE == 0` gate and drops to `gemmlaunch!`'s scalar kernel.
# Measured cost of being on that path, interleaved:
#
#     1280 x 1536 x 1280  (all %16)     0.134 ms   37.64 TF/s
#     1288 x 1544 x 1288  (none %16)    0.875 ms    5.86 TF/s      6.43x
#
# So a single ragged extent is worth ~6x on that matmul, silently. Ragged M/K
# does not come from transformer model dims (768/1280/512 are all %16) — it comes
# from convolutions routed through im2col, where `K = Cin*kh*kw` and an RGB stem
# is 3*3*3 = 27. Whisper's stem was fixed by hand once (a 2.44x); this asks how
# many more there are.
#
# HOW IT COUNTS, and why it needs no instrumentation. Rather than patch
# `mm_coopmat_plan`, run a step and look at which GEMM kernels the device
# actually compiled. A scalar GEMM pipeline in `Lava.kernel_stats` after a step
# means something declined — the kernel cannot exist otherwise.
#
# THE NEGATIVE CONTROL, which this tool would be worthless without: a shape known
# to decline is pushed through first, and the census must SEE it. Without that,
# "no declines found" is indistinguishable from "the detector does not work" —
# the exact failure mode that made an earlier audit here report all-clean because
# its pattern matched every line.
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions

"""Names of GEMM pipelines that only exist when the tensor-core path declined."""
const SCALAR_GEMM_KERNELS = ("strided_gemm_kernel", "scalar_gemm_staged_kernel")

"""
    gemm_kernel_census(ctx) -> (scalar, coopmat)

Compiled pipeline names split into the ones that mean "declined" and the rest.
Read after a step; `Lava.clear_kernel_cache!()` before it if you want the census
to describe that step alone.
"""
function gemm_kernel_census(ctx)
    scalar = String[]
    coopmat = String[]
    # `kernel_stats` takes a LavaLinkedKernel, NOT a context — the cache is
    # `linked_kernel_cache(ctx)`, a Dict of key => LavaLinkedKernel.
    #
    # And the field to read is `source`, NOT `name`: `name` is the SPIR-V entry
    # point, which is `"main"` for every kernel Lava emits. Reading `name` made
    # this census report zero declines for a matmul that had visibly declined —
    # caught only by the planted control below, which is why it exists.
    for (_, linked) in Lava.linked_kernel_cache(ctx)
        s = Lava.kernel_stats(linked)
        nm = isempty(s.source) ? s.name : s.source
        low = lowercase(nm)
        (occursin("gemm", low) || occursin("matmul", low)) || continue
        push!(any(t -> occursin(t, nm), SCALAR_GEMM_KERNELS) ? scalar : coopmat, nm)
    end
    (unique(scalar), unique(coopmat))
end

"""
    control_declines(back) -> Bool

Push a deliberately ragged fp16 matmul through `mul!` so the census has
something it MUST detect. Returns whether a scalar GEMM kernel appeared.
"""
function control_declines(back)
    ctx = Lava.vk_context()
    A = KA.allocate(back, Float16, 200, 184); fill!(A, Float16(0.01))
    B = KA.allocate(back, Float16, 184, 216); fill!(B, Float16(0.01))
    C = KA.allocate(back, Float32, 200, 216); fill!(C, 0f0)
    mul!(C, A, B)
    KA.synchronize(back)
    !isempty(first(gemm_kernel_census(ctx)))
end

"""
    report_census(label, ctx)

Print the split. A `Decline` shows up as a scalar GEMM pipeline; the shapes
themselves need `mm_coopmat_plan` instrumentation, which this deliberately does
not add — the first question is whether ANY model declines at all.
"""
function report_census(label, ctx)
    scalar, coopmat = gemm_kernel_census(ctx)
    @printf("%-16s coopmat: %d   SCALAR (declined): %d\n", label, length(coopmat), length(scalar))
    for s in scalar
        println("    declined -> ", s)
    end
end

# ── THE INSTRUMENT IS VALIDATED, both directions (2026-08-07):
#
#     ragged  200x184x216 fp16  ->  gpu_scalar_gemm_staged_kernel   (declined)
#     aligned 256x256x256 fp16  ->  gpu_coopmat_gemm_kernel         (tensor cores)
#
# Which also confirms by KERNEL IDENTITY, not just by timing, that a ragged fp16
# matmul lands on the scalar path — the 5.86 TF/s route measured against 37.64
# for the same work aligned.
#
# The first version of this file read `s.name` and reported ZERO declines for
# that same ragged matmul, because `name` is the SPIR-V entry point and is
# `"main"` for every kernel Lava emits. Only the planted control caught it. A
# census with no control is a way of writing "all clean" without checking.
#
# ── PER-MODEL CENSUS: pending. Run with the GPU otherwise idle, `control_declines`
# first, and `Lava.clear_kernel_cache!()` between models so each census describes
# its own step.
