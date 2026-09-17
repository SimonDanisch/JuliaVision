# What does our GEMM actually DO for each of the encoder's six shapes?
#
# Facts before theories. The gap to cuBLAS is a flat 61-76% across every shape
# (`gemm_vs_cublas.jl`), and a flat deficit is not explained by a shape-specific
# cliff — so the question is what the kernel is at each shape, not which shape is
# unlucky.
#
# Per shape: the tiling the chooser selects, the registers and shared memory the
# DRIVER reports for the pipeline that results, and the occupancy those imply.
# `pipeline_exec_stats` needs the properties extension enabled BEFORE the device
# exists, so this file must not be `include`d after anything has made a device.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
using Lava
using Mantle: LavaBackend   # Mantle owns it; Lava does not re-export it
Mantle.enable_pipeline_executable_properties!()
using DNNKernels, KernelAbstractions, Printf
const KA = KernelAbstractions
const DK = DNNKernels

const BACKEND = LavaBackend()
const WS = nothing   # `scratch!(nothing, backend, …)` allocates directly
const CTX = DK.Ctx(BACKEND; ws = WS)

const SHAPES = [(2304, 4096,  576, 24.4),
                ( 576, 4096, 2304, 24.4),
                (1728, 4096,  576, 17.8),
                ( 576, 4096,  576,  6.1),
                ( 288, 16384, 1152, 4.1),
                (1152, 16384,  288, 4.1)]

# An Ada SM has 65536 registers and 100 KB of shared memory available to
# compute. Occupancy in workgroups per SM is whichever of the two binds first.
const REGS_PER_SM = 65536
const SHARED_PER_SM = 100 * 1024

pipes() = Mantle.vk_context().caches.pipelines

"""Driver statistics for the pipelines this shape newly compiles."""
function facts(M, N, K)
    A = KA.allocate(BACKEND, Float16, M, K); fill!(A, Float16(0.01))
    B = KA.allocate(BACKEND, Float16, K, N); fill!(B, Float16(0.01))
    C = KA.allocate(BACKEND, Float16, M, N)
    before = Set(keys(pipes()))
    DK.reset!(WS)
    DK.matmul!(CTX, C, A, B, nothing)
    KA.synchronize(BACKEND)
    fresh = [k for k in keys(pipes()) if !(k in before)]
    out = NamedTuple[]
    for k in fresh
        s = Mantle.pipeline_exec_stats(pipes()[k])
        s === nothing && continue
        d = Dict(String(r.name) => r.value for r in s.raw_stats if !(r.value isa Bool))
        push!(out, (regs = get(d, "Register Count", -1),
                    shared = get(d, "Shared Memory Size", -1),
                    binary = get(d, "Binary Size", -1),
                    spill = get(d, "Local Memory Size", -1)))
    end
    A = B = C = nothing; GC.gc()
    out
end

@printf("%-22s %6s %6s %8s %8s %7s %s\n",
        "M x N x K", "share", "regs", "shared", "spill", "binary", "occupancy (wg/SM)")
for (M, N, K, share) in SHAPES
    fs = facts(M, N, K)
    if isempty(fs)
        @printf("%-22s %5.1f%%  (no new pipeline — already compiled)\n", "$M x $N x $K", share)
        continue
    end
    for f in fs
        # Workgroup size is not in the driver stats; report the register- and
        # shared-limited counts per 128 and per 256 threads so the binding
        # constraint is visible either way.
        occ(nt) = min(REGS_PER_SM ÷ max(f.regs * nt, 1),
                      f.shared > 0 ? SHARED_PER_SM ÷ f.shared : 99)
        @printf("%-22s %5.1f%% %6d %8d %8d %7d   %d@128thr  %d@256thr%s\n",
                "$M x $N x $K", share, f.regs, f.shared, f.spill, f.binary,
                occ(128), occ(256), f.spill > 16 ? "   <-- SPILLING" : "")
    end
    flush(stdout)
end
println("\nDONE")
