# One convolution, repeated, for profiling under Nsight Compute:
#   /opt/nvidia/nsight-compute/2026.2.1/ncu --target-processes all \
#       --set full julia --project=. tools/ncu_conv.jl
#
# The question this exists to answer: the same `convolution_igemm!` source is
# 2.5-2.8x slower under Lava than under CUDA.jl, and register pressure,
# workgroup size, shared memory and loop unrolling have all been eliminated as
# causes. What is left needs a hardware profiler.

using Lava, KernelAbstractions, DNNKernels
const KA = KernelAbstractions

be = LavaBackend()
OW, OH, Cout, N, KW, KH, Cin, s, p = 15, 8, 256, 1, 3, 3, 256, 1, 1
Wid = (OW - 1) * s + KW - 2p
Hei = (OH - 1) * s + KH - 2p

x = KA.allocate(be, Float16, Wid, Hei, Cin, N); fill!(x, Float16(0.1))
w = KA.allocate(be, Float16, KW, KH, Cin, Cout); fill!(w, Float16(0.05))
b = KA.allocate(be, Float32, Cout); fill!(b, 0f0)
o = KA.allocate(be, Float16, OW, OH, Cout, N)

DNNKernels.convolution_igemm!(o, x, w, b, (s, s), (p, p), (1, 1))
KA.synchronize(be)
for _ in 1:20
    DNNKernels.convolution_igemm!(o, x, w, b, (s, s), (p, p), (1, 1))
end
KA.synchronize(be)
println("done")
