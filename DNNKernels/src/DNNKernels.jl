"""
    DNNKernels

Runtime and kernel sources for running exported ATen graphs in the same
execution graph as graphics passes. See lava-dnn.md.

Holds no SPIR-V and no per-model instantiation list: kernel *sources* live
here and are shared across models, while kernel *instantiations* live in the
generated model package.
"""
module DNNKernels

using JSON3
using KernelAbstractions
using Lava
# The API this package builds GPU execution with — placement, lifetimes,
# barriers, recording and replay. Not `using`: `Mantle.run!`, `Mantle.Buffer` and
# `Mantle.storage` all collide with names here or in Base, and `M.` at the use
# site says which one is meant. A dependency and not an extension because there
# is no path through this package that does not go through Mantle: every buffer
# is a `Mantle.Buffer` or a `Mantle.Transient.Buffer`, and every dispatch is
# declared into a `Mantle.Graph`.
import Mantle
const M = Mantle
# `runonce!` is Mantle's verb for one-shot work — declare a graph, plan it, run
# it once, free the plan. See its docstring there for why such work is a graph
# and not a launch.
using Mantle: runonce!
# The intrinsics the kernels here are written against. Macro-free: a kernel is a
# plain function using `KI.get_global_id()` rather than a `@kernel` with
# `@index`, so `dispatch!` takes the function itself and Mantle compiles it for
# whichever backend the graph is on.
import KernelInterface as KI
# `@private T (dims)` was KernelAbstractions' spelling for a per-workitem array,
# and Lava lowers it to exactly this — see `Lava/src/device/ndrange.jl`. A
# dependency of this package's own, not `import Lava: StaticArrays`: Lava's body is
# gated on a Vulkan loader, so on a Mac `Lava.StaticArrays` does not exist, the
# import bound nothing, and every kernel naming `StaticArrays.MArray` failed to
# compile for Metal (Qwen-Image's softmax). One environment has one StaticArrays,
# so the type is the one Lava lowers either way.
import StaticArrays
using LinearAlgebra: mul!, transpose
using Random
import AcceleratedKernels as AK
import Atomix
import GPUArrays

export loadgraph, launch!, readsafetensors, verifygraph, Model, matte, step!
export ConvRotQInt8Matrix, ConvRotQInt8HostMatrix, convrotqint8
export W4A8ConvRotHostMatrix, w4a8convrot
export readgguf, GGUFFile, GGUFTensor, gguftensor, ggufbytes
export PTQ1Matrix, ptq1matrix, ptq1mul!, ptq1_getrows!, ptq1_dequant
export hadamard!, gated_delta_net!, depthwise_conv4!

include("assets.jl")
include("safetensors.jl")
include("gguf.jl")
include("graph.jl")
include("fusedop.jl")
include("context.jl")
include("kernelplans.jl")
include("launch.jl")
include("kernels/extern/conv.jl")
include("kernels/extern/conv_implicit.jl")
include("kernels/extern/conv_coopmat.jl")
include("quant.jl")
include("ptq1.jl")
include("gated_delta_net.jl")
include("q8gemm.jl")
include("w8a8.jl")
include("kernels/extern/matmul.jl")
include("kernels/extern/attention.jl")
include("kernels/extern/flash.jl")
include("kernels/extern/flash_cm2.jl")
include("kernels/extern/flash_rows.jl")
include("maskedprefill.jl")
include("kernels/extern/lstm.jl")       # aten::lstm kept whole, loop in-kernel
include("kernels/extern/spectral.jl")   # STFT + mel, on Lava's FFT
include("kernels/layernorm.jl")
include("kernels/resample.jl")
include("ops.jl")
include("memory.jl")
include("plan.jl")
include("hoistcasts.jl")
include("foldbn.jl")
include("foldrelu.jl")
include("foldconvpad.jl")
include("foldoutcasts.jl")
include("foldincasts.jl")
include("dropclones.jl")
include("dce.jl")
include("fuse.jl")
include("fusepass.jl")
include("fuseattention.jl")
include("fusemaskedattention.jl")
include("foldcache.jl")
include("fusegroupedrms.jl")
include("fuseqkv.jl")
include("fuserope.jl")
include("fuseswiglu.jl")
include("kernels/elementwise.jl")
include("kernels/shapeops.jl")
include("kernels/batchnorm.jl")
include("kernels/conv_igemm.jl")
include("emit.jl")
include("halfconvs.jl")    # after emit.jl: it declares emitop! methods
include("driver.jl")
include("flowmatch.jl")
include("wan.jl")
# `sam2.jl` moved to SAM2Runner. It arrived here in `7273481` "Import LavaDNN as
# DNNKernels" — the rename described half the package and moved nothing — and it
# is a model driver, not a kernel. Eleven `*Runner` packages already say where
# model code goes; this one just predates them.
include("verify.jl")

end # module
