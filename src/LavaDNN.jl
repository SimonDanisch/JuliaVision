"""
    LavaDNN

Runtime and kernel sources for running exported ATen graphs in the same
execution graph as graphics passes. See lava-dnn.md.

Holds no SPIR-V and no per-model instantiation list: kernel *sources* live
here and are shared across models, while kernel *instantiations* live in the
generated model package.
"""
module LavaDNN

using JSON3
using KernelAbstractions
using Lava
using LinearAlgebra: mul!
import AcceleratedKernels as AK
import Atomix
import GPUArrays

export loadgraph, execute!, launch!, readsafetensors, verifygraph, Model, matte, step!

include("safetensors.jl")
include("graph.jl")
include("workspace.jl")
include("launch.jl")
include("kernels/extern/conv.jl")
include("kernels/extern/conv_implicit.jl")
include("kernels/extern/conv_coopmat.jl")
include("kernels/extern/matmul.jl")
include("kernels/extern/attention.jl")
include("kernels/resample.jl")
include("execute.jl")
include("ops.jl")
include("memory.jl")
include("plan.jl")
include("hoistcasts.jl")
include("foldbn.jl")
include("foldrelu.jl")
include("dce.jl")
include("fuse.jl")
include("driver.jl")
include("verify.jl")

end # module
