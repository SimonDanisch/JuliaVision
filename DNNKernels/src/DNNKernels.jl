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
using LinearAlgebra: mul!
import AcceleratedKernels as AK
import Atomix
import GPUArrays

export loadgraph, execute!, launch!, readsafetensors, verifygraph, Model, matte, step!
export findasset, assetpath
export KERNELS_VERSION

"""
    KERNELS_VERSION

Generation of the frozen SPIR-V cache, shared by **every** model that runs on
this runtime.

One version for all of them, not one per model, because they share kernels: a
broadcast over a `LavaArray`, `ndmap!`, AcceleratedKernels' reductions. Keyed per
package those would be frozen once per package under different versions — the
same bytes, several times, and a cache miss for whichever model ran second.
Shared, the first workload to reach a kernel freezes it and the rest hit it.

**Bump this after editing any kernel** any graph reaches, in DNNKernels or in Lava.
Nothing detects a stale entry; see `Lava/src/runtime/frozen_cache.jl` for why
that is deliberate.
"""
# "2": the staged cooperative-matrix GEMM (new kernels, new tilings), `splitidx`
# in place of `%`/`÷` in the GEMM and flash staging indices, and two changes in
# Lava's emitter that alter the SPIR-V of *every* kernel — `NonPrivatePointer` on
# workgroup accesses, and plain `Workgroup` variables where no type needs an
# explicit layout.
const KERNELS_VERSION = "3"

include("assets.jl")
include("safetensors.jl")
include("graph.jl")
include("workspace.jl")
include("launch.jl")
include("kernels/extern/conv.jl")
include("kernels/extern/conv_implicit.jl")
include("kernels/extern/conv_coopmat.jl")
include("kernels/extern/matmul.jl")
include("kernels/extern/attention.jl")
include("kernels/extern/flash.jl")
include("kernels/layernorm.jl")
include("kernels/resample.jl")
include("execute.jl")
include("ops.jl")
include("memory.jl")
include("plan.jl")
include("hoistcasts.jl")
include("foldbn.jl")
include("foldrelu.jl")
include("foldoutcasts.jl")
include("dce.jl")
include("fuse.jl")
include("driver.jl")
include("wan.jl")
include("sam2.jl")
include("verify.jl")

end # module
