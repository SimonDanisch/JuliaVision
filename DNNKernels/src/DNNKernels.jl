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
using LinearAlgebra: mul!, transpose
using Random
import AcceleratedKernels as AK
import Atomix
import GPUArrays

export loadgraph, execute!, launch!, readsafetensors, verifygraph, Model, matte, step!
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
#
# "5": kernels were edited AFTER "4"'s entries were already frozen — the norm
# reductions widened their accumulator, `padgemm` changed every im2col's row
# count, forward 1-D convolution reaches the coopmat path, and `scatteradd_kernel!`
# is new. A frozen entry is keyed by module, name, argument types, workgroup and
# this version — NOT by the kernel body — so an edit without a bump silently
# loads the old SPIR-V. That is not theoretical: it faulted the device
# ("device was lost ... a dispatch wrote out of bounds") deterministically at the
# same timeline on every precompile, and cleared the moment the cache was emptied.
#
# "4": the fused LSTM became one `Val`-parameterised kernel instead of a family
# `@eval`ed per `(H, reverse)`, so both its name and its signature changed. The
# key would have changed with it and the old entries merely orphaned rather than
# stale — but working that out from how `frozen_key` is derived is exactly the
# reasoning the version exists to make unnecessary, and getting it wrong is
# silent.
#
# "6": `erf`, `geluexact` and `gelutanh` now evaluate in `accum(T)`, and Lava's
# `strided_gemm_kernel!` no longer accumulates in the destination's precision.
# Both change the SPIR-V of kernels a graph reaches. The Lava half is covered by
# `module_build_id` in `frozen_key` — but the DNNKernels half is NOT: a broadcast
# over `geluexact` compiles a kernel whose *function* belongs to Lava and whose
# *argument types* name `typeof(geluexact)`, and a type name does not change when
# its method body does. That is the gap the build-id key still leaves, and this
# constant is what closes it.
const KERNELS_VERSION = "6"

include("assets.jl")
include("safetensors.jl")
include("graph.jl")
include("context.jl")
include("kernelplans.jl")
include("workspace.jl")
include("launch.jl")
include("kernels/extern/conv.jl")
include("kernels/extern/conv_implicit.jl")
include("kernels/extern/conv_coopmat.jl")
include("kernels/extern/matmul.jl")
include("kernels/extern/attention.jl")
include("kernels/extern/flash.jl")
include("kernels/extern/lstm.jl")       # aten::lstm kept whole, loop in-kernel
include("kernels/extern/spectral.jl")   # STFT + mel, on Lava's FFT
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
# `sam2.jl` moved to SAM2Runner. It arrived here in `7273481` "Import LavaDNN as
# DNNKernels" — the rename described half the package and moved nothing — and it
# is a model driver, not a kernel. Eleven `*Runner` packages already say where
# model code goes; this one just predates them.
include("verify.jl")

end # module
