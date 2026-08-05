"""
Stage 1 of lava-dnn.md's verification order: CPU backend, same KA source,
layer-by-layer against the PyTorch reference activations.

Two precisions, because they answer different questions:

  autocast  what MatAnyone2 actually ships (inference_matanyone2.py wraps main
            in @safe_autocast_decorator, so conv/matmul are fp16 while
            reductions and cat stay fp32). torch.export bakes that policy into
            the graph as explicit _to_copy nodes, so matching it is a matter of
            honouring the declared dtypes, not of reimplementing a policy.
            This is the one that says "we reproduce the product".

  fp32      the same graphs traced without autocast, TF32 off. Error floor
            ~1e-6 instead of ~1e-3, so a real bug stands out by orders of
            magnitude. This is the one that localises a fault.

    julia --project=. dev/JuliaVision/DNNKernels/test/runtests.jl

Regenerate with:
    uv run tools/export_graphs.py --precision autocast
    uv run tools/export_graphs.py --precision fp32
    uv run tools/convert_weights.py
    uv run tools/dump_refs.py --precision autocast --max-size 128
    uv run tools/dump_refs.py --precision fp32     --max-size 128
"""

using Test
using DNNKernels
using KernelAbstractions

const JSON3 = DNNKernels.JSON3   # not a direct dep of the driving project

# The layer-by-layer parity gate MOVED to `MatAnyoneRunner/test/test_parity.jl`.
# It needs MatAnyone's weights, references and manifest, so it made this suite
# unrunnable without a model package installed — the dependency backwards.
# A runner owns its graphs and weights and tests them; what stays here are unit
# tests of DNNKernels' own passes.
#
# It had also been silently skipping: pointed at the `matanyone-refs` artifact,
# which is bound in no `Artifacts.toml` in this repository, it went from 61
# assertions to 1 and stayed green. See that file for how to bind it.

# Host-only, so it belongs in stage 1 with the rest of this file: what the
# static slab may contain, and that nothing overlaps inside it.
include(joinpath(@__DIR__, "test_plan.jl"))
# Also host-only: a graph rewrite, checked against the real exported graph.
include(joinpath(@__DIR__, "test_foldoutcasts.jl"))
# The one rewrite that runs the ops it folds — here on the CPU backend.
include(joinpath(@__DIR__, "test_constfold.jl"))
# Instrumentation rides on the context: two runs, two measurements, no crosstalk.
include(joinpath(@__DIR__, "test_diagnostics.jl"))
# The non-overlapping transposed convolution against the gather it replaces.
include(joinpath(@__DIR__, "test_convtranspose_gemm.jl"))
# Element functions that must not be evaluated at the operand's own precision.
include(joinpath(@__DIR__, "test_activation_width.jl"))
# Which advanced-index forms are an outer product and which are genuinely paired.
include(joinpath(@__DIR__, "test_index_tensor.jl"))
# `bmm` reaches the same capability dispatch a 2-D matmul does. It did not, and
# that one line was 79.6% of Depth Anything's forward pass.
include(joinpath(@__DIR__, "test_batchedmatmul.jl"))
