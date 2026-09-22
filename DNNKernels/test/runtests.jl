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
# It had also been silently skipping, and the diagnosis in this comment was
# wrong: `matanyone-refs` IS bound, in `MatAnyoneRunner/Artifacts.toml`, with a
# download URL. Bound but not *installed*, one transient download failure is
# indistinguishable from an unbound artifact at the call site — so it
# read as "no fixtures" and went from 61 assertions to 1, still green.
# `ensure_artifact_installed` succeeds on retry; the gate then runs and passes.

# Which buffers a declared plan may place at all, over every exported graph.
# Needs a device, because `Mantle.Place` does the placement. Two silent faults
# from running SAM 2 end to
# end are in there: two output views placed on one another's bytes, and an
# eagerly-evaluated scratch belonging to no pass.
# What a declared view costs: the stride arithmetic that decides between a
# descriptor and a copy, and the refusals that keep a strided reshape from
# reading the wrong elements.
include(joinpath(@__DIR__, "test_stridedview.jl"))
include(joinpath(@__DIR__, "test_dynamic_slice.jl"))
include(joinpath(@__DIR__, "test_plan.jl"))
# What a checkpoint costs while it is being uploaded, which is a reachability
# question and not a byte count. See the file.
include(joinpath(@__DIR__, "test_upload_frees_host.jl"))
# Also host-only: a graph rewrite, checked against the real exported graph.
include(joinpath(@__DIR__, "test_foldoutcasts.jl"))
include(joinpath(@__DIR__, "test_fusegroupedrms.jl"))
include(joinpath(@__DIR__, "test_cat_interleave.jl"))
include(joinpath(@__DIR__, "test_fusepairrope.jl"))
include(joinpath(@__DIR__, "test_swiglu_stacked.jl"))
# The one rewrite that runs the ops it folds — here on the CPU backend.
include(joinpath(@__DIR__, "test_constfold.jl"))
# Instrumentation rides on the context: two runs, two measurements, no crosstalk.
include(joinpath(@__DIR__, "test_diagnostics.jl"))
# The non-overlapping transposed convolution against the gather it replaces.
include(joinpath(@__DIR__, "test_convtranspose_gemm.jl"))
include(joinpath(@__DIR__, "test_conv1d_library.jl"))
# Element functions that must not be evaluated at the operand's own precision.
include(joinpath(@__DIR__, "test_activation_width.jl"))
# Which advanced-index forms are an outer product and which are genuinely paired.
include(joinpath(@__DIR__, "test_index_tensor.jl"))
include(joinpath(@__DIR__, "test_index_recovery.jl"))
# `bmm` reaches the same capability dispatch a 2-D matmul does. It did not, and
# that one line was 79.6% of Depth Anything's forward pass.
include(joinpath(@__DIR__, "test_batchedmatmul.jl"))
# Elementwise fusion as a graph rewrite, and the `FusedOp` it emits. Asserts on
# the numbers rather than on op counts: the failure this pass shipped first was a
# binary op handed one argument, which runs and returns the wrong answer.
include(joinpath(@__DIR__, "test_fusepass.jl"))

# ── The three attention paths, and the plan that chooses between them.
#
# All three are registered here. Unregistered, `test_flash.jl` and
# `test_coopmat_attention.jl` sat on `Lava.DeviceCaps(dev; …)` after `caps`
# started returning Mantle's struct, which is a `MethodError` on the first line
# that reaches it. A test nothing runs is not a test.
#
# GPU-only, and they skip themselves without one.
include(joinpath(@__DIR__, "test_flash.jl"))
include(joinpath(@__DIR__, "test_coopmat_attention.jl"))
include(joinpath(@__DIR__, "test_flash_cm2.jl"))
# An elementwise op that allocates its own output instead of taking the planned
# one. Values stayed correct, so nothing here caught it for as long as it shipped.
include(joinpath(@__DIR__, "test_clamp_planned.jl"))
include(joinpath(@__DIR__, "test_no_kernel_alloc.jl"))
# `_fused_rms_norm`, `topk` and `scatter.src` — the entire new-op surface of
# Hunyuan3D-2.1's 3.05B-parameter denoiser, against their definitions. Host-only.
# The overflow case in there found a real fault in the RMS norm fallback on its
# first run, so it is not a formality.
include(joinpath(@__DIR__, "test_hunyuan3d_ops.jl"))
# An op attribute the exporter files under either its position or its name, and
# a runtime that read only one of them — so no graph in the tree had ever taken
# gelu's tanh branch. Host-only.
include(joinpath(@__DIR__, "test_atenarg.jl"))
# `arange` with a fractional start (DINOv3's patch centres, which threw) and with
# a non-unit step (which silently produced one element too few). Host-only.
include(joinpath(@__DIR__, "test_arange.jl"))
include(joinpath(@__DIR__, "test_recorded_call.jl"))
# The gate between the one-kernel norms and their six-pass fallback. It named one
# backend's array type, so it was a no-op to test on that backend and wrong on
# every other.
include(joinpath(@__DIR__, "test_norm_gate.jl"))
include(joinpath(@__DIR__, "test_q8gemm.jl"))
include(joinpath(@__DIR__, "test_bonsai_primitives.jl"))
include(joinpath(@__DIR__, "test_masked_flash.jl"))
include(joinpath(@__DIR__, "test_safesoftmax_attention.jl"))
include(joinpath(@__DIR__, "test_convrot_q8.jl"))
include(joinpath(@__DIR__, "test_conv_coopmat_chunk.jl"))
include(joinpath(@__DIR__, "test_foldconvpad.jl"))
# Host-side gates a second backend exposed: which devices the matrix kernels are
# for, and what a unary op does to an integer operand. Neither needs a device.
include(joinpath(@__DIR__, "test_coopmat_gate.jl"))
include(joinpath(@__DIR__, "test_deferred_add.jl"))
include(joinpath(@__DIR__, "test_deform_conv.jl"))
include(joinpath(@__DIR__, "test_int_unary.jl"))
include(joinpath(@__DIR__, "test_w8a8.jl"))
include(joinpath(@__DIR__, "test_masked_prefill.jl"))
# This file existed and was never listed here, so its assertions had never run —
# including the one its own docstring calls "the whole test". It covers the tiled
# transpose, which is now also how every transposed weight reaches the device.
include(joinpath(@__DIR__, "test_transposeLE.jl"))
# And so did this one, which is the third time: 165 lines over `tilecopy!`,
# `folddims!`, `bnstats!`, `blockcopy!` and the `gather` walk, self-driving over
# `Mantle.eachbackend()` and reached by nothing. Its own header says "a test
# nothing runs is not a test".
include(joinpath(@__DIR__, "test_declared_ops.jl"))
# Which passes the declared attention declares, over every backend: both of its
# products go through `Mantle.native_batched_gemm_dispatch!` where the device offers
# one, and the apply product carries softmax's row-sum divide as a fused epilogue —
# a route the rest of this suite only reaches through `LavaBackend()`.
include(joinpath(@__DIR__, "test_declared_attention.jl"))
# `aten::lstm` against torch's definition rather than against the other route:
# both routes share the kernel, so only an independent reference can check the
# recurrence.
include(joinpath(@__DIR__, "test_lstm.jl"))
# Where a composite op's result shapes come from when the export gives two
# answers. Host-only.
include(joinpath(@__DIR__, "test_multiresult.jl"))
