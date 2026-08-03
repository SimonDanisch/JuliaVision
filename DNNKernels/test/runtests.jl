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
using MatAnyoneRunner

const JSON3 = DNNKernels.JSON3   # not a direct dep of the driving project

# From the `matanyone-refs` artifact, not from a `gen/` tree. This used to walk
# up the filesystem, which made the single most valuable test in the repository
# — layer by layer against PyTorch, error floor ~1e-6 — run only on the machine
# that had generated the references, and quietly not exist anywhere else.
#
# Everything comes from `MatAnyoneRunner`, which owns MatAnyone's artifacts. This
# file never names a directory: it asks for the precisions that have references,
# then for the graph, the references and the manifest at each one. Where inside
# an artifact those live is the runner's business, and a re-export that moves
# them must not break this file.
#
# `matanyoneprecisions()` returns EMPTY rather than throwing when the
# `matanyone-refs` artifact is not bound, so the loop below runs zero times and
# the testset records a skip — instead of an `@assert` that took the other five
# test files down with it before they loaded.
const NAMES = ["encode_image", "transform_key", "encode_mask_deep", "encode_mask_shallow",
               "pixel_fusion", "pred_uncertainty", "segment", "readout_query"]

const PRECISIONS = MatAnyoneRunner.matanyoneprecisions()
weights = isempty(PRECISIONS) ? nothing : MatAnyoneRunner.matanyoneweights()

@testset "DNNKernels vs PyTorch, layer by layer" begin
    if isempty(PRECISIONS)
        # A skip, not a silent pass. A testset that runs zero assertions is
        # indistinguishable from one that runs and succeeds — this file's own
        # `test_foldoutcasts.jl` reported `Total 0` as green for 4172 assertions.
        @info "no `matanyone-refs` artifact; the parity gate is SKIPPED, not passing"
        @test_skip !isempty(PRECISIONS)
    end
    @testset "$precision" for precision in PRECISIONS
        manifest = MatAnyoneRunner.matanyonemanifest(precision)
        H, W_ = manifest.resolution
        dims = (h = cld(H, 16), w = cld(W_, 16))
        refs = MatAnyoneRunner.matanyonerefs(precision)

        @testset "$name" for name in NAMES
            g = MatAnyoneRunner.matanyonegraph(name, precision)
            impl, missing_ops = DNNKernels.coverage(g)
            @test isempty(missing_ops)

            ok, diffs, ties = verifygraph(g, refs, weights; dims)
            if !ok
                f = first(diffs)
                @info "first mismatch" precision graph=name index=f.index id=f.id aten=f.aten maxabs=f.maxabs rel=f.relative
            end
            @test ok

            # Tie-sensitive predicates are expected in readout_query only, and
            # only at a handful of elements; a systematically wrong predicate
            # would show up as a large fraction. See object_transformer.py:193.
            for (id, flipped, total) in ties
                @test flipped / total < 0.05
            end
        end
    end
end

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
