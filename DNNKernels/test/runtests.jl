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

const GEN = normpath(joinpath(@__DIR__, "..", "..", "..", "gen"))
const NAMES = ["encode_image", "transform_key", "encode_mask_deep", "encode_mask_shallow",
               "pixel_fusion", "pred_uncertainty", "segment", "readout_query"]

const PRECISIONS = filter(p -> isfile(joinpath(GEN, "refs-$p.safetensors")),
                          ["autocast", "fp32"])
@assert !isempty(PRECISIONS) "no refs-*.safetensors in $GEN; see the header"

weights = readsafetensors(joinpath(GEN, "weights.safetensors"))

@testset "DNNKernels vs PyTorch, layer by layer" begin
    @testset "$precision" for precision in PRECISIONS
        manifest = JSON3.read(read(joinpath(GEN, "refs_manifest-$precision.json"), String))
        H, W_ = manifest.resolution
        dims = (h = cld(H, 16), w = cld(W_, 16))
        refs = readsafetensors(joinpath(GEN, "refs-$precision.safetensors"))
        graphs = joinpath(GEN, "graphs", "aten-$precision")

        @testset "$name" for name in NAMES
            path = joinpath(graphs, "$name.json")
            impl, missing_ops = DNNKernels.coverage(path)
            @test isempty(missing_ops)

            ok, diffs, ties = verifygraph(path, refs, weights; dims)
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
