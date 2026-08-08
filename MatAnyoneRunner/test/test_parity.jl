"""
MatAnyone against PyTorch, layer by layer — the parity gate.

**Moved here from `DNNKernels/test/runtests.jl`.** Its subject is DNNKernels'
execution correctness, but it structurally needs *this* model's weights,
references and manifest, and a kernel library whose suite cannot run without a
model package installed has the dependency backwards. A runner owns its graphs
and weights and tests them; DNNKernels unit-tests its passes on constructed
graphs whose answers are known.

Error floor is ~1e-6, which is why this is worth guarding: it catches a wrong
kernel that a shape check and a smoke test both wave through.

**It was silently skipping for a while.** The artifacts refactor pointed it at
`matanyone-refs`, which is bound in `MatAnyoneRunner/Artifacts.toml` but was
not *installed*, and a first download attempt failed transiently —
indistinguishable from an unbound artifact at the call site, because
`matanyoneprecisions()` returns empty for both. The testset went from 61
assertions to 1 and stayed green. It now runs the full gate; if the refs ever
go missing again, regenerate and re-bind with:

    uv run tools/dump_refs.py --precision autocast --max-size 128
    uv run tools/dump_refs.py --precision fp32     --max-size 128
    julia --project=. tools/make_artifacts.jl matanyone-refs
"""

using Test
using DNNKernels
using DNNKernels: verifygraph
using MatAnyoneRunner

const NAMES = ["encode_image", "transform_key", "encode_mask_deep", "encode_mask_shallow",
               "pixel_fusion", "pred_uncertainty", "segment", "readout_query"]

const PRECISIONS = MatAnyoneRunner.matanyoneprecisions()
weights = isempty(PRECISIONS) ? nothing : MatAnyoneRunner.matanyoneweights()

@testset "MatAnyone vs PyTorch, layer by layer" begin
    if isempty(PRECISIONS)
        # A skip, not a silent pass. A testset that runs zero assertions is
        # indistinguishable from one that runs and succeeds.
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
