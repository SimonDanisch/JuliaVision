"""
What the static slab is allowed to contain.

The plan is the one place in this package where a mistake is silent: an offset
that overlaps a buffer still being read produces a *plausible* tensor, not a
crash, and the error surfaces as a slightly wrong mask several graphs later. So
the invariant is asserted here rather than trusted.

Run standalone:
    julia --project=. dev/JuliaVision/DNNKernels/test/test_plan.jl
"""

using Test
using DNNKernels
using DNNKernels: loadgraph, planslab, checkslab, lifetimes, fusableset, evalshape, alignup
include("fixtures.jl")

# Nothing here names a directory. `Fixtures` owns the two layouts (SAM 2's JSONs
# are flat, MatAnyone's sit under `graphs/`), which is exactly the kind of detail
# a caller must not have to know. Two models because planning is shape-driven and
# one of them would not exercise it.

"""Every graph we can find, with the dims it is planned at.

Named rather than globbed: the export drops `op_histogram.json` next to the
graphs and it is not one.
"""
function testgraphs()
    out = Tuple{String,Any,Any}[]
    for n in ("encode_image", "transform_key", "encode_mask_deep",
              "encode_mask_shallow", "pixel_fusion", "pred_uncertainty",
              "segment", "readout_query")
        Fixtures.have(n) && push!(out, (n, Fixtures.matanyone(n), (h = 32, w = 32)))
    end
    for n in ("sam2_encoder", "sam2_decoder")
        push!(out, (n, Fixtures.sam2(n), (res = 1024,)))
    end
    out
end

const GRAPHS = testgraphs()
@assert !isempty(GRAPHS) "no exported graphs under $GEN — nothing to plan"

@testset "static plan" begin
    @testset "$name" for (name, g, dims) in GRAPHS
        plan = planslab(g, dims)

        # The invariant everything else rests on.
        nplanned, conflicts = checkslab(g, dims, plan)
        @test conflicts == 0

        # A fused value is returned by `emit` as the Broadcasted itself and never
        # reaches `dest`, so reserving bytes for it reserves them for something
        # that does not exist. On SAM 2's encoder that was 1 334 MB of a 2 020 MB
        # slab.
        lazy = fusableset(g)
        @test isempty(intersect(keys(plan.offsets), lazy))

        # Escaping values must not be planned either: a step chains graphs
        # through one slab, and an output the next graph reads cannot live in
        # memory that graph will overwrite.
        @test isempty(intersect(keys(plan.offsets), Set(g.outputs)))

        # Reuse actually happens: the slab is smaller than laying every planned
        # buffer end to end. (Equality would mean no two lifetimes are disjoint,
        # which no real graph manages.)
        summed = sum(values(plan.sizes); init = 0)
        @test plan.bytes <= summed
        nplanned > 4 && @test plan.bytes < summed

        # Nothing lands outside the slab it was sized for.
        @test all(id -> plan.offsets[id] + plan.sizes[id] <= plan.bytes, keys(plan.offsets))
    end

    @testset "greedy placement is at the liveness bound" begin
        # The planner is greedy, so it is allowed to be worse than optimal — but
        # if it ever drifts far above the bound that is a planner bug, not a
        # graph property. The bound is the largest total of bytes simultaneously
        # live at any single op.
        for (name, g, dims) in GRAPHS
            plan = planslab(g, dims)
            isempty(plan.offsets) && continue
            lt = lifetimes(g, fusableset(g))
            root(id) = (i = findlast('.', id); i === nothing ? id : id[1:(i - 1)])
            nops = length(g.ops)
            live = zeros(Int, nops)
            for id in keys(plan.offsets)
                k = root(id)
                haskey(lt, k) || continue
                a, b = lt[k]
                for i in max(a, 1):min(b, nops)
                    live[i] += plan.sizes[id]
                end
            end
            bound = maximum(live; init = 0)
            # A true lower bound: it ignores that a buffer needs *contiguous*
            # bytes, so greedy is expected to sit above it — 1.00x on SAM 2's
            # encoder, 1.31x on the worst of MatAnyone's graphs. This is a
            # regression guard on the planner, not a claim of optimality.
            @test plan.bytes >= bound
            @test plan.bytes <= 1.6 * bound
        end
    end
end

# ── Every generated block size is reachable ──────────────────────────────────
#
# `kernel-library-review.md` finding 5: `ATTN_BLOCKS` generates one kernel per
# entry, and the dispatchers used to be two hand-written `tk == 32 && return …`
# chains beside it. Adding a block size generated a kernel and silently did not
# dispatch to it — dead code that read as live, and nothing failed.
#
# The dispatchers are now folded out of the same tuple, so the two cannot
# disagree by construction. This asserts it anyway, because "by construction"
# is exactly the claim that rots when someone writes the next chain by hand.
@testset "every ATTN_BLOCKS entry has a kernel and a dispatch arm" begin
    for B in DNNKernels.ATTN_BLOCKS
        for prefix in ("attn_scores_b", "attn_apply_b")
            @test isdefined(DNNKernels, Symbol(prefix, B, "!"))
        end
    end

    # And the arms actually name those kernels. Reading the lowered source is the
    # only way to see a *missing* arm: a dispatcher with a fallback answers every
    # `tk`, so calling it can never reveal that one size fell through to the
    # wrong kernel.
    for (f, prefix) in ((DNNKernels.scoresblocked!, "attn_scores_b"),
                        (DNNKernels.applyblocked!,  "attn_apply_b"))
        body = string(Base.uncompressed_ast(only(methods(f))).code)
        for B in DNNKernels.ATTN_BLOCKS
            @test occursin(string(prefix, B, "!"), body)
        end
    end
end
