"""
What the static slab is allowed to contain.

The plan is the one place in this package where a mistake is silent: an offset
that overlaps a buffer still being read produces a *plausible* tensor, not a
crash, and the error surfaces as a slightly wrong mask several graphs later. So
the invariant is asserted here rather than trusted.

Run standalone:
    julia --project=. dev/LavaDNN/test/test_plan.jl
"""

using Test
using LavaDNN
using LavaDNN: loadgraph, planslab, checkslab, lifetimes, fusableset, evalshape, alignup

const GEN = normpath(joinpath(@__DIR__, "..", "..", "..", "gen"))

"""Every graph we can find, with the dims it is planned at.

Named rather than globbed: the export drops `op_histogram.json` next to the
graphs and it is not one.
"""
function testgraphs()
    out = Tuple{String,Any,Any}[]
    aten = joinpath(GEN, "graphs", "aten-autocast")
    if isdir(aten)
        for n in ("encode_image", "transform_key", "encode_mask_deep",
                  "encode_mask_shallow", "pixel_fusion", "pred_uncertainty",
                  "segment", "readout_query")
            f = joinpath(aten, "$n.json")
            isfile(f) && push!(out, (n, loadgraph(f), (h = 32, w = 32)))
        end
    end
    sam = joinpath(GEN, "graphs", "sam2-large")
    if isdir(sam)
        for n in ("sam2_encoder", "sam2_decoder")
            f = joinpath(sam, "$n.json")
            isfile(f) && push!(out, (n, loadgraph(f), (res = 1024,)))
        end
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
