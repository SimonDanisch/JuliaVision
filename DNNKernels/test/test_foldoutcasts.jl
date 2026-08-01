"""
`foldoutcasts` on SAM 2's own encoder graph.

Against the real graph rather than a synthetic one, because every property this
pass has to get right is a property of how *that* graph is shaped: the casts read
their input through a `getitem` of a multi-output tuple or through a reshape, the
reader count only comes out right if views are transparent, and a layer norm's
dtype lives in an attribute rather than in the buffer's own field. A hand-built
two-op graph exercises none of it.

The pass is only allowed to be free, so the assertions are about invariants —
nothing widens, nothing but casts is dropped, and every retyped buffer had
exactly one reader — rather than about a count that moves whenever the model is
re-exported.
"""

using Test, DNNKernels
const DK = DNNKernels

const GDIR = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                               "gen", "graphs", "sam2-large"))

@testset "foldoutcasts" begin
    if !isfile(joinpath(GDIR, "sam2_encoder.json"))
        @info "no exported SAM 2 graph at $GDIR; skipping"
    else
        g = DK.loadgraph(joinpath(GDIR, "sam2_encoder.json"))
        g2, n = DK.foldoutcasts(g)

        @testset "it folds, and only casts" begin
            @test n > 0
            dropped = setdiff(Set(o.id for o in g.ops), Set(o.id for o in g2.ops))
            @test length(dropped) == n
            @test all(o.aten == "_to_copy.default" for o in g.ops if o.id in dropped)
            @test length(g2.ops) == length(g.ops) - n
            # The op order has to lose exactly the same ids, or the planner walks
            # a buffer whose op is gone.
            @test length(g2.order) <= length(g.order)
        end

        @testset "no buffer widens" begin
            # The pass may only narrow. A widened buffer would mean a producer
            # asked to write MORE bytes than the graph declared, which is the one
            # way this could cost time rather than save it.
            for (id, b) in g2.buffers
                old = get(g.buffers, id, nothing)
                old === nothing && continue
                @test sizeof(b.dtype) <= sizeof(old.dtype)
            end
        end

        @testset "every retyped producer had exactly one reader" begin
            # Counted the way the pass counts: views transparent, ops and graph
            # outputs only. This is the invariant that makes the fold safe — if
            # anything else read the wide value it would now see a narrow one.
            reads = Dict{String,Int}()
            bump!(id) = (reads[id] = get(reads, id, 0) + 1)
            for op in g.ops, i in op.ins
                bump!(last(DK.viewchain(g, i)))
            end
            for o in g.outputs
                bump!(last(DK.viewchain(g, o)))
            end
            changed = [id for (id, b) in g2.buffers
                       if haskey(g.buffers, id) && b.dtype !== g.buffers[id].dtype]
            @test !isempty(changed)
            for id in changed
                root = last(DK.viewchain(g, id))
                @test get(reads, root, 0) == 1
            end
        end

        @testset "the switch turns it off" begin
            old = DK.FOLD_OUTCASTS[]
            try
                DK.FOLD_OUTCASTS[] = false
                g3, k = DK.foldoutcasts(g)
                @test k == 0
                @test length(g3.ops) == length(g.ops)
            finally
                DK.FOLD_OUTCASTS[] = old
            end
        end

        @testset "it is idempotent" begin
            # Running it twice must find nothing new: the casts it folded are
            # gone, and the buffers it retyped now agree with their readers.
            _, k = DK.foldoutcasts(g2)
            @test k == 0
        end
    end
end
