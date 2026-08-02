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
            # A keyword, so there is no module state to save and restore — which
            # a failing `@test` inside the old `try` would have skipped, leaving
            # the fold off for every test after it.
            g3, k = DK.foldoutcasts(g; enabled = false)
            @test k == 0
            @test length(g3.ops) == length(g.ops)
        end

        @testset "it is idempotent" begin
            # Running it twice must find nothing new: the casts it folded are
            # gone, and the buffers it retyped now agree with their readers.
            _, k = DK.foldoutcasts(g2)
            @test k == 0
        end
    end
end

@testset "foldrelu folds gelu into addmm" begin
    if !isfile(joinpath(GDIR, "sam2_encoder.json"))
        @info "no exported SAM 2 graph at $GDIR; skipping"
    else
        g = DK.loadgraph(joinpath(GDIR, "sam2_encoder.json"))
        g2, n = DK.foldrelu(g)

        # The reason this needed the whole view chain rather than
        # `resolvealias`: the graph reshapes the `addmm` result, so the shape is
        # `gelu <- view.default <- addmm` and an alias-only resolve finds no
        # producer at all. It folded 0 of 48 until that was fixed.
        @test n >= 48
        @test count(o -> o.aten == "gelu.default", g2.ops) == 0
        @test count(o -> get(o.attrs, "act", "") == "gelu", g2.ops) == 48
        @test all(o -> o.aten == "addmm.default",
                  (o for o in g2.ops if get(o.attrs, "act", "") == "gelu"))

        @testset "the alias keeps the gelu's own shape" begin
            # Aliasing onto the `addmm` buffer would be a rank error waiting to
            # happen: `addmm_2` is `[65536, 576]` and the gelu it feeds is
            # `[1, 256, 256, 576]`. The alias has to target the *view*.
            for o in g.ops
                o.aten == "gelu.default" || continue
                b = g2.buffers[o.out]
                @test b.kind === :view
                @test b.viewop == "alias.default"
                @test g2.buffers[b.of].shape == g.buffers[o.out].shape
            end
        end

        @testset "the tanh approximation is a different function and does not fold" begin
            # `gelu(approximate="tanh")` differs from the exact form by ~1e-3.
            # Folding it in as if it were the same would be a silent wrong answer.
            gl = first(o for o in g.ops if o.aten == "gelu.default")
            attrs = Dict{String,Any}(gl.attrs); attrs["arg1"] = "tanh"
            ops = [o.id == gl.id ? DK.Op(o.id, o.aten, o.ins, o.out, attrs) : o
                   for o in g.ops]
            gt = DK.Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers,
                          g.order, ops, g.fusion)
            _, nt = DK.foldrelu(gt)
            @test nt == n - 1
        end
    end
end
