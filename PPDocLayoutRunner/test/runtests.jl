using Test, PPDocLayoutRunner, DNNKernels, ColorTypes
import PNGFiles
using PPDocLayoutRunner: plan, orderranks

# What the exporter leaves beside the graph: PyTorch's answers on the page it
# exported with. Not in the artifact, so this check runs where the export was
# made and is skipped elsewhere.
const GEN = let dir = normpath(joinpath(@__DIR__, "..", "..", "gen", "graphs"))
    isdir(dir) ? dir : normpath(joinpath(@__DIR__, "..", "..", "..", "..", "gen", "graphs"))
end
reference() = (p = joinpath(GEN, "ppdoclayoutv3", "reference.safetensors"); isfile(p) ? readsafetensors(p) : nothing)

sigmoid(x) = 1 / (1 + exp(-x))
kept(logits) = findall(q -> maximum(sigmoid.(view(logits, :, q))) >= 0.5f0, axes(logits, 2))

@testset "PPDocLayoutRunner" begin
    if !PPDocLayoutRunner.ready()
        @info "PP-DocLayoutV3 export not installed; nothing to test"
        return
    end
    m = doclayout()
    try
        @testset "against PyTorch" begin
            ref = reference()
            if ref !== nothing
                input = only(plan(m).inputs)
                copyto!(input, ref["input"])
                logits, boxes, order = Array.(DNNKernels.call(m.model, "ppdoclayoutv3", input; dims = (;)))
                # The regions kept, their kinds' logits and boxes, and their order.
                # The low-scoring queries are not compared: among anchors of
                # near-equal score the encoder's top-k may pick others.
                keep = kept(logits)
                @test keep == kept(ref["logits"])
                @test maximum(abs.(logits[:, keep] .- ref["logits"][:, keep])) < 0.02
                @test maximum(abs.(boxes[:, keep] .- ref["boxes"][:, keep])) * 800 < 0.5
                order = dropdims(order; dims = 3)
                expected = dropdims(ref["order"]; dims = 3)
                @test sortperm(orderranks(order)[keep]) == sortperm(orderranks(expected)[keep])
            end
        end

        invoice = permutedims(PNGFiles.load(joinpath(@__DIR__, "invoice.png")))
        @testset "an invoice" begin
            found = regions(m, invoice)
            kinds = [r.kind for r in found]
            # Its title, its table and its footer, in reading order.
            title, table, footer = findfirst(==(:paragraph_title), kinds), findfirst(==(:table), kinds),
                                   findfirst(==(:footer), kinds)
            @test title !== nothing && table !== nothing && footer !== nothing
            @test title < table < footer
            # The table spans the page's width below the title, in page pixels.
            x0, y0, x1, y1 = found[table].box
            @test 100 < x0 < 140 && 1100 < x1 < 1140 && 680 < y0 < 720 && 970 < y1 < 1010
            @test all(r -> r.score >= 0.5f0, found)
            # The same page again is the same answer, to float rounding.
            again = regions(m, invoice)
            @test [r.kind for r in again] == kinds
            @test all(((a, b),) -> maximum(abs.(a.box .- b.box)) < 1, zip(again, found))
        end

        @testset "an empty page" begin
            @test isempty(regions(m, fill(RGB{Float32}(0.97f0, 0.97f0, 0.96f0), 620, 877)))
        end
    finally
        close(m)
    end
end
