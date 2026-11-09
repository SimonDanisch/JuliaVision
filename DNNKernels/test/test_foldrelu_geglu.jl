"""
`foldrelu` folds a `gelu` into the `addmm` it activates only when it activates the
WHOLE result.

GEGLU (`x, gate = proj(x).chunk(2); gelu(gate) * x`, IndexTTS-2.5's emotion
perceiver) applies it to one half of a split. Following every view up to the
`addmm` folded it into both halves, so the block computed `gelu(x) * gelu(gate)`.
A plain reshape of the whole result still folds.
"""

using Test, DNNKernels

const DKG = DNNKernels

gbuf(id, kind, shape; of = "", viewop = "") =
    DKG.Buffer(id, kind, Any[shape...], Float32, "", (0, 0), of, viewop, Dict{String,Any}())

"""`addmm -> <views> -> gelu`, the gelu reading `gate`."""
function geglugraph(split::Bool)
    bufs = Dict{String,DKG.Buffer}(
        "x" => gbuf("x", :external, (4, 8)),
        "w" => gbuf("w", :weight, (8, 16)),
        "b" => gbuf("b", :weight, (16,)),
        "y" => gbuf("y", :transient, (4, 16)),
        "h" => gbuf("h", :transient, split ? (4, 8) : (4, 16)),
    )
    ops = DKG.Op[DKG.Op("y", "addmm.default", ["b", "x", "w"], "y", Dict{String,Any}())]
    if split
        # One reader of `y` (the split), two of the split: the shape the export
        # gives `chunk(2)`.
        bufs["s"] = gbuf("s", :view, (4, 16); of = "y", viewop = "split.Tensor")
        bufs["a"] = gbuf("a", :view, (4, 8); of = "s", viewop = "slice.Tensor")
        bufs["gate"] = gbuf("gate", :view, (4, 8); of = "s", viewop = "slice.Tensor")
        bufs["out"] = gbuf("out", :transient, (4, 8))
        push!(ops, DKG.Op("h", "gelu.default", ["gate"], "h", Dict{String,Any}()))
        push!(ops, DKG.Op("out", "mul.Tensor", ["h", "a"], "out", Dict{String,Any}()))
        order, outputs = ["y", "s", "a", "gate", "h", "out"], ["out"]
    else
        bufs["gate"] = gbuf("gate", :view, (4, 16); of = "y", viewop = "view.default")
        push!(ops, DKG.Op("h", "gelu.default", ["gate"], "h", Dict{String,Any}()))
        order, outputs = ["y", "gate", "h"], ["h"]
    end
    return DKG.Graph("geglu", String[], ["x"], outputs, bufs, order, ops, Vector{Vector{String}}())
end

@testset "foldrelu: gelu on one half of a split stays a separate op" begin
    g, n = DKG.foldrelu(geglugraph(true))
    @test n == 0
    @test !haskey(only(o for o in g.ops if o.aten == "addmm.default").attrs, "act")
    @test any(o -> o.aten == "gelu.default", g.ops)
end

@testset "foldrelu: gelu on a reshape of the whole addmm folds" begin
    g, n = DKG.foldrelu(geglugraph(false))
    @test n == 1
    @test only(o for o in g.ops if o.aten == "addmm.default").attrs["act"] == "gelu"
    @test !any(o -> o.aten == "gelu.default", g.ops)
end
