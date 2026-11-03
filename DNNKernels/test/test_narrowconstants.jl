"""
`narrowconstants`: a constant table computed in the type its readers get, and nothing else touched.
"""

using Test, DNNKernels
const DKN = DNNKernels

# `ids` as (id, dtype, kind, of) buffers and `ops` as (aten, ins, out, attrs).
function narrowgraph(ids, ops; inputs = String[], outputs = String[])
    buffers = Dict(id => DKN.Buffer(id, kind, Any[4], T, "", (0, 0), of, of == "" ? "" : "view.default", Dict{String,Any}())
                   for (id, T, kind, of) in ids)
    return DKN.Graph("g", String[], inputs, outputs, buffers, [first(i) for i in ids],
                     [DKN.Op(out, aten, ins, out, attrs) for (aten, ins, out, attrs) in ops])
end
tablebuffers(extra...) = [("a", Float64, :transient, ""), ("v", Float64, :view, "a"), ("s", Float64, :transient, ""),
                          ("c", Float32, :transient, ""), extra...]
tableops(extra...) = [("arange.start_step", String[], "a", Dict{String,Any}("dtype" => "float64")),
                      ("sin.default", ["v"], "s", Dict{String,Any}()),
                      ("_to_copy.default", ["s"], "c", Dict{String,Any}("dtype" => "float32")), extra...]

@testset "narrowconstants" begin
    @testset "a table read as Float32 is computed in Float32" begin
        g, n = DKN.narrowconstants(narrowgraph(tablebuffers(), tableops()))
        @test n == 3
        @test all(id -> g.buffers[id].dtype === Float32, ("a", "v", "s", "c"))
        @test g.ops[1].attrs["dtype"] == "float32"
        @test g.ops[3].attrs["dtype"] == "float32"
    end

    @testset "what really reads a double stays a double" begin
        refused(g) = (g2 = first(DKN.narrowconstants(g)); all(id -> g2.buffers[id].dtype === Float64, ("a", "v", "s")))
        # A reader that computes with it instead of converting it.
        @test refused(narrowgraph(tablebuffers(("x", Float32, :transient, "")),
                                  tableops(("mul.Tensor", ["s", "c"], "x", Dict{String,Any}()))))
        # A graph output.
        @test refused(narrowgraph(tablebuffers(), tableops(); outputs = ["s"]))
        # Written from something that is not a constant.
        w = [("a", Float64, :weight, ""), ("v", Float64, :view, "a"), ("s", Float64, :transient, ""), ("c", Float32, :transient, "")]
        @test refused(narrowgraph(w, tableops()[2:end]))
        # A view of the table that is not itself a double.
        @test refused(narrowgraph(tablebuffers(("w", Float32, :view, "a")), tableops()))
    end

    @testset "a graph without doubles comes back as it was" begin
        g = narrowgraph([("c", Float32, :transient, "")], [("arange.start_step", String[], "c", Dict{String,Any}())])
        g2, n = DKN.narrowconstants(g)
        @test n == 0
        @test g2 === g
    end
end
