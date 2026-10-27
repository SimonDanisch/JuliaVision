module GenericFusionTests
using Test, DNNKernels
import Mantle as M
const DK = DNNKernels

# `(x + y) * y`: one elementwise group, so the fused graph is what Model runs.
function tinygraph()
    b(id, kind) = DK.Buffer(id, kind, Any[8], Float32, "", (0, 0), "", "", Dict{String,Any}())
    buffers = Dict("x" => b("x", :external), "y" => b("y", :external),
                   "s" => b("s", :transient), "o" => b("o", :transient))
    ops = [DK.Op("add", "add.Tensor", ["x", "y"], "s", Dict{String,Any}()),
           DK.Op("mul", "mul.Tensor", ["s", "y"], "o", Dict{String,Any}())]
    DK.Graph("tiny", String[], ["x", "y"], ["o"], buffers, ["x", "y", "s", "o"], ops)
end

# The shared pipeline's dict form fed `Model` a Dict{String,Any} for a day, and
# every fused Model build threw a MethodError. Build and run one.
function run(dev)
    @testset "Generic fusion inside Model" begin
        fused, report = genericfusion(Dict("tiny" => tinygraph()))
        @test fused isa Dict{String,DK.Graph}
        @test report.elementwise == 1
        patterns, _ = patternfusion(Dict{String,Any}("tiny" => tinygraph()))
        @test first(genericfusion(patterns)) isa Dict{String,DK.Graph}
        m = DK.Model(Dict("tiny" => tinygraph()), Dict{String,Any}(); backend = dev)
        @test length(m.graphs["tiny"].ops) == 1
        xh = randn(Float32, 8); yh = randn(Float32, 8)
        out = only(DK.call(m, "tiny", DK.toback(m.backend, xh), DK.toback(m.backend, yh); dims = (;)))
        @test Array(out) ≈ (xh .+ yh) .* yh
    end
end
end
