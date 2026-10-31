"""
The ops PP-DocLayoutV3 brought: `topk` over a long axis (the detector keeps 300
of 13125 anchors, past `topk_body`'s O(k*n) limit, so `topk_large_kernel!`),
`min.dim`, `any.dims` and `pow.Scalar`. Each emitted alone on the device and
checked against Julia's own.
"""

using Test, Random
import DNNKernels, Mantle
const DK = DNNKernels

"""
Emit one op over `inputs` (name => host array) into `outs` (name => (T, dims)),
run it, and return the outputs on the host.
"""
function emitone(dev, aten, inputs, ins, attrs, outs)
    g = Mantle.Graph(dev)
    res = Dict{String, Any}(k => Mantle.Buffer(dev, v) for (k, v) in inputs)
    for (k, (T, dims)) in outs
        res[k] = Mantle.Buffer(dev, T, dims)
    end
    op = DK.Op("v", aten, ins, "v", Dict{String, Any}(attrs))
    emitctx = DK.EmitCtx(DK.Graph("v", String[], String[], String[], Dict{String, DK.Buffer}(), String[], DK.Op[],
                                  Vector{Vector{String}}()),
                         g, dev, NamedTuple(), res, Set{String}(), Ref("v"), Any[], Dict{String, Any}())
    DK.emitop!(emitctx, op, Val(Symbol(aten)))
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    Mantle.run!(plan)
    Mantle.waitidle(dev)
    result = Dict(k => Array(Mantle.storage(res[k])) for k in keys(outs))
    Mantle.free!(plan)
    foreach(Mantle.free!, values(res))
    return result
end

function layoutops(be)
    dev = Mantle.todevice(be)
    @testset "detector ops on $(nameof(typeof(dev)))" begin
        Random.seed!(7)
        # torch's (2, 13125) with k = 300 along dim 1, which is Julia's first.
        for a in (randn(Float32, 13125, 2), Float32.(rand(-20:20, 13125, 2)))   # the second full of ties
            r = emitone(dev, "topk.default", Dict("a" => a), ["a"], ("arg1" => 300, "arg2" => 1),
                        Dict("v#0" => (Float32, (300, 2)), "v#1" => (Int64, (300, 2))))
            for j in 1:2
                expected = sortperm(a[:, j]; rev = true, alg = Base.Sort.DEFAULT_STABLE)[1:300]
                @test r["v#1"][:, j] == expected .- 1
                @test r["v#0"][:, j] == a[expected, j]
            end
            # Only where they are, as the detector reads it.
            only_inds = emitone(dev, "topk.default", Dict("a" => a), ["a"], ("arg1" => 300, "arg2" => 1),
                                Dict("v#1" => (Int64, (300, 2))))
            @test only_inds["v#1"] == r["v#1"]
        end

        a = randn(Float32, 5, 7, 3)
        r = emitone(dev, "min.dim", Dict("a" => a), ["a"], ("arg1" => -1,),
                    Dict("v#0" => (Float32, (7, 3)), "v#1" => (Int64, (7, 3))))
        @test r["v#0"] == dropdims(minimum(a; dims = 1); dims = 1)
        @test r["v#1"] == dropdims(map(i -> i[1] - 1, argmin(a; dims = 1)); dims = 1)

        b = rand(Float32, 4, 5, 6) .> 0.97f0
        r = emitone(dev, "any.dims", Dict("b" => b), ["b"], ("arg1" => [-2, -1],), Dict("v" => (Bool, (6,))))
        @test r["v"] == vec(any(b; dims = (1, 2)))

        x = rand(Float32, 64) .* 4
        r = emitone(dev, "pow.Scalar", Dict("x" => x), ["x"], ("arg0" => 10000,), Dict("v" => (Float32, (64,))))
        @test r["v"] ≈ 10000.0f0 .^ x rtol = 1e-5
    end
end

let bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the detector ops on the device"
    foreach(layoutops, bes)
end
