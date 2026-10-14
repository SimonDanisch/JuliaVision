"""
`aten::max.dim` on the device: the maximum along an axis, and where it was.

`maxdim_body` returned the index as `oftype(best, bi - 1)`, the VALUES' element
type, so an fp16 graph could not name an index past 2048: Float16 holds only even
integers from there, and index 2999 came back 3000. Stored into torch's int64
indices the round trip also carried an `InexactError` path whose error object is
a GPU allocation, which Lava can only lower to an undefined pointer and warns
about at compile. The index is an integer now and the store converts it to
whatever the buffer holds.
"""

using Test, Logging, Random
import DNNKernels, Mantle
const DK = DNNKernels

"""
Emit `max.dim` over torch dim `tdim` of `a` into a values and an indices buffer
of type `IT`, with the reduced axis kept (`keepdim`) or dropped as torch drops it.
"""
function devicemaxdim(dev, a, tdim, IT; keepdim::Bool = true)
    d = ndims(a) - tdim
    sz = keepdim ? ntuple(k -> k == d ? 1 : size(a, k), ndims(a)) :
         Tuple(size(a, k) for k in 1:ndims(a) if k != d)
    g = Mantle.Graph(dev)
    res = Dict{String,Any}("a" => Mantle.Buffer(dev, a),
                           "v#0" => Mantle.Buffer(dev, eltype(a), sz),
                           "v#1" => Mantle.Buffer(dev, IT, sz))
    op = DK.Op("v", "max.dim", ["a"], "v", Dict{String,Any}("arg1" => tdim))
    emitctx = DK.EmitCtx(DK.Graph("v", String[], String[], String[],
                                  Dict{String,DK.Buffer}(), String[], DK.Op[],
                                  Vector{Vector{String}}()),
                         g, dev, NamedTuple(), res, Set{String}(), Ref("v"), Any[],
                         Dict{String,Any}())
    DK.emitop!(emitctx, op, Val(Symbol("max.dim")))
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    Mantle.run!(plan)
    Mantle.waitidle(dev)
    vals, inds = Array(Mantle.storage(res["v#0"])), Array(Mantle.storage(res["v#1"]))
    Mantle.free!(plan)
    foreach(Mantle.free!, values(res))
    return vals, inds, d
end

function devicemaxdims(be)
    dev = Mantle.todevice(be)
    @testset "max.dim on $(nameof(typeof(dev)))" begin
        Random.seed!(4)
        a = randn(Float32, 5, 7, 3, 2)
        for (tdim, IT) in ((1, Int64), (3, Int64), (2, Float32))
            vals, inds, d = devicemaxdim(dev, a, tdim, IT)
            @test vals == maximum(a; dims = d)
            @test inds == IT.(map(i -> i[d] - 1, argmax(a; dims = d)))
        end
        # fp16 values over an axis longer than 2048, the maximum at an odd index
        # past it: the old body returned 3000 for 2999. No warning at compile,
        # which the old body's `InexactError` path was.
        h = fill(Float16(0), 2, 3001, 1, 1)
        h[1, 3000, 1, 1] = Float16(5)
        h[2, 2052, 1, 1] = Float16(5)
        vals, inds, d = @test_logs min_level = Logging.Warn devicemaxdim(dev, h, 2, Int64)
        @test vec(vals) == [Float16(5), Float16(5)]
        @test vec(inds) == [2999, 2051]
        # Without `keepdim` the outputs have one axis fewer than the input. The
        # body indexed the input with the output's coordinates, so every slice
        # read the first: PP-OCRv6's recogniser, `max(dim=2)` of `(1, T, 18710)`,
        # returned the blank class at every one of its T steps.
        p = randn(Float32, 18710, 40, 1)
        vals, inds, d = devicemaxdim(dev, p, 2, Int64; keepdim = false)
        @test size(vals) == (40, 1)
        @test vals == dropdims(maximum(p; dims = 1); dims = 1)
        @test inds == dropdims(map(i -> i[1] - 1, argmax(p; dims = 1)); dims = 1)
    end
end

let bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping max.dim on the device"
    foreach(devicemaxdims, bes)
end
