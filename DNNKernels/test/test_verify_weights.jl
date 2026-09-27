"""
`verifygraph` on a graph whose reference dump names a WEIGHT.

A constant-folded weight has no op, so `verifygraph` compares it on its own,
against `weights[key]`. A model's resident weights are `Mantle.Buffer`s, which
`tohost` passes through untouched, so that comparison iterated a `Buffer` and
threw a `MethodError` in `maxerr`: SAM 2's layer-by-layer gate errored there on
the first folded weight its refs name, before comparing anything. The weight is
unwrapped with `Mantle.storage` now, as `declaredvalues` unwraps every result.
"""

using Test
import DNNKernels, Mantle
const DK = DNNKernels

vbuf(id, kind, shape; key = "") =
    DK.Buffer(id, kind, Any[shape...], Float32, key, (0, 1), "", "", Dict{String,Any}())

function verifyweights(be)
    dev = Mantle.todevice(be)
    @testset "a referenced weight is compared on $(nameof(typeof(dev)))" begin
        bufs = Dict("x" => vbuf("x", :external, [4, 3]),
                    "w" => vbuf("w", :weight, [4, 3]; key = "w"),
                    "y" => vbuf("y", :transient, [4, 3]))
        g = DK.Graph("g", String[], ["x"], ["y"], bufs, ["x", "w", "y"],
                     [DK.Op("y", "add.Tensor", ["x", "w"], "y", Dict{String,Any}())],
                     Vector{String}[])
        x = rand(Float32, 3, 4)
        w = rand(Float32, 3, 4)
        refs = Dict{String,Any}("g/in0" => x, "g/node/w" => w, "g/node/y" => x .+ w)
        weights = Dict{String,Any}("w" => Mantle.Buffer(dev, w))
        ok, diffs, _ = DK.verifygraph(g, refs, weights; dims = (;), device = dev, verbose = false)
        @test ok
        @test isempty(diffs)
        Mantle.free!(weights["w"])
    end
end

let bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the referenced-weight comparison"
    foreach(verifyweights, bes)
end
