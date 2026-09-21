"""
`constant_pad_nd` folded into the convolution that reads it.

The convolution kernel has always substituted zero outside its input, so an
explicit zero pad in front of a `padding=0` convolution is a full extra copy of
the input, one pixel wider on each side, for nothing. The Qwen-Image 2.1 VAE
exports forty of them and they are 661 ms of a 4.3 s decode.

What is checked is the REFUSALS as much as the fold, because a convolution
given a padding it should not have is wrong in a way the shapes still agree
with.
"""

using Test, DNNKernels

const DKP = DNNKernels

padbuf(id, kind, shape, T) =
    DKP.Buffer(id, kind, Any[shape...], T, "", (0, 4), "", "", Dict{String,Any}())

"""`pad -> conv` as a graph, with whatever attributes the case wants."""
function padgraph(; amount = [1, 1, 1, 1], value = 0, convpad = [0, 0],
                   transposed = false, extrareader = false)
    W = H = 8
    # torch pads the LAST axis first, so `amount[1:2]` is width and `[3:4]`,
    # where they exist, is height.
    dw = amount[1] + amount[2]
    dh = length(amount) >= 4 ? amount[3] + amount[4] : 0
    bufs = Dict{String,DKP.Buffer}(
        "x"   => padbuf("x", :input, [1, 4, H, W], Float16),
        "pd"  => padbuf("pd", :op, [1, 4, H + dh, W + dw], Float16),
        "w"   => padbuf("w", :weight, [8, 4, 3, 3], Float16),
        "y"   => padbuf("y", :op, [1, 8, H, W], Float16),
    )
    ops = DKP.Op[
        DKP.Op("pd", "constant_pad_nd.default", ["x"], "pd",
               Dict{String,Any}("arg1" => Any[amount...], "arg2" => value)),
        DKP.Op("y", "convolution.default", ["pd", "w"], "y",
               Dict{String,Any}("arg3" => Any[1, 1], "arg4" => Any[convpad...],
                                "arg5" => Any[1, 1], "arg6" => transposed,
                                "arg7" => Any[0, 0], "arg8" => 1)),
    ]
    order = ["pd", "y"]
    if extrareader
        bufs["z"] = padbuf("z", :op, [1, 4, H + 2, W + 2], Float16)
        push!(ops, DKP.Op("z", "relu.default", ["pd"], "z", Dict{String,Any}()))
        push!(order, "z")
    end
    DKP.Graph("g", String[], ["x"], extrareader ? ["y", "z"] : ["y"],
              bufs, order, ops, Vector{String}[])
end

@testset "an explicit zero pad becomes the convolution's own" begin
    g, n = DKP.foldconvpad(padgraph())
    @test n == 1
    @test length(g.ops) == 1
    conv = only(g.ops)
    @test conv.id == "y"
    @test conv.ins[1] == "x"                 # reads the pad's input directly
    @test DKP.ints(conv.attrs["arg4"]) == [1, 1]
    @test !haskey(g.buffers, "pd")

    # Asymmetric on the width axis: `arg4` is one number per axis and cannot
    # say it, so the pad stays.
    @test DKP.foldconvpad(padgraph(; amount = [1, 2, 1, 1]))[2] == 0
    @test DKP.foldconvpad(padgraph(; amount = [1, 1, 0, 2]))[2] == 0
    # A non-zero fill is not what the convolution substitutes.
    @test DKP.foldconvpad(padgraph(; value = 1))[2] == 0
    # The convolution already pads; adding them is right arithmetically and is
    # not what this does.
    @test DKP.foldconvpad(padgraph(; convpad = [1, 1]))[2] == 0
    # Transposed convolutions read `arg4` as output cropping, not input padding.
    @test DKP.foldconvpad(padgraph(; transposed = true))[2] == 0
    # A second reader still needs the padded tensor.
    @test DKP.foldconvpad(padgraph(; extrareader = true))[2] == 0

    # Padding only the innermost axis is two numbers, and the other axis is
    # then zero.
    g2, n2 = DKP.foldconvpad(padgraph(; amount = [1, 1]))
    @test n2 == 1
    @test DKP.ints(only(g2.ops).attrs["arg4"]) == [0, 1]

    # Padding a non-spatial axis as well is more than `arg4` can carry.
    @test DKP.foldconvpad(padgraph(; amount = [1, 1, 1, 1, 1, 1]))[2] == 0
end
