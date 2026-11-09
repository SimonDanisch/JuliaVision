using Test, Random, Mantle, DNNKernels

const MSI = Mantle
const DSI = DNNKernels

@testset "Float32 convolution via bounded im2col and scalar GEMM" begin
    dev = isdefined(Main, :MANTLE_TEST_BACKEND) ? MSI.Device(Main.MANTLE_TEST_BACKEND) : MSI.Device()
    rng = Random.Xoshiro(130)
    for (batch, hasbias, act) in ((1, true, :relu), (2, false, :none))
        # Non-tile-sized output and simultaneous stride, dilation and padding.
        # Two images also exercise the NPQ -> (W,H,C,N) epilogue permutation.
        ow, ci, co, kw = 257, 256, 256, 3
        sx, px, dx = 2, 1, 2
        width = sx * (ow - 1) - 2px + dx * (kw - 1) + 1
        xh = 0.02f0 .* randn(rng, Float32, width, 1, ci, batch)
        wh = 0.02f0 .* randn(rng, Float32, kw, 1, ci, co)
        bh = hasbias ? 0.02f0 .* randn(rng, Float32, co) : nothing
        x64, w64 = Float64.(xh), Float64.(wh)
        b64 = bh === nothing ? nothing : Float64.(bh)
        od = (ow, 1, co, batch)
        want = map(CartesianIndices(od)) do i
            v = DSI.conv2d(Tuple(i), x64, w64, b64, Val(sx), Val(1), Val(px), Val(0), Val(dx), Val(1), Val(1))
            act === :relu ? max(v, 0.0) : v
        end
        g = MSI.Graph(dev)
        res = Dict{String,Any}("x" => MSI.Buffer(dev, xh), "w" => MSI.Buffer(dev, wh),
                               "y" => MSI.Buffer(dev, Float32, od))
        ins = ["x", "w"]
        bh === nothing || (res["b"] = MSI.Buffer(dev, bh); push!(ins, "b"))
        attrs = Dict{String,Any}("arg3" => [1, sx], "arg4" => [0, px],
            "arg5" => [1, dx], "arg6" => false, "arg7" => [0, 0], "arg8" => 1, "act" => string(act))
        op = DSI.Op("y", "convolution.default", ins, "y", attrs)
        aten = DSI.Graph("scalarconv", String[], String[], String[], Dict{String,DSI.Buffer}(),
                         String[], DSI.Op[], Vector{Vector{String}}())
        ctx = DSI.EmitCtx(aten, g, dev, (;), res, Set{String}(), Ref("y"), Any[], Dict{String,Any}())
        pl = nothing
        try
            @test DSI.scalarconvworth(res["y"], res["x"], res["w"])
            fill!(res["y"], NaN32)
            MSI.waitidle(dev)
            # Exercise the shared declaration, independent of a backend library
            # choosing its own convolution before the portable selector.
            DSI.emitconvscalar!(ctx, op, res["x"], res["w"], get(res, "b", nothing), res["y"],
                                (sx, 1), (px, 0), (dx, 1), act)
            future = MSI.Readback(g, res["y"])
            pl = MSI.record!(MSI.Plan(g; budget=Inf))
            n = MSI.submissions(dev)
            MSI.run!(pl)
            got = fetch(future)
            @test MSI.submissions(dev) == n + 1
            @test all(isfinite, got)
            @test maximum(abs.(got .- want)) < 1e-5
        finally
            pl === nothing || MSI.free!(pl)
            foreach(MSI.free!, values(res))
        end
    end
end
