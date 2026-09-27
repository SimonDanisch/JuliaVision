"""
`halfconvs`: convolutions on fp16 operands in an fp32 graph.

The Qwen-Image 2.1 VAE is the case it was written for. Exported at fp16 it
decoded unscaled `randn` latents to 80.9% NaN; its residual stream reaches 3.5e5
while every normalised convolution input stays under 23. A pass that cast every
convolution input would have been fine on the VAE's real latents and wrong on
those: ROCm returned NaN, and Vulkan returned no NaN and a wrong picture, which
is why the checks below compare against the fp32 graph and do not stop at NaN.

The graph here has one convolution of each kind the pass tells apart:

  * `y1` reads RMS norm -> scale -> gamma -> SiLU, bounded by the graph itself
    (`sqrt(C) * max|gamma|`), so its operands are cast as they are;
  * `y2` is a 3x3 over a raw input, unbounded, so it is scaled at run time;
  * `y3` is a 1x1 over the same input, unbounded, and stays fp32.
"""

using Test, DNNKernels, Mantle

const DKH = DNNKernels

hcbuf(id, kind, shape, T; key = "") =
    DKH.Buffer(id, kind, Any[shape...], T, key, (0, 0), "", "", Dict{String,Any}())

convattrs(pad) = Dict{String,Any}("arg3" => Any[1, 1], "arg4" => Any[pad, pad],
                                  "arg5" => Any[1, 1], "arg6" => false,
                                  "arg7" => Any[0, 0], "arg8" => 1)

function hcgraph(; C = 32, Co = 32, H = 16, W = 16)
    act, T = [1, C, H, W], Float32
    bufs = Dict{String,DKH.Buffer}(
        "x" => hcbuf("x", :external, act, T), "r" => hcbuf("r", :external, act, T),
        "n" => hcbuf("n", :transient, [1, 1, H, W], T),
        "nc" => hcbuf("nc", :transient, [1, 1, H, W], T),
        (id => hcbuf(id, :transient, act, T) for id in ("d", "s1", "g", "sg", "a"))...,
        "gamma" => hcbuf("gamma", :weight, [C, 1, 1], T; key = "gamma"),
        (id => hcbuf(id, :transient, [1, Co, H, W], T) for id in ("y1", "y2", "y3"))...,
        "w1" => hcbuf("w1", :weight, [Co, C, 3, 3], T; key = "w1"),
        "w2" => hcbuf("w2", :weight, [Co, C, 3, 3], T; key = "w2"),
        "w3" => hcbuf("w3", :weight, [Co, C, 1, 1], T; key = "w3"),
        (id => hcbuf(id, :weight, [Co], T; key = id) for id in ("b1", "b2", "b3"))...)
    ops = DKH.Op[
        DKH.Op("n", "linalg_vector_norm.default", ["x"], "n",
               Dict{String,Any}("arg1" => 2, "arg2" => Any[1], "arg3" => true)),
        DKH.Op("nc", "clamp.default", ["n"], "nc", Dict{String,Any}("arg1" => 1e-12)),
        DKH.Op("d", "div.Tensor", ["x", "nc"], "d", Dict{String,Any}()),
        DKH.Op("s1", "mul.Tensor", ["d"], "s1", Dict{String,Any}("arg1" => sqrt(C))),
        DKH.Op("g", "mul.Tensor", ["s1", "gamma"], "g", Dict{String,Any}()),
        DKH.Op("sg", "sigmoid.default", ["g"], "sg", Dict{String,Any}()),
        DKH.Op("a", "mul.Tensor", ["g", "sg"], "a", Dict{String,Any}()),
        DKH.Op("y1", "convolution.default", ["a", "w1", "b1"], "y1", convattrs(1)),
        DKH.Op("y2", "convolution.default", ["r", "w2", "b2"], "y2", convattrs(1)),
        DKH.Op("y3", "convolution.default", ["r", "w3", "b3"], "y3", convattrs(0)),
    ]
    order = ["x", "r", "gamma", "w1", "w2", "w3", "b1", "b2", "b3",
             "n", "nc", "d", "s1", "g", "sg", "a", "y1", "y2", "y3"]
    DKH.Graph("hc", String[], ["x", "r"], ["y1", "y2", "y3"], bufs, order, ops, Vector{String}[])
end

function hcweights(; C = 32, Co = 32)
    rnd(dims...) = randn(Float32, dims...) ./ sqrt(Float32(prod(dims[1:end-1])))
    Dict{String,Any}("gamma" => 1f0 .+ 0.5f0 .* randn(Float32, 1, 1, C),
                     "w1" => rnd(3, 3, C, Co), "w2" => rnd(3, 3, C, Co), "w3" => rnd(1, 1, C, Co),
                     "b1" => randn(Float32, Co), "b2" => randn(Float32, Co), "b3" => randn(Float32, Co))
end

@testset "halfconvs tells the three kinds of convolution apart" begin
    g, w = hcgraph(), hcweights()
    bound = DKH.magnitudebounds(g, w)
    # |x / max(‖x‖, eps)| <= 1, times sqrt(C), times max|gamma|; SiLU never grows it.
    @test bound("a") ≈ sqrt(32) * maximum(abs, w["gamma"])
    @test bound("sg") == 1.0
    @test bound("r") == Inf
    @test bound("y1") == Inf

    g2, n = DKH.halfconvs(g, w)
    @test n == (; plain = 1, scaled = 1, kept = 1)
    conv(id) = only(op for op in g2.ops if op.id == id && op.aten == "convolution.default")
    @test g2.buffers[conv("y1").ins[1]].dtype === Float16
    @test g2.buffers[conv("y1").ins[2]].dtype === Float16
    @test conv("y1").out == "y1"                          # the result stays the fp32 buffer
    @test g2.buffers[conv("y2").ins[1]].dtype === Float16
    @test length(conv("y2").ins) == 2                     # the bias moved to the rescale
    rescale = only(op for op in g2.ops if op.aten == "halfconvs.rescale")
    @test rescale.out == "y2" && rescale.ins[3] == "b2"
    @test conv("y3").ins == ["r", "w3", "b3"]             # untouched
    @test all(b -> b.dtype === Float32, (g2.buffers[id] for id in ("y1", "y2", "y3")))

    # A weight outside fp16's range keeps its convolution in fp32 whatever the input.
    wbig = hcweights(); wbig["w1"][1] = 1f5
    @test DKH.halfconvs(g, wbig)[2].plain == 0
end

function halfconvscheck(backend)
    @testset "halfconvs keeps fp32's range — $(nameof(typeof(backend)))" begin
        w = hcweights()
        m32 = DKH.Model(Dict("hc" => hcgraph()), copy(w); backend)
        m16 = DKH.Model(Dict("hc" => hcgraph()), copy(w); backend, halfconvs = true)
        # `r` well past fp16's 65504, then back inside it through the SAME plan:
        # the scale is taken from the input at run time, not when the plan was
        # recorded.
        for rscale in (1f5, 1f0)
            x = randn(Float32, 16, 16, 32, 1)
            r = rscale .* randn(Float32, 16, 16, 32, 1)
            want = map(Array, DKH.call(m32, "hc", DKH.toback(backend, x), DKH.toback(backend, r); dims = (;)))
            got = map(Array, DKH.call(m16, "hc", DKH.toback(backend, x), DKH.toback(backend, r); dims = (;)))
            for (k, (a, b)) in enumerate(zip(got, want))
                @test eltype(a) === Float32
                @test all(isfinite, a)
                # fp16 operands, fp32 accumulation: relative to the result's scale.
                @test maximum(abs.(a .- b)) / maximum(abs.(b)) < 5f-3
            end
            @test got[3] == want[3]                    # the 1x1 is the fp32 convolution
        end
    end
end

foreach(halfconvscheck, Mantle.eachbackend())
