"""
A modulated deformable convolution, built as a matrix once instead of per output
channel.

The direct kernel walks one thread per output element and re-gathers every tap for
every OUTPUT channel. Nothing in that gather depends on the output channel — the
offsets and the mask are indexed by pixel, deform group and tap, and the bilinear
sample by the INPUT channel — so at `Cout = 64` it reads the same sample
sixty-four times. The note on the kernel used to say im2col "does not apply,
because there is no fixed matrix to build": the PATTERN is data-dependent, the
matrix is not, and building it once turns the rest into a GEMM.

Measured on BasicVSR++, whose `SecondOrderDeformableAlignment` runs sixteen a clip:
52.4 ms each and 71% of the whole frame, against 3.4 ms. The model goes 1005 ->
161 ms.

The two routes agree to a few ULPs, not bit for bit. The im2col holds `m * sample`
and the GEMM multiplies it by the same weights, but a GEMM is free to reassociate
its reduction and does: on BasicVSR++'s own shapes (`K = 1152`) the results are in
fact bit-identical, and at the small `K` here they differ in the last place. So the
tolerance is tight rather than exact — 1e-6 relative, which is far below anything
a wrong tap order, deform group or mask index could produce, all of which move
whole terms.
"""

using Test, DNNKernels, Mantle, KernelAbstractions

const DKF = DNNKernels

# A weight is looked up by its `key`, not by its id, so a weight buffer names itself.
dbuf(id, kind, shape, T) =
    DKF.Buffer(id, kind, Any[shape...], T, kind === :weight ? id : "", (0, 4), "", "",
               Dict{String,Any}())

"""A one-op `deform_conv2d` graph at the given geometry.

Shapes are TORCH's, because `Buffer.shape` is and `evalshape` reverses it: `x` is
`(N, Cin, H, W)` and the weight `(Cout, Cin, KH, KW)`, which reversed are the
`(W, H, Cin, N)` and `(KW, KH, Cin, Cout)` the kernels read. Writing them in Julia
order instead builds a graph whose every extent is wrong and whose two routes then
disagree for a reason that has nothing to do with either.
"""
function deformgraph(; W = 8, H = 8, Cin = 4, Cout = 6, K = 3, dg = 2,
                      stride = 1, pad = 1, T = Float32)
    OW = (W + 2pad - K) ÷ stride + 1
    OH = (H + 2pad - K) ÷ stride + 1
    bufs = Dict{String,DKF.Buffer}(
        "x"      => dbuf("x", :input, [1, Cin, H, W], T),
        "w"      => dbuf("w", :weight, [Cout, Cin, K, K], T),
        "offset" => dbuf("offset", :input, [1, 2 * dg * K * K, OH, OW], T),
        "mask"   => dbuf("mask", :input, [1, dg * K * K, OH, OW], T),
        "bias"   => dbuf("bias", :weight, [Cout], T),
        "y"      => dbuf("y", :transient, [1, Cout, OH, OW], T))
    ins = ["x", "w", "offset", "mask", "bias"]
    attrs = Dict{String,Any}("arg5" => stride, "arg6" => stride,
                             "arg7" => pad, "arg8" => pad,
                             "arg9" => 1, "arg10" => 1,
                             "arg11" => 1, "arg12" => dg, "arg13" => true)
    DKF.Graph("deform", String[], ["x", "offset", "mask"], ["y"], bufs,
              ["x", "w", "offset", "mask", "bias", "y"],
              [DKF.Op("y", "torchvision.deform_conv2d.default", ins, "y", attrs)],
              Vector{String}[])
end

for be in Mantle.eachbackend()
    # `local`: a `dev` global exists in the test session, and a soft-scope
    # assignment to a name that is also a global is a warning rather than a rule.
    local dev = Mantle.todevice(be)
    @testset "im2col and the direct kernel agree — $(nameof(typeof(be)))" begin
        for (dg, stride, Cin, Cout) in ((2, 1, 4, 6), (1, 1, 4, 6), (2, 2, 4, 6), (2, 1, 8, 4))
            g = deformgraph(; dg, stride, Cin, Cout)
            # Weights and inputs that exercise the gather: offsets large enough to
            # leave the grid, so the out-of-range branch is taken too.
            wsz = DKF.evalshape(g.buffers["w"].shape, (;))
            # `0.5f0`, not `0.5`: a Float64 literal promotes the whole array and Metal
            # has no `double`, so the upload refuses rather than the kernel.
            wts = Dict{String,Any}(
                "w" => reshape(Float32.((1:prod(wsz)) .% 17) ./ 17f0 .- 0.5f0, wsz),
                "bias" => Float32.((1:DKF.evalshape(g.buffers["bias"].shape, (;))[1]) ./ 7f0))
            function run(useim2col)
                DKF.DEFORM_IM2COL[] = useim2col
                mg, ec = DKF.emitgraph(dev, g, DKF.residentweights(dev, g, wts), (;))
                p = Mantle.Plan(mg); Mantle.record!(p)
                for id in g.inputs
                    s = Mantle.storage(ec.res[id])
                    v = id == "offset" ?
                        Float32.(3 .* sin.(1:length(s))) :       # off the grid on purpose
                        Float32.((1:length(s)) .% 23) ./ 23 .- 0.5
                    copyto!(s, convert(Array{eltype(s)}, reshape(v, size(s))))
                end
                Mantle.run!(p); Mantle.waitidle(dev)
                o = DKF.tohost(Mantle.storage(ec.res["y"]))
                names = [String(x.name) for x in mg.passes]
                Mantle.free!(p); DKF.freeowned!(ec)
                (o, names)
            end
            a, na = run(true)
            b, nb = run(false)
            DKF.DEFORM_IM2COL[] = true
            @test all(isfinite, a)
            sc = maximum(abs, b)
            @test maximum(abs, Float64.(a) .- Float64.(b)) / sc < 1e-6
            # …and that it really was the other route each time, so an accidental
            # no-op switch cannot make this pass by comparing a thing with itself.
            @test any(n -> endswith(n, ".im2col"), na)
            @test !any(n -> endswith(n, ".im2col"), nb)
        end
    end
end
