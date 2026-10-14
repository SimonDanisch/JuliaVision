"""
What porting PP-OCRv6 found, one test each.

  * A batch norm folded into a TRANSPOSED convolution scaled the weight's input
    channels: the output-channel axis of `(kx, ky, Cout, Cin)` is the third, not
    the last. With `Cin == Cout` every shape agreed and the detector's head came
    out 5.04 off.
  * Grouped and depthwise convolutions had no declared lowering and refused.
  * `mean.dim` over a whole plane gave each output one thread: a global pool of
    a 320x448 map per channel was 651 ms. Now one workgroup per output.
  * (`max.dim` without `keepdim` is in `test_maxdim.jl`.)
"""

using Test, Random
import DNNKernels, Mantle
const DKR = DNNKernels

rbuf(id, kind, shape, T; key = "", of = "", viewop = "", attrs = Dict{String,Any}()) =
    DKR.Buffer(id, kind, Any[shape...], T, key, (0, 0), of, viewop, attrs)

"""`(transposed) convolution -> batch norm -> getitem 0`, the shape an export has."""
function convbngraph(; transposed::Bool, C = 4)
    bufs = Dict{String,DKR.Buffer}(
        "x" => rbuf("x", :input, [1, C, 8, 8], Float32),
        "w" => rbuf("w", :weight, transposed ? [C, C, 2, 2] : [C, C, 3, 3], Float32; key = "w"),
        "g" => rbuf("g", :weight, [C], Float32; key = "g"),
        "b" => rbuf("b", :weight, [C], Float32; key = "b"),
        "m" => rbuf("m", :weight, [C], Float32; key = "m"),
        "v" => rbuf("v", :weight, [C], Float32; key = "v"),
        "c" => rbuf("c", :transient, [1, C, 8, 8], Float32),
        "bn" => rbuf("bn", :transient, [1, C, 8, 8], Float32),
        "bn0" => rbuf("bn0", :view, [1, C, 8, 8], Float32; of = "bn", viewop = "getitem",
                      attrs = Dict{String,Any}("arg1" => 0)),
    )
    ops = DKR.Op[
        DKR.Op("c", "convolution.default", ["x", "w"], "c",
               Dict{String,Any}("arg3" => Any[transposed ? 2 : 1, transposed ? 2 : 1],
                                "arg4" => Any[transposed ? 0 : 1, transposed ? 0 : 1],
                                "arg5" => Any[1, 1], "arg6" => transposed,
                                "arg7" => Any[0, 0], "arg8" => 1)),
        DKR.Op("bn", "_native_batch_norm_legit_no_training.default", ["c", "g", "b", "m", "v"], "bn",
               Dict{String,Any}("arg5" => 0.1, "arg6" => 1e-5)),
    ]
    DKR.Graph("g", String[], ["x"], ["bn0"], bufs, ["c", "bn"], ops, Vector{String}[])
end

@testset "batch norm folds into a transposed convolution's output channels" begin
    Random.seed!(7)
    C = 4
    for transposed in (false, true)
        # Julia's layout: `(kx, ky, Cin, Cout)`, or `(kx, ky, Cout, Cin)` transposed.
        W = randn(Float32, (transposed ? (2, 2) : (3, 3))..., C, C)
        γ, β, μ, v = rand(Float32, C) .+ 0.5f0, randn(Float32, C), randn(Float32, C), rand(Float32, C) .+ 0.5f0
        weights = Dict{String,Any}("w" => W, "g" => γ, "b" => β, "m" => μ, "v" => v)
        gs, w2, n = DKR.foldbatchnorm(Dict("g" => convbngraph(; transposed, C)), weights)
        @test n == 1
        s = γ ./ sqrt.(v .+ 1.0f-5)
        out = transposed ? 3 : 4
        expected = W .* reshape(s, ntuple(k -> k == out ? C : 1, 4))
        @test w2["g|c|foldbn_weight"] ≈ expected
        @test w2["g|c|foldbn_bias"] ≈ β .- μ .* s
    end
    # Grouped transposed weights interleave groups with output channels: not folded.
    g = convbngraph(; transposed = true)
    c = g.ops[1]
    g.ops[1] = DKR.Op(c.id, c.aten, c.ins, c.out, merge(c.attrs, Dict{String,Any}("arg8" => 2)))
    weights = Dict{String,Any}("w" => randn(Float32, 2, 2, 2, 4), "g" => ones(Float32, 4), "b" => zeros(Float32, 4),
                               "m" => zeros(Float32, 4), "v" => ones(Float32, 4))
    @test DKR.foldbatchnorm(Dict("g" => g), weights)[3] == 0
end

"""Emit one op over host inputs on `dev`, run it, and return its result."""
function deviceop(dev, aten, ins, outshape, attrs)
    g = Mantle.Graph(dev)
    res = Dict{String,Any}(k => Mantle.Buffer(dev, v) for (k, v) in ins)
    res["y"] = Mantle.Buffer(dev, Float32, outshape)
    op = DKR.Op("y", aten, first.(ins), "y", attrs)
    emitctx = DKR.EmitCtx(DKR.Graph("y", String[], String[], String[], Dict{String,DKR.Buffer}(),
                                    String[], DKR.Op[], Vector{Vector{String}}()),
                          g, dev, NamedTuple(), res, Set{String}(), Ref("y"), Any[], Dict{String,Any}())
    DKR.emitop!(emitctx, op, Val(Symbol(aten)))
    plan = Mantle.Plan(g)
    Mantle.record!(plan)
    Mantle.run!(plan)
    Mantle.waitidle(dev)
    y = Array(Mantle.storage(res["y"]))
    Mantle.free!(plan)
    foreach(Mantle.free!, values(res))
    return y
end

"""A grouped 2-D convolution on the host, `(W, H, C, N)` and `(kx, ky, Cin/g, Cout)`."""
function hostconv(x, w, b, s, p, groups)
    Wd, Hd, _, N = size(x)
    KX, KY, CI, Cout = size(w)
    OW, OH = (Wd + 2p - KX) ÷ s + 1, (Hd + 2p - KY) ÷ s + 1
    y = zeros(Float32, OW, OH, Cout, N)
    per = Cout ÷ groups
    for n in 1:N, co in 1:Cout, oy in 1:OH, ox in 1:OW
        g = (co - 1) ÷ per
        acc = b[co]
        for ci in 1:CI, ky in 1:KY, kx in 1:KX
            ix, iy = (ox - 1) * s - p + kx, (oy - 1) * s - p + ky
            (1 <= ix <= Wd && 1 <= iy <= Hd) || continue
            acc += x[ix, iy, g * CI + ci, n] * w[kx, ky, ci, co]
        end
        y[ox, oy, co, n] = acc
    end
    return y
end

function grouped_and_pooled(be)
    dev = Mantle.todevice(be)
    @testset "grouped convolution and global pool on $(nameof(typeof(dev)))" begin
        Random.seed!(11)
        for (groups, C, k, s) in ((8, 8, 3, 1), (8, 8, 5, 2), (2, 8, 3, 1))
            x = randn(Float32, 17, 13, C, 1)
            w = randn(Float32, k, k, C ÷ groups, C)
            b = randn(Float32, C)
            p = k ÷ 2
            ref = hostconv(x, w, b, s, p, groups)
            y = deviceop(dev, "convolution.default", ["x" => x, "w" => w, "b" => b], size(ref),
                         Dict{String,Any}("arg3" => Any[s, s], "arg4" => Any[p, p], "arg5" => Any[1, 1],
                                          "arg6" => false, "arg7" => Any[0, 0], "arg8" => groups))
            @test y ≈ ref rtol = 1e-4
        end
        # Torch's dims 2 and 3 of (N, C, H, W) are Julia's first two. Long enough
        # for the workgroup reduction, and short enough for the per-thread one.
        for (w, h) in ((96, 80), (8, 8))
            x = randn(Float32, w, h, 5, 2)
            y = deviceop(dev, "mean.dim", ["a" => x], (1, 1, 5, 2),
                         Dict{String,Any}("arg1" => Any[2, 3], "arg2" => true))
            @test y ≈ sum(x; dims = (1, 2)) ./ (w * h) rtol = 1e-5
        end
    end
end

let bes = Mantle.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the grouped convolution and pool"
    foreach(grouped_and_pooled, bes)
end
