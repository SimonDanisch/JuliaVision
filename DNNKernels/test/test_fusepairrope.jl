"""
The other rotary spelling: a rotation of adjacent PAIRS.

`fuserope` collapses `cat(-x[H/2:], x[:H/2]) * sin + x * cos`, which is how the
HF models write it. Qwen-Image 2.1 writes the complex form — view the head as
`H/2` pairs, rotate each, interleave the results back — and nothing matched it,
so its two rotations stayed eight ops each.

What that cost was the LAYOUT rather than the arithmetic. The two components
are a stride-2 read each, so both were materialised; the interleave was another
pass. One layer at 1024² measured 1.95 ms of squeezes and 1.16 of interleave on
top of 1.84 of arithmetic, against 1.6 ms for both rotations as one kernel, and
the layer went 177.9 ms to 169.0.

Checked here: that the pattern is recognised, that it computes the rotation,
and that the arrangements which are NOT this rotation are declined — the last
being the point, since a loose match silently changes what a model computes.
"""

using Test, DNNKernels, Mantle

const DKP = DNNKernels

pbuf(id, kind, shape, T; key = "", of = "", viewop = "", attrs = Dict{String,Any}()) =
    DKP.Buffer(id, kind, Any[shape...], T, key, (0, 9), of, viewop, attrs)

"""
The exported chain, as a graph. `swap` puts the addition's two products the
other way round (still the same function); `crossed` gives the subtraction its
components the wrong way round (a different function, and it must decline).
"""
function pairropegraph(; L = 4, H = 2, P = 2, T = Float32,
                       swap = false, crossed = false, extrareader = false,
                       tailcast = true, sharedview = true)
    C = 2P
    A(p...) = Dict{String,Any}(p...)
    b = Dict{String,DKP.Buffer}()
    b["x"] = pbuf("x", :external, (1, L, H, C), T)
    b["cos"] = pbuf("cos", :external, (1, L, 1, P), T)
    b["sin"] = pbuf("sin", :external, (1, L, 1, P), T)
    b["v"] = pbuf("v", :view, (1, L, H, P, 2), T; of = "x", viewop = "view.default")
    b["v2"] = pbuf("v2", :view, (1, L, H, P, 2), T; of = "x", viewop = "view.default")
    for (id, lo) in (("s0", 0), ("s1", 1))
        parent = (id == "s1" && !sharedview) ? "v2" : "v"
        b[id] = pbuf(id, :view, (1, L, H, P, 1), T; of = parent, viewop = "slice.Tensor",
                     attrs = A("arg1" => 4, "arg2" => lo, "arg3" => lo + 1))
    end
    b["a"] = pbuf("a", :view, (1, L, H, P), T; of = "s0", viewop = "squeeze.dims")
    b["b"] = pbuf("b", :view, (1, L, H, P), T; of = "s1", viewop = "squeeze.dims")
    for id in ("ac", "bs", "bc", "as", "sub", "add")
        b[id] = pbuf(id, :transient, (1, L, H, P), T)
    end
    b["usub"] = pbuf("usub", :view, (1, L, H, P, 1), T; of = "sub", viewop = "unsqueeze.default")
    b["uadd"] = pbuf("uadd", :view, (1, L, H, P, 1), T; of = "add", viewop = "unsqueeze.default")
    b["cat"] = pbuf("cat", :transient, (1, L, H, P, 2), T)
    b["catv"] = pbuf("catv", :view, (1, L, H, C), T; of = "cat", viewop = "view.default")
    b["out"] = pbuf("out", :transient, (1, L, H, C), Float16)

    # `crossed` swaps which component the subtraction scales by cosine.
    first_, second_ = crossed ? ("b", "a") : ("a", "b")
    ops = DKP.Op[
        DKP.Op("ac", "mul.Tensor", [first_, "cos"], "ac", A()),
        DKP.Op("bs", "mul.Tensor", [second_, "sin"], "bs", A()),
        DKP.Op("sub", "sub.Tensor", ["ac", "bs"], "sub", A()),
        DKP.Op("bc", "mul.Tensor", ["b", "cos"], "bc", A()),
        DKP.Op("as", "mul.Tensor", ["a", "sin"], "as", A()),
        DKP.Op("add", "add.Tensor", swap ? ["as", "bc"] : ["bc", "as"], "add", A()),
        DKP.Op("cat", "cat.default", ["usub", "uadd"], "cat",
               A("arg1" => -1, "arg0" => ["\$usub", "\$uadd"]))]
    outid = "cat"
    if tailcast
        push!(ops, DKP.Op("out", "_to_copy.default", ["catv"], "out", A()))
        outid = "out"
    end
    if extrareader
        b["peek"] = pbuf("peek", :transient, (1, L, H, P), T)
        push!(ops, DKP.Op("peek", "clone.default", ["sub"], "peek", A()))
    end
    outs = extrareader ? [outid, "peek"] : [outid]
    DKP.Graph("pairrope", String[], ["x", "cos", "sin"], outs, b, collect(keys(b)), ops)
end

@testset "the interleaved rotary is recognised" begin
    g, n = DKP.fusepairrope(pairropegraph())
    @test n == 1
    op = only(o for o in g.ops if o.aten == "fused.pairrope")
    @test op.ins == ["x", "cos", "sin"]
    @test op.attrs["P"] == 2
    # The narrowing cast on the way out is the kernel's store.
    @test op.out == "out"
    @test !any(o -> o.aten in ("mul.Tensor", "sub.Tensor", "add.Tensor", "cat.default"), g.ops)

    # The addition's two products may come either way round.
    @test last(DKP.fusepairrope(pairropegraph(swap = true))) == 1

    # Without the cast it writes the concatenation's own buffer.
    g2, n2 = DKP.fusepairrope(pairropegraph(tailcast = false))
    @test n2 == 1
    @test only(o for o in g2.ops if o.aten == "fused.pairrope").out == "cat"
end

@testset "what is not this rotation is declined" begin
    # The subtraction scaling the wrong component by cosine is a different
    # function, not a different spelling.
    @test last(DKP.fusepairrope(pairropegraph(crossed = true))) == 0
    # Two components of two different views are not a pair.
    @test last(DKP.fusepairrope(pairropegraph(sharedview = false))) == 0
    # Something outside the group reading an intermediate keeps it alive.
    @test last(DKP.fusepairrope(pairropegraph(extrareader = true))) == 0
    # `FUSEROPE` turns it off with the other one.
    DKP.FUSEROPE[] = false
    @test last(DKP.fusepairrope(Dict("g" => pairropegraph()))) == 0
    DKP.FUSEROPE[] = true
end

@testset "the fused rotation computes the rotation" begin
    backend = Mantle.LavaBackend()
    L, H, P = 4, 2, 2
    C = 2P
    chain = pairropegraph(; L, H, P)
    fused, n = DKP.fusepairrope(chain)
    @test n == 1
    fused, _ = DKP.dropdead(fused)

    # Julia order is the reverse of the graph's: x is `(C, H, L, 1)`.
    xh = Float32.(reshape(1:(C * H * L), C, H, L, 1)) ./ 8
    ch = Float32.(reshape([cos(0.1i) for i in 1:(P * L)], P, 1, L, 1))
    sh = Float32.(reshape([sin(0.1i) for i in 1:(P * L)], P, 1, L, 1))
    args = (DKP.toback(backend, xh), DKP.toback(backend, ch), DKP.toback(backend, sh))

    mc = DKP.Model(Dict("pairrope" => chain), Dict{String,Any}(), backend, 1, 1, 1)
    mf = DKP.Model(Dict("pairrope" => fused), Dict{String,Any}(), backend, 1, 1, 1)
    gotc = Array(only(DKP.call(mc, "pairrope", args...; dims = (;))))
    gotf = Array(only(DKP.call(mf, "pairrope", args...; dims = (;))))

    want = similar(gotf)
    for l in 1:L, h in 1:H, j in 1:P
        a, b = xh[2j - 1, h, l, 1], xh[2j, h, l, 1]
        cv, sv = ch[j, 1, l, 1], sh[j, 1, l, 1]
        want[2j - 1, h, l, 1] = Float16(a * cv - b * sv)
        want[2j, h, l, 1] = Float16(a * sv + b * cv)
    end
    @test gotf == want
    # And the chain it replaced, to the fp16 output's own resolution. Not
    # bit-identical: the elementwise fuser contracts the chain's two products
    # into a `muladd` and this does not.
    @test maximum(abs, gotf .- gotc) <= 2eps(Float16) * maximum(abs, Float32.(gotc))

    # The INTERPRETED runner too, which is a second implementation of the same
    # op and the one an unrecorded model reaches. A wrong head count there is
    # invisible until something runs without a plan.
    interp = Array(DKP.execute!(fused, Dict(zip(fused.inputs, args)),
                                Dict{String,Any}(); dims = (;), backend)[
                       DKP.viewroot(fused, only(fused.outputs))])
    @test interp == want
end
