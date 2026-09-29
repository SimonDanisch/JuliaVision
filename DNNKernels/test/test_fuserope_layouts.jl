"""
`fused.rope` reads its tables the way the graph broadcast them, in every layout.

`fuserope` collapses `x * cos + cat(-x[H/2:], x[:H/2]) * sin` into one op, and
the kernel behind it assumed the tables were `(H, T)` against `x` as
`(H, T, heads, batch)` in Julia order, which is how every HF language model
lays out a query. Qwen3-VL's vision tower rotates `(H, heads, tokens)` with
`(H, 1, tokens)` tables: its tokens are on the THIRD axis, and the kernel read
the cosine of token `h` for every head `h`. The tower's output came out 99%
wrong against PyTorch, with nothing raised; the model ran fine with the rewrite
turned off. A batch whose tables differ per sample was wrong the same way.

Each layout here is the exported chain, fused and unfused, against the rotation
computed on the host.
"""

using Test, DNNKernels, Mantle

const DKR = DNNKernels

rbuf(id, kind, shape, T; of = "", viewop = "", attrs = Dict{String,Any}()) =
    DKR.Buffer(id, kind, Any[shape...], T, "", (0, 9), of, viewop, attrs)

"""The half-rotation as `run_decompositions` leaves it, over torch shapes."""
function ropegraph(xshape, cshape)
    D = last(xshape)
    half = D ÷ 2
    nd = length(xshape)
    A(p...) = Dict{String,Any}(p...)
    hshape = (xshape[1:end-1]..., half)
    b = Dict{String,DKR.Buffer}()
    b["x"] = rbuf("x", :external, xshape, Float32)
    b["cos"] = rbuf("cos", :external, cshape, Float32)
    b["sin"] = rbuf("sin", :external, cshape, Float32)
    b["lo"] = rbuf("lo", :view, hshape, Float32; of = "x", viewop = "slice.Tensor",
                   attrs = A("arg1" => nd - 1, "arg2" => 0, "arg3" => half))
    b["hi"] = rbuf("hi", :view, hshape, Float32; of = "x", viewop = "slice.Tensor",
                   attrs = A("arg1" => nd - 1, "arg2" => half, "arg3" => D))
    b["neg"] = rbuf("neg", :transient, hshape, Float32)
    b["cat"] = rbuf("cat", :transient, xshape, Float32)
    b["xs"] = rbuf("xs", :transient, xshape, Float32)
    b["xc"] = rbuf("xc", :transient, xshape, Float32)
    b["out"] = rbuf("out", :transient, xshape, Float32)
    ops = DKR.Op[
        DKR.Op("neg", "neg.default", ["hi"], "neg", A()),
        DKR.Op("cat", "cat.default", ["neg", "lo"], "cat", A("arg1" => -1, "arg0" => ["\$neg", "\$lo"])),
        DKR.Op("xs", "mul.Tensor", ["cat", "sin"], "xs", A()),
        DKR.Op("xc", "mul.Tensor", ["x", "cos"], "xc", A()),
        DKR.Op("out", "add.Tensor", ["xc", "xs"], "out", A())]
    DKR.Graph("rope", String[], ["x", "cos", "sin"], ["out"], b, collect(keys(b)), ops)
end

"""`x * cos + rotate_half(x) * sin` in torch index order, broadcasting the tables."""
function hostrope(x, c, s)
    D = size(x, ndims(x))
    half = D ÷ 2
    rot = cat(-selectdim(x, ndims(x), half+1:D), selectdim(x, ndims(x), 1:half); dims = ndims(x))
    x .* c .+ rot .* s
end

torchorder(a) = permutedims(a, ndims(a):-1:1)

@testset "fused.rope in $name" for (name, xshape, cshape) in (
        ("an HF language model's layout", (1, 2, 5, 8), (1, 1, 5, 8)),
        ("a vision tower's layout", (6, 3, 8), (6, 1, 8)),
        ("a batch with tables per sample", (2, 2, 3, 8), (2, 1, 3, 8)))
    backend = Mantle.LavaBackend()
    chain = ropegraph(xshape, cshape)
    fused, n = DKR.fuserope(chain)
    @test n == 1
    fused, _ = DKR.dropdead(fused)
    @test any(o -> o.aten == "fused.rope", fused.ops)

    # Values in torch order, handed over in Julia order.
    xt = Float32.(reshape(1:prod(xshape), xshape...)) ./ 16
    ct = Float32.(reshape([cos(0.3i) for i in 1:prod(cshape)], cshape...))
    st = Float32.(reshape([sin(0.3i) for i in 1:prod(cshape)], cshape...))
    want = torchorder(hostrope(xt, ct, st))
    args = (DKR.toback(backend, torchorder(xt)), DKR.toback(backend, torchorder(ct)),
            DKR.toback(backend, torchorder(st)))

    mc = DKR.Model(Dict("rope" => chain), Dict{String,Any}(), backend, 1, 1, 1)
    mf = DKR.Model(Dict("rope" => fused), Dict{String,Any}(), backend, 1, 1, 1)
    gotc = Array(only(DKR.call(mc, "rope", args...; dims = (;))))
    gotf = Array(only(DKR.call(mf, "rope", args...; dims = (;))))
    @test size(gotf) == size(want)
    @test maximum(abs, gotc .- want) <= 1f-5 * maximum(abs, want)
    @test maximum(abs, gotf .- want) <= 1f-5 * maximum(abs, want)
end

@testset "tables that do not broadcast are refused, not guessed" begin
    backend = Mantle.LavaBackend()
    # Torch would refuse this multiply; a graph that says it is one is wrong.
    fused, n = DKR.fuserope(ropegraph((4, 2, 8), (2, 4, 8)))
    @test n == 1
    fused, _ = DKR.dropdead(fused)
    m = DKR.Model(Dict("rope" => fused), Dict{String,Any}(), backend, 1, 1, 1)
    z(s) = DKR.toback(backend, zeros(Float32, reverse(s)...))
    @test_throws ErrorException DKR.call(m, "rope", z((4, 2, 8)), z((2, 4, 8)), z((2, 4, 8)); dims = (;))
end
