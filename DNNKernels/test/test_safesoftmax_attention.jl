"""
Attention as `torch.export` writes it when `scaled_dot_product_attention` is
exported rather than lowered: `_safe_softmax`, with the product in fp32 and the
scale split between q and k.

    q32 = to_copy(q16); qs = q32 * sq          k32 = to_copy(k16); ks = kᵀ * sk
    scores = bmm(qs, ks)
    p      = softmax(scores, -1)
    p      = where(!any(scores != -inf, -1), zeros_like(p), p)   # the NaN guard
    out    = bmm(p, v)

Qwen-Image 2.1 exports every one of its 32 joint attentions this way, and none
of them fused: the guard broke the `softmax -> bmm` adjacency, and the fp32
operands have no flash plan, so a 4096-query attention materialised 2.12 GB of
scores per layer and the model did not fit.

Two halves, and the second is the one that cost the time:

  * the REWRITE, host-only — that the guard is recognised, that the scaling is
    folded out to reach the fp16 operands, and that both are declined where
    dropping them would change the result.
  * the two RUNNERS agreeing. A folded scale lives in the op's attribute, and
    the interpreted `runop!` used to ignore it and run at `scale = 1` while the
    declared `emitsdpa!` read it. Same graph, same weights, `rel 1.8` apart, and
    every layer downstream of it still finite and plausible.
"""

using Test, DNNKernels, Mantle, Random, LinearAlgebra, KernelAbstractions

const DKA = DNNKernels

# (H, Lq, Lk, D) for the rewrite tests. Shapes are torch's, as a graph carries
# them; `Lq != Lk` on purpose — a joint attention queries the image tokens over
# the whole [text; image] sequence, and a matcher that assumes one length
# declines exactly the attentions this pass exists for.
const SS_H, SS_LQ, SS_LK, SS_D = 2, 4, 8, 16
const SS_SQ, SS_SK = 0.25, 0.5

"""The exported chain as a graph, with `tweak` applied to buffers and ops."""
function safesoftmaxgraph(; mask::Bool = false, fill::Real = 0,
                          outputs::Vector{String} = ["out"],
                          extraops::Vector = [])
    H, Lq, Lk, D = SS_H, SS_LQ, SS_LK, SS_D
    buf(id, shape; kind = :transient, dtype = Float16, of = "", viewop = "") =
        DKA.Buffer(id, kind, Any[shape...], dtype, "", (0, 0), of,
                   isempty(viewop) && !isempty(of) ? "view.default" : viewop,
                   Dict{String,Any}())
    bs = [
        buf("q16", (1, H, Lq, D); kind = :external),
        buf("k16", (1, H, Lk, D); kind = :external),
        buf("v16", (1, H, Lk, D); kind = :external),
        buf("q32", (1, H, Lq, D); dtype = Float32),
        buf("k32", (1, H, Lk, D); dtype = Float32),
        buf("v32", (1, H, Lk, D); dtype = Float32),
        buf("qs", (1, H, Lq, D); dtype = Float32),
        # k reaches its `mul` through a permute, so the operand walk has to find
        # the pre-image of a TRANSPOSED view — see `flashoperand`.
        buf("kp", (1, H, D, Lk); kind = :view, dtype = Float32, of = "k32",
            viewop = "permute.default"),
        buf("ks", (1, H, D, Lk); dtype = Float32),
        buf("q3", (H, Lq, D); kind = :view, dtype = Float32, of = "qs"),
        buf("k3", (H, D, Lk); kind = :view, dtype = Float32, of = "ks"),
        buf("v3", (H, Lk, D); kind = :view, dtype = Float32, of = "v32"),
        buf("scores", (H, Lq, Lk); dtype = Float32),
        buf("scores4", (1, H, Lq, Lk); kind = :view, dtype = Float32, of = "scores"),
        buf("masked", (1, H, Lq, Lk); dtype = Float32),
        buf("mask", (1, 1, Lq, Lk); kind = :external, dtype = Float32),
        buf("p", (1, H, Lq, Lk); dtype = Float32),
        buf("eqb", (1, H, Lq, Lk); dtype = Bool),
        buf("ln1", (1, H, Lq, Lk); dtype = Bool),
        buf("an", (1, H, Lq, 1); dtype = Bool),
        buf("ln2", (1, H, Lq, 1); dtype = Bool),
        buf("zeros", (1, H, Lq, Lk); dtype = Float32),
        buf("wh", (1, H, Lq, Lk); dtype = Float32),
        buf("wh3", (H, Lq, Lk); kind = :view, dtype = Float32, of = "wh"),
        buf("out", (H, Lq, D); dtype = Float32),
    ]
    buffers = Dict(b.id => b for b in bs)
    A(kv...) = Dict{String,Any}(kv...)
    softmaxin = mask ? "masked" : "scores4"
    ops = DKA.Op[
        DKA.Op("q32", "_to_copy.default", ["q16"], "q32", A("dtype" => "float32")),
        DKA.Op("k32", "_to_copy.default", ["k16"], "k32", A("dtype" => "float32")),
        DKA.Op("v32", "_to_copy.default", ["v16"], "v32", A("dtype" => "float32")),
        DKA.Op("qs", "mul.Scalar", ["q32"], "qs", A("arg1" => SS_SQ)),
        DKA.Op("ks", "mul.Scalar", ["kp"], "ks", A("arg1" => SS_SK)),
        DKA.Op("scores", "bmm.default", ["q3", "k3"], "scores", A()),
    ]
    mask && push!(ops, DKA.Op("masked", "add.Tensor", ["scores4", "mask"], "masked", A()))
    append!(ops, DKA.Op[
        DKA.Op("p", "_softmax.default", [softmaxin], "p", A("arg1" => -1, "arg2" => false)),
        DKA.Op("eqb", "eq.Scalar", ["scores4"], "eqb", A("arg1" => "-inf")),
        DKA.Op("ln1", "logical_not.default", ["eqb"], "ln1", A()),
        DKA.Op("an", "any.dim", ["ln1"], "an", A("arg1" => -1, "arg2" => true)),
        DKA.Op("ln2", "logical_not.default", ["an"], "ln2", A()),
        DKA.Op("zeros", "full_like.default", ["p"], "zeros", A("arg1" => fill)),
        DKA.Op("wh", "where.self", ["ln2", "zeros", "p"], "wh", A()),
        DKA.Op("out", "bmm.default", ["wh3", "v3"], "out", A()),
    ])
    append!(ops, extraops)
    inputs = mask ? ["q16", "k16", "v16", "mask"] : ["q16", "k16", "v16"]
    DKA.Graph("safesoftmax", String[], inputs, outputs, buffers,
              collect(keys(buffers)), ops)
end

@testset "safe-softmax attention fusion" begin
    fused, n = DKA.fuseattention(safesoftmaxgraph())
    @test n == 1
    op = only(o for o in fused.ops if o.aten == "fused.sdpa")
    # The fp16 pre-images, not the fp32 copies the `bmm` read: an fp32 operand
    # has no flash plan at all.
    @test op.ins == ["q16", "k16", "v16"]
    @test op.attrs["scale"] ≈ SS_SQ * SS_SK
    @test op.out == "out"
    # The guard, the softmax and the first product are gone with it.
    @test !any(o -> o.aten in ("_softmax.default", "where.self", "any.dim",
                               "eq.Scalar", "full_like.default"), fused.ops)

    # An additive mask makes the guard load-bearing: a fully masked row really
    # is all `-inf` there, and `softmax` of it is `NaN` the guard replaces.
    @test last(DKA.fuseattention(safesoftmaxgraph(mask = true))) == 0
    # A guard that fills with something other than zero is not this pattern.
    @test last(DKA.fuseattention(safesoftmaxgraph(fill = 1))) == 0
    # Something outside the group reading the scores keeps them alive.
    @test last(DKA.fuseattention(safesoftmaxgraph(outputs = ["out", "p"]))) == 0
    extra = [DKA.Op("peek", "clone.default", ["scores4"], "peek", Dict{String,Any}())]
    @test last(DKA.fuseattention(safesoftmaxgraph(extraops = extra))) == 0
end

# `fuseattn` exists because a fused attention and a RECORDED graph are not
# always both available: `emitgraph` needs the chosen plan to have a declared
# form, and `CoopMatSDPAPlan` has none. Qwen-Image 2.1's VAE decoder is the
# case, one mid-block head 1152 wide, so `qwenimagevae(record = true)` asks for
# the chain instead of the fusion. It is a switch on that one pass and not on
# `fuse`, which would also take the elementwise groups with it.
@testset "the attention fusion can be switched off on its own" begin
    fused = DKA.Model(Dict("g" => safesoftmaxgraph()), Dict{String,Any}();
                      backend = KernelAbstractions.CPU()).graphs["g"]
    @test count(o -> o.aten == "fused.sdpa", fused.ops) == 1
    @test !any(o -> o.aten == "_softmax.default", fused.ops)

    chain = DKA.Model(Dict("g" => safesoftmaxgraph()), Dict{String,Any}();
                      backend = KernelAbstractions.CPU(), fuseattn = false).graphs["g"]
    @test !any(o -> o.aten == "fused.sdpa", chain.ops)
    # And the chain it declined to fuse is still all there to run.
    @test count(o -> o.aten == "_softmax.default", chain.ops) == 1
    @test count(o -> o.aten == "bmm.default", chain.ops) == 2
end

# The two runners on one graph. `scale` is an attribute here rather than a `mul`
# left in the graph, and both paths have to read it.
@testset "folded attention scale, interpreted and declared" begin
    backend = Mantle.LavaBackend()
    caps = DNNKernels.caps(backend)
    if caps.coopmat && caps.coopmatsubgroup == 32
        H, Lq, Lk, D = 2, 64, 128, 128
        rng = MersenneTwister(5)
        qh = randn(rng, Float16, D, Lq, H, 1) .* Float16(0.3)
        kh = randn(rng, Float16, D, Lk, H, 1) .* Float16(0.3)
        vh = randn(rng, Float16, D, Lk, H, 1)
        scale = 1 / sqrt(D)

        buf(id, shape; kind = :transient, dtype = Float16) =
            DKA.Buffer(id, kind, Any[shape...], dtype, "", (0, 0), "", "",
                       Dict{String,Any}())
        buffers = Dict(b.id => b for b in (
            buf("q", (1, H, Lq, D); kind = :external),
            buf("k", (1, H, Lk, D); kind = :external),
            buf("v", (1, H, Lk, D); kind = :external),
            buf("out", (H, Lq, D); dtype = Float32)))
        ops = [DKA.Op("out", "fused.sdpa", ["q", "k", "v"], "out",
                      Dict{String,Any}("scale" => scale))]
        g = DKA.Graph("sdpascale", String[], ["q", "k", "v"], ["out"], buffers,
                      collect(keys(buffers)), ops)

        want = zeros(Float32, D, Lq, H)
        for h in 1:H
            s = (transpose(Float32.(kh[:, :, h, 1])) * Float32.(qh[:, :, h, 1])) .* Float32(scale)
            s .-= maximum(s; dims = 1)
            p = exp.(s); p ./= sum(p; dims = 1)
            want[:, :, h] = Float32.(vh[:, :, h, 1]) * p
        end

        q, k, v = map(a -> DNNKernels.toback(backend, a), (qh, kh, vh))
        vals = DNNKernels.execute!(g, Dict("q" => q, "k" => k, "v" => v),
                                   Dict{String,Any}(); dims = (;), backend)
        got = Float32.(Array(vals["out"]))
        @test maximum(abs, reshape(got, D, Lq, H) .- want) < 0.01maximum(abs, want)

        plan = DNNKernels.planfor(Mantle.todevice(backend), g, Dict{String,Any}(), (;))
        rec = Float32.(Array(first(DNNKernels.replay!(plan, "sdpascale", (q, k, v)))))
        @test maximum(abs, reshape(rec, D, Lq, H) .- want) < 0.01maximum(abs, want)
        # The failure this pins is the two paths disagreeing, not either of them
        # being slightly off: they ran at different scales and stayed finite.
        @test maximum(abs, reshape(rec, D, Lq, H) .- reshape(got, D, Lq, H)) <
              0.01maximum(abs, want)
        Mantle.free!(plan.plan)
    else
        @test_skip false
    end
end

# Which strided attention operands the kernel reads in place. Pure shape
# arithmetic, so it needs no device.
@testset "an interleaved attention operand is copied, a sliced one is not" begin
    E, Lq, H = 128, 4096, 32
    root = zeros(Float16, 1)
    # `[B, H, L, E]`: the `(E, L)` plane is contiguous, which is the case the
    # root-and-four-strides form exists for.
    planar = DKA.StridedOperand(root, (E, Lq, H, 1), (1, E, E * Lq, E * Lq * H), 0)
    @test DKA.flashplanar(planar)
    # A slice of the tokens keeps that: same strides, fewer of them.
    @test DKA.flashplanar(DKA.StridedOperand(root, (E, 2048, H, 1),
                                             (1, E, E * Lq, E * Lq * H), E * 64))
    # `[B, L, H, E]` permuted to `[B, H, L, E]` — a QKV projection's own layout.
    # Consecutive tokens are `E * H` apart and every tile row is its own burst.
    interleaved = DKA.StridedOperand(root, (E, Lq, H, 1), (1, E * H, E, E * H * Lq), 0)
    @test !DKA.flashplanar(interleaved)
    # An operand whose head dimension is not contiguous either.
    @test !DKA.flashplanar(DKA.StridedOperand(root, (E, Lq, H, 1),
                                              (2, 2E, 2E * Lq, 2E * Lq * H), 0))
end
