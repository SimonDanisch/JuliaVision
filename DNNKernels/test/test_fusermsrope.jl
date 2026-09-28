"""
`fusermsrope`: a grouped RMS norm whose only reader is an interleaved rotary,
directly or through a widening cast, becomes one `fused.rmsrope`.

Qwen-Image 2.1 normalises and rotates every query and key head, and apart that
was four passes over each: a copy of the head out of the QKV product, the norm,
the fp32 cast and the rotary. 140 ms of a denoising step went to 37 ms. The
reduction runs in the norm's own order and the rounding sits where the graph put
it, so the fused op is the chain to the bit, and that is what this pins.
"""

using Test, DNNKernels, Mantle, Random

const DKM = DNNKernels

@testset "a head norm and its rotary are one pass" begin
    backend = first(Mantle.eachbackend())
    dev = Mantle.todevice(backend)
    A(p...) = Dict{String,Any}(p...)
    buf(id, kind, shape, T; of = "", viewop = "", attrs = A(), key = "") =
        DKM.Buffer(id, kind, Any[shape...], T, key, (0, 0), of, viewop, attrs)
    N, H, C = 37, 4, 128
    P = C ÷ 2

    # torch order throughout. `strided` reads the heads out of a stacked QKV
    # product as Qwen-Image does; otherwise the norm's input is its own tensor.
    function chain(; strided::Bool, cast::Bool = true, midround::Bool = true,
                   readers::Int = 1)
        bs = [buf("gamma", :weight, (C,), Float16; key = "gamma"),
              buf("cos", :external, (1, N, 1, P), Float32),
              buf("sin", :external, (1, N, 1, P), Float32),
              buf("nrm", :transient, (1, N, H, C), Float16),
              buf("wide", :transient, (1, N, H, C), Float32),
              buf("rot", :transient, (1, N, H, C), Float16),
              buf("other", :transient, (1, N, H, C), Float16)]
        if strided
            append!(bs, [buf("qkv", :external, (N, 3H * C), Float16),
                         buf("qs", :view, (N, H * C), Float16; of = "qkv",
                             viewop = "slice.Tensor",
                             attrs = A("arg1" => 1, "arg2" => H * C, "arg3" => 2H * C)),
                         buf("x", :view, (1, N, H, C), Float16; of = "qs",
                             viewop = "view.default", attrs = A("arg1" => Any[1, N, H, C]))])
        else
            push!(bs, buf("x", :external, (1, N, H, C), Float16))
        end
        rin = cast ? "wide" : "nrm"
        ops = [DKM.Op("nrm", "fused.groupedrms", ["x", "gamma"], "nrm",
                      A("C" => C, "ng" => 1, "eps" => 1e-6, "midround" => midround))]
        cast && push!(ops, DKM.Op("wide", "_to_copy.default", ["nrm"], "wide", A()))
        push!(ops, DKM.Op("rot", "fused.pairrope", [rin, "cos", "sin"], "rot", A("P" => P)))
        readers == 2 && push!(ops, DKM.Op("other", "clamp.default", ["nrm"], "other",
                                          A("arg1" => -1000, "arg2" => 1000)))
        inputs = [strided ? "qkv" : "x", "cos", "sin"]
        outputs = readers == 2 ? ["rot", "other"] : ["rot"]
        DKM.Graph("rr", String[], inputs, outputs, Dict(b.id => b for b in bs),
                  [b.id for b in bs], ops)
    end

    # The rewrite, and a norm it must leave alone because something else reads it.
    g, n = DKM.fusermsrope(chain(; strided = true))
    @test n == 1
    op = only(o for o in g.ops if o.aten == "fused.rmsrope")
    @test op.ins == ["x", "gamma", "cos", "sin"] && op.out == "rot"
    @test op.attrs["P"] == P && op.attrs["normdtype"] === Float16 && op.attrs["midround"]
    @test length(g.ops) == 1
    @test DKM.fusermsrope(chain(; strided = true, readers = 2))[2] == 0

    rng = MersenneTwister(21)
    γ = Float16.(1 .+ 0.3f0 .* randn(rng, Float32, C))
    weights = Dict{String,Any}("gamma" => DKM.toback(backend, γ))
    θ = randn(rng, Float32, P, 1, N, 1)
    cs, sn = DKM.toback(backend, cos.(θ)), DKM.toback(backend, sin.(θ))
    function run(g, x)
        plan = DKM.planfor(dev, g, weights, (;))
        got = Array(first(DKM.replay!(plan, "rr", (x, cs, sn))))
        names = [String(p.pass.name) for p in plan.plan.passes]
        Mantle.free!(plan.plan)
        got, names
    end
    for strided in (true, false), cast in (true, false), midround in (true, false)
        x = DKM.toback(backend, Float16.(randn(rng, Float32,
                                               strided ? (3H * C, N) : (C, H, N, 1)) .* 3f0))
        want, _ = run(chain(; strided, cast, midround), x)
        got, names = run(first(DKM.fusermsrope(chain(; strided, cast, midround))), x)
        @test got == want
        # One pass for the op, and nothing that copies the heads out first.
        @test count(==("rot"), names) == 1
        @test !any(n -> n in ("nrm", "wide") || startswith(n, "x.") || startswith(n, "qs."),
                   names)
    end
end
