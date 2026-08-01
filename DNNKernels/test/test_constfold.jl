"""
The transitive `hoistconstants` — the method that folds whole *subgraphs*.

Two halves, because the pass has two kinds of thing to get right and they are not
testable by the same means.

**Structure**, against SAM 2's own encoder: the set of constant ops has to be
closed (an op is in it only if every producer feeding it is too), it must never
reach an external input, and it must refuse anything whose extent is only known
at a resolution. Those are properties of how a real exported graph is shaped —
tuple outputs, views over weights, casts on top of casts — and a hand-built graph
exercises none of them.

**Semantics**, against a four-op synthetic graph on the CPU backend: that the
folded graph computes what the unfolded one computed. The real encoder would say
the same thing far more convincingly, but folding it on the CPU backend costs
6.5 s, almost all of it JIT for kernels nothing else needs; the GPU end-to-end
check lives in `bench_sam2.jl`, where the encoder's six outputs are compared
against PyTorch on every run. What is left for here is the rewrite itself, and
four ops show that as well as six hundred.

The guard gets its own case in both directions. A pass that trades slab space for
resident weights is only worth having when the trade is a win, and "it happened
to be a win on SAM 2" is not the same claim.
"""

using Test, DNNKernels, KernelAbstractions
const DK = DNNKernels
const KA = KernelAbstractions

const CFDIR = normpath(joinpath(@__DIR__, "..", "..", "..", "..",
                                "gen", "graphs", "sam2-large"))

cfbuf(id, kind, shape, dtype; key = "") =
    DK.Buffer(id, kind, Any[shape...], dtype, key, (0, 0), "", "", Dict{String,Any}())

"""`c1 = w1*w2; c2 = c1*w2; out = c2 + x` — a constant chain with one live tail."""
function cfgraph(; cshape = (4, 4))
    bufs = Dict{String,DK.Buffer}(
        "w1"  => cfbuf("w1", :weight, cshape, Float32; key = "w1"),
        "w2"  => cfbuf("w2", :weight, cshape, Float32; key = "w2"),
        "x"   => cfbuf("x", :external, (4, 4), Float32),
        "c1"  => cfbuf("c1", :transient, cshape, Float32),
        "c2"  => cfbuf("c2", :transient, cshape, Float32),
        "out" => cfbuf("out", :transient, (4, 4), Float32))
    ops = [DK.Op("c1", "mul.Tensor", ["w1", "w2"], "c1", Dict{String,Any}()),
           DK.Op("c2", "mul.Tensor", ["c1", "w2"], "c2", Dict{String,Any}()),
           DK.Op("out", "add.Tensor", ["c2", "x"], "out", Dict{String,Any}())]
    DK.Graph("t", String[], ["x"], ["out"], bufs,
             ["w1", "w2", "x", "c1", "c2", "out"], ops, Vector{Vector{String}}())
end

@testset "hoistconstants over subgraphs" begin

    @testset "the constant set is closed and reaches no input" begin
        if !isfile(joinpath(CFDIR, "sam2_encoder.json"))
            @info "no exported SAM 2 graph at $CFDIR; skipping"
        else
            g = DK.loadgraph(joinpath(CFDIR, "sam2_encoder.json"))
            ops = DK.constops(g)
            @test !isempty(ops)
            byid = Dict(o.id => o for o in g.ops)
            made = Set(DK.rootbuffer(g, o.out) for o in g.ops if o.id in ops)

            for id in ops
                o = byid[id]
                for i in o.ins
                    r = DK.rootbuffer(g, i)
                    b = g.buffers[r]
                    # Closed: every input is a weight or something the set makes.
                    @test b.kind === :weight || r in made
                    # And never the image, or anything resolved from `dims`.
                    @test b.kind !== :external
                    @test b.kind !== :host
                    @test DK.concreteshape(b)
                end
                @test DK.concreteshape(g.buffers[o.out])
            end

            # Everything that escapes was made inside, or the rewrite would be
            # promoting a buffer it never computed.
            esc = DK.constescaping(g, ops)
            @test !isempty(esc)
            @test all(e -> e in made, esc)

            # The subgraph is self-contained: every free variable is a weight,
            # everything else it reads it also makes. That — not the set of
            # buffer *kinds* it contains — is what makes it runnable on its own.
            # `:external` in an exported graph marks a value that leaves for
            # another graph, which is most of the encoder's outputs, so it turns
            # up here legitimately.
            sg = DK.constsubgraph(g, ops, sort(collect(esc)))
            @test length(sg.ops) == length(ops)
            inside = Set(o.out for o in sg.ops)
            for o in sg.ops, i in o.ins
                r = DK.rootbuffer(sg, i)
                @test haskey(sg.buffers, r)
                @test sg.buffers[r].kind === :weight || r in inside
            end
            # A graph input is the caller's and must never be frozen.
            @test all(o -> !(o.out in g.inputs), sg.ops)
        end
    end

    @testset "it folds the chain and keeps the tail" begin
        g = cfgraph()
        w = Dict{String,Any}("w1" => Float32[i + j for i in 1:4, j in 1:4],
                             "w2" => Float32[i - 2j for i in 1:4, j in 1:4])
        x = Float32[0.5i * j for i in 1:4, j in 1:4]

        want = DK.execute!(g, Dict{String,Any}("x" => x), copy(w);
                           dims = NamedTuple(), backend = KA.CPU())["out"]

        w2 = Dict{String,Any}(w)
        g2, n = DK.hoistconstants(g, w2, KA.CPU())
        @test n == 2                                   # c1 and c2, not `out`
        @test length(g2.ops) == 1
        @test only(g2.ops).id == "out"
        @test g2.buffers["c2"].kind === :weight        # the tail became a weight
        @test g2.buffers["c1"].kind === :transient     # the interior did not
        @test haskey(w2, g2.buffers["c2"].key)

        got = DK.execute!(g2, Dict{String,Any}("x" => x), w2;
                          dims = NamedTuple(), backend = KA.CPU())["out"]
        @test got == want
        # The graph still answers to a different input, i.e. the fold froze the
        # constant part and *only* the constant part.
        x2 = Float32[i + 3j for i in 1:4, j in 1:4]
        @test DK.execute!(g2, Dict{String,Any}("x" => x2), w2;
                          dims = NamedTuple(), backend = KA.CPU())["out"] ==
              DK.execute!(g, Dict{String,Any}("x" => x2), copy(w);
                          dims = NamedTuple(), backend = KA.CPU())["out"]
    end

    @testset "it refuses a fold that would not pay" begin
        # One constant op whose result escapes: the payload is exactly what the
        # slab stops holding, so there is nothing in it.
        g = cfgraph()
        bufs = Dict{String,DK.Buffer}(g.buffers)
        ops = [DK.Op("c1", "mul.Tensor", ["w1", "w2"], "c1", Dict{String,Any}()),
               DK.Op("out", "add.Tensor", ["c1", "x"], "out", Dict{String,Any}())]
        g1 = DK.Graph(g.name, g.symbols, g.inputs, g.outputs, bufs,
                      ["w1", "w2", "x", "c1", "out"], ops, Vector{Vector{String}}())
        w = Dict{String,Any}("w1" => zeros(Float32, 4, 4), "w2" => zeros(Float32, 4, 4))
        g1b, n = DK.hoistconstants(g1, w, KA.CPU())
        @test n == 0
        @test length(g1b.ops) == 2
        @test g1b.buffers["c1"].kind === :transient
    end

    @testset "a 0-d buffer is four bytes, not a MethodError" begin
        # `shape` for a scalar is an empty `Vector{Any}`; `Int.` of that is still
        # `Vector{Any}`, and `prod` of an empty `Vector{Any}` has no identity to
        # return. It throws, and it throws while *sizing* the fold, which is
        # before the guard that would have declined it — so a single scalar
        # anywhere in a graph took `Model` down with it. MatAnyone's
        # `readout_query` has 49 of them.
        @test DK.constbytes(cfbuf("s", :transient, (), Float32)) == 4
        @test DK.constbytes(cfbuf("s", :transient, (), Float16)) == 2
        @test DK.concreteshape(cfbuf("s", :transient, (), Float32))

        # And end to end: a graph whose constant chain passes through a scalar.
        bufs = Dict{String,DK.Buffer}(
            "w"   => cfbuf("w", :weight, (4, 4), Float32; key = "w"),
            "s"   => cfbuf("s", :weight, (), Float32; key = "s"),
            "x"   => cfbuf("x", :external, (4, 4), Float32),
            "c1"  => cfbuf("c1", :transient, (4, 4), Float32),
            "c2"  => cfbuf("c2", :transient, (4, 4), Float32),
            "out" => cfbuf("out", :transient, (4, 4), Float32))
        ops = [DK.Op("c1", "mul.Tensor", ["w", "s"], "c1", Dict{String,Any}()),
               DK.Op("c2", "mul.Tensor", ["c1", "w"], "c2", Dict{String,Any}()),
               DK.Op("out", "add.Tensor", ["c2", "x"], "out", Dict{String,Any}())]
        g = DK.Graph("t0", String[], ["x"], ["out"], bufs,
                     ["w", "s", "x", "c1", "c2", "out"], ops, Vector{Vector{String}}())
        w = Dict{String,Any}("w" => Float32[i + j for i in 1:4, j in 1:4],
                             "s" => fill(2.5f0))
        x = Float32[0.5i for i in 1:4, j in 1:4]
        want = DK.execute!(g, Dict{String,Any}("x" => x), copy(w);
                           dims = NamedTuple(), backend = KA.CPU())["out"]
        w2 = Dict{String,Any}(w)
        g2, n = DK.hoistconstants(g, w2, KA.CPU())
        @test n == 2
        @test DK.execute!(g2, Dict{String,Any}("x" => x), w2;
                          dims = NamedTuple(), backend = KA.CPU())["out"] == want
    end

    @testset "it refuses an extent that is only known at a resolution" begin
        # `scratchfor` keys the slab on `dims`, so one `Model` serves several
        # resolutions; a value that depends on one is not a constant of the model.
        g = cfgraph(; cshape = ("res", 4))
        @test isempty(DK.constops(g))
        w = Dict{String,Any}("w1" => zeros(Float32, 4, 4), "w2" => zeros(Float32, 4, 4))
        g2, n = DK.hoistconstants(g, w, KA.CPU())
        @test n == 0
        @test length(g2.ops) == length(g.ops)
    end
end
