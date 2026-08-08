"""
`FusedOp`, `fuseops` and `foldepilogue`.

Two kinds of test, because the pass has two kinds of failure.

**Synthetic graphs, run both ways.** The bug this pass shipped first was silent:
`operand` resolves a position to either a scalar attribute or the next unconsumed
tensor input, and building the arguments from `op.ins` alone dropped the scalars.
`add.Tensor(x, 1e-6)` became `+(x)` — unary plus, the identity — and
`sub.Tensor(x, 1)` became `-(x)`, negation. Neither throws. On SAM 2's decoder it
came out as mask logits off by 1.96, which reads exactly like a precision problem.
So the assertions here run the graph fused and unfused and compare the numbers;
an invariant about op counts would have been perfectly green.

**`FusedOp` itself**, where the failure is allocation rather than wrongness. The
natural recursive implementation is type stable at every depth and starts heap
allocating at four functions deep, which inside a kernel is per element.

The real graphs are covered by `SAM2Runner`'s parity suite, which has the weights;
this file has graphs only, so it builds what it needs.
"""

using Test, DNNKernels, KernelAbstractions
const DK = DNNKernels
using DNNKernels: FusedOp, In, Tmp, Konst, Cast, Buffer, Op, Graph

@testset "FusedOp" begin
    fo = FusedOp((+, *), ((In(1), In(2)), (Tmp(1), In(3))))       # (a+b)*c, a DAG
    @test fo(2f0, 3f0, 4f0) === 20f0
    @test DK.nargs(fo) == 3
    @test length(fo) == 2

    # A chain that changes element type — the ordinary case, since `_to_copy` is
    # the most frequent op in an exported graph.
    g = FusedOp((*, Cast(Float16), +),
                ((In(1), Konst(2f0)), (Tmp(1),), (Tmp(2), Konst(Float16(1)))))
    @test g(3f0) === Float16(7)

    @test FusedOp((+, *), ((In(1), Konst(1f0)), (Tmp(1), Tmp(1))))(4f0) === 25f0

    # A forward reference is refused at construction: inside a kernel it is a
    # MethodError several frames into a launch.
    @test_throws ArgumentError FusedOp((+, *), ((In(1), Tmp(2)), (Tmp(1), In(2))))
    @test_throws ArgumentError FusedOp((+,), ((In(1), In(2)), (In(1),)))
    @test_throws ArgumentError FusedOp((), ())
end

@testset "FusedOp does not allocate at any depth" begin
    # The recursive implementation was 0/0/48/112/176/304 bytes at depths
    # 2..8 while being type stable throughout, which is why this asserts on
    # allocation and not on inference.
    mk(n) = FusedOp(ntuple(_ -> +, n),
                    ntuple(k -> k == 1 ? (In(1), In(2)) : (Tmp(k - 1), In(2)), n))
    for n in 2:8
        f = let c = mk(n); (x, y) -> c(x, y); end
        f(1f0, 2f0)                                   # compile
        @test @allocated(f(1f0, 2f0)) == 0
        @test Base.return_types(f, Tuple{Float32,Float32})[1] === Float32
    end
end

"""A graph of elementwise ops over one input, with scalars carried in attributes
the way an exported graph carries them."""
function chaingraph()
    b(id, kind, dtype; of = "", viewop = "") =
        Buffer(id, kind, Any[4, 4], dtype, "", (0, 0), of, viewop, Dict{String,Any}())
    buffers = Dict{String,Buffer}(
        "x"  => b("x",  :external,  Float32),
        "t1" => b("t1", :transient, Float32),
        "t2" => b("t2", :transient, Float32),
        "t3" => b("t3", :transient, Float32),
        "y"  => b("y",  :transient, Float32))
    ops = [
        # add with a SCALAR second operand — the case that was dropped
        Op("o1", "add.Tensor", ["x"], "t1", Dict{String,Any}("arg1" => 1.0e-6)),
        Op("o2", "sqrt.default", ["t1"], "t2", Dict{String,Any}()),
        # non-commutative, with the scalar first: `sub(2, t2)`, not `sub(t2, 2)`
        Op("o3", "sub.Tensor", ["t2"], "t3", Dict{String,Any}("arg0" => 2.0f0)),
        Op("o4", "mul.Tensor", ["t3"], "y",  Dict{String,Any}("arg1" => 3.0f0)),
    ]
    Graph("chain", String[], ["x"], ["y"], buffers,
          ["x", "t1", "t2", "t3", "y"], ops, Vector{Vector{String}}())
end

@testset "fuseops preserves the arithmetic" begin
    g = chaingraph()
    gf, n = DK.fuseops(g)
    @test n == 3                                    # four ops become one
    @test length(gf.ops) == 1
    @test only(gf.ops).aten == "fused.elementwise"

    fo = only(gf.ops).attrs["fused"]
    @test fo isa FusedOp
    @test length(fo) == 4
    @test DK.nargs(fo) == 1                          # only `x` comes in as a tensor

    # The whole point: the scalars survived, in the right positions.
    x = 4.0f0
    @test fo(x) ≈ (2.0f0 - sqrt(x + 1.0f-6)) * 3.0f0
    # and `sub` did not become negation
    @test fo(x) != -(sqrt(x + 1.0f-6)) * 3.0f0

    # Against the unfused graph, on real arrays.
    be = CPU()
    inp = Dict{String,Any}("x" => Float32[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
    ref = DK.execute!(g,  inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    got = DK.execute!(gf, inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    @test Array(got["y"]) == Array(ref["y"])
end

@testset "fuseops refuses what it cannot model" begin
    # A non-unit `alpha` is not `+`, and a symbolic scalar needs `dims` the passes
    # do not have. Both are left alone rather than half-understood.
    g = chaingraph()
    ops = copy(g.ops)
    ops[1] = Op("o1", "add.Tensor", ["x"], "t1",
                Dict{String,Any}("arg1" => 1.0e-6, "alpha" => 2))
    g2 = Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers, g.order, ops, g.fusion)
    @test !DK.fusable(g2, ops[1])

    ops[1] = Op("o1", "add.Tensor", ["x"], "t1", Dict{String,Any}("arg1" => "s0*2"))
    g3 = Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers, g.order, ops, g.fusion)
    @test !DK.fusable(g3, ops[1])

    # An op carrying a folded activation is no longer just its own arithmetic.
    ops[1] = Op("o1", "add.Tensor", ["x"], "t1",
                Dict{String,Any}("arg1" => 1.0e-6, "act" => "relu"))
    g4 = Graph(g.name, g.symbols, g.inputs, g.outputs, g.buffers, g.order, ops, g.fusion)
    @test !DK.fusable(g4, ops[1])
end

@testset "fuseops does not absorb an op SMALLER than the group's output" begin
    # The decomposed layer norm from SAM 2's decoder, in miniature:
    #
    #     m (4,1) -> add -> sqrt -> t2 (4,1) -> div -> y (4,4)
    #                                        x (4,4) ------^
    #
    # A fused expression is evaluated once per output element, so absorbing
    # `add`/`sqrt` into `div`'s group would evaluate that `sqrt` four times —
    # once per element of the broadcast axis. On the real graph the factor is
    # 64, the length of the normalised axis. The guard refuses the absorb, and
    # the backwards walk then fuses `add -> sqrt` on their OWN shape instead.
    b(id, kind, shape) =
        Buffer(id, kind, shape, Float32, "", (0, 0), "", "", Dict{String,Any}())
    buffers = Dict{String,Buffer}(
        "x"  => b("x",  :external,  Any[4, 4]),
        "m"  => b("m",  :external,  Any[4, 1]),
        "t1" => b("t1", :transient, Any[4, 1]),
        "t2" => b("t2", :transient, Any[4, 1]),
        "y"  => b("y",  :transient, Any[4, 4]))
    ops = [
        Op("o1", "add.Tensor",   ["m"],     "t1", Dict{String,Any}("arg1" => 1.0e-6)),
        Op("o2", "sqrt.default", ["t1"],    "t2", Dict{String,Any}()),
        Op("o3", "div.Tensor",   ["x","t2"],"y",  Dict{String,Any}()),
    ]
    g = Graph("layernorm", String[], ["x", "m"], ["y"], buffers,
              ["x", "m", "t1", "t2", "y"], ops, Vector{Vector{String}}())
    gf, n = DK.fuseops(g)
    @test n == 1                     # add+sqrt fuse; div does NOT join them
    @test length(gf.ops) == 2
    @test gf.ops[1].aten == "fused.elementwise" && gf.ops[1].out == "t2"
    @test gf.ops[2].aten == "div.Tensor"

    be = CPU()
    inp = Dict{String,Any}("x" => Float32[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16],
                           "m" => Float32[1; 2; 3; 4;;])
    ref = DK.execute!(g,  inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    got = DK.execute!(gf, inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    @test Array(got["y"]) == Array(ref["y"])
end

@testset "unaryfused is unary in TENSORS, not in operands" begin
    g = chaingraph()
    byid = Dict(o.id => o for o in g.ops)
    # `add.Tensor(x, 1e-6)` has arity 2 and one tensor, so it can be an epilogue.
    fo, inid = DK.unaryfused(g, byid["o1"])
    @test fo isa FusedOp && DK.nargs(fo) == 1
    @test fo(4f0) ≈ 4f0 + 1.0f-6
    # The operand id comes back because `foldpremap` folds into the CONSUMER,
    # which then has to read this op's input instead of its output.
    @test inid == "x"
    # Rounded to the declared dtype: the Float64 scalar must not widen the result,
    # because folding removes the store that would have narrowed it.
    @test fo(4f0) isa Float32

    # Two tensors cannot: `epi` sees one value inside the GEMM's store.
    two = Op("o5", "mul.Tensor", ["x", "t1"], "y", Dict{String,Any}())
    @test DK.unaryfused(g, two) === nothing
end

"""`prod(1 - x)` over one input — the shape `foldpremap` actually finds in
MatAnyone's `readout_query` and `segment`."""
function premapgraph(; extrareader::Bool = false)
    b(id, kind) = Buffer(id, kind, Any[4, 4], Float32, "", (0, 0), "", "", Dict{String,Any}())
    buffers = Dict{String,Buffer}("x" => b("x", :external), "s" => b("s", :transient),
                                  "y" => b("y", :transient), "z" => b("z", :transient))
    ops = Op[
        Op("o1", "sub.Tensor", ["x"], "s", Dict{String,Any}("arg0" => 1.0f0)),
        Op("o2", "prod.dim_int", ["s"], "y", Dict{String,Any}("arg1" => 0)),
    ]
    # A second consumer of `s` means the mapped tensor has to exist anyway.
    extrareader && push!(ops, Op("o3", "mul.Tensor", ["s"], "z", Dict{String,Any}("arg1" => 2.0f0)))
    Graph("premap", String[], ["x"], extrareader ? ["y", "z"] : ["y"], buffers,
          ["x", "s", "y", "z"], ops, Vector{Vector{String}}())
end

@testset "foldpremap folds the map into the reduction" begin
    g = premapgraph()
    gp, n = DK.foldpremap(g)
    @test n == 1
    @test length(gp.ops) == 1                        # the `sub` is gone
    red = only(gp.ops)
    @test red.aten == "prod.dim_int"
    @test red.ins == ["x"]                           # reads the operand now, not `s`
    @test red.attrs["premap"] isa FusedOp
    @test red.attrs["premap"](0.25f0) ≈ 0.75f0

    # Same numbers as the unfused graph, on real arrays.
    be = CPU()
    inp = Dict{String,Any}("x" => Float32[0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8;
                                          0.9 0.15 0.25 0.35; 0.45 0.55 0.65 0.75])
    ref = DK.execute!(g,  inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    got = DK.execute!(gp, inp, Dict{String,Any}(); dims = NamedTuple(), backend = be,
                      lazy = Set{String}())
    @test Array(got["y"]) ≈ Array(ref["y"])
    @test maximum(abs.(Array(got["y"]) .- Array(ref["y"]))) == 0

    # A value with a second reader has to be materialised regardless, so folding
    # it would compute it twice rather than save a pass.
    g2, n2 = DK.foldpremap(premapgraph(; extrareader = true))
    @test n2 == 0
    @test length(g2.ops) == 3

    # `any`/`all` are not premappable: a map changes the question, not the method.
    gany = premapgraph()
    ops = [gany.ops[1], Op("o2", "any.dim", ["s"], "y", Dict{String,Any}("arg1" => 0))]
    g3 = Graph(gany.name, gany.symbols, gany.inputs, gany.outputs, gany.buffers,
               gany.order, ops, gany.fusion)
    @test DK.foldpremap(g3)[2] == 0
end
