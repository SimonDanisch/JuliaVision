"""
An aten argument reaches the graph JSON under one of *two* names, and the
runtime has to accept both.

## What went wrong

`export_graphs.py` files a traced call's positional arguments as `arg0`, `arg1`,
… and its keyword arguments under their own names, and nothing normalises the
two — which of the two a given op gets is decided by how the model's source
happened to write the call, not by anything about the op.

`runop!(::Val{Symbol("gelu.default")})` tested `arg1 == "tanh"`. Every graph in
`gen/graphs` that has a tanh gelu records it as `{"approximate": "tanh"}`, so the
test never once fired: Wan's DiT and TRELLIS.2's sparse-structure flow model both
ran torch's *exact* gelu where the reference ran the approximation. `foldrelu`
read the same key to decide a gelu was the exact form and therefore foldable, so
it folded tanh gelus into an `addmm` epilogue as `geluexact` as well — a second
path to the same wrong answer, and its comment argued the fold was benign on the
grounds that the epilogue computes "the same computation the deleted op did",
which holds only when the activation is the same function.

## What it cost

The two forms differ by ~1e-3 and it accumulates with depth, in every precision.
TRELLIS.2's torso disagreed with PyTorch by 0.0125% of range at 3 blocks and
0.114% at 30 — and identically in fp32 and fp16, which is what said the cause was
not rounding. Making the same substitution inside PyTorch reproduces both figures
to three digits (0.012521% and 0.109699%, `tools/measure_gelu_cost.py`), so the
gelu was the entire disagreement.

## The assertions

`atenarg` itself, then the two consumers, each in both spellings and each against
the *other* activation so that a regression cannot pass by returning a plausible
number. The last case is the one that would have caught this: it asserts a graph
carrying `approximate` computes something the exact form does not.

Through the DECLARED path, which is the only one there is: `planfor` emits the
graph and `replay!` runs it. `tools/audit_attr_keys.py` is the sweep for the
rest of the op surface.
"""

using Test, DNNKernels, Mantle
const DKA = DNNKernels

aabuf(id, kind, shape, dtype; of = "", viewop = "") =
    DKA.Buffer(id, kind, Any[shape...], dtype, "", (0, 0), of, viewop,
               Dict{String,Any}())

"""A one-op graph `out = gelu(x)` carrying `attrs`."""
function gelugraph(attrs::Dict{String,Any}; n = 8)
    bufs = Dict{String,DKA.Buffer}(
        "x"   => aabuf("x", :external, (n,), Float32),
        "out" => aabuf("out", :transient, (n,), Float32))
    ops = [DKA.Op("g", "gelu.default", ["x"], "out", attrs)]
    DKA.Graph("t", String[], ["x"], ["out"], bufs, ["x", "out"], ops,
              Vector{Vector{String}}())
end

"""`addmm -> view -> gelu`, the shape `foldrelu` folds."""
function foldgraph(attrs::Dict{String,Any})
    bufs = Dict{String,DKA.Buffer}(
        "bias" => aabuf("bias", :weight, (4,), Float32),
        "a"    => aabuf("a", :external, (4, 4), Float32),
        "b"    => aabuf("b", :weight, (4, 4), Float32),
        "mm"   => aabuf("mm", :transient, (4, 4), Float32),
        "v"    => aabuf("v", :view, (16,), Float32; of = "mm", viewop = "view.default"),
        "out"  => aabuf("out", :transient, (16,), Float32))
    ops = [DKA.Op("mm", "addmm.default", ["bias", "a", "b"], "mm", Dict{String,Any}()),
           DKA.Op("g", "gelu.default", ["v"], "out", attrs)]
    DKA.Graph("t", String[], ["a"], ["out"], bufs,
              ["bias", "a", "b", "mm", "v", "out"], ops, Vector{Vector{String}}())
end

@testset "aten arguments under either name" begin
    x = Float32[-4, -3, -2, -1, 0, 1, 2, 3]

    @testset "atenarg reads both spellings" begin
        pos = DKA.Op("g", "gelu.default", ["x"], "out",
                     Dict{String,Any}("arg1" => "tanh"))
        kw = DKA.Op("g", "gelu.default", ["x"], "out",
                    Dict{String,Any}("approximate" => "tanh"))
        none = DKA.Op("g", "gelu.default", ["x"], "out", Dict{String,Any}())
        @test DKA.atenarg(pos, 1, "approximate", "none") == "tanh"
        @test DKA.atenarg(kw, 1, "approximate", "none") == "tanh"
        @test DKA.atenarg(none, 1, "approximate", "none") == "none"
        # Position wins when a graph somehow carries both: it is what the older
        # graphs on disk mean, and torch never emits an argument twice.
        both = DKA.Op("g", "gelu.default", ["x"], "out",
                      Dict{String,Any}("arg1" => "none", "approximate" => "tanh"))
        @test DKA.atenarg(both, 1, "approximate", "none") == "none"
    end

    @testset "gelu selects the approximation the graph asked for" begin
        # The two forms have to disagree, or nothing below can fail.
        @test !all(DKA.gelutanh.(x) .≈ DKA.geluexact.(x))

        function run(attrs)
            dev = Mantle.todevice(Mantle.LavaBackend())
            plan = DKA.planfor(dev, gelugraph(attrs), Dict{String,Any}(), NamedTuple())
            out = Array(first(DKA.replay!(plan, "g", (DKA.toback(Mantle.LavaBackend(), x),))))
            Mantle.free!(plan.plan)
            out
        end

        # `atol` and not `==`, and not `≈` either. The device's `tanh` differs
        # from the host's in the last ulp or two — 2.4e-8 at x = -4 — so exact
        # equality held only while this ran the same Julia source on the CPU.
        # Plain `≈` is no good the other way: it compares vector norms, and the
        # gap between the two gelu FORMS is 2.5e-4 against a norm of 3.7, which
        # is inside the default tolerance, so it would accept either answer.
        # 1e-6 sits between the two: 40x above the device's noise and 100x
        # below the difference this test exists to see.
        samevals(a, b) = all(isapprox.(a, b; atol = 1e-6))
        @test samevals(run(Dict{String,Any}()), DKA.geluexact.(x))
        @test samevals(run(Dict{String,Any}("arg1" => "tanh")), DKA.gelutanh.(x))
        # The spelling every tanh gelu in the tree actually uses, and the one a
        # missing case falls through to the exact form on.
        @test samevals(run(Dict{String,Any}("approximate" => "tanh")), DKA.gelutanh.(x))
        @test samevals(run(Dict{String,Any}("approximate" => "none")), DKA.geluexact.(x))
        # …and the tolerance has to REFUSE the other form, or none of the four
        # above means anything.
        @test !samevals(DKA.gelutanh.(x), DKA.geluexact.(x))
    end

    @testset "keepdim reads both spellings too" begin
        # Not a bug that was hit — every graph in the tree carries it as `arg2` —
        # but `keepdim=True` is a keyword in nearly every PyTorch source that
        # writes one, so it is one export away from being the same fault.
        mk(attrs) = DKA.Op("r", "sum.dim_IntList", ["x"], "out", attrs)
        @test DKA.keepdim(mk(Dict{String,Any}("arg2" => true)))
        @test DKA.keepdim(mk(Dict{String,Any}("keepdim" => true)))
        @test !DKA.keepdim(mk(Dict{String,Any}("keepdim" => false)))
        @test !DKA.keepdim(mk(Dict{String,Any}()))
    end

    # The other thing read off an op: WHICH AXIS it names.
    #
    # `jdim` maps a torch dim (0-based, `-n:n-1`) onto the reversed Julia shape.
    # The arithmetic alone answers 0 for `d == n` and `n + 1` for `d == -(n+1)`,
    # an axis that does not exist, returned as confidently as one that does, and
    # it goes straight into `colstrides` indexing from there. Five of its
    # thirty-five call sites had grown their own range check, each worded
    # differently; the ops that needed one were not the ops whose author thought
    # of it. torch raises `IndexError` here, so refusing matches the producer.
    @testset "jdim refuses an axis that does not exist" begin
        for n in 1:4, d in 0:(n - 1)
            @test DKA.jdim(d, n) == n - d              # 0-based from the front
            @test DKA.jdim(d - n, n) == n - d          # and the negative form
            @test 1 <= DKA.jdim(d, n) <= n
        end
        for (d, n) in ((1, 1), (3, 3), (4, 3), (-2, 1), (-4, 3))
            @test_throws "out of range" DKA.jdim(d, n)
        end
        # The message names both numbers, since neither alone says what is wrong.
        @test_throws ["3", "2-d"] DKA.jdim(3, 2)
    end

    @testset "foldrelu folds the exact form only" begin
        _, n = DKA.foldrelu(foldgraph(Dict{String,Any}()))
        @test n == 1
        for attrs in (Dict{String,Any}("arg1" => "tanh"),
                      Dict{String,Any}("approximate" => "tanh"))
            g, n = DKA.foldrelu(foldgraph(attrs))
            @test n == 0
            # and the op it declined to fold is still there to do the work
            @test any(o -> o.aten == "gelu.default", g.ops)
            @test !haskey(only(filter(o -> o.aten == "addmm.default", g.ops)).attrs, "act")
        end
    end
end
