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

Host-only — the CPU backend runs the same op source the GPU does.
`tools/audit_attr_keys.py` is the sweep for the rest of the op surface.
"""

using Test, DNNKernels, KernelAbstractions
const DKA = DNNKernels
const KAB = KernelAbstractions

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

        run(attrs) = DKA.execute!(gelugraph(attrs), Dict{String,Any}("x" => x),
                                  Dict{String,Any}(); dims = NamedTuple(),
                                  backend = KAB.CPU())["out"]

        @test run(Dict{String,Any}()) == DKA.geluexact.(x)
        @test run(Dict{String,Any}("arg1" => "tanh")) == DKA.gelutanh.(x)
        # The spelling every tanh gelu in the tree actually uses, and the one
        # that used to fall through to the exact form.
        @test run(Dict{String,Any}("approximate" => "tanh")) == DKA.gelutanh.(x)
        @test run(Dict{String,Any}("approximate" => "none")) == DKA.geluexact.(x)
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
