"""
`aten::slice.Tensor` whose START depends on a graph symbol.

## What went wrong

A length-generic export slices with a bound that is not a number. Qwen-Image
2.1's denoiser runs a joint `[text; image]` sequence through every block and
then takes the image half back out with `query[:, prompt_tokens:]`. Exported at
a fixed prompt length that is `slice.Tensor(dim=1, start=22, end=MAX)`; exported
with the prompt length symbolic the start is a `SymInt`, and
`export_graphs.convert` wrote the node into `in` for the dependency and dropped
the attribute slot that said WHICH argument it was.

Neither side then complained. The view's own declared shape is right, so it had
the right extent and began at the schema default of 0: every image token read
the 22 rows belonging to the text plus the first 4074 image rows. The text half
of the model's output stayed bit-exact and the image half was quietly wrong,
which is why it survived a shape check, a NaN check and an end-to-end run that
produced a plausible image.

## The assertions

Two one-op graphs, a declared one and an interpreted one, that slice a symbolic
offset off a `(t + 4, 3)` input and must return the last 4 rows for any `t`. The
bug is a wrong OFFSET with a right SHAPE, so asserting the size is not enough
and every assertion here compares values.

`arg2` as `"\$s"` is the spelling `export_graphs` now emits and `intattr` has
always resolved; the failing form is `arg2` absent, which is also covered so
that a regression cannot reintroduce it as a silent default.
"""

using Test, DNNKernels, KernelAbstractions, Mantle
const DKR = DNNKernels
const KAR = KernelAbstractions

"""
`out = x[start:, :]` as a four-buffer graph, where `start` is the host scalar `s`.

`shape` is in torch order, so the sliced axis is torch dim 0. `arg2 = nothing`
reproduces the dropped attribute.

`planfor` directly rather than through `Model`: a `Model` runs the host passes
first, and a host scalar that only a VIEW reads looks dead to them. Every real
export also feeds its `sym_size_int` to an op — Qwen-Image 2.1's goes to a
`full` and two `arange`s — so that pruning never fires in practice, and pinning
it here would test the passes rather than the view.
"""
function sliceparent(t::Int; start = "\$s", interpreted::Bool)
    rows, cols = t + 4, 3
    x = Float32.(reshape(1:(rows * cols), cols, rows))   # Julia (cols, rows)
    attrs = Dict{String,Any}("arg1" => 0)
    start === nothing || (attrs["arg2"] = start)
    bufs = Dict{String,DKR.Buffer}(
        "x" => DKR.Buffer("x", :external, Any["t + 4", cols], Float32, "", (0, 0),
                          "", "", Dict{String,Any}()),
        "s" => DKR.Buffer("s", :host, Any[], Int, "", (0, 0), "", "",
                          Dict{String,Any}("expr" => "t")),
        "v" => DKR.Buffer("v", :view, Any[4, cols], Float32, "", (0, 0),
                          "x", "slice.Tensor", attrs),
        "out" => DKR.Buffer("out", :external, Any[4, cols], Float32, "", (0, 1),
                            "", "", Dict{String,Any}()))
    g = DKR.Graph("s", ["t"], ["x"], ["out"], bufs, ["x", "s", "v", "out"],
                  [DKR.Op("c", "clone.default", ["v"], "out", Dict{String,Any}())],
                  Vector{Vector{String}}())
    dims = (; t)
    interpreted && return DKR.execute!(g, Dict{String,Any}("x" => x),
                                       Dict{String,Any}(); dims, backend = KAR.CPU())["out"]
    dev = Mantle.todevice(Mantle.LavaBackend())
    plan = DKR.planfor(dev, g, Dict{String,Any}(), dims)
    out = Array(first(DKR.replay!(plan, "s", (DKR.toback(Mantle.LavaBackend(), x),))))
    Mantle.free!(plan.plan)
    out
end

@testset "a slice whose start is a graph symbol" begin
    # `x` is (3, t+4) in Julia. The last four rows of the torch-order tensor are
    # the last four COLUMNS here, and they are what `x[t:, :]` must return.
    for t in (0, 1, 22, 37)
        want = Float32.(reshape(1:((t + 4) * 3), 3, t + 4))[:, (t + 1):(t + 4)]

        @testset "t = $t" begin
            @test sliceparent(t; interpreted = true) == want
            @test sliceparent(t; interpreted = false) == want
        end
    end

    # The failing form, kept so a regression cannot quietly restore the default.
    # With the attribute gone the view still has the right shape, so only the
    # values distinguish a correct slice from one that began at 0.
    @testset "a dropped bound is not silently zero" begin
        t = 22
        want = Float32.(reshape(1:((t + 4) * 3), 3, t + 4))[:, (t + 1):(t + 4)]
        wrong = Float32.(reshape(1:((t + 4) * 3), 3, t + 4))[:, 1:4]
        @test want != wrong
        for interpreted in (true, false)
            got = sliceparent(t; start = nothing, interpreted)
            @test size(got) == size(want)      # the shape never caught it
            @test got == wrong                 # and this is what it returned
        end
    end
end
