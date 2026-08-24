"""
`aten::arange.start_step`, whose three scalars are not integers and whose length
is not a Julia range's.

## What went wrong

DINOv3's rotary embedding asks for `arange(0.5, 32, 1)` — the patch centres — and
the op pushed all three arguments through `intattr`, which threw
`InexactError: Int64(0.5)` on the first TRELLIS.2 conditioner graph that reached
it.

Fixing that exposed a second, latent fault in the same three lines: the element
count was `length(start:step:stop-step)`, which agrees with aten only when the
step is 1. `arange(0, 5, 2)` is `[0, 2, 4]` and that formula gives two elements.
Nothing exported so far has a non-unit integer step, so it had never been wrong
in practice — it was wrong in waiting.

## The assertions

Four calls run as one-op graphs on the CPU backend: the fractional start that
threw, the non-unit step that miscounted, the unit-step form every other graph in
the tree uses, and a fractional step. Against `ceil((end - start) / step)`, which
is aten's own definition, and against the values rather than only the length.

Host-only. `alloc` sizes the output from the count the op computes, not from the
declared buffer shape, so a miscount shows up as a wrong-length result here
exactly as it would on a device.
"""

using Test, DNNKernels, KernelAbstractions
const DKR = DNNKernels
const KAR = KernelAbstractions

"""A one-op graph `out = arange(start, stop, step)`, run on the CPU."""
function runarange(; start = nothing, stop, step = nothing, n, T = Float32)
    attrs = Dict{String,Any}("arg1" => stop)
    start === nothing || (attrs["arg0"] = start)
    step === nothing || (attrs["arg2"] = step)
    bufs = Dict{String,DKR.Buffer}(
        "out" => DKR.Buffer("out", :transient, Any[n], T, "", (0, 0), "", "",
                            Dict{String,Any}()))
    g = DKR.Graph("t", String[], String[], ["out"], bufs, ["out"],
                  [DKR.Op("a", "arange.start_step", String[], "out", attrs)],
                  Vector{Vector{String}}())
    DKR.execute!(g, Dict{String,Any}(), Dict{String,Any}();
                 dims = NamedTuple(), backend = KAR.CPU())["out"]
end

@testset "arange.start_step" begin
    @testset "a fractional start is not an Int" begin
        # DINOv3's own call. It threw InexactError before it ever produced a
        # number, so the assertion is that it runs at all — and then that the
        # centres are the centres.
        got = runarange(start = 0.5, stop = 32, step = 1, n = 32)
        @test length(got) == 32
        @test got ≈ Float32.(0.5:1:31.5)
    end

    @testset "the length is aten's, not a Julia range's" begin
        # `length(0:2:5-2)` is 2. aten's `ceil((5 - 0) / 2)` is 3.
        got = runarange(start = 0, stop = 5, step = 2, n = 3)
        @test length(got) == 3
        @test got == Float32[0, 2, 4]

        got = runarange(start = 1, stop = 10, step = 3, n = 3)
        @test got == Float32[1, 4, 7]
    end

    @testset "the unit step every other graph uses is unchanged" begin
        # TRELLIS.2's flow model: `arange(0, 128)`, no step attr at all.
        got = runarange(stop = 128, n = 128)
        @test length(got) == 128
        @test got == Float32.(0:127)
    end

    @testset "a fractional step" begin
        got = runarange(start = 0, stop = 1, step = 0.25, n = 4)
        @test got ≈ Float32[0, 0.25, 0.5, 0.75]
    end
end
