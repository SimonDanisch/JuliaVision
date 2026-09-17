"""
Diagnostics ride on the context, not on the module.

Module-level `Ref`s holding this state make every test that touches them save
and restore, which a failing test skips. What this file asserts is the property
that makes that unnecessary: **two runs in one process can be instrumented
differently and neither sees the other's measurements.** A global cannot do
that, so this is the test a regression to one would fail.

Host-only, on the CPU backend: nothing here is about what a kernel computes.

Run standalone:
    julia --project=. dev/JuliaVision/DNNKernels/test/test_diagnostics.jl
"""

using Test
using DNNKernels
using DNNKernels: Diagnostics, Ctx, loadgraph, execute!, launch!, planmisses
using KernelAbstractions
include("fixtures.jl")
const DK = DNNKernels
const KA = KernelAbstractions

# The graph comes from this package's own `matanyone` binding, not from a runner
# and not from `matanyone-refs`. Asking for `transform_key` at **fp32** reaches
# only into the refs artifact, unbound on every machine, so the
# `execute!` half of this file silently skipped: 25 tests became 13. Nothing here
# depends on the precision; it counts launches.

@testset "diagnostics on the context" begin
    @testset "a fresh one is off, and off costs nothing to ask" begin
        d = Diagnostics()
        @test d.optimes === nothing
        @test d.opdouble == ""
        @test d.opdoublefilter === nothing
        @test d.planmisses === nothing
        @test d.launches === nothing
        @test isempty(planmisses(d))
    end

    @testset "the launch probe reaches `launch!` through the context" begin
        back = KA.CPU()
        d = Diagnostics(launches = Dict{Tuple{Base.Dims,Base.Dims},
                                        Tuple{Int,Base.Dims}}())
        ctx = Ctx(back; diag = d)
        out = zeros(Float32, 64)
        # A 1-D destination on purpose: `launch!` flattens every multi-dimensional
        # linearly-indexable output, and the probe records the shape of the N-D
        # form, which is the condition the probe is for.
        launch!(ctx, (I, v) -> v, out, 2.0f0)
        launch!(ctx, (I, v) -> v, out, 3.0f0)
        @test all(==(3.0f0), out)
        @test length(d.launches) == 1
        (count, groups) = first(values(d.launches))
        @test count == 2
        @test all(>(0), groups)
        # A second context with its own diagnostics does not see those two.
        d2 = Diagnostics(launches = Dict{Tuple{Base.Dims,Base.Dims},
                                         Tuple{Int,Base.Dims}}())
        launch!(Ctx(back; diag = d2), (I, v) -> v, out, 1.0f0)
        @test sum(first, values(d2.launches)) == 1
        @test sum(first, values(d.launches)) == 2
        # And a context with the default diagnostics records nothing at all.
        launch!(Ctx(back), (I, v) -> v, out, 0.0f0)
        @test sum(first, values(d.launches)) == 2
    end

    # ── what the DECLARED path can be asked ─────────────────────────────────
    #
    # `optimes` and `opdouble` instrument the interpreted run, and neither has a
    # declared counterpart: a declared graph has PASSES, not ops, and Mantle
    # times them from the device's own query pool (`Mantle.timings`, tested
    # there), while running one op twice inside a recorded plan is not something
    # a plan can be asked to do.
    #
    # So what is asserted above is the launch probe on the context, which is
    # what `Diagnostics` instruments on both paths.
end
