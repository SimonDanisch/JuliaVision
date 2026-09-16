"""
Diagnostics ride on the context, not on the module.

Five module-level `Ref`s used to hold this state — `OPTIMES`, `OPDOUBLE`,
`OPDOUBLEFILTER`, `PLAN_MISSES`, `LAUNCH_PROBE` — and the tests that used them
had to save and restore, which a failing test would skip. What this file asserts
is the property that replaced them: **two runs in one process can be instrumented
differently and neither sees the other's measurements.** A global cannot do that,
so this is the test a regression to one would fail.

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
# and not from `matanyone-refs`. It used to ask for `transform_key` at **fp32**,
# which lives only in the refs artifact — unbound on every machine — so the
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
        # form — the same condition it recorded before this moved off a global.
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

    # ── the `execute!` half is gone, with the path it instrumented ───────────
    #
    # Three testsets lived here: per-op timings accumulating into a
    # `Diagnostics`, two instruments not crosstalking, and `opdouble` running a
    # named op twice for the differential-ablation measurement. All three drove
    # `execute!`, which is the interpreted run — and `execute!` cannot run: it
    # calls `reset!(ctx.ws)` per op and `ctx.ws` is a `Workspace`, which went
    # with that path on 2026-09-15.
    #
    # `optimes` is not a facility the declared path has, and not because it was
    # dropped: a declared graph has PASSES, not ops, and Mantle times them from
    # the device's own query pool (`Mantle.timings`, tested there). An op is a
    # host-side notion here and several of them become one pass. `opdouble` has
    # no declared meaning either — running one op twice inside a recorded plan
    # is not a thing a plan can be asked to do.
    #
    # What survives is above: the launch probe on the context, which is what
    # `Diagnostics` still instruments.
end
