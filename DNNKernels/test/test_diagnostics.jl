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
using MatAnyoneRunner
const DK = DNNKernels
const KA = KernelAbstractions

# The runner hands back the graph; this file never names a path inside an
# artifact. `matanyoneprecisions()` is empty when the references are not bound,
# which the guard below turns into a recorded skip.
const HAVE_FP32 = "fp32" in MatAnyoneRunner.matanyoneprecisions()

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

    if !HAVE_FP32
        @info "no `matanyone-refs` artifact at fp32; skipping the execute! half"
    else
        g = MatAnyoneRunner.matanyonegraph("transform_key", "fp32")
        dims = (h = 8, w = 8)
        weights = Dict{String,Any}()
        for id in g.order
            b = g.buffers[id]
            b.kind === :weight || continue
            weights[b.key] = zeros(b.dtype, DK.evalshape(b.shape, dims)...)
        end
        inputs = Dict{String,Any}(i => zeros(g.buffers[i].dtype,
                                             DK.evalshape(g.buffers[i].shape, dims)...)
                                  for i in g.inputs)
        run(d) = execute!(g, inputs, weights; dims, backend = KA.CPU(), diag = d)

        @testset "op timings accumulate into the object they were given" begin
            d = Diagnostics(optimes = Dict{String,Tuple{Int,Float64}}())
            run(d)
            @test !isempty(d.optimes)
            @test sum(first, values(d.optimes)) == length(g.ops)
            @test all(t -> t[2] >= 0, values(d.optimes))
            # Same object, second run: the counts add up rather than reset.
            run(d)
            @test sum(first, values(d.optimes)) == 2 * length(g.ops)
        end

        @testset "two runs, two instruments, no crosstalk" begin
            timed = Diagnostics(optimes = Dict{String,Tuple{Int,Float64}}())
            missed = Diagnostics(planmisses = Dict{String,Tuple{Int,Int}}())
            run(timed)
            run(missed)
            # Each recorded only what it was asked for, and nothing of the other's.
            @test !isempty(timed.optimes)
            @test timed.planmisses === nothing
            # No plan was supplied, so every output `dest` handed out is a miss —
            # which is what makes this a usable assertion rather than a vacuous one.
            @test !isempty(missed.planmisses)
            @test missed.optimes === nothing
            @test first(planmisses(missed)).bytes >= last(planmisses(missed)).bytes
        end

        @testset "`opdouble` runs the named op twice, `opdoublefilter` narrows it" begin
            # An idempotent op run twice must not change the answer: that is the
            # whole basis of the differential-ablation measurement.
            base = run(Diagnostics())
            aten = g.ops[1].aten
            doubled = run(Diagnostics(opdouble = aten))
            @test base[g.outputs[1]] == doubled[g.outputs[1]]

            seen = String[]
            d = Diagnostics(opdouble = "*",
                            opdoublefilter = (ctx, op) -> (push!(seen, op.aten); false))
            run(d)
            # The filter saw every op and refused every one, so nothing doubled.
            @test length(seen) == length(g.ops)
            @test run(Diagnostics())[g.outputs[1]] == base[g.outputs[1]]
        end
    end
end
