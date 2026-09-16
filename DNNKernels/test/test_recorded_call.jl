# `call`, which declares a plan once per `(name, dims)` and replays it.
#
# Recording is not a mode. It was `Model(...; record = true)` with a list of
# preconditions and an immediate path to fall back to; `call` now emits, plans,
# records and replays, and there is nothing else for it to do. So what is under
# test here is the replay contract itself: the same plan answering repeated
# calls, and refusing a call it was not declared for.
#
# Two testsets went with the facilities they drove. `scratchfor(m, dims)[5]` and
# `flip!` reached into the interpreted run's slab and its `Recycler`, and
# `opdouble` ran one aten twice to attribute device time — a plan has PASSES and
# no way to be asked for that. See `test_diagnostics.jl` for the same note from
# the instrumentation side.

using Test, DNNKernels, Mantle, KernelAbstractions

@testset "a replayed plan carries its dtype conversions" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    # `clone` returns the input dtype and the declared output is narrower, so
    # the conversion is a pass of its own. This is the stale Whisper encoder
    # failure in miniature: a conversion that happened outside the capture scope
    # was not in the plan, so the replay returned the first call's values.
    buffers = Dict(id => DK.Buffer(id, kind, Any[128], T, "", (0, 2), "", "",
                                  Dict{String,Any}())
                   for (id, kind, T) in (("x", :external, Float32),
                                         ("y", :transient, Float16)))
    op = DK.Op("clone", "clone.default", ["x"], "y", Dict{String,Any}())
    g = DK.Graph("cast", String[], ["x"], ["y"], buffers, ["x", "y"], [op])
    m = DK.Model(Dict("cast" => g), Dict{String,Any}(), backend, 1, 1, 1)
    x = DK.toback(backend, fill(1f0, 128))
    @test Array(only(DK.call(m, "cast", x; dims=(;)))) == fill(Float16(1), 128)
    # The SAME buffer, written in place: the plan reads what it was declared
    # against, so a replay has to see the new bytes.
    fill!(x, 2f0)
    @test Array(only(DK.call(m, "cast", x; dims=(;)))) == fill(Float16(2), 128)
    # A DIFFERENT buffer of the same shape: `call` copies it into the one the
    # plan holds rather than re-recording.
    x2 = DK.toback(backend, fill(3f0, 128))
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(3), 128)
    fill!(x2, 4f0)
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(4), 128)
    # And the wrong arity, shape or dtype is refused rather than made to fit.
    @test_throws ErrorException DK.call(m, "cast"; dims=(;))
    @test_throws ArgumentError DK.call(m, "cast",
        DK.toback(backend, zeros(Float32, 64)); dims=(;))
    @test_throws ArgumentError DK.call(m, "cast",
        DK.toback(backend, zeros(Float16, 128)); dims=(;))
    # One plan, not one per call.
    @test count(v -> v isa DK.RecordedPlan, values(m.scratch)) == 1
end

# A wrong-shaped input used to RUN. The graph is built around its own
# declaration, so it produced a declaration-shaped result, recorded a plan
# around the mistake, and then rejected the CORRECT shape with "replay input
# shape or dtype changed" — one bad call poisoning the entry for everyone after
# it. The order here is that scenario: bad, then good.
@testset "call checks inputs against the declared shape" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    for (declared, dims, goodsize, badsize) in ((Any[1, 8], (;), (8, 1), (8, 2)),
                                                (Any[1, "n"], (; n = 8), (8, 1), (9, 1)))
        buffers = Dict(id => DK.Buffer(id, kind, declared, Float32, "", (0, 2), "", "",
                                       Dict{String,Any}())
                       for (id, kind) in (("x", :external), ("y", :transient)))
        op = DK.Op("relu", "relu.default", ["x"], "y", Dict{String,Any}())
        g = DK.Graph("act", collect(String, filter(s -> s isa String, declared)),
                     ["x"], ["y"], buffers, ["x", "y"], [op])
        m = DK.Model(Dict("act" => g), Dict{String,Any}(), backend, 1, 1, 1)
        bad = DK.toback(backend, fill(-1f0, badsize...))
        good = DK.toback(backend, fill(-1f0, goodsize...))
        @test_throws ArgumentError DK.call(m, "act", bad; dims)
        # Still usable afterwards: the bad call must not have recorded anything.
        for _ in 1:2
            @test all(iszero, Array(only(DK.call(m, "act", good; dims))))
        end
        @test_throws ArgumentError DK.call(m, "act", bad; dims)
    end
end

@testset "host randomness has no declared form" begin
    DK = DNNKernels
    back = Mantle.LavaBackend()
    b = DK.Buffer("y", :transient, Any[128], Float32, "", (0, 1), "", "", Dict{String,Any}())
    op = DK.Op("noise", "rand.default", String[], "y", Dict{String,Any}())
    g = DK.Graph("noise", String[], String[], ["y"], Dict("y"=>b), ["y"], [op])
    m = DK.Model(Dict("noise"=>g), Dict{String,Any}(), back, 1, 1, 1)
    # A host draw cannot be replayed, and this used to be a recording-time
    # refusal with `ZeroNoise` as the escape hatch — captured as a device fill.
    # Declared, there is no capture and no escape hatch: `rand.default` has no
    # `emitop!`, so the graph is refused whichever `NoiseSource` it was given.
    # The declared form would be a device RNG, which is a kernel.
    for noise in (DK.RandomNoise(), DK.ZeroNoise())
        err = try
            DK.call(m, "noise"; dims=(;), noise)
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("rand.default", sprint(showerror, err))
    end
end
