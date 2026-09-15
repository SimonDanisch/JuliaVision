using Test, DNNKernels, Mantle, KernelAbstractions

@testset "recorded calls capture dtype conversions" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    # clone returns the input dtype; the declared output requires a separate
    # conversion kernel. This is the stale Whisper encoder failure in miniature.
    buffers = Dict(id => DK.Buffer(id, kind, Any[128], T, "", (0, 2), "", "",
                                  Dict{String,Any}())
                   for (id, kind, T) in (("x", :external, Float32),
                                         ("y", :transient, Float16)))
    op = DK.Op("clone", "clone.default", ["x"], "y", Dict{String,Any}())
    g = DK.Graph("cast", String[], ["x"], ["y"], buffers, ["x", "y"], [op])
    m = DK.Model(Dict("cast" => g), Dict{String,Any}(), backend, 1, 1, 1; record=true)
    x = DK.toback(backend, fill(1f0, 128))
    @test Array(only(DK.call(m, "cast", x; dims=(;)))) == fill(Float16(1), 128)
    fill!(x, 2f0)
    @test Array(only(DK.call(m, "cast", x; dims=(;)))) == fill(Float16(2), 128)
    x2 = DK.toback(backend, fill(3f0, 128))
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(3), 128)
    @test_throws ErrorException DK.call(m, "cast"; dims=(;))
    @test_throws ArgumentError DK.call(m, "cast", DK.toback(backend, zeros(Float32, 64)); dims=(;))
    @test_throws ArgumentError DK.call(m, "cast", DK.toback(backend, zeros(Float16, 128)); dims=(;))
    rec = DK.scratchfor(m, (;))[5]["cast"]
    DK.flip!(rec)
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(3), 128)
    fill!(x2, 4f0)
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(4), 128)
    DK.flip!(rec)
    @test Array(only(DK.call(m, "cast", x2; dims=(;)))) == fill(Float16(4), 128)
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
        m = DK.Model(Dict("act" => g), Dict{String,Any}(), backend, 1, 1, 1; record=true)
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

# `opdouble` attributes device time by running one aten twice and watching the
# step grow. The recording path used to call `runop!` directly and drop it, so a
# recorded model reported every op as free -- and a measurement that silently
# reads zero is worse than one that refuses.
@testset "a recording honours opdouble" begin
    DK = DNNKernels
    backend = Mantle.LavaBackend()
    buffers = Dict(id => DK.Buffer(id, kind, Any[128], Float32, "", (0, 2), "", "",
                                   Dict{String,Any}())
                   for (id, kind) in (("x", :external), ("y", :transient)))
    op = DK.Op("relu", "relu.default", ["x"], "y", Dict{String,Any}())
    g = DK.Graph("act", String[], ["x"], ["y"], buffers, ["x", "y"], [op])
    x = DK.toback(backend, fill(-1f0, 128))
    counts = map(("", "relu.default")) do double
        m = DK.Model(Dict("act" => g), Dict{String,Any}(), backend, 1, 1, 1; record=true)
        m.diag.opdouble = double
        # The first call runs immediately and then records; the second replays.
        # Relu of -1 is 0 either way, run twice or once.
        for _ in 1:2
            @test all(iszero, Array(only(DK.call(m, "act", x; dims=(;)))))
        end
        plan = only(v for v in values(m.scratch) if v isa DK.RecordedPlan).plan
        sum(length(p.dispatches) for p in plan.graph.passes)
    end
    @test counts[1] == 1
    @test counts[2] == 2      # the doubled dispatch is IN the plan, not just the run
end

@testset "recording refuses frozen host randomness" begin
    DK = DNNKernels
    back = Mantle.LavaBackend()
    b = DK.Buffer("y", :transient, Any[128], Float32, "", (0, 1), "", "", Dict{String,Any}())
    op = DK.Op("noise", "rand.default", String[], "y", Dict{String,Any}())
    g = DK.Graph("noise", String[], String[], ["y"], Dict("y"=>b), ["y"], [op])
    m = DK.Model(Dict("noise"=>g), Dict{String,Any}(), back, 1, 1, 1; record=true)
    @test_throws ArgumentError DK.call(m, "noise"; dims=(;))
    for _ in 1:2
        @test all(iszero, Array(only(DK.call(m, "noise"; dims=(;), noise=DK.ZeroNoise()))))
    end
    @test_throws ArgumentError DK.call(m, "noise"; dims=(;))
end
