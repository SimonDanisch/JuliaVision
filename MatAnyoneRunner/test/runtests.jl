"""
Same claim as `SAM2Runner`'s test, same reason it runs in a subprocess: Julia's
compile-time counter is per-process, so a call in *this* one has already paid
whatever there was to pay.

Additionally checks that the shared `KERNELS_VERSION` is doing its job — the
kernels this model has in common with SAM 2 must be hits, not a second copy.
"""

using Test, MatAnyoneRunner

const SUBPROCESS = """
using MatAnyoneRunner, Lava, DNNKernels, KernelAbstractions
using DNNKernels: toback
const KA = KernelAbstractions
backend = LavaBackend()
model = MatAnyoneRunner.matanyonemodel(; backend)
W, H = 128, 96
image = toback(backend, fill(0.5f0, W, H, 3, 1))
host = zeros(Float32, W, H); host[(W÷4):(3W÷4), (H÷4):(3H÷4)] .= 255.0f0
mask = toback(backend, host)
Lava.frozen_reset_stats!()
c0 = Base.cumulative_compile_time_ns()
t = @elapsed begin
    a = MatAnyoneRunner.runmatanyone(model, image, mask)
    ah = Array(a); KA.synchronize(backend)
end
c1 = Base.cumulative_compile_time_ns()
s = Lava.frozen_stats()
println("RESULT ", (; wall = t, compile = (c1[1] - c0[1]) / 1e9,
                     hits = s.hits, misses = s.misses, version = s.version,
                     finite = all(isfinite, ah), inrange = all(x -> 0 <= x <= 1, ah)))
"""

@testset "MatAnyoneRunner: first call does not compile" begin
    if !MatAnyoneRunner.ready()
        @info "no MatAnyone assets; skipping"
    else
        script = tempname() * ".jl"
        write(script, SUBPROCESS)
        out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`, String)
        i = findfirst(l -> startswith(l, "RESULT "), split(out, '\n'))
        @test i !== nothing
        r = eval(Meta.parse(split(out, '\n')[i][8:end]))
        @info "MatAnyone2 first propagation in a fresh process" r
        @test r.finite
        @test r.inrange                             # an alpha matte, not garbage
        @test r.misses == 0                         # every kernel came from disk
        @test r.hits > 0
        @test r.compile < 5.0
        @test r.wall < 20.0
        @test r.version == DNNKernels.KERNELS_VERSION  # the shared generation
    end
end

# The parity gate: MatAnyone against PyTorch layer by layer, error floor ~1e-6.
# Moved from DNNKernels/test — it needs this model's weights and references, and
# a kernel library's suite should not require a model package to run.
include("test_parity.jl")
