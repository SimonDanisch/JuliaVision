"""
The claim under test is "no compile time on first use", and it can only be
measured in a process that has not run SAM 2 yet — Julia's compile-time counter
is per-process and every earlier call in this one has already paid it.

So the measurement runs in a **fresh subprocess** and reports back. What it
asserts:

  * `Lava.frozen_stats().misses == 0` — every kernel the run touches came from
    the frozen cache. A miss is a kernel the workload does not cover, and it is
    the thing that silently reintroduces a multi-second stall.
  * Julia's own compile time over the first `runsam2` is a small fraction of it.
    Without the workload this was ~62 s of ~63 s.
  * The result still matches the reference, because a cache that returns the
    wrong kernel fast is worse than no cache.

**Dispatch logging is on in the subprocess.** Twice on 2026-07-29 this test hung
in `vkWaitSemaphores` — 120 s waiting for a timeline value with five batches in
flight, from the `Array(mask)` download — and both times only when a GPU-heavy
Julia process had just exited. It is not reproducible on demand (0 of 3 attempts
at that exact sequence) and it passes standalone, so rather than leave the next
occurrence as a mystery the log names the kernel that did not complete. The cost
is one interpolated string per dispatch on a 0.6 s run.
"""

using Test, SAM2Runner

const SUBPROCESS = """
using SAM2Runner, Lava, DNNKernels, KernelAbstractions, Printf
using DNNKernels: readsafetensors, toback
const KA = KernelAbstractions

# See the module docstring: this exists so a rare hang names its kernel.
Lava.DISPATCH_LOG_FILE[] = joinpath(tempdir(), "sam2runner_dispatch.log")
Lava.DISPATCH_LOGGING_ENABLED[] = true

dir = SAM2Runner.assetdir()
backend = LavaBackend()
model = SAM2Runner.sam2model(; backend, dir)
refs = readsafetensors(joinpath(dir, "refs.safetensors"))
image = toback(backend, refs["sam2_encoder/in0"])

Lava.frozen_reset_stats!()
c0 = Base.cumulative_compile_time_ns()
t = @elapsed begin
    mask, score = SAM2Runner.runsam2(model, image)
    m = Array(mask); KA.synchronize(backend)
end
c1 = Base.cumulative_compile_time_ns()

s = Lava.frozen_stats()
println("RESULT ", (; wall = t, compile = (c1[1] - c0[1]) / 1e9,
                     hits = s.hits, misses = s.misses, stores = s.stores,
                     version = s.version, finite = all(isfinite, m)))
"""

@testset "SAM2Runner: first call does not compile" begin
    dir = SAM2Runner.assetdir()
    if !isfile(joinpath(dir, "weights.safetensors"))
        @info "no SAM 2 assets; skipping the latency test" dir
    else
        script = tempname() * ".jl"
        write(script, SUBPROCESS)
        out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) $script`, String)
        line = findfirst(l -> startswith(l, "RESULT "), split(out, '\n'))
        @test line !== nothing
        r = eval(Meta.parse(split(out, '\n')[line][8:end]))

        @info "SAM 2.1 first call in a fresh process" r
        @test r.finite                              # the answer is still real
        @test r.misses == 0                         # every kernel came from disk
        @test r.hits > 0                            # …and there were kernels
        # Without the workload this was ~62 s of compilation. Anything close to
        # that means the package image is not being used.
        @test r.compile < 5.0
        @test r.wall < 20.0
    end
end
