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

using Test, SAM2Runner, Random
using DNNKernels: verifygraph
import Lava

const SUBPROCESS = """
using SAM2Runner, Lava, DNNKernels, KernelAbstractions, Printf
using DNNKernels: readsafetensors, toback
const KA = KernelAbstractions

backend = LavaBackend()
# See the module docstring: this exists so a rare hang names its kernel. These
# are `ctx.diag` fields now, not module-level `Ref`s, so they are set AFTER the
# backend exists — there is no context to carry them before that.
let d = Lava.vk_context().diag
    d.dispatch_log_file = joinpath(tempdir(), "sam2runner_dispatch.log")
    d.dispatch_logging = true
end
model = SAM2Runner.sam2model(; backend)
refs = SAM2Runner.sam2refs()
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
    if !SAM2Runner.ready()
        @info "no SAM 2 assets; skipping the latency test"
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

# The mask resample, against the literal form of its own definition.
#
# It is the largest host-side cost of a click — 8.1 ms at 1920x1080 against 3.3
# for the decode — and the rewrite that took it to 1.04 ms is entirely about
# what does *not* vary per pixel. That is exactly the kind of change that is
# correct on the interior and wrong on an edge, so the reference below is the
# original loop, kept verbatim, and the comparison is over every pixel rather
# than a tolerance: bilinear-then-threshold has no rounding slack, a wrong tap
# flips a pixel outright.
#
# No GPU and no assets, so it always runs.
"""The reference's threshold input, so a disagreement can be judged by how close
to zero it was rather than only counted."""
function referencevalue(lg, w, h)
    mw, mh = size(lg)
    out = Matrix{Float32}(undef, w, h)
    @inbounds for j in 1:h, i in 1:w
        fx = ((i - 0.5) / w) * mw + 0.5
        fy = ((j - 0.5) / h) * mh + 0.5
        x0 = clamp(floor(Int, fx), 1, mw); y0 = clamp(floor(Int, fy), 1, mh)
        x1 = min(x0 + 1, mw); y1 = min(y0 + 1, mh)
        tx = Float32(fx - x0); ty = Float32(fy - y0)
        out[i, j] = (1 - tx) * (1 - ty) * lg[x0, y0] + tx * (1 - ty) * lg[x1, y0] +
                    (1 - tx) * ty * lg[x0, y1] + tx * ty * lg[x1, y1]
    end
    out
end

function referencemask(lg, w, h)
    mw, mh = size(lg)
    out = Matrix{UInt8}(undef, w, h)
    @inbounds for j in 1:h, i in 1:w
        fx = ((i - 0.5) / w) * mw + 0.5
        fy = ((j - 0.5) / h) * mh + 0.5
        x0 = clamp(floor(Int, fx), 1, mw); y0 = clamp(floor(Int, fy), 1, mh)
        x1 = min(x0 + 1, mw); y1 = min(y0 + 1, mh)
        tx = Float32(fx - x0); ty = Float32(fy - y0)
        v = (1 - tx) * (1 - ty) * lg[x0, y0] + tx * (1 - ty) * lg[x1, y0] +
            (1 - tx) * ty * lg[x0, y1] + tx * ty * lg[x1, y1]
        out[i, j] = v > 0 ? 0xff : 0x00
    end
    out
end

@testset "maskatframe matches its reference" begin
    # Seeded: the assertion below is about a *tie* at the threshold, and an
    # unseeded draw turns that into a test that fails once every few hundred
    # runs on a different pixel each time.
    Random.seed!(20260802)
    lg = Float32.(3 .* randn(256, 256))
    # Upsampling, downsampling, 1:1, and the degenerate sizes where x1 == x0.
    # **Not bit-equality, and the reason is worth stating.** The reference sums
    # four products; the shipped form blends two rows and then interpolates. Those
    # are algebraically the same and not the same in floating point, so a pixel
    # whose interpolated value sits *at* zero can round either side of the `> 0`
    # threshold. Measured: one pixel in 8 294 400 at 3840x2160, where the
    # reference gets +1.19e-07 and the row-blend gets exactly 0.0.
    #
    # So the contract is: identical wherever the value is meaningfully non-zero,
    # and disagreements confined to the zero crossing. An indexing bug would put
    # them anywhere.
    @testset "$w x $h" for (w, h) in [(1920, 1080), (256, 256), (64, 40),
                                      (1, 1), (1, 300), (300, 1), (3840, 2160)]
        got = SAM2Runner.maskatframe(lg, w, h)
        want = referencemask(lg, w, h)
        val = referencevalue(lg, w, h)
        differ = got .!= want
        @test count(differ) <= max(1, length(got) ÷ 1_000_000)
        # Every disagreement is a value that rounds to zero in fp32.
        @test all(abs.(val[differ]) .<= 1f-6)
        # And away from the crossing they agree exactly.
        solid = abs.(val) .> 1f-3
        @test got[solid] == want[solid]
    end

    # A logit field that is zero somewhere: `v > 0` is the whole output, so a
    # field that never crosses zero would agree with almost any implementation.
    flat = zeros(Float32, 256, 256)
    flat[100:160, 100:160] .= 1f0
    m = SAM2Runner.maskatframe(flat, 1920, 1080)
    @test m == referencemask(flat, 1920, 1080)
    @test 0 < count(==(0xff), m) < length(m)      # it really does have both

    # Non-square logits, since nothing in the signature promises 256x256.
    odd = Float32.(randn(37, 91))
    @test SAM2Runner.maskatframe(odd, 640, 480) == referencemask(odd, 640, 480)
end

# The layer-by-layer gate against PyTorch, on the graphs as they SHIP — after
# every rewrite pass, on the GPU. `tools/verify_sam2.jl` covers the raw export
# on the CPU; this one is what catches a rewrite pass or a kernel drifting.
# The decoder is fully green. The encoder carries one known output-level
# mismatch, pinned so its moving in EITHER direction is loud.
@testset "graphs vs PyTorch, layer by layer" begin
    if !SAM2Runner.ready()
        @info "no SAM 2 assets; the layer-by-layer gate is SKIPPED, not passing"
        @test_skip SAM2Runner.ready()
    else
        back = Lava.LavaBackend()
        sam = SAM2Runner.sam2model(; backend = back)
        refs = SAM2Runner.sam2refs()

        ok, diffs, _ = verifygraph(sam.model.graphs["sam2_decoder"], refs,
                                   sam.model.weights;
                                   dims = (res = 1024,), backend = back,
                                   verbose = false)
        ok || @info "decoder first mismatch" first(diffs)
        @test ok

        oke, diffse, _ = verifygraph(sam.model.graphs["sam2_encoder"], refs,
                                     sam.model.weights;
                                     dims = (res = 1024,), backend = back,
                                     verbose = false)
        # Known, and so far unlocalised: the dump covers the encoder at its six
        # outputs only, so `add_129` carries the accumulated fp16 difference of
        # the 543 unchecked ops before it. `runsam2`'s score is unaffected and
        # the value is identical fused and unfused. Localising it is
        # `uv run tools/dump_sam2_refs.py --nodes all` — which needs a torch
        # install this machine does not have. Until then the gate pins the
        # signature: same node, same magnitude, and a change either way fails.
        f = first(diffse)
        @test !oke
        @test f.id == "add_129"
        @test 0.2 < f.maxabs < 0.6
    end
end

# The decoder's baked-plan path. It lived in `DNNKernels/test` while `sam2.jl`
# did; it tests SAM 2's decode and belongs with it.
include("test_replay_decode.jl")
