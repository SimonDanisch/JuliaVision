# WHY does the coopmat2 GEMM lose? An ablation, not a story.
#
# The first A/B said 37.9 TFLOP/s staged against 29.1 for the best coopmat2
# tiling, and the explanation written next to it — "less reuse per lane" — was a
# guess. This file tests the three ways the port actually differs from
# `mul_mm_cm2.comp`, one at a time, plus the one hardware fact that silently
# costs a factor of four:
#
#   1. **No unrolling.** The reference issues eight k-steps per iteration; the
#      port issued one, so every `muladd` waits on the previous one to retire.
#   2. **Clamping on every load.** The reference keeps an unclamped layout and
#      uses it whenever the tile is entirely in bounds. The port clamped always,
#      which bounds-checks every element of every load.
#   3. **Tile size.** 64x64 is 16 accumulator components a lane at 256
#      invocations; the staged kernel holds 32.
#   4. **Spill.** A cooperative-matrix accumulator that does not fit in registers
#      is spilled to local memory by the driver, silently, and the kernel then
#      runs at a quarter speed with no other symptom. `pipeline_exec_stats` is
#      the only way to see it without Nsight.
#
# Read the columns against each other. Each variant computes the same product and
# is checked against the same reference before it is timed.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
# Before the first pipeline is created, or the properties are not collected.
using Lava
Mantle.enable_pipeline_executable_properties!()
include(joinpath(@__DIR__, "gemm_lab.jl"))
using Printf

# The two dominant shapes, both of which divide every tile below — so the
# unclamped variants are legal and the comparison is like for like.
const WHYSHAPES = [(2304, 4096, 576), (576, 4096, 2304)]

# The first three isolate unrolling and clamping at the tile the first port used.
# The last three are **llama.cpp's own coopmat2 warptiles**, read out of
# `ggml-vulkan.cpp` rather than invented:
#
#     l_warptile = { 256, 128, 256, 64 }   256 threads, BM=128, BN=256, BK=64
#     m_warptile = { 256, 128, 128, 64 }   256 threads, BM=128, BN=128, BK=64
#     s_warptile = { 128,  64,  64, 64 }   128 threads, BM=64,  BN=64,  BK=64
#
# Note `BK = 64`, where the first port used 32 and 16, and note that the large
# tile is `128 x 256` — four to eight times the work per workgroup, i.e. four to
# eight times the reuse per byte loaded. If the gap closes here it was a bad
# port; if it does not, it is a fact about the approach.
const VARIANTS = [
    ("64x64/32  clamp,roll",   (64, 64, 32, 256), false, true),
    ("64x64/32  clamp,unroll", (64, 64, 32, 256), true,  true),
    # Is it the BIG TILE or the DEEP K that pins the register file? Every
    # `BK = 64` row above came back at 254-255 registers — the driver's ceiling —
    # while `64x64/32` used 62. These two separate the variables: same tile at
    # `BK = 32`, and the same `BK = 64` without unrolling.
    ("128x128/32 @256",        (128, 128, 32, 256), true, nothing),
    ("256x128/32 @256",        (256, 128, 32, 256), true, nothing),
    ("64x64/64 @128 noroll",   (64, 64, 64, 128), false, nothing),
    # Unclamped is only LEGAL where the tile divides the shape; `nothing` lets
    # the launcher decide, which is what a shipped path would do. Forcing it
    # `false` on a shape it does not divide reads out of bounds — that is what
    # the `WRONG` rows in the first run of this file were, and they were the
    # harness's fault rather than the kernel's.
    ("64x64/32  auto,unroll",  (64, 64, 32, 256), true,  nothing),
    ("s 64x64/64   @128",      (64, 64, 64, 128), true,  nothing),
    ("m 128x128/64 @256",      (128, 128, 64, 256), true, nothing),
    ("l 128x256/64 @256",      (128, 256, 64, 256), true, nothing),
]

# The pipeline cache is `ctx.caches.pipelines`, NOT the module-level
# `Lava.PIPELINE_CACHE` — that was one of the twelve globals that moved onto the
# context. `gemm_lab.jl`'s own `kernelstats` still reaches for the old name and
# would throw the same `UndefVarError` the first time anyone called it.
pipecache() = Mantle.vk_context().caches.pipelines

function stats_for(f)
    before = Set(keys(pipecache()))
    f(); KA.synchronize(BACKEND)
    fresh = [k for k in keys(pipecache()) if !(k in before)]
    isempty(fresh) && return nothing
    # `(; registers, scratch_bytes, raw_stats)`; scratch is the spill.
    Mantle.pipeline_exec_stats(pipecache()[fresh[1]])
end

function main()
    warmclock(16)
    for (M, N, K) in WHYSHAPES
        hA = rand(Float16, M, K) .- Float16(0.5)
        hB = rand(Float16, K, N) .- Float16(0.5)
        A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, hA)
        B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, hB)
        C = KA.allocate(BACKEND, Float16, M, N)
        rows = 1:min(M, 64)
        ref = Float32.(hA[rows, :]) * Float32.(hB)

        fs = Function[() -> Mantle.coopmat_gemm!(C, A, B, M, N, K)]
        names = ["staged"]
        for (nm, t, un, cl) in VARIANTS
            push!(fs, () -> Mantle.coopmat_gemm_cm2!(C, A, B, M, N, K;
                                                   tiling = t, unroll = un, clamp = cl))
            push!(names, nm)
        end
        # The other combination: tensor loads into SUBGROUP-scope tiles, keeping
        # the staged kernel's 4x4 register block. The ablation says a
        # workgroup-scope matrix is what the allocator handles badly, so this
        # asks whether the loads are worth anything when the tiles are the ones
        # it handles well.
        for nw in (2, 4)
            push!(fs, () -> Mantle.coopmat_gemm_cm2_sg!(C, A, B, M, N, K; nw = nw))
            push!(names, "sg 4x4 block @$(nw)sg")
        end

        @printf("\n%d x %d x %d\n", M, N, K)
        @printf("  %-22s %9s %9s %8s %8s %9s\n",
                "variant", "TFLOP/s", "vs staged", "regs", "spill", "relerr")
        base = 0.0
        for (i, f) in enumerate(fs)
            # Stats FIRST: `stats_for` reports the pipelines this call created,
            # and after a correctness run there are none left to create — which
            # is why the first version of this file printed "-" in every row.
            st = stats_for(f)
            fill!(C, Float16(0)); f(); KA.synchronize(BACKEND)
            got = Float32.(Array(C)[rows, :])
            err = maximum(abs.(got .- ref)) / max(1f-6, maximum(abs.(ref)))
            for _ in 1:3; f(); end; KA.synchronize(BACKEND)
            best = Inf
            for _ in 1:7
                t0 = time_ns(); for _ in 1:9; f(); end
                KA.synchronize(BACKEND)
                best = min(best, (time_ns() - t0) / 1e9 / 9)
            end
            tf = tflops(M, N, K, best)
            i == 1 && (base = tf)
            @printf("  %-22s %9.1f %8.1f%% %8s %8s %9.1e %s\n", names[i], tf,
                    100 * (tf - base) / base,
                    st === nothing ? "-" : string(something(st.registers, "-")),
                    st === nothing ? "-" : string(something(st.scratch_bytes, "-")),
                    err, err > 2e-2 ? "<-- WRONG" : "")
            flush(stdout)
        end
        A = B = C = nothing; GC.gc()
    end
end

main()
