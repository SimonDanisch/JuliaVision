# Bench harness for the DNNKernels inference loop. Reload after a restart with
#   include("tools/lavadnn_bench.jl")
using DNNKernels, KernelAbstractions, Lava, Statistics, LinearAlgebra, Logging
const KA = KernelAbstractions
const GENDIR = joinpath(@__DIR__, "..", "gen")

buildmodel(dir; backend=LavaBackend()) =
    Model(joinpath(GENDIR, "graphs", dir), joinpath(GENDIR, "weights.safetensors"); backend)

"""
    e2einputs(backend) -> (image, mask, refalpha)

The first frame of the reference clip and its mask, on the device. Random noise
is useless for checking a matte: the segmentation head makes a hard decision per
pixel, so two numerically-close runs can differ by 1.0 on a frame with no
structure in it. `refalpha` is PyTorch's own output for that frame.
"""
function e2einputs(backend)
    d = readsafetensors(joinpath(GENDIR, "e2e-fp32.safetensors"))
    # Stored frame-major: `frames` is (T, W, H, C) and `alpha` is (T, W, H).
    frames, mask, alpha = d["frames"], d["mask"], d["alpha"]
    f1 = permutedims(collect(view(frames, 1:1, :, :, :)), (2, 3, 4, 1))
    (DNNKernels.toback(backend, f1), DNNKernels.toback(backend, collect(mask)),
     collect(view(alpha, 1, :, :)))
end

"""Model + a state warmed through the mask-ingest and first-frame paths."""
function warmmodel(m, image, mask)
    s = DNNKernels.initstate(m, size(image, 1), size(image, 2))
    DNNKernels.step!(m, s, image; mask)
    for _ in 1:3
        DNNKernels.step!(m, s, image; firstframe=true)
    end
    KA.synchronize(m.backend)
    (m, s)
end

"""Median ms per `step!`, synchronising each step."""
function benchsteps(m, s, image; n=25)
    DNNKernels.step!(m, s, image); KA.synchronize(m.backend)
    ts = Float64[]
    for _ in 1:n
        t = time_ns()
        DNNKernels.step!(m, s, image)
        KA.synchronize(m.backend)
        push!(ts, (time_ns() - t) / 1e6)
    end
    median(ts)
end

"""Host-side (record) time vs wall, to see whether the step is CPU- or GPU-bound."""
function splitcost(m, s, image; n=20)
    DNNKernels.step!(m, s, image); KA.synchronize(m.backend)
    host = Float64[]; wall = Float64[]
    for _ in 1:n
        t = time_ns()
        DNNKernels.step!(m, s, image)
        push!(host, (time_ns() - t) / 1e6)
        KA.synchronize(m.backend)
        push!(wall, (time_ns() - t) / 1e6)
    end
    (host=median(host), wall=median(wall))
end

"""Best-of-`rounds` timing; anything less mismeasures a cold kernel by 10x."""
function timeit(f; n=100, rounds=5, backend=LavaBackend())
    best = Inf
    for _ in 1:rounds
        f(); KA.synchronize(backend)
        t = time_ns()
        for _ in 1:n; f(); end
        KA.synchronize(backend)
        best = min(best, (time_ns() - t) / 1e6 / n)
    end
    best
end
