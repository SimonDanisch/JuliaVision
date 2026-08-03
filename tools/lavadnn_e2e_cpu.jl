"""
End-to-end driver check on the CPU backend, against PyTorch's own alpha.

`dev/JuliaVision/DNNKernels/test/runtests.jl` calls `verifygraph` on the graph
JSON directly, so it never constructs a `Model` — which means it does not
exercise `foldbatchnorm`, `foldrelu`, `hoistcasts` or `dropdead` at all. Those
rewrite the graph before it ever runs, and a fault in one of them is invisible
to a green suite. This is the check that covers them, and it needs no GPU.

    julia --project=. tools/lavadnn_e2e_cpu.jl [fp32|autocast]
"""

using DNNKernels, KernelAbstractions, Logging

const GEN = normpath(joinpath(@__DIR__, "..", "gen"))

function main(precision = "fp32")
    d = readsafetensors(joinpath(GEN, "e2e-fp32.safetensors"))
    # Stored frame-major: `frames` is (T, W, H, C), `alpha` is (T, W, H).
    frames, mask, alpha = d["frames"], d["mask"], d["alpha"]
    img = permutedims(collect(view(frames, 1:1, :, :, :)), (2, 3, 4, 1))
    ref = collect(view(alpha, 1, :, :))

    m = with_logger(ConsoleLogger(stderr, Logging.Debug)) do
        Model(joinpath(GEN, "graphs", "aten-$precision"),
              joinpath(GEN, "weights.safetensors"))
    end
    s = DNNKernels.initstate(m, size(img, 1), size(img, 2))
    DNNKernels.step!(m, s, img; mask = collect(mask))
    for _ in 1:3
        DNNKernels.step!(m, s, img; firstframe = true)
    end
    a = DNNKernels.step!(m, s, img)

    err = abs.(a .- ref)
    mean = sum(err) / length(err)
    big = count(>(0.1), err)
    println("e2e $precision: mean|Δ| = ", mean, "  max = ", maximum(err),
            "  px>0.1 = ", big, "/", length(err))
    # The fp32 oracle sits at ~1e-6 per-op; the driver's own reductions and the
    # tie-sensitive predicate push the end-to-end figure to ~3e-4. Anything an
    # order of magnitude above that is a real fault in a graph pass.
    ok = mean < 2e-3 && big < length(err) ÷ 100
    println(ok ? "PASS" : "FAIL")
    ok
end

exit(main(isempty(ARGS) ? "fp32" : ARGS[1]) ? 0 : 1)
