# Per-op-family device cost by differential ablation. See `DNNKernels.OPDOUBLE`.
include(joinpath(@__DIR__, "lavadnn_bench.jl"))

const FAMILIES = ["convolution.default", "relu.default", "add.Tensor", "addmm.default",
                  "_to_copy.default", "cat.default", "mul.Tensor",
                  "native_layer_norm.default", "sigmoid.default", "div.Tensor",
                  "sub.Tensor", "clone.default", "upsample_bilinear2d.vec",
                  "bmm.default", "_adaptive_avg_pool2d.default", "mean.dim",
                  "_softmax.default", "slice_scatter.default"]

"""
    famcost(model, state, image, names) -> (base, drift, costs)

Cost of each op family, each measured against a baseline taken *immediately
around it*.

The interleaving is not caution, it is required. A single baseline at the start
of the run made every family come out negative — doubling an op appeared to make
the step faster — because the card idles at 555 MHz and boosts to 3105 while the
benchmark runs, so everything measured later beats anything measured earlier.
Bracketing each family with its own baseline cancels any drift slower than one
measurement triple.
"""
function famcost(m, s, image, names=FAMILIES; n=15, reps=3)
    DNNKernels.OPDOUBLE[] = ""
    bench() = median([benchsteps(m, s, image; n) for _ in 1:reps])
    for _ in 1:3; bench(); end          # let the clocks settle before anything counts
    base = bench()
    out = Tuple{String,Float64}[]
    drift = 0.0
    for nm in names
        b1 = bench()
        DNNKernels.OPDOUBLE[] = nm
        t = bench()
        DNNKernels.OPDOUBLE[] = ""
        b2 = bench()
        push!(out, (nm, t - (b1 + b2) / 2))
        drift = max(drift, abs(b2 - b1))
    end
    (base=base, drift=drift, costs=sort(out, by=p -> -p[2]))
end

function showfam(r)
    println("base ", round(r.base, digits=2), " ms   drift across the run ",
            round(r.drift, digits=2), " ms")
    for (nm, c) in r.costs
        println(rpad(nm, 34), lpad(round(c, digits=2), 7), " ms  (",
                round(100c / r.base, digits=1), "%)")
    end
    println("accounted ", round(sum(c for (_, c) in r.costs), digits=2),
            " of ", round(r.base, digits=2), " ms")
end
