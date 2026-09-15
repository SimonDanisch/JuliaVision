# Sweep the registered direct-int8 tilings over the four weight shapes a Horizon
# layer actually runs, at the prompt widths where prefill still trails. Needs no
# model weights: the shapes are what matters, and random int8 of the same extent
# streams the same bytes.
#
# `(m, k)` is the weight as `matmul!` sees it -- m output channels, k reduction
# -- and `n` is the prompt width. Timing is a recorded replay, because the
# immediate launch path hides tile differences behind its own dispatch cost.
using DNNKernels, Mantle, KernelAbstractions, Statistics, Random, Test

const HORIZON_LAYER_GEMMS = ((53248, 5120, "swiglu gate+up"),
                             (10240, 5120, "qkv"),
                             (5120, 8192, "attention out"),
                             (5120, 26624, "ffn down"))

function bench_q8_tiles(backend; shapes = HORIZON_LAYER_GEMMS, widths = (128, 256),
                        repeats = 5, inner = 5)
    dev = DNNKernels.caps(backend)
    dev.coopmat && dev.coopmatsubgroup == 32 && dev.tile == 16 ||
        error("no cooperative matrix support to sweep")
    rng = MersenneTwister(97)
    for (m, k, label) in shapes
        a = DNNKernels.quantizeint8(backend,
            DNNKernels.toback(backend, randn(rng, Float16, m, k) .* Float16(.02)))
        for n in widths
            b = DNNKernels.toback(backend, randn(rng, Float16, k, n) .* Float16(.1))
            c = KernelAbstractions.allocate(backend, Float16, m, n)
            chosen = DNNKernels.q8gemm_tiling(dev, a, b, c)
            reference = nothing
            results = []
            for cfg in sort(collect(keys(DNNKernels.Q8_GEMM_KERNELS)))
                stm, stn, wm, wn, bk, pad = cfg
                bm, bn = 16stm * wm, 16stn * wn
                m % bm == 0 && n % bn == 0 && k % bk == 0 || continue
                shared = 2 * ((bm + pad) * bk + (bk + pad) * bn)
                32wm * wn <= dev.workgrouplimit && shared <= dev.sharedbudget || continue
                run() = (DNNKernels.q8gemm!(c, a, b; tiling = cfg); nothing)
                run(); KernelAbstractions.synchronize(backend)
                got = Array(c)
                if reference === nothing
                    reference = got
                else
                    @test maximum(abs, Float32.(got) .- Float32.(reference)) <=
                          .01maximum(abs, Float32.(reference))
                end
                g = Mantle.Graph(Mantle.Device(backend))
                Mantle.record_into(run, g, "q8gemm")
                plan = Mantle.record!(Mantle.Plan(g))
                times = Float64[]
                for _ in 1:repeats
                    push!(times, 1000 * @elapsed(begin
                        for _ in 1:inner; Mantle.run!(plan); end
                        KernelAbstractions.synchronize(backend)
                    end) / inner)
                end
                Mantle.free!(plan)
                push!(results, (median(times), cfg))
            end
            sort!(results)
            best, bestcfg = results[1]
            current = findfirst(r -> r[2] == chosen, results)
            now = current === nothing ? NaN : results[current][1]
            println("Q8TILE $label m=$m k=$k n=$n chosen=$chosen chosen_ms=$now ",
                    "best=$bestcfg best_ms=$best gain=$(round(100*(now-best)/now, digits=1))%")
            for (t, cfg) in results
                println("   $cfg  $(round(t, digits=3)) ms")
            end
            flush(stdout)
        end
    end
    nothing
end
