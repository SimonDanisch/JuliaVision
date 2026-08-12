# Is the coopmat2 flash kernel bound by its softmax or by its products?
#
# The tile sweep is exhausted (`BR` x `BC` x `NT` all measured) and the kernel
# sits at 255 registers — the driver's cap — which says it is saturated but not
# WHY it is 2x off PyTorch's 24.9 TFLOP/s. This removes work one layer at a time
# and computes wrong answers on purpose, which is what an ablation is for.
#
#   :all         the kernel
#   :norescale   drops the `O` rescale — the `Br x EP` smear and the multiply,
#                the largest live transient (40 registers of 255)
#   :noreduce    drops the two row reductions, keeps every per-element pass
#   :noexp       the mirror: keeps both reductions, drops the per-element chain
#   :nosoftmax   drops the whole online softmax; `S` feeds the second product
#
# Read the DIFFERENCES. Each variant is a different amount of work and only the
# gaps mean anything; the absolute numbers of the broken ones are meaningless.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
include(joinpath(@__DIR__, "attn_lab.jl"))
using Printf

const DEV = DK.caps(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

function ablate(E, Lq, Lk, H, B; n = 7, reps = 5)
    q, k, v, out = operands(E, Lq, Lk, H, B)
    scale = Float32(1 / sqrt(E))
    plan = DK.flashcm2_plan(DEV, q, k, v, nothing)
    plan isa DK.FlashCM2Plan || (println("  declined: ", plan.reason); return)
    vs = [(a, () -> (DK.reset!(WS);
                     DK.sdpaflashcm2!(CTX, out, plan, q, k, v, scale; abl = a)))
          for a in (:all, :noreduce, :noexp, :norescale, :nosoftmax)]
    for _ in 1:4, (_, f) in vs; f(); end
    KA.synchronize(BACKEND)
    # INTERLEAVED — one rep of every variant, then the next. Timing each
    # variant in its own block let the leading one absorb the ramp: the fused
    # A/B read 2.305 ms for a configuration that measures 0.235.
    acc = [Float64[] for _ in vs]
    for _ in 1:reps, (i, (_, f)) in enumerate(vs)
        t0 = time_ns(); for _ in 1:n; f(); end
        KA.synchronize(BACKEND); push!(acc[i], (time_ns() - t0) / 1e6 / n)
    end
    best = map(minimum, acc)
    g = gflop(E, Lq, Lk, H, B)
    @printf("\nE%d L%dx%d H%d B%d   (%.1f GFLOP)\n", E, Lq, Lk, H, B, g)
    for (i, (a, _)) in enumerate(vs)
        @printf("  %-11s %8.3f ms  %6.2f TF/s   %+7.1f%% vs :all\n",
                a, best[i], g / best[i], 100 * (best[i] - best[1]) / best[1])
    end
    flush(stdout)
end

warmclock(16)
ablate(72, 4096, 4096, 8, 1)
ablate(72, 256, 256, 8, 16)
println("\nABLDONE")
