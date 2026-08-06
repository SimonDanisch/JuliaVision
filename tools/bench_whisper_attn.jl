# Is Whisper's attention slow because it is CLAMPED?
#
# The attribution puts `_scaled_dot_product_flash_attention` at 44.68 ms of a
# 124.66 ms encode (35.8%), and the arithmetic is only 0.369 TFLOP — so it runs
# at 8.3 TF/s against a measured 107.3 TF/s ceiling, 7.7%. That is the worst
# efficiency of any family in the model, worse than the GEMM's 25%.
#
# The obvious suspect is the flag that made it run at all. Whisper's audio
# context is 1500, which divides no block, so `clampattn = true` is what lifts
# `Decline(:extent)` — and every one of the 32 attentions therefore takes the
# BOUNDS-CHECKED path. `flashcm_tiling` also takes `clamp`, so the clamped call
# may not even be choosing the same tiling.
#
# The alternative is the trick `Lava.gemm_padn` already applies to the GEMM's
# N = 1500: pad the sequence to 1536 (= 24 * 64) and run UNCLAMPED. That is 2.4%
# more arithmetic, which the comparison below charges to the padded arm honestly
# by reporting achieved TF/s over each arm's OWN flop count.
#
# Two arms, interleaved via `compare` — never two `bench` calls, see measure.jl.
#
# ── RESULT 2026-08-06: THIS HARNESS CANNOT ANSWER THE QUESTION. Keep the file,
# but do not believe a number from it without fixing the clock first.
#
#     L1500 clamped     2.505 ms  ±57.3%  clock 0.39   4.6 TF/s
#     L1536 unclamped   2.345 ms  ±52.5%  clock 0.39   5.2 TF/s
#     ratio 1.068x   trustworthy=false
#
# `clock 0.39` is the finding. A ~2.5 ms kernel with a `synchronize` between
# samples never holds the card awake, so `plateau` measured its gate against a
# DOWNCLOCKED card and `floor = 0.90` of that low plateau kept all 15 samples.
# The arms are internally consistent and jointly meaningless: this harness prices
# the 32 attentions at 80.2 ms/encode where the in-situ `opdouble` attribution
# prices them at 44.68. Same code, 1.8x apart, because of the clock alone.
#
# That also settles the question it was built for — the clamp is NOT why
# attention runs at 8.3 TF/s, since a 6.8% difference sits far inside a 57%
# spread. Attention questions here have to be asked IN SITU (tools/attrib_whisper.jl),
# which is the same conclusion `opdoublefilter`'s docstring already records for
# standalone convolution microbenchmarks: "identical code timed one shape at
# 16 us and 116 us in consecutive runs".

using DNNKernels, Lava, KernelAbstractions, Printf
const DK = DNNKernels
const KA = KernelAbstractions
include("measure.jl")

back = LavaBackend()

E, H, B = 64, 20, 1          # head dim, heads, batch — Whisper large-v3-turbo
scale = 1.0f0 / sqrt(Float32(E))

"""Dense fp16 q/k/v at sequence length `L`, in the kernel's own `(E, L, H, B)`."""
function operands(L)
    q = KA.allocate(back, Float16, E, L, H, B)
    k = KA.allocate(back, Float16, E, L, H, B)
    v = KA.allocate(back, Float16, E, L, H, B)
    for a in (q, k, v)
        copyto!(a, Float16.(randn(Float32, E, L, H, B) .* 0.1f0))
    end
    (q, k, v)
end

flops(L) = 2 * 2 * L * L * E * H * B      # QK' and PV, 2 flops per MAC

function arm(L, clampattn)
    ctx = DK.Ctx(back; ws = DK.Workspace(back), clampattn)
    q, k, v = operands(L)
    plan, k2, v2 = DK.sdpaplan(ctx, q, k, v, nothing)
    out = KA.allocate(back, Float32, E, L, H, B)
    (; L, clampattn, plan, run = () -> DK.sdpa!(ctx, plan, out, q, k2, v2, nothing, scale))
end

a = arm(1500, true)      # what ships
b = arm(1536, false)     # padded to the block, unclamped

for x in (a, b)
    p = x.plan
    @printf("L=%-5d clamp=%-6s plan=%s\n", x.L, x.clampattn,
            p isa DK.FlashCMPlan ?
                "FlashCM(BR=$(p.BR), BC=$(p.BC), NW=$(p.NW), nsplit=$(p.nsplit), clamp=$(p.clamp))" :
                string(p))
end
println()

ra, rb, ratio, trust = compare(a.run, b.run;
                              labels = ("L1500 clamped", "L1536 unclamped"),
                              sync = () -> KA.synchronize(back))
for (r, x) in ((ra, a), (rb, b))
    @printf("%-18s %8.3f ms  ±%5.1f%%  clock %4.2f  kept %2d  %6.1f TF/s\n",
            r.label, 1e3r.median, 100r.spread, r.clock, r.kept,
            flops(x.L) / r.median / 1e12)
end
@printf("\nratio %.3fx   trustworthy=%s  (effect must exceed BOTH spreads)\n", ratio, trust)
@printf("32 attentions per encode: %.1f ms vs %.1f ms  ->  %+.1f ms on a 124.7 ms encode\n",
        32e3ra.median, 32e3rb.median, 32e3 * (rb.median - ra.median))
