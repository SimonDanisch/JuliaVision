# Does `opdouble` cost anything when it matches NOTHING?
#
# The two Whisper attributions disagreed about their own null control: run 1 read
# -0.60 ms, run 2 read +5.35 ms with a 2.8% spread. If a non-matching `opdouble`
# systematically slows the step, every attribution carries that offset and the
# per-family numbers are all shifted.
#
# The code says it should be free. `timeop!` short-circuits on `isempty` and then
# does ONE string compare per op:
#
#     d = ctx.diag.opdouble
#     if !isempty(d) && (d == "*" || op.aten == d) && opdoublewanted(ctx, op)
#
# 681 ops against a name of a different length is a length check each — order 7 us
# a step, not 5 ms. So either the reading was drift, or something else is gated.
#
# ALTERNATING arms, not one block each, because that is the whole lesson of
# measure.jl: a block per arm gives each its own stretch of the clock ramp.

using WhisperRunner, DNNKernels, Lava, KernelAbstractions, Statistics, Printf
const KA = KernelAbstractions

backend = LavaBackend()
w = WhisperRunner.whispermodel(; backend)
mel = rand(Float32, 3000, 128, 1)
enc() = (WhisperRunner.encode(w, mel); KA.synchronize(backend))
enc(); enc(); enc()

ROUNDS = 12
off, nul = Float64[], Float64[]
for _ in 1:ROUNDS
    w.model.diag.opdouble = ""
    t0 = time_ns(); enc(); push!(off, (time_ns() - t0) / 1e6)

    w.model.diag.opdouble = "no_such_aten_zzz"
    t0 = time_ns(); enc(); push!(nul, (time_ns() - t0) / 1e6)
end
w.model.diag.opdouble = ""

stat(v) = (sort!(copy(v)); (med = v[cld(end, 2)], lo = first(v), hi = last(v)))
so, sn = stat(off), stat(nul)
@printf("opdouble = \"\"                med %.2f  min %.2f  max %.2f  (n=%d)\n",
        so.med, so.lo, so.hi, ROUNDS)
@printf("opdouble = no-such-aten      med %.2f  min %.2f  max %.2f\n", sn.med, sn.lo, sn.hi)
@printf("\ndifference of medians  %+.2f ms  (%.1f%%)\n",
        sn.med - so.med, 100(sn.med - so.med) / so.med)

# Paired: each round contributes one difference, so drift common to both arms
# cancels. This is the number that decides it.
d = nul .- off
@printf("paired differences     med %+.2f  min %+.2f  max %+.2f  mean %+.2f ms\n",
        stat(d).med, stat(d).lo, stat(d).hi, mean(d))
@printf("\nverdict: %s\n", abs(stat(d).med) < 1.0 ?
        "FREE — the +5.35 ms in one earlier run was drift, not the flag" :
        "NOT free — every opdouble attribution carries this offset, re-read them")
