# Is held-`O` (with `:perelem`) worth turning on for SAM 2's two dominant
# attention shapes?
#
# `kernelplans.jl` calls held "the largest measured win left in this kernel",
# gated on three rescales fitting under 128 registers; this card reports
# `cooperativeMatrixPerElementOperations`, which is the documented way past that.
#
# REFUSES TO MEASURE ON A BUSY CARD. Two earlier attempts disagreed by 90% in
# opposite directions because unrelated jobs took the GPU mid-run; a number from
# a contended card is worse than no number.
#
# CHECKS CORRECTNESS FIRST, and prints it beside every time. `nrsc < 3` rescales
# only the first `nrsc` of the three held `O` tiles — it is a DIAGNOSTIC and it is
# wrong (measured: 1.58% relative error), so it runs faster for the same reason a
# kernel that skips work always does. Reported speed for a variant that does not
# match the reference is not a result. 1.58% is small enough to pass a loose
# tolerance and to leave mask IoU looking fine, which is exactly why the check is
# here and not left to judgement.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":1")
ENV["XAUTHORITY"] = get(ENV, "XAUTHORITY", "/run/user/1000/xauth_OlqIBB")
include(joinpath(@__DIR__, "attn_lab.jl"))
using Printf

# Another COMPUTE job on the card, in MiB — not the desktop. Summing every
# compute app aborts constantly here: the compositor, plasmashell and a browser
# hold ~1 GB between them and swing by hundreds of megabytes on their own, which
# discarded two runs of `flash_cm2_ab.jl` before it produced a single number.
# What matters is a second job doing arithmetic.
function othergpumib(; me = getpid())
    out = read(`nvidia-smi --query-compute-apps=pid,used_memory,process_name
                --format=csv,noheader,nounits`, String)
    total = 0
    for line in split(out, '\n')
        isempty(strip(line)) && continue
        f = strip.(split(line, ','))
        length(f) >= 3 || continue
        pid = tryparse(Int, f[1]); pid === nothing && continue
        pid == me && continue
        occursin(r"julia|python"i, f[3]) || continue
        total += something(tryparse(Int, f[2]), 0)
    end
    total
end

const MINE = othergpumib()
busy() = othergpumib() - MINE > 400

busy() && error("GPU is busy ($(othergpumib()) MiB across other processes) — not measuring")

const DEV = DK.caps(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

function ab(E, Lq, Lk, H, B; n = 9, reps = 10)
    q, k, v, out = operands(E, Lq, Lk, H, B)
    scale = 1 / sqrt(E)
    base = DK.flashcm_plan(DEV, q, k, v, nothing)
    held = DK.flashcm_plan(DEV, q, k, v, nothing; held = true, rescale = :perelem)
    vs = Tuple{String,Function}[]
    push!(vs, ("held=false", () -> (DK.reset!(WS); DK.sdpaflashcm!(CTX, out, base, q, k, v, scale))))
    for nr in (1, 2, 3)
        push!(vs, ("held nrsc=$nr",
            () -> (DK.reset!(WS); DK.sdpaflashcm!(CTX, out, held, q, k, v, scale; nrsc = nr))))
    end
    for _ in 1:6, (_, f) in vs; f(); end          # warm every variant's pipeline
    KA.synchronize(BACKEND)

    # correctness vs the held=false reference, before any timing
    DK.reset!(WS); vs[1][2](); KA.synchronize(BACKEND); ref = Array(out)
    errs = Float64[]
    for (_, f) in vs
        DK.reset!(WS); f(); KA.synchronize(BACKEND)
        push!(errs, maximum(abs.(Array(out) .- ref)) / max(maximum(abs.(ref)), eps()))
    end
    acc = [Float64[] for _ in vs]
    for _ in 1:reps, (i, vf) in enumerate(vs)     # interleaved: drift hits all arms alike
        t0 = time_ns(); for _ in 1:n; vf[2](); end
        KA.synchronize(BACKEND); push!(acc[i], (time_ns() - t0) / 1e6 / n)
    end
    busy() && error("GPU became busy during the run — discarding")
    [(vs[i][1], minimum(acc[i]), sort(acc[i])[end ÷ 2 + 1], errs[i]) for i in eachindex(vs)]
end

warmclock(16)
for (E, Lq, Lk, H, B, calls) in SHAPES
    g = gflop(E, Lq, Lk, H, B)
    @printf("\nE%d L%dx%d H%d B%d  (x%d per encode)\n", E, Lq, Lk, H, B, calls)
    rows = ab(E, Lq, Lk, H, B)
    b = rows[1][2]
    for (nm, mn, md, er) in rows
        ok = er <= 1e-3
        @printf("   %-14s min %7.3f  med %7.3f  %6.2f TF/s  %+6.1f%%   relerr %.2e %s\n",
                nm, mn, md, g / mn, 100 * (mn - b) / b, er,
                ok ? "" : "<-- WRONG, speed meaningless")
    end
    # spread is the honesty check: if min and med diverge wildly the card moved
    @printf("   spread(med/min) baseline %.2fx\n", rows[1][3] / rows[1][2])
end
println("\nDONE")
