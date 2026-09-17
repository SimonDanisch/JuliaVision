# Is the coopmat2 flash kernel faster than the one that ships?
#
# `attn_flash_cm!` (subgroup scope, everything staged through `@localmem`) against
# `attn_flash_cm2!` (workgroup scope, tensor loads, no shared memory), on SAM 2's
# two dominant attention shapes.
#
# Same three rules as `flash_held_ab.jl`, for the same reasons:
#
#   * REFUSES TO MEASURE ON A BUSY CARD — two earlier A/Bs disagreed by 90% in
#     opposite directions because another job took the GPU mid-run.
#   * CHECKS CORRECTNESS FIRST and prints the error beside every time. A variant
#     that does not match is not slower or faster, it is wrong.
#   * INTERLEAVED, so clock drift hits every arm alike, and the median/min spread
#     is printed as the honesty check.
#
# The cm2 arms sweep the workgroup size because it decides the granularity, and
# the granularity decides the padding: at 256 invocations `EP` must be a multiple
# of 32, so `E = 72` pads to 96 and 25% of both products is padding. At 128 it is
# a multiple of 16, so `EP = 80` — less waste per lane, more registers per lane.
# Which wins is not derivable, which is why both run.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
include(joinpath(@__DIR__, "attn_lab.jl"))
using Printf

# Another COMPUTE job on the card, in MiB — not the desktop.
#
# `flash_held_ab.jl` sums every compute app and aborts on a 400 MiB swing, and on
# this machine that fires constantly: the compositor, plasmashell and a browser
# hold ~1 GB between them and move by hundreds of megabytes on their own. Two
# runs of this A/B were discarded by that before producing a single number, which
# is a guard measuring the wrong thing rather than a busy card. What matters is a
# second job doing arithmetic, and those are `julia` and `python` here.
function otherjobmib(; me = getpid())
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

const MINE = otherjobmib()
busy() = otherjobmib() - MINE > 400

busy() && error("another compute job is on the card ($(otherjobmib()) MiB) — not measuring")

const DEV = DK.caps(BACKEND)
const CTX = DK.Ctx(BACKEND; ws = WS)

# (BR, BC, NT) for the cm2 kernel. `flashcm2fits` refuses the illegal ones, so a
# tiling that cannot run is skipped with a line rather than failing the run.
#
# 2026-08-11, first sweep, against cm1 at 4.436 / 0.416 ms:
#
#     tiling            4096x4096          256x256
#     32x64/256 EP96      +49.2%            +16.6%
#     32x32/256 EP96     +109.1%            +57.3%
#     64x64/256 EP96      +38.3%             +3.2%
#     32x64/128 EP80       +9.5%            -24.5%
#     32x32/128 EP80       +7.1%            -15.8%
#     64x64/128 EP80      -13.1%            -45.2%   <- best of the six
#
# `NT = 128` beat `NT = 256` at every tiling, by a lot. The granularity table is
# why: 256 invocations force `N` to a multiple of 32, so `E = 72` pads to 96 and
# a quarter of BOTH products is padding, where 128 allows `EP = 80` and 10%. The
# per-lane register saving that made 256 look attractive does not pay for that.
#
# Second sweep, same baselines:
#
#     tiling             4096x4096          256x256
#     64x64/128           -12.6%            -45.1%
#     64x128/128          +26.6%            -22.2%
#     128x64/128          -16.4%            -38.7%
#     64x32/128           -27.4%            -37.4%   <- best long
#     96x64/128            +8.6%            -24.0%
#     64x64/64            +61.6%             +5.7%
#     128x128/128         +85.2%            +17.4%
#
# No single tiling wins both, and the gap is worth about a millisecond an encode,
# so the chooser needs a rule. THIRD sweep: the three survivors over four shapes,
# to find where they cross rather than interpolating between two workloads — the
# trap `gemm-tiling-aliasing-cliff` records, where a weighted mean over shapes
# ranked the tilings backwards.
# `BC = 16` is the untried point, and it is aimed at the register ceiling rather
# than at the tile: `pipeline_exec_stats` says the shipped `64x32/128` sits at
# **255 registers** — the driver's cap — against cm1's 115, so it runs two
# 128-thread workgroups per SM where cm1 runs two of 256. Six of the eight live
# transients are `Br x Bc` (S, P, rowmax, rowsum, d, eM), so halving `Bc` halves
# each of them; `O` (Br x EP) and `eMdiag` do not move. Legal at `NT = 128`,
# where the N and K granularities are both 16.
const CM2 = [(64, 32, 128), (64, 16, 128), (32, 16, 128), (128, 16, 128)]

# (E, Lq, Lk, H, B, calls per encode). The last two are not in `attn_lab`'s
# `SHAPES`: 1024 to place the crossover, and the DECODER's cross-attention, whose
# `Lq = 23` divides no tile at all — cm1 needs `clamp` for it and takes the
# three-pass path in production, while this kernel's clamping loads and masked
# softmax handle it with no switch.
const SWEEP = [(72, 4096, 4096, 8, 1, 3),
               (72, 256, 256, 8, 16, 32)]

function ab(E, Lq, Lk, H, B; n = 9, reps = 10)
    q, k, v, out = operands(E, Lq, Lk, H, B)
    scale = 1 / sqrt(E)
    vs = Tuple{String,Function}[]
    # The three-pass path is always available and is the reference below, so it
    # is also the honest floor to compare against on a shape cm1 declines.
    push!(vs, ("3-pass",
               () -> (DK.reset!(WS);
                      DK.sdpa!(CTX, DK.Decline(:threepass), out, q, k, v, nothing, scale))))
    # `clamp` so cm1 is offered the decoder's `Lq = 23` too, rather than the row
    # simply being absent.
    base = DK.flashcm_plan(DEV, q, k, v, nothing)
    if base isa DK.FlashCMPlan
        push!(vs, (@sprintf("cm1 %dx%d/%d", base.BR, base.BC, base.NW),
                   () -> (DK.reset!(WS); DK.sdpaflashcm!(CTX, out, base, q, k, v, scale))))
    else
        @printf("   cm1 declines (%s)\n", base.reason)
    end
    for (BR, BC, NT) in CM2
        DK.flashcm2fits(DEV, E, BR, BC, NT) || continue
        push!(vs, (@sprintf("cm2 %dx%d/%d EP%d", BR, BC, NT, DK.cm2pad(DEV, E, NT)),
                   () -> (DK.reset!(WS);
                          DK.sdpaflashcm2!(CTX, out, q, k, v, scale; BR, BC, NT))))
    end
    for _ in 1:4, (_, f) in vs; f(); end          # warm every variant's pipeline
    KA.synchronize(BACKEND)

    # Correctness before timing, against the three-pass path — which is the one
    # implementation here that nothing in this comparison shares code with.
    DK.reset!(WS)
    ref3 = similar(out)
    # `Decline` selects the three-pass method — the one implementation in this
    # comparison that shares no code with either flash kernel.
    DK.sdpa!(CTX, DK.Decline(:threepass), ref3, q, k, v, nothing, scale)
    KA.synchronize(BACKEND)
    ref = Array(ref3)
    scaleref = max(maximum(abs.(ref)), eps())
    errs = Float64[]
    for (_, f) in vs
        fill!(out, Float32(NaN)); DK.reset!(WS); f(); KA.synchronize(BACKEND)
        push!(errs, maximum(abs.(Array(out) .- ref)) / scaleref)
    end

    acc = [Float64[] for _ in vs]
    for _ in 1:reps, (i, vf) in enumerate(vs)     # interleaved
        t0 = time_ns(); for _ in 1:n; vf[2](); end
        KA.synchronize(BACKEND); push!(acc[i], (time_ns() - t0) / 1e6 / n)
    end
    busy() && error("GPU became busy during the run — discarding")
    [(vs[i][1], minimum(acc[i]), sort(acc[i])[end ÷ 2 + 1], errs[i]) for i in eachindex(vs)]
end

warmclock(16)
for (E, Lq, Lk, H, B, calls) in SWEEP
    g = gflop(E, Lq, Lk, H, B)
    @printf("\nE%d L%dx%d H%d B%d  (x%d per encode, %.1f GFLOP)\n", E, Lq, Lk, H, B, calls, g)
    # Flushed line by line, because a run redirected to a file block-buffers and
    # a kernel that takes the device down loses everything still in the buffer —
    # which reports an empty log and exit code 0.
    flush(stdout)
    rows = ab(E, Lq, Lk, H, B)
    # Percentages are against what SHIPS today, which is cm1 wherever it applies
    # and the three-pass path where it declines — not against whichever row
    # happens to be first.
    ib = something(findfirst(r -> startswith(r[1], "cm1"), rows), 1)
    b = rows[ib][2]
    for (nm, mn, md, er) in rows
        ok = er <= 5e-3
        @printf("   %-20s min %7.3f  med %7.3f  %6.2f TF/s  %+6.1f%%   relerr %.2e %s\n",
                nm, mn, md, g / mn, 100 * (mn - b) / b, er,
                ok ? "" : "<-- WRONG, speed meaningless")
    end
    @printf("   spread(med/min) baseline %.2fx\n", rows[ib][3] / rows[ib][2])
    flush(stdout)
end
println("\nDONE")
