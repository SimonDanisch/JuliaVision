# Does a given formulation of the cm2 flash kernel pay for itself?
#
# Each entry of `VARIANTS` is a set of keyword arguments to `sdpaflashcm2!`, and
# every one is run at every legal tiling in `CM2`, paired against the BASELINE
# variant AT THE SAME TILING — so a row reports the value of the formulation and
# not of the tile. Adding a formulation is a line here, not another tool.
#
# The ablation in `flash_cm2_ablate.jl` says the softmax — not the two matrix
# products — is 70% of this kernel: the products alone run at 33.9 TF/s, which is
# our GEMM's rate, while the whole kernel does 10.3. Per key block the three-pass
# formulation issues EIGHT per-element passes over `Br x Bc` matrices plus three
# reductions, for about 1/80th of the arithmetic.
#
# `Lava.coopmat_perelement` with a second matrix operand collapses them:
#
#     scale then mask          2 passes -> 1   (cm2scalemask)
#     max(rowmax, Mold)        3 passes -> 1   (cm2max2:    neg, nrelu, add)
#     exp(Mold - M)            1 pass   -> 1   (cm2expdiff, was exp(-relu(d)))
#     exp(S - M) then mask0    3 passes -> 1   (cm2expdiff: neg, add, exp, mask0)
#
# Eight passes become four, and it LOSES by about 31% — which is what sent the
# ablation looking again and found the real cost: the two ROW REDUCTIONS are
# 38.1% of the kernel against those six per-element passes' 13.5%.
#
# `osum` is the answer to that. `EP` is `E` rounded up to the operand
# granularity, so at SAM 2's `E = 72` the accumulator has eight columns the
# clamping store never writes. A column of ones in V's padding makes one of them
# accumulate the row sum, already rescaled — deleting a reduction per key block
# and the `Br x Bc` running total with it.
#
# Every formulation lives in `attn_flash_cm2!` behind a `Val`, so these are
# binary differences in ONE kernel and not kernels that drifted apart.
#
# Same three rules as `flash_cm2_ab.jl`: refuse a busy card, check correctness
# before believing a time, interleave the arms.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
include(joinpath(@__DIR__, "attn_lab.jl"))
using Printf

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

# The shipped tiling, plus the two the third sweep put next to it. Fewer passes
# frees registers, and the register ceiling is what made every bigger tile lose —
# so the tiling that wins fused is not necessarily the one that wins unfused, and
# a single-tiling A/B could not see that.
# `BC = 80` is not a rounder number that got left out — it is `EP` at `E = 72,
# NT = 128`, and at `BC == EP` the `Br x EP` smear becomes the identity, so the
# THIRD reduction per key block disappears with it. Legal because 80 is a
# multiple of the N and K granularity (16), which a sweep over powers of two
# cannot reach.
const CM2 = [(64, 32, 128), (64, 16, 128), (128, 64, 128), (64, 64, 128),
             (64, 80, 128), (32, 80, 128), (128, 80, 128)]

# `VARIANTS` is in `attn_lab.jl` — `flash_cm2_regs.jl` reads the driver's
# register count for the same list, and the two only line up if there is one.
# `osum` removes a reduction PER KEY BLOCK, so the tiling that wins with it is
# not necessarily the one that won without: that is why this sweeps tiles.
const BASELINE = first(VARIANTS)[1]

# The encoder's two dominant shapes, plus the decoder's cross-attention whose
# `Lq = 23` divides no tile — the masked path is exercised hardest there, and the
# fused form DROPS the explicit `mask0` pass, so it is the correctness case that
# matters most.
const SWEEP = [(72, 4096, 4096, 8, 1, 3),
               (72, 256, 256, 8, 16, 32),
               (72, 23, 4096, 8, 1, 1)]

function ab(E, Lq, Lk, H, B; n = 9, reps = 10)
    q, k, v, out = operands(E, Lq, Lk, H, B)
    scale = 1 / sqrt(E)
    vs = Tuple{String,Function}[]
    for (BR, BC, NT) in CM2
        DK.flashcm2fits(DEV, E, BR, BC, NT) || continue
        plan = DK.flashcm2_plan(DEV, q, k, v, nothing; BR, BC, NT)
        plan isa DK.FlashCM2Plan || continue
        for (nm, kw) in VARIANTS
            push!(vs, (@sprintf("%dx%d/%d %s", BR, BC, NT, nm),
                       () -> (DK.reset!(WS);
                              DK.sdpaflashcm2!(CTX, out, plan, q, k, v, scale; kw...))))
        end
    end
    isempty(vs) && return Tuple{String,Float64,Float64,Float64}[]
    for _ in 1:4, (_, f) in vs; f(); end
    KA.synchronize(BACKEND)

    # Reference is the generic three-pass SDPA — the one implementation here that
    # shares no code with either formulation of the kernel.
    DK.reset!(WS)
    ref3 = similar(out)
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
    for _ in 1:reps, (i, vf) in enumerate(vs)
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
    flush(stdout)
    rows = ab(E, Lq, Lk, H, B)
    isempty(rows) && (println("   no legal tiling"); continue)
    # Each row is compared with ITS OWN baseline twin at the same tiling, so the
    # number is the value of the formulation and not of the tile.
    tileof(nm) = first(split(nm, ' '))
    base = Dict(tileof(nm) => mn for (nm, mn, _, _) in rows if endswith(nm, BASELINE))
    for (nm, mn, md, er) in rows
        tile = tileof(nm)
        b = get(base, tile, mn)
        ok = er <= 5e-3
        @printf("   %-18s min %7.3f  med %7.3f  %6.2f TF/s  %+6.1f%%   relerr %.2e %s\n",
                nm, mn, md, g / mn, 100 * (mn - b) / b, er,
                ok ? "" : "<-- WRONG, speed meaningless")
    end
    flush(stdout)
end
println("\nDONE")
