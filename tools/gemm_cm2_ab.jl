# Is the coopmat2 GEMM faster than the staged one that ships?
#
# `Mantle.coopmat_gemm!` (staged through `@localmem`, per-subgroup 16x16 tiles)
# against `Mantle.coopmat_gemm_cm2!` (workgroup-scope matrices, tensor-addressed
# loads, no shared memory), on SAM 2's own `addmm` shapes weighted by their share
# of the encoder's arithmetic.
#
# Why now: the same treatment took SAM 2's attention 43.19 -> 28.85 ms, and
# `addmm` is what is left — 57.8 ms of 126.8 serialised, at 35 TFLOP/s against
# the 44 a cuBLAS-class kernel reaches on these shapes.
#
# Why it might still lose, and the reason this file checks rather than assumes:
# the tensor-addressed GEMM has been tried here before and lost (2.3x slower on
# Whisper's shapes; padding beat it). That was at SUBGROUP scope, before
# workgroup-scope matrices existed — a premise that has changed, which is not the
# same as a conclusion that has.
#
# Same three rules as the flash A/B: refuse a busy card, check correctness before
# timing, interleave.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":99")
include(joinpath(@__DIR__, "gemm_lab.jl"))
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
busy() && error("another compute job is on the card — not measuring")

"Interleaved medians for the two GEMMs on one shape, correctness first."
function ab(M, N, K; n = 9, reps = 6)
    hA = rand(Float16, M, K) .- Float16(0.5)
    hB = rand(Float16, K, N) .- Float16(0.5)
    A = KA.allocate(BACKEND, Float16, M, K); copyto!(A, hA)
    B = KA.allocate(BACKEND, Float16, K, N); copyto!(B, hB)
    C1 = KA.allocate(BACKEND, Float16, M, N)
    C2 = KA.allocate(BACKEND, Float16, M, N)

    vs = Pair{String,Function}[]
    push!(vs, "staged" => (() -> Mantle.coopmat_gemm!(C1, A, B, M, N, K)))
    # Sweep the tile: a `64x64` accumulator is 16 components a lane at 256
    # invocations, against the staged kernel's 32, so the first question is
    # whether this kernel is simply too small to reuse anything.
    for t in TILINGS
        push!(vs, @sprintf("%dx%d/%d", t[1], t[2], t[3]) =>
                  (() -> Mantle.coopmat_gemm_cm2!(C2, A, B, M, N, K; tiling = t)))
    end

    # A Float32 CPU reference over the first rows, as `gemm_lab` does: the whole
    # product at 2304x4096 is not worth the host time and the rows are enough to
    # catch an orientation or a k-loop error.
    rows = 1:min(M, 64)
    ref = Float32.(hA[rows, :]) * Float32.(hB)
    errs = Float64[]
    for (i, (_, f)) in enumerate(vs)
        fill!(C1, Float16(0)); fill!(C2, Float16(0))
        f(); KA.synchronize(BACKEND)
        got = Float32.(Array(i == 1 ? C1 : C2)[rows, :])
        push!(errs, maximum(abs.(got .- ref)) / max(1f-6, maximum(abs.(ref))))
    end

    for _ in 1:3, (_, f) in vs; f(); end
    KA.synchronize(BACKEND)
    acc = [Float64[] for _ in vs]
    for _ in 1:reps, (i, vf) in enumerate(vs)
        t0 = time_ns(); for _ in 1:n; vf[2](); end
        KA.synchronize(BACKEND); push!(acc[i], (time_ns() - t0) / 1e9 / n)
    end
    busy() && error("GPU became busy during the run — discarding")
    A = B = C1 = C2 = nothing; GC.gc()
    [(vs[i][1], minimum(acc[i]), errs[i]) for i in eachindex(vs)]
end

# (BM, BN, BK, NT). Every extent must be a multiple of the granularity at `NT`
# invocations: 32 for M and N, 16 for K, on this card at 256.
const TILINGS = [(64, 64, 32, 256), (128, 64, 32, 256), (64, 128, 32, 256),
                 (128, 128, 32, 256), (128, 128, 16, 256)]

# In a function, not at top level: a bare `for` gives `names` and `tot` their own
# locals and the accumulation is lost — the same soft-scope trap that ate the
# flash A/B's result line.
function main()
    warmclock(16)
    names = String[]
    tot = Float64[]
    wsum = sum(s for (_, _, _, s) in SHAPES)
    for (M, N, K, share) in SHAPES
        rows = ab(M, N, K)
        if isempty(names)
            names = [r[1] for r in rows]
            tot = zeros(length(rows))
            @printf("%-20s %6s", "M x N x K", "share")
            for nm in names; @printf(" %11s", nm); end
            println("   TFLOP/s, ERR = wrong")
        end
        tf = [tflops(M, N, K, r[2]) for r in rows]
        tot .+= share .* tf
        @printf("%-20s %5.1f%%", "$(M)x$(N)x$(K)", share)
        for (i, x) in enumerate(tf)
            @printf(" %11s", rows[i][3] > 2e-2 ? "ERR" : @sprintf("%.1f", x))
        end
        println()
        flush(stdout)
    end
    @printf("%-20s %6s", "weighted mean", "")
    for x in tot; @printf(" %11.1f", x / wsum); end
    @printf("   (staged is column 1; cuBLAS %.1f)\n", CUBLAS_TFLOPS)
end

main()
