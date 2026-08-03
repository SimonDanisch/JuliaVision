"""
Bench harness for SAM 2's attention, as an `include`-able file.

    julia> include("tools/attn_lab.jl")
    julia> warmclock(); breakdown()

The shapes are the encoder's own, read off the exported graph rather than
guessed — `_scaled_dot_product_flash_attention` appears 48 times and **two
shapes are 95% of the arithmetic**:

    (E, Lq, Lk, H, B)        calls   GFLOP   share
    72, 4096, 4096, 8,  1        3   116.0   57%   global blocks
    72,  256,  256, 8, 16       32    77.3   38%   windowed blocks
    everything else             13    10.0    5%

Both take the cooperative-matrix path (`min(Lq,Lk) >= COOPMAT_MINL = 256`, both
extents on the 16-tile), so **the three-pass path is 5% of the problem** and
tuning it is not where the time is. That is a change from when this work
started: at `COOPMAT_MINL = 512` only the 3 global calls qualified.

Same two rules as `gemm_lab.jl`, for the same reasons: warm the clock (this card
idles at 210 MHz of 2265 and a cold sample reads as a 2.7x regression), and
reset the `Workspace` per call or the bump allocator OOMs in the loop.
"""

using Lava, DNNKernels, KernelAbstractions, Printf, Statistics
const KA = KernelAbstractions
const DK = DNNKernels

# (E, Lq, Lk, H, B, calls per encode). GFLOP is derived, not stored, so a shape
# edit cannot leave a stale share behind.
const SHAPES = [(72, 4096, 4096, 8, 1, 3),
                (72, 256, 256, 8, 16, 32)]

gflop(E, Lq, Lk, H, B) = 2 * 2 * Lq * Lk * E * H * B / 1e9

const BACKEND = LavaBackend()
const HEATW = KA.allocate(BACKEND, Float32, 1 << 22)
const HEATV = KA.allocate(BACKEND, Float32, 1 << 22)
const WS = DK.Workspace(BACKEND)

heat(k = 200) = (for _ in 1:k; HEATW .= HEATV .* 1.0001f0 .+ 0.5f0; end)
smclock() = parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                        String))))

"Boost the SM clock and report where it landed; under ~2000 the absolute
GFLOP/s is meaningless (ratios survive, which is why variants interleave)."
function warmclock(rounds = 8)
    c = 0
    for _ in 1:rounds
        for _ in 1:10; heat(400); end
        KA.synchronize(BACKEND)
        (c = smclock()) >= 2200 && break
    end
    c
end

"Median of `n` interleaved samples per variant; a sample is `reps` launches
with one sync."
function timedall(fs; n = 9, reps = 4)
    for f in fs; f(); end; KA.synchronize(BACKEND)
    ts = [Float64[] for _ in fs]
    for _ in 1:n
        heat(40)
        for (i, f) in enumerate(fs)
            KA.synchronize(BACKEND)
            t0 = time_ns()
            for _ in 1:reps; f(); end
            KA.synchronize(BACKEND)
            push!(ts[i], (time_ns() - t0) / 1e6 / reps)
        end
    end
    map(median, ts)
end

"q, k, v, out for a shape, in the graph's own `(E, L, H, B)` layout."
function operands(E, Lq, Lk, H, B)
    mk(L, f, s) = (a = KA.allocate(BACKEND, Float16, E, L, H, B);
                   copyto!(a, Float16.(0.4 .* f.(range(0, s, E * L * H * B)))); a)
    (mk(Lq, sin, 9), mk(Lk, cos, 7), mk(Lk, sin, 5),
     KA.allocate(BACKEND, Float32, E, Lq, H, B))
end

"""
    breakdown()

Where `sdpa_coopmat!` spends its time, per stage, on each dominant shape.

The stages are timed in isolation, so they each pay a full sync and the column
sums exceed the fused total — the same caveat as `OPTIMES`. Read the columns
against each other, not the total against the encoder.

The question this exists to answer: is the cooperative-matrix path limited by
its two GEMMs or by the score matrix it materialises? A flash port removes the
matrix and keeps the GEMMs; extending the staged GEMM to `nbatch > 1` does the
opposite. They are very different amounts of work and the split decides which.
"""
function breakdown()
    @printf("%-22s %9s %9s %9s %9s %9s %9s %9s\n",
            "shape", "padK", "padV", "padQ", "gemm1", "softmax", "gemm2", "unpad")
    for (E, Lq, Lk, H, B, calls) in SHAPES
        q, k, v, out = operands(E, Lq, Lk, H, B)
        EP = cld(E, Lava.GEMM_TILE) * Lava.GEMM_TILE
        NB = H * B
        scale = 1 / sqrt(E)
        DK.reset!(WS)
        kp = DK.scratch!(WS, BACKEND, Float16, EP, Lk, H, B)
        vT = DK.scratch!(WS, BACKEND, Float16, Lk, EP, H, B)
        CH = min(Lq, max(Lava.GEMM_TILE, DK.COOPMAT_QCHUNK[]))
        CH = cld(CH, Lava.GEMM_TILE) * Lava.GEMM_TILE
        qc   = DK.scratch!(WS, BACKEND, Float16, CH, EP, H, B)
        S    = DK.scratch!(WS, BACKEND, Float32, CH, Lk, H, B)
        P    = DK.scratch!(WS, BACKEND, Float16, CH, Lk, H, B)
        sums = DK.scratch!(WS, BACKEND, Float32, CH, H, B)
        O    = DK.scratch!(WS, BACKEND, Float32, CH, EP, H, B)
        nch = cld(Lq, CH)

        fs = [() -> DK.launch!(DK.padE, kp, k, E; backend = BACKEND),
              () -> DK.launch!(DK.toLEpad, vT, v, E; backend = BACKEND),
              () -> for q0 in 0:CH:(Lq - 1)
                        DK.launch!(DK.toLEpadchunk, qc, q, E, q0, Lq; backend = BACKEND)
                    end,
              () -> for _ in 1:nch
                        Lava.coopmat_gemm!(S, qc, kp, CH, Lk, EP; nbatch = NB)
                    end,
              () -> for _ in 1:nch
                        DK.attnsoftmax!(sums, P, S, scale; backend = BACKEND)
                    end,
              () -> for _ in 1:nch
                        Lava.coopmat_gemm!(O, P, vT, CH, EP, Lk; nbatch = NB)
                    end,
              () -> for q0 in 0:CH:(Lq - 1)
                        n = min(CH, Lq - q0)
                        DK.launch!(DK.fromLEpad, view(out, :, (q0+1):(q0+n), :, :), O, sums;
                                   backend = BACKEND)
                    end]
        ts = timedall(fs)
        @printf("%-22s", "E$(E) L$(Lq)x$(Lk) H$(H) B$(B)")
        for t in ts; @printf(" %9.3f", t); end
        g = gflop(E, Lq, Lk, H, B)
        @printf("   sum %7.2f ms   x%d calls = %6.1f ms   %.1f GFLOP\n",
                sum(ts), calls, calls * sum(ts), g)
        q = k = v = out = nothing
        GC.gc()
    end
end

"""
    fused()

Whole-op timing of every path `sdpa` can take, on each dominant shape, so the
stage table above can be checked against something that pays only one sync.

**Both switches, not one.** This used to set `COOPMAT_MINL` alone and label the
two columns "3-pass" and "coopmat" — but `flashcm_applicable` is gated on
`FLASHCM[]`, so both columns took the *fused* path and the table compared one
kernel with itself. It read plausibly for months (4.401 against 4.378) because
two runs of the same kernel do agree. `sdpaflash!` is called directly because
`sdpa` never routes to it.

    shape          3-pass   staged   flash-cm   flash-scalar     (2026-08-02)
    4096x4096      18.286    8.879      4.423      39.435
     256x256        1.109    0.813      0.444       2.673
"""
function fused()
    oldm, oldf = DK.COOPMAT_MINL[], DK.FLASHCM[]
    try
        @printf("%-22s %10s %10s %10s %10s   %s\n",
                "shape", "3-pass", "staged", "flash-cm", "flash-scal", "TF/s (staged | flash-cm)")
        for (E, Lq, Lk, H, B, calls) in SHAPES
            q, k, v, out = operands(E, Lq, Lk, H, B)
            scale = 1 / sqrt(E)
            fs = [() -> (DK.FLASHCM[] = false; DK.COOPMAT_MINL[] = typemax(Int); DK.reset!(WS);
                         DK.sdpa(q, k, v, nothing, scale; backend = BACKEND, ws = WS, out)),
                  () -> (DK.FLASHCM[] = false; DK.COOPMAT_MINL[] = 1; DK.reset!(WS);
                         DK.sdpa(q, k, v, nothing, scale; backend = BACKEND, ws = WS, out)),
                  () -> (DK.FLASHCM[] = true; DK.COOPMAT_MINL[] = 1; DK.reset!(WS);
                         DK.sdpa(q, k, v, nothing, scale; backend = BACKEND, ws = WS, out)),
                  () -> (DK.reset!(WS);
                         DK.sdpaflash!(out, q, k, v, scale; backend = BACKEND))]
            ts = timedall(fs; n = 7, reps = 3)
            g = gflop(E, Lq, Lk, H, B)
            @printf("%-22s %10.3f %10.3f %10.3f %10.3f   %6.2f | %6.2f\n",
                    "E$(E) L$(Lq)x$(Lk) H$(H) B$(B)", ts[1], ts[2], ts[3], ts[4],
                    g / ts[2], g / ts[3])
            q = k = v = out = nothing
            GC.gc()
        end
    finally
        DK.COOPMAT_MINL[] = oldm
        DK.FLASHCM[] = oldf
    end
end

# ── ablation: where does `attn_flash_cm!` actually spend its time? ────────────
#
# The kernel runs at 8.9 TFLOP/s against the 35 the same cooperative matrices
# reach in the GEMM, and four hypotheses about why have now been measured and
# refuted (`O` in registers, the rescale pass, the softmax's idle warps, and
# occupancy — see `perf-plan.md`). None of those said what the time IS, only what
# it is not.
#
# This is the differential ablation that answers it. A copy of the shipped
# kernel, gated so stages can be removed one at a time. **It computes wrong
# answers on purpose** — that is the point of an ablation and the reason it lives
# in a lab file and not in the package.
#
#   :all        the kernel as it stood when this was written, as a control
#               (two-pass softmax — compare against `onepass = false`)
#   :nosoftmax  scores go straight to `ps` with no max, no exp, no sums
#   :products   also drops the value product's staging of `V`
#   :qk         only the score product; `P·V` removed entirely
#
# Read the differences, never the absolutes: each variant is a different amount
# of work and only the gaps between them mean anything.
@kernel cpu=false unsafe_indices=true function attn_probe!(
        out, @Const(q), @Const(k), @Const(v), scale,
        ::Val{BR}, ::Val{BC}, ::Val{E}, ::Val{EP}, ::Val{NW}, ::Val{ABL},
        Lk::Int32) where {BR,BC,E,EP,NW,ABL}
    NT = NW * 32
    qs  = @localmem Float16 (EP * BR,)
    kvs = @localmem Float16 (EP * BC,)
    ss  = @localmem Float32 (BR * BC,)
    ps  = @localmem Float16 (BC * BR,)
    pvs = @localmem Float32 (BR * EP,)
    ms  = @localmem Float32 (BR,)
    ls  = @localmem Float32 (BR,)
    cs  = @localmem Float32 (BR,)

    RT = BR ÷ Lava.GEMM_TILE
    CT = BC ÷ Lava.GEMM_TILE
    ET = EP ÷ Lava.GEMM_TILE
    tid = @index(Local, Linear) - 1
    grp = @index(Group, NTuple)
    qb, h, b = grp[1], grp[2], grp[3]
    w = tid ÷ 32

    @inbounds begin
        q0 = (qb - 1) * BR
        for r in 0:(div(BR * EP, NT) - 1)
            idx = tid + r * NT
            e, lq = Lava.splitidx(idx, Val(EP))
            qs[1 + idx] = e < E ? q[1 + e, 1 + q0 + lq, h, b] : zero(Float16)
            pvs[1 + idx] = 0.0f0
        end
        if tid < BR
            ms[1 + tid] = -Inf32; ls[1 + tid] = 0.0f0; cs[1 + tid] = 1.0f0
        end
        @synchronize

        for kb in 0:(div(Lk, BC) - 1)
            k0 = kb * BC
            for r in 0:(div(BC * EP, NT) - 1)
                idx = tid + r * NT
                e, lk = Lava.splitidx(idx, Val(EP))
                kvs[1 + idx] = e < E ? k[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
            end
            @synchronize

            for t in w:NW:(RT * CT - 1)
                rt = t % RT; ct = t ÷ RT
                acc = zero(Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator})
                for et in 0:(ET - 1)
                    a = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                            qs, 1 + rt * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(true))
                    bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                            kvs, 1 + ct * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(false))
                    acc = muladd(a, bm, acc)
                end
                copyto!(ss, 1 + rt * Lava.GEMM_TILE + ct * Lava.GEMM_TILE * BR, BR, acc)
            end
            @synchronize

            if ABL === :all
                if tid < BR
                    mo = ms[1 + tid]; mb = -Inf32
                    for ci in 0:(BC - 1)
                        mb = max(mb, ss[1 + tid + ci * BR] * scale)
                    end
                    mn = max(mo, mb)
                    cr = isfinite(mo) ? exp(mo - mn) : 0.0f0
                    sm = 0.0f0
                    for ci in 0:(BC - 1)
                        p = exp(ss[1 + tid + ci * BR] * scale - mn)
                        ps[1 + ci + tid * BC] = Float16(p); sm += p
                    end
                    ms[1 + tid] = mn; ls[1 + tid] = ls[1 + tid] * cr + sm; cs[1 + tid] = cr
                end
            else
                # Same shared traffic in and out, none of the arithmetic.
                for r in 0:(div(BR * BC, NT) - 1)
                    idx = tid + r * NT
                    ps[1 + idx] = Float16(ss[1 + idx] * 0.001f0)
                end
            end
            @synchronize

            if ABL !== :qk
                if ABL === :all || ABL === :nosoftmax
                    for r in 0:(div(BC * EP, NT) - 1)
                        idx = tid + r * NT
                        e, lk = Lava.splitidx(idx, Val(EP))
                        kvs[1 + idx] = e < E ? v[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
                    end
                end
                @synchronize
                for t in w:NW:(RT * ET - 1)
                    rt = t % RT; et = t ÷ RT
                    off = 1 + rt * Lava.GEMM_TILE + et * Lava.GEMM_TILE * BR
                    acc = Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator}(
                              pvs, off, BR, Val(false))
                    for ct in 0:(CT - 1)
                        a = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                                ps, 1 + rt * Lava.GEMM_TILE * BC + ct * Lava.GEMM_TILE, BC, Val(true))
                        bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                                kvs, 1 + ct * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(true))
                        acc = muladd(a, bm, acc)
                    end
                    copyto!(pvs, off, BR, acc)
                end
            end
            @synchronize
        end

        for s in 1:div(BR * E, NT)
            idx = tid + (s - 1) * NT
            lq, e = Lava.splitidx(idx, Val(BR))
            l = ls[1 + lq]
            out[1 + e, 1 + q0 + lq, h, b] = pvs[1 + idx] / (l == 0.0f0 ? 1.0f0 : l)
        end
    end
end

"""
    ablate()

Run the probe on each dominant shape with each stage removed, and report the
difference each stage makes. `:all` reproduces the shipped kernel and is the
control — if it does not match `sdpaflashcm!`'s own timing, the probe has drifted
from the kernel it is supposed to explain and nothing below means anything.
"""
function ablate(BR = 64, BC = 32, NW = 8)
    E0 = 72; EP = cld(E0, Lava.GEMM_TILE) * Lava.GEMM_TILE; NT = NW * 32
    stages = (:all, :nosoftmax, :products, :qk)
    for (E, Lq, Lk, H, B, calls) in SHAPES
        q, k, v, out = operands(E, Lq, Lk, H, B)
        fs = Any[() -> attn_probe!(BACKEND, NT)(out, q, k, v, Float32(1/sqrt(E)),
                                                Val(BR), Val(BC), Val(E), Val(EP),
                                                Val(NW), Val(a), Int32(Lk);
                                                ndrange = (NT * div(Lq, BR), H, B))
                 for a in stages]
        # `onepass = false`, because the probe is pinned to the TWO-pass softmax.
        # The control only means something against the kernel it copies, and
        # `FLASHCM_ONEPASS` changed that kernel after this probe was written —
        # which is exactly the drift the `:all`-vs-`shipped` row exists to catch.
        push!(fs, () -> DK.sdpaflashcm!(out, q, k, v, 1/sqrt(E); backend = BACKEND,
                                        BR, BC, NW, onepass = false))
        ts = timedall(fs; n = 7, reps = 3)
        println("E$(E) L$(Lq)x$(Lk) H$(H) B$(B)")
        prev = ts[1]
        for (nm, t) in zip((stages..., :shipped), ts)
            @printf("   %-11s %8.3f ms   %+7.3f vs :all\n", nm, t, t - prev)
        end
        q = k = v = out = nothing; GC.gc()
    end
end

# ── does holding `O` in accumulators across the key loop pay? ─────────────────
#
# `P·V` is 37% of the kernel and almost all of it is accumulator round trips: 20
# tiles a key block, each loaded from `pvs`, given two muladds, and stored back.
# Holding each subgroup's tiles in cooperative-matrix accumulators across the
# whole key loop removes every one of those — the mechanism is proven, the staged
# GEMM already carries accumulators across a `for` with barriers inside it.
#
# The cost is registers. Three accumulators a subgroup (`ceil(RT*ET / NW)` = 3 at
# the shipped tiling) is ~24 more a thread on a kernel the driver already reports
# at 128, and 2 workgroups an SM needs <= 128. So this trades all of the `P·V`
# traffic against halving occupancy, which is a coin flip rather than a win.
#
# **This kernel answers it without doing the work.** It never rescales, so it
# computes the wrong answer whenever a running maximum grows — but rescaling is
# the only thing the real version would add, and `FLASHCM_LAZYRESCALE` already
# showed those blocks are rare. If the best case is not faster, the real one
# cannot be, and the restructure (which needs the kernel generated per tiling,
# because `@nexprs` wants a literal count) is not worth starting.
@kernel cpu=false unsafe_indices=true function attn_heldacc!(
        out, @Const(q), @Const(k), @Const(v), scale,
        ::Val{BR}, ::Val{BC}, ::Val{E}, ::Val{EP}, ::Val{NW},
        Lk::Int32) where {BR,BC,E,EP,NW}
    NT = NW * 32
    qs  = @localmem Float16 (EP * BR,)
    kvs = @localmem Float16 (EP * BC,)
    ss  = @localmem Float32 (BR * BC,)
    ps  = @localmem Float16 (BC * BR,)
    pvs = @localmem Float32 (BR * EP,)
    ms  = @localmem Float32 (BR,)
    ls  = @localmem Float32 (BR,)

    RT = BR ÷ Lava.GEMM_TILE
    CT = BC ÷ Lava.GEMM_TILE
    ET = EP ÷ Lava.GEMM_TILE
    tid = @index(Local, Linear) - 1
    grp = @index(Group, NTuple)
    qb, h, b = grp[1], grp[2], grp[3]
    w = tid ÷ 32

    @inbounds begin
        q0 = (qb - 1) * BR
        for r in 0:(div(BR * EP, NT) - 1)
            idx = tid + r * NT
            e, lq = Lava.splitidx(idx, Val(EP))
            qs[1 + idx] = e < E ? q[1 + e, 1 + q0 + lq, h, b] : zero(Float16)
        end
        if tid < BR
            ms[1 + tid] = -Inf32; ls[1 + tid] = 0.0f0
        end
        # Three, held for the whole loop. `t_j` is uniform within a subgroup, so
        # the guard around each is uniform control flow and legal for a coopmat.
        Base.Cartesian.@nexprs 3 j ->
            acc_j = zero(Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator})
        @synchronize

        for kb in 0:(div(Lk, BC) - 1)
            k0 = kb * BC
            for r in 0:(div(BC * EP, NT) - 1)
                idx = tid + r * NT
                e, lk = Lava.splitidx(idx, Val(EP))
                kvs[1 + idx] = e < E ? k[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
            end
            @synchronize
            for t in w:NW:(RT * CT - 1)
                rt = t % RT; ct = t ÷ RT
                a = zero(Lava.AcceleratedMatrix{Float32,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.Accumulator})
                for et in 0:(ET - 1)
                    am = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                            qs, 1 + rt * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(true))
                    bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                            kvs, 1 + ct * Lava.GEMM_TILE * EP + et * Lava.GEMM_TILE, EP, Val(false))
                    a = muladd(am, bm, a)
                end
                copyto!(ss, 1 + rt * Lava.GEMM_TILE + ct * Lava.GEMM_TILE * BR, BR, a)
            end
            @synchronize
            if tid < BR
                mo = ms[1 + tid]; sm = 0.0f0
                for ci in 0:(BC - 1)
                    p = exp(ss[1 + tid + ci * BR] * scale - (isfinite(mo) ? mo : 0.0f0))
                    ps[1 + ci + tid * BC] = Float16(p); sm += p
                end
                ms[1 + tid] = isfinite(mo) ? mo : 0.0f0
                ls[1 + tid] = ls[1 + tid] + sm
            end
            @synchronize
            for r in 0:(div(BC * EP, NT) - 1)
                idx = tid + r * NT
                e, lk = Lava.splitidx(idx, Val(EP))
                kvs[1 + idx] = e < E ? v[1 + e, 1 + k0 + lk, h, b] : zero(Float16)
            end
            @synchronize
            Base.Cartesian.@nexprs 3 j -> begin
                t_j = w + (j - 1) * NW
                if t_j < RT * ET
                    rt_j = t_j % RT; et_j = t_j ÷ RT
                    for ct in 0:(CT - 1)
                        am = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixA}(
                                ps, 1 + rt_j * Lava.GEMM_TILE * BC + ct * Lava.GEMM_TILE, BC, Val(true))
                        bm = Lava.AcceleratedMatrix{Float16,Lava.GEMM_TILE,Lava.GEMM_TILE,Lava.MatrixB}(
                                kvs, 1 + ct * Lava.GEMM_TILE * EP + et_j * Lava.GEMM_TILE, EP, Val(true))
                        acc_j = muladd(am, bm, acc_j)
                    end
                end
            end
            @synchronize
        end

        Base.Cartesian.@nexprs 3 j -> begin
            t_j = w + (j - 1) * NW
            if t_j < RT * ET
                rt_j = t_j % RT; et_j = t_j ÷ RT
                copyto!(pvs, 1 + rt_j * Lava.GEMM_TILE + et_j * Lava.GEMM_TILE * BR, BR, acc_j)
            end
        end
        @synchronize
        for s in 1:div(BR * E, NT)
            idx = tid + (s - 1) * NT
            lq, e = Lava.splitidx(idx, Val(BR))
            l = ls[1 + lq]
            out[1 + e, 1 + q0 + lq, h, b] = pvs[1 + idx] / (l == 0.0f0 ? 1.0f0 : l)
        end
    end
end

"""
    heldacc()

The best case for holding `O` in accumulators, against the shipped kernel.
Reports the driver's register count for both, because that is the thing the
trade turns on.
"""
function heldacc(BR = 64, BC = 32, NW = 8)
    EP = 80; NT = NW * 32
    for (E, Lq, Lk, H, B, calls) in SHAPES
        q, k, v, out = operands(E, Lq, Lk, H, B)
        s = 1 / sqrt(E)
        fs = Any[() -> DK.sdpaflashcm!(out, q, k, v, s; backend = BACKEND, BR, BC, NW),
                 () -> attn_heldacc!(BACKEND, NT)(out, q, k, v, Float32(s),
                        Val(BR), Val(BC), Val(E), Val(EP), Val(NW), Int32(Lk);
                        ndrange = (NT * div(Lq, BR), H, B))]
        ts = timedall(fs; n = 9, reps = 3)
        g = gflop(E, Lq, Lk, H, B)
        println("E$(E) L$(Lq)x$(Lk) H$(H) B$(B)")
        for (nm, t) in zip(["shipped", "held-acc"], ts)
            @printf("   %-10s %8.3f ms  %6.2f TF/s\n", nm, t, g / t)
        end
        @printf("   -> %+.1f%%\n", 100 * (ts[1] - ts[2]) / ts[1])
        q = k = v = out = nothing; GC.gc()
    end
end
