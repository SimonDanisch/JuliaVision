"""
Convolution cost for the model's real shapes, measured so the number survives a
restart.

Standalone conv timings in this repo were wrong by 5-7x across sessions — the
same five shapes read 216/290/200/271/18 us once and 31/42/34/17/20 us the next
run. The cause is that Lava compiles a cooperative-matrix kernel per tile shape
on first use, and a plain best-of-N does not exclude it: the compile lands in one
round, the minimum comes from another, and whether you caught the warm state is
luck. An entire "1x1 256->256 is 15x slower than 1x1 1024->256" anomaly was
manufactured that way and does not exist.

So this harness does not take a best-of-N. It repeats a shape until the running
minimum stops improving (`converge`), which is the only stopping rule that
distinguishes "warm" from "lucky", and it reports how many rounds that took so a
shape that never settles is visible rather than silently averaged.

The check that matters is at the bottom: the shapes' total, weighted by how often
each occurs in a step, against the in-situ `OPDOUBLE`-through-replay figure of
**6.62 ms**.

**It does not reconstruct it, and that is the finding.** Weighted total 13.87 ms
against an in-situ 6.62 ms — 2.1x high.

Rotating the destination and workspace across 8 slots (below), on the theory that
a self-dependent loop was serialising im2col against the previous GEMM, moved it
to 13.10 ms. **Still 2x, so that hypothesis was wrong.** Worse, per-shape numbers
swing across runs of identical code: `1x1 256->256 @15x8` read 16.0 us and then
116.1 us, with the convergence rule reporting it settled both times. Something
outside the timed loop dominates — clock state, contention, or allocation
placement — and the convergence rule does not detect it.

So: **do not tune convolution against this file's timings.** They are not stable
enough to compare a change against. What the tool reliably produces is its own
falsification — the cross-check against the in-situ figure — and until the total
reconstructs ~6.62 ms, any conclusion drawn here is about the harness.

The only conv number that has survived every check is the in-situ one:
`OPDOUBLE` through `Lava.replay!`, 6.62 ms for 134 convolutions, 2.53 TFLOP/s.
Attribute there; use this only once it agrees.

    julia --project=. tools/conv_bench.jl
"""

using DNNKernels, Lava, KernelAbstractions, Printf
const KA = KernelAbstractions

"Shapes as they occur in the autocast graphs at 240x128 — (W, H, Cin, Cout, K, count)."
const CONVSHAPES = [
    (15,  8,  256,  256, 3, 30),
    (60, 32,   64,   64, 3, 11),
    (30, 16,  128,  128, 3, 11),
    (15,  8,  256,  256, 1,  9),
    (15,  8, 1024,  256, 1,  7),
    (30, 16,  128,  128, 1,  6),
    (60, 32,   64,   64, 1,  6),
]

"""
    converge(f; iters, maxrounds, patience) -> (min_ms, rounds)

Run `f` in blocks of `iters` until the running minimum has not improved for
`patience` consecutive blocks. Returns that minimum and the number of blocks it
took — a shape needing many rounds is still specializing and its number is not
yet trustworthy.
"""
function converge(f, backend; iters = 50, maxrounds = 40, patience = 4, tol = 0.01)
    f(); KA.synchronize(backend)
    best = Inf; stale = 0; rounds = 0
    while rounds < maxrounds
        rounds += 1
        t = time_ns()
        for _ in 1:iters; f(); end
        KA.synchronize(backend)
        ms = (time_ns() - t) / 1e6 / iters
        if ms < best * (1 - tol)
            best = min(best, ms); stale = 0
        else
            best = min(best, ms); stale += 1
            stale >= patience && break
        end
    end
    (best, rounds)
end

function main(; backend = LavaBackend())
    @printf("%-28s %9s %8s %9s %7s %9s\n",
            "shape", "us", "rounds", "TFLOP/s", "count", "ms/step")
    total = 0.0
    for (W, H, Cin, Cout, K, count) in CONVSHAPES
        x = KA.allocate(backend, Float16, W, H, Cin, 1); fill!(x, Float16(0.5))
        w = KA.allocate(backend, Float16, K, K, Cin, Cout); fill!(w, Float16(0.01))
        b = KA.allocate(backend, Float16, Cout); fill!(b, Float16(0))
        # Rotate destination AND workspace. With one of each, iteration k's
        # im2col overwrites the scratch iteration k-1's GEMM is still reading, so
        # the barrier between them serialises three dispatches that the model
        # overlaps — consecutive convolutions there write different slab regions.
        # Measuring one shape in a tight self-dependent loop measures the hazard.
        nbuf = 8
        outs = [KA.allocate(backend, Float16, W, H, Cout, 1) for _ in 1:nbuf]
        wss  = [DNNKernels.Workspace(backend) for _ in 1:nbuf]
        slot = Ref(0)
        function f()
            i = (slot[] = slot[] % nbuf + 1)
            DNNKernels.convolution!(outs[i], x, w, b, (1, 1), (K ÷ 2, K ÷ 2), (1, 1), 1;
                                 ws = wss[i], act = :none)
        end
        ms, rounds = converge(f, backend)
        flops = 2.0 * W * H * Cout * Cin * K * K
        contrib = ms * count
        total += contrib
        @printf("%-28s %9.1f %8d %9.2f %7d %9.2f\n",
                "$(K)x$(K) $(Cin)->$(Cout) @$(W)x$(H)", ms * 1000, rounds,
                flops / (ms * 1e-3) / 1e12, count, contrib)
    end
    @printf("\nweighted total %.2f ms across %d convolutions\n",
            total, sum(c for (_,_,_,_,_,c) in CONVSHAPES))
    println("in-situ reference (OPDOUBLE through replay): 6.62 ms for 134 convolutions")
    println("A standalone total far below the reference means the microbenchmark is")
    println("missing something the model pays — im2col on cold caches, workspace")
    println("growth, or barriers between the three dispatches a convolution records.")
    return total
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
