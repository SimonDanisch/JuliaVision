"""
What this tree gets on one operation against what the vendor stack gets, per
shape, as a table you can re-run.

Every performance note in `plans/` before 2026-09-21 was self-relative — this
tree against itself — and two of the conclusions drawn that way did not survive
a vendor comparison (`plans/2026-09-21-rocm-baseline.md`). This exists so the
gap is a tracked number rather than an occasional discovery.

    julia --project=. tools/gap_vs_rocm.jl            # writes tmp/baseline/ours.json
    tmp/baseline/preload.sh tools/gap_vs_rocm.py      # times the same shapes, prints the table

Two processes because they need different runtimes, and one JSON between them
so the SHAPES are stated once, here. `preload.sh` is why the torch side runs at
all on this box; `plans/2026-09-21-rocm-baseline.md` has that story.

Shapes are the ones the models in this tree actually execute, named after the
op that runs them, so a row that regresses says where to look.
"""

using DNNKernels, Printf, Statistics, Random
const DK = DNNKernels
# Through the package, so this runs from any environment DNNKernels loads in.
const JSON3 = DNNKernels.JSON3
const KA = DK.KernelAbstractions
const M = DK.Mantle

"""Minimum of `n` timed runs, after `warm` untimed ones. Minimum, not mean: a
GPU that shares a machine has a floor and no ceiling."""
function best(backend, f; n = 15, warm = 5)
    for _ in 1:warm; f(); end
    KA.synchronize(backend)
    t = Inf
    for _ in 1:n
        KA.synchronize(backend)
        t0 = time_ns(); f(); KA.synchronize(backend)
        t = min(t, (time_ns() - t0) / 1e6)
    end
    t
end

f16(backend, dims...; scale = 0.3f0) =
    (a = KA.allocate(backend, Float16, dims...);
     copyto!(a, Float16.(randn(Float32, dims...) .* scale)); a)

"""
`(E, Lq, Lk, H, B, calls)` attentions, named for where they run, with how many
times one encode runs each.

Read off the exported graphs, not chosen. `sam2-encoder-global` used to be here
as `E = 64, B = 1` and was BOTH wrong: SAM 2's head dim is 72, and that shape is
3 of the encoder's 48 attention calls. Optimising against it moved the
microbenchmark 23% and the model not at all. The windowed shape below is 32 of
the 48 and is where the time is.
"""
const ATTN = [("qwen-denoiser", 128, 4096, 4118, 32, 1, 60),
              ("qwen-denoiser-square", 128, 4096, 4096, 32, 1, 60),
              ("sam2-windowed-256", 72, 256, 256, 8, 16, 32),
              ("sam2-encoder-global", 72, 4096, 4096, 8, 1, 3),
              ("sam2-windowed-16", 72, 16, 16, 4, 1024, 6)]

"""`(Cin, Cout, H, W)` 3x3 stride-1 convolutions, all of them from the
Qwen-Image 2.1 VAE decoder, which is 82% convolution."""
const CONV = [("vae-288-1024", 288, 288, 1024, 1024),
              ("vae-144-1024", 144, 144, 1024, 1024),
              ("vae-288-144-1024", 288, 144, 1024, 1024),
              ("vae-576-512", 576, 576, 512, 512),
              ("vae-1152-256", 1152, 1152, 256, 256),
              ("vae-1152-128", 1152, 1152, 128, 128)]

"""`(M, N, K)` fp16 products."""
const GEMM = [("square-4096", 4096, 4096, 4096),
              ("denoiser-qkv", 12288, 4224, 4096),
              ("denoiser-gateup", 24576, 4224, 4096)]

function main()
    backend = M.LavaBackend()
    ctx = DK.Ctx(backend)
    caps = DK.caps(backend)
    rows = Any[]

    for (name, E, Lq, Lk, H, B, calls) in ATTN
        q, k, v = f16(backend, E, Lq, H, B), f16(backend, E, Lk, H, B), f16(backend, E, Lk, H, B)
        out = KA.allocate(backend, Float32, E, Lq, H, B); fill!(out, 0f0)
        # **The plan the MODEL would get, which is unclamped first.**
        # `emitsdpa!` asks `flashcm_plan` with no clamp and only retries through
        # `flashcm_padded_plan` when that declines; this file used to force
        # `clamp = Lk % 32 != 0` instead, which is not the same rule and is
        # wrong wherever the chosen `BC` divides `Lk` anyway. `sam2-windowed-16`
        # is exactly that: `Lk = 16` fails `% 32` but the tiling is `(16, 16)`,
        # and the clamped plan is a different tiling on the staged kernel —
        # **1.319 ms against 0.568**, so this row reported a shape SAM 2 does
        # not run, 2.3x slow.
        plan = DK.flashcm_plan(caps, q, k, v, nothing)
        plan isa DK.FlashCMPlan ||
            (plan = DK.flashcm_padded_plan(caps, q, k, v, nothing))
        if plan isa DK.FlashCMPlan
            t = best(backend, () -> DK.sdpaflashcm!(ctx, out, plan, q, k, v, Float32(1/sqrt(E))))
            push!(rows, (kind = "attention", name = name, ms = t,
                         flops = 2.0 * 2 * Lq * Lk * E * H * B,
                         params = Dict("E" => E, "Lq" => Lq, "Lk" => Lk, "H" => H,
                                       "B" => B, "calls" => calls)))
        end
        q = k = v = out = nothing; GC.gc()
    end

    for (name, Cin, Cout, H, W) in CONV
        x = f16(backend, W, H, Cin, 1)
        w = f16(backend, 3, 3, Cin, Cout; scale = 0.05f0)
        out = KA.allocate(backend, Float16, W, H, Cout, 1); fill!(out, Float16(0))
        t = best(backend, () -> DK.convolution!(ctx, out, x, w, nothing, (1, 1), (1, 1), (1, 1), 1); n = 8, warm = 3)
        push!(rows, (kind = "conv3x3", name = name, ms = t,
                     flops = 2.0 * Cin * Cout * 9 * H * W,
                     params = Dict("Cin" => Cin, "Cout" => Cout, "H" => H, "W" => W)))
        x = w = out = nothing; GC.gc()
    end

    for (name, Mm, Nn, Kk) in GEMM
        a, b = f16(backend, Mm, Kk), f16(backend, Kk, Nn)
        _, sk = M.coopmat_gemm_shape(Mm, Nn, Kk)
        c = KA.allocate(backend, Float32, Mm, Nn, max(sk, 1))
        t = best(backend, () -> M.coopmat_gemm!(c, a, b, Mm, Nn, Kk; partials = c, reduce = false); n = 8, warm = 3)
        push!(rows, (kind = "gemm-fp16", name = name, ms = t,
                     flops = 2.0 * Mm * Nn * Kk,
                     params = Dict("M" => Mm, "N" => Nn, "K" => Kk)))
        a = b = c = nothing; GC.gc()
    end

    for (name, T, nbytes) in (("copy-fp16-512MiB", Float16, 512 << 20),
                              ("copy-fp32-512MiB", Float32, 512 << 20))
        n = nbytes ÷ sizeof(T)
        src = KA.allocate(backend, T, n); dst = KA.allocate(backend, T, n)
        t = best(backend, () -> copyto!(dst, src))
        push!(rows, (kind = "copy", name = name, ms = t, flops = 0.0,
                     params = Dict("bytes" => nbytes, "eltype" => string(T))))
        src = dst = nothing; GC.gc()
    end

    out = joinpath(@__DIR__, "..", "tmp", "baseline", "ours.json")
    mkpath(dirname(out))
    open(out, "w") do io
        JSON3.pretty(io, (rows = rows,))
    end
    @printf("%-10s %-22s %10s %10s\n", "kind", "name", "ms", "TFLOP/s")
    for r in rows
        @printf("%-10s %-22s %10.2f %10s\n", r.kind, r.name, r.ms,
                r.flops == 0 ? "-" : @sprintf("%.2f", r.flops / (r.ms * 1e-3) / 1e12))
    end
    println("\nwrote ", out)
end

main()
