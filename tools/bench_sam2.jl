"""
SAM 2.1 on Lava: does the GPU agree with PyTorch, and how fast is it.

    julia --project=. tools/bench_sam2.jl [gpu|cpu]

Two questions in one run, because they need the same setup:

  * **parity** — the encoder's six outputs and the decoder's masks and IoU
    scores against `dump_sam2_refs.py`'s recording. `verify_sam2.jl` checks the
    decoder node by node on the CPU backend; this is the end-to-end check, and
    the only one that covers the encoder, whose 1493 ops at 1024x1024 are not
    something to run on a CPU backend.
  * **cost** — encode and decode timed separately, against
    `sam2_pytorch_baseline.py`. Separately because the editor pays them at
    different rates: encode once per marked frame, decode per click.

A mask is compared as a mask, not as a tensor: what matters is whether the same
pixels are selected, so the headline number is IoU of `logit > 0` against the
reference, with the worst logit difference alongside it to catch a near-miss
that happens to threshold the same way.
"""

using DNNKernels, KernelAbstractions, Statistics, Printf
# `SAM2`, `encode`, `decode` and `prompt` moved from DNNKernels to SAM2Runner
# (JuliaVision `5b59cd7`, "the model drivers leave the kernel library"). They are
# model code, not kernels. This tool followed them here rather than through the
# stale binding, which still *imported* under the old name and then failed with
# `UndefVarError` at first use.
using SAM2Runner: SAM2, encode, decode, prompt
using DNNKernels: readsafetensors, toback
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "sam2-large"))
const RES = 1024

"""
Device memory in use, in MiB, or `nothing` where `nvidia-smi` is not available.

Sampled **before the Vulkan device exists** so the difference is this process's,
not the desktop's. It is what the memory goal is stated in: `perf-plan.md` tracks
"VRAM steady" against PyTorch's 1 756 MB reserved, and 90% of that is a ceiling of
1 951 MB — currently met with 11 MB to spare, which is thin enough that it is
reported here beside the speed rather than left to be rediscovered.
"""
vram() = try
    parse(Int, first(split(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader`,
                                String))))
catch ex
    # `nvidia-smi` is simply absent on any non-NVIDIA machine, and that is the
    # whole reason this returns `nothing` instead of failing. Narrowed to that:
    # an nvidia-smi that IS present and answers something unparseable is a real
    # problem, and used to come back as "VRAM unknown" like a missing binary.
    ex isa Union{Base.IOError, SystemError, Base.ProcessFailedException} || rethrow()
    nothing
end

const VRAM_BASE = vram()
const VRAM_CEILING = 1951        # 1756 MB PyTorch / 0.9

mode = isempty(ARGS) ? "gpu" : lowercase(ARGS[1])
backend = if mode == "gpu"
    using Lava
    LavaBackend()
else
    KA.CPU()
end

@info "loading SAM 2.1 on $mode"
t0 = time()
sam = SAM2(joinpath(DIR, "sam2_encoder.json") |> dirname,
           joinpath(DIR, "weights.safetensors"); backend, res=RES)
@printf("  loaded in %.1f s\n", time() - t0)

refs = readsafetensors(joinpath(DIR, "refs.safetensors"))
image = toback(backend, refs["sam2_encoder/in0"])
point = toback(backend, refs["sam2_decoder/in3"])
label = toback(backend, refs["sam2_decoder/in4"])

# ---------------------------------------------------------------- parity
feats = encode(sam, image)
KA.synchronize(backend)

# Node names come from the export; re-exporting SAM 2 renumbers them, which is
# why these are looked up rather than assumed — a stale name is a KeyError, not
# a silently skipped check.
encnames = ["convolution_5", "convolution_6", "add_129",
            "_to_copy_599", "_to_copy_595", "repeat_2"]
println("\nencoder outputs vs PyTorch:")
encworst = 0.0
for (i, n) in enumerate(encnames)
    want = refs["sam2_encoder/node/$n"]
    got = Array(feats[i])
    size(got) == size(want) || error("$n: got $(size(got)), reference $(size(want))")
    e = maximum(abs, got .- want)
    rel = e / max(maximum(abs, want), eps(Float32))
    global encworst = max(encworst, rel)   # `for` at file scope is a soft scope
    @printf("  %-16s %-22s max abs %.3e   relative %.2e\n", n, string(size(got)), e, rel)
end

masks, iou = decode(sam, feats, point, label)
KA.synchronize(backend)
wantm = refs["sam2_decoder/node/slice_2"]
wanti = refs["sam2_decoder/node/slice_3"]
gotm, goti = Array(masks), Array(iou)

println("\ndecoder vs PyTorch:")
@printf("  logits    max abs %.3e over %s\n", maximum(abs, gotm .- wantm), string(size(gotm)))
@printf("  iou       got %s\n            want %s\n",
        string(round.(goti[:, 1]; digits=4)), string(round.(wanti[:, 1]; digits=4)))
for i in 1:size(gotm, 3)
    a, b = view(gotm, :, :, i, 1) .> 0, view(wantm, :, :, i, 1) .> 0
    inter, uni = count(a .& b), count(a .| b)
    @printf("  mask %d    IoU %.5f   covers %.1f%% (reference %.1f%%)\n",
            i, uni == 0 ? 1.0 : inter / uni, 100count(a) / length(a), 100count(b) / length(b))
end

# ---------------------------------------------------------------- cost
"""Median of `iters` timed runs after `warmup`, each synchronized."""
function timed(f, warmup, iters)
    for _ in 1:warmup; f(); end
    KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:iters
        t = time_ns(); f(); KA.synchronize(backend)
        push!(ts, (time_ns() - t) / 1e6)
    end
    (min=minimum(ts), p50=median(ts), mean=mean(ts))
end

"""
Live device memory in MiB, once trimming stops changing it.

Settle rather than guess: `live_bytes` counts pool *blocks*, so it reports the
allocator's transient high-water mark until the finalizers have run and the empty
blocks have gone back to the driver — a function of GC timing rather than of
demand, and worth 256 MiB here. `trim_gpu_pool!` does both.

A function rather than a top-level loop because under Julia 1.12's soft scope the
`for` body's `live = l` bound a *new local* and left the outer name undefined, so
this whole block died with an `UndefVarError` after the benchmark had already run.
"""
function settledlive(tries = 8)
    live = typemax(Int)
    for _ in 1:tries
        Lava.trim_gpu_pool!()
        l = Lava.gpu_memory_usage().live_bytes ÷ 2^20
        l == live && return l
        live = l
    end
    return live
end

# Settled once before any of it, so the three rows are measured under the same
# allocator conditions. Without this the first loop runs while the parity
# section's temporaries are still resident and pays the pool growth — a decode
# median of 8.76 ms against a min of 3.85, on three runs out of seven, always on
# whichever loop happened to go first. The rows are meant to be compared with
# each other; that made them incomparable.
mode == "gpu" && settledlive()

println("\ncost:")
e = timed(() -> encode(sam, image), 3, mode == "gpu" ? 20 : 1)
d = timed(() -> decode(sam, feats, point, label), 3, mode == "gpu" ? 50 : 1)
@printf("  encode  p50 %7.2f ms   min %7.2f\n", e.p50, e.min)
@printf("  decode  p50 %7.2f ms   min %7.2f\n", d.p50, d.min)


# Settled HERE, before the click loop, and printed further down. This process is
# the only thing anywhere that runs both decode paths: the loop below captures,
# and a held capture pins the transients of the decode it recorded so the pool
# cannot recycle them, while the 53 *uncaptured* decodes above have a live set of
# their own. Two sets at once reads 1225 MiB against 1161, and describes no
# configuration that exists — releasing the capture afterwards gives back only
# the empty blocks. Over the editor's real pattern (encode a frame, click it,
# move on) a held capture costs 4 MiB: 1161 -> 1165, measured separately.
livemib = mode == "gpu" ? settledlive() : 0

# The decode row hands `decode` the reference tensors directly, which is what the
# parity check needs but is *not* what a click costs. A click arrives through
# `prompt`, whose persistent buffers are what let the decode be captured once and
# replayed thereafter, so it is measurably cheaper. Both are reported: the first
# is comparable to PyTorch's decoder, the second is what the editor pays per
# click. Quote `min` when the two disagree by more than a few percent — this card
# also drives the desktop, and a compositor burst inflates a whole process's
# medians (seen once in three runs, on both rows at once).
if mode == "gpu"
    c = timed(3, 50) do
        pt, lb = prompt(sam, [(0.5, 0.5)], [true])
        decode(sam, feats, pt, lb)
    end
    @printf("  click   p50 %7.2f ms   min %7.2f   (prompt + decode, replayed)\n",
            c.p50, c.min)
end

# Steady state, after the timing loops have grown every workspace they are going
# to grow. Two GCs because the first only queues the finalizers.
#
# **Two numbers, and they are not interchangeable.** `live` is what Lava has
# handed out and is what `perf-plan.md` tracks against PyTorch's 1 756 MB — its
# recorded 2 234 comes with a workspace/slab/weights/pool breakdown, which is
# Lava-side accounting. `reserved` is what the driver has taken from the card,
# which includes the allocator's pool: that is a high-water mark and depends on
# allocation order, and has been seen to differ by 3 GB between two runs whose
# `live` agreed within 10%. Quote `live`; watch `reserved` for leaks.
if mode == "gpu"
    live = livemib
    @printf("\nmemory:\n  live     %d MiB   ceiling %d   %s\n",
            live, VRAM_CEILING,
            live <= VRAM_CEILING ? "OK ($(VRAM_CEILING - live) MiB spare)" :
                                   "OVER BY $(live - VRAM_CEILING) MiB")
    VRAM_BASE === nothing ||
        @printf("  reserved %d MiB   (driver total incl. pool; order-dependent)\n",
                vram() - VRAM_BASE)
end

base = joinpath(DIR, "pytorch_baseline.json")
if isfile(base)
    # DNNKernels' JSON3, not a direct dependency of this script's environment.
    b = DNNKernels.JSON3.read(read(base, String))
    @printf("\nagainst PyTorch on %s:\n", b.device)
    @printf("  encode  %7.2f ms pytorch -> %5.1f%% of its speed\n",
            b.encode_ms.p50, 100 * b.encode_ms.p50 / e.p50)
    @printf("  decode  %7.2f ms pytorch -> %5.1f%% of its speed\n",
            b.decode_ms.p50, 100 * b.decode_ms.p50 / d.p50)
else
    println("\n(no pytorch_baseline.json — run tools/sam2_pytorch_baseline.py)")
end
