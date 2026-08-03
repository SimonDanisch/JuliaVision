"""
One repeatable VRAM number for SAM 2.1, and the spread that makes it one.

    julia --project=. tools/memcheck.jl

The memory goal is "within 90% of PyTorch's 1 756 MB", i.e. **≤ 1 951 MB**, and
two separate claims about reaching it were retracted in one afternoon because the
number moved under them:

  * `nvidia-smi` and `Lava.gpu_memory_usage().live_bytes` are different
    quantities. The first includes the allocator's pool, whose high-water mark
    depends on allocation *order*: two runs whose `live` agreed within 10%
    reported 1 940 and 4 849 MB reserved, and one reported a **negative** delta
    because another process freed memory mid-run.
  * `live` itself needs settling. GPU buffers go through finalizers, so a plain
    `GC.gc()` pair reads 2 393 MiB where a full collection reaches 2 288 and
    holds.
  * and even settled, two `bench_sam2.jl` runs of identical code measured 2 800
    and 2 096.

So this reports **encode-only and encode+decode separately** — the recorded
history in `perf-plan.md` is encoder-only, and PyTorch's figure is not — and it
reports every repeat rather than a mean, because the spread is the finding when
the spread is 700 MB.

Run it more than once from a cold process. A single number from this file is
worth no more than a single timing was.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics
using DNNKernels: readsafetensors, SAM2, encode, decode, toback
using Lava
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "sam2-large"))
const CEILING = 1951        # 1756 MB PyTorch / 0.9
const DECODES = 20          # past the warm-up; the steady state is flat well before this

"""
Live bytes in MiB, after finalizers have run **and** dead pool capacity has been
handed back.

`live_bytes` counts the allocator's pool blocks, not the sub-allocations handed
out of them, so without the trim it reports the pool's transient high-water mark
— which is a function of GC timing, not of what the model needs. On SAM 2.1 that
was the difference between 2 224 and 1 968 MiB, and it is why the same workload
measured 2 800 in one session and 2 096 in another.
"""
function settled(; tries = 8)
    prev = typemax(Int)
    for _ in 1:tries
        trim_gpu_pool!()
        l = Lava.gpu_memory_usage().live_bytes ÷ 2^20
        l == prev && return l
        prev = l
    end
    return prev
end

function main(repeats = 3)
    backend = LavaBackend()
    Lava.FLUSH_TIMEOUT_NS[] = 25_000_000_000
    sam = SAM2(DIR, joinpath(DIR, "weights.safetensors"); backend, res = 1024)
    refs = readsafetensors(joinpath(DIR, "refs.safetensors"))
    image = toback(backend, refs["sam2_encoder/in0"])
    point = toback(backend, refs["sam2_decoder/in3"])
    label = toback(backend, refs["sam2_decoder/in4"])

    @printf("%-8s %12s %14s\n", "repeat", "encode MiB", "enc+dec MiB")
    enc, both = Int[], Int[]
    for r in 1:repeats
        feats = encode(sam, image)
        KA.synchronize(backend)
        e = settled()
        for _ in 1:DECODES
            decode(sam, feats, point, label)
            KA.synchronize(backend)
        end
        b = settled()
        push!(enc, e); push!(both, b)
        @printf("%-8d %12d %14d\n", r, e, b)
    end
    @printf("\nencode only   median %d MiB, spread %d\n", median(enc), maximum(enc) - minimum(enc))
    @printf("encode+decode median %d MiB, spread %d   ceiling %d -> %s\n",
            median(both), maximum(both) - minimum(both), CEILING,
            median(both) <= CEILING ? "OK" : "OVER BY $(Int(median(both)) - CEILING) MiB")
    println("\nA spread of the same order as the gap to the ceiling means no memory ",
            "change smaller than that spread can be demonstrated at all.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
