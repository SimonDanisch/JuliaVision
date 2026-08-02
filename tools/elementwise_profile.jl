"""
Where the elementwise/layout time actually goes, and which launch shapes cause it.

    julia --project=. tools/elementwise_profile.jl

`perf-plan.md`'s GPU-time breakdown puts **elementwise and layout at ~113 ms of
the 260 ms encode (43%)** — larger than matmul, and the bucket nobody has taken
apart. Attention, by contrast, is 33 ms. Since PyTorch does the whole encode in
87.6 ms, no amount of attention work reaches the 90% target on its own; this
bucket has to come down, so the first job is to know what is in it.

Two signals, one run, because they answer different halves:

  * `Lava.with_dispatch_timing` — which *kernel* costs what. It is the arbiter:
    isolated microbenchmarks in this project have three times shown a win the
    encode did not move at all.
  * `DNNKernels.LAUNCH_PROBE` — which *launch site and shape* produced it. A grid
    of 64 workgroups on a 48-SM card leaves the device idle however good the
    kernel is, and that is invisible in a timing table.

Prints the ranked kernels, the elementwise/layout subtotal, and every recorded
launch whose grid cannot fill the device — the launches worth fixing first.
"""

using SAM2Runner, DNNKernels, Lava, KernelAbstractions, Printf
const KA = KernelAbstractions

# 48 SMs on the RTX 4000 Ada. A grid below this cannot occupy the card at all,
# whatever the kernel does; one at 1-2x is still leaving most scheduling slack.
const SMS = 48

# Which kernel families make up each bucket in perf-plan.md's table. Matched as
# substrings of the dispatch name.
const BUCKETS = ["matmul"      => ["coopmat_gemm", "mm_epilogue", "conv2d_igemm"],
                 "elementwise" => ["ndmap", "broadcast", "permutedims", "layernorm"],
                 "attention"   => ["attn_"]]

function bucketof(name)
    for (bucket, keys) in BUCKETS
        any(k -> occursin(k, name), keys) && return bucket
    end
    return "other"
end

function main()
    backend = LavaBackend()
    @info "loading SAM 2.1"
    model = SAM2Runner.sam2model(; backend)
    img = KA.allocate(backend, Float32, 1024, 1024, 3, 1)
    copyto!(img, fill(0.5f0, 1024, 1024, 3, 1))

    # Warm up enough to raise the clock and compile every kernel, but NOT so long
    # that the allocator's state becomes the thing being measured — a 1.5 s
    # warm-up once flattered a replacement by 8x (see perf-plan.md).
    DNNKernels.encode(model, img); KA.synchronize(backend)
    DNNKernels.encode(model, img); KA.synchronize(backend)

    probe = Dict{Any,Any}()
    bprobe = Dict{Any,Any}()
    DNNKernels.LAUNCH_PROBE[] = probe
    Lava.BROADCAST_PROBE[] = bprobe
    report = Lava.with_dispatch_timing() do
        DNNKernels.encode(model, img)
        KA.synchronize(backend)
    end
    DNNKernels.LAUNCH_PROBE[] = nothing
    Lava.BROADCAST_PROBE[] = nothing

    sort!(report; by = r -> -r.total_ns)
    total = sum(r -> r.total_ns, report; init = 0.0)
    @printf("\n%-60s %6s %9s %7s  %s\n", "kernel", "calls", "GPU ms", "share", "bucket")
    for r in report
        r.total_ns / 1e6 < 0.5 && continue
        @printf("%-60s %6d %9.2f %6.1f%%  %s\n", first(r.name, 60), r.n_dispatches,
                r.total_ns / 1e6, 100 * r.total_ns / total, bucketof(r.name))
    end

    # One row per kernel, summed over launch geometries. The per-dispatch table
    # above splits a kernel across every `groups=` it was launched with — five
    # `lava_broadcast_flat_mixed!` rows had to be added up by hand to compare the
    # family against its previous total, which is exactly the arithmetic that
    # makes a comparison wrong.
    fams = Dict{String,Tuple{Int,Float64}}()
    for r in report
        m = match(r"gpu_([A-Za-z0-9_]+!)", r.name)
        k = m === nothing ? r.name : m.captures[1]
        c, ms = get(fams, k, (0, 0.0))
        fams[k] = (c + r.n_dispatches, ms + r.total_ns / 1e6)
    end
    @printf("\n%-32s %6s %9s %7s  %s\n", "kernel family", "calls", "GPU ms", "share", "bucket")
    for (k, (c, ms)) in sort!(collect(fams); by = kv -> -kv.second[2])
        ms < 0.3 && continue
        @printf("%-32s %6d %9.2f %6.1f%%  %s\n", k, c, ms, 100 * ms * 1e6 / total, bucketof(k))
    end
    @printf("\ntotal %.1f ms\n", total / 1e6)
    for (b, _) in BUCKETS
        ms = sum(r -> bucketof(r.name) == b ? r.total_ns : 0.0, report; init = 0.0)
        @printf("  %-12s %7.1f ms  %5.1f%%\n", b, ms / 1e6, 100 * ms / total)
    end
    ms = sum(r -> bucketof(r.name) == "other" ? r.total_ns : 0.0, report; init = 0.0)
    @printf("  %-12s %7.1f ms  %5.1f%%\n", "other", ms / 1e6, 100 * ms / total)

    # What feeds each broadcast path. `lava_broadcast_flat_mixed!` is the
    # expensive one (40 ms on SAM 2's encoder) and the recorded finding is that
    # 203 of 203 of its dispatches are `dest .= PermutedDimsArray(a, perm)`
    # moving ~2.4 GB at 60 GB/s where a plain copy runs at 150-270. This says
    # which shapes and which `perm`, which is what decides whether a tiled
    # staging transpose can help: with `perm[1] == 1` the fast axis is already
    # contiguous, so the cost is locality across the outer axes, not a transpose.
    counts = Dict{Symbol,Int}()
    for ((path, _, _), n) in bprobe
        counts[path] = get(counts, path, 0) + n
    end
    println("\nbroadcast dispatches by path: ",
            join(("$k=$v" for (k, v) in sort!(collect(counts); by = kv -> -kv.second)), "  "))
    # ALL of them, not the top 20. Which permutations are present decides which
    # kernel is worth writing, and the tail is where a family hides: the ported
    # tiled transpose targets `perm[1] != 1`, and a truncated list cannot say
    # whether the encoder still issues any.
    mixed = sort!([kv for kv in bprobe if kv.first[1] === :mixed]; by = kv -> -kv.second)
    isempty(mixed) || @printf("\n%8s  %-24s %s\n", "calls", "dest size", "leaves (size, kind, perm/linear)")
    for ((_, dsz, leaves), n) in mixed
        @printf("%8d  %-24s %s\n", n, string(dsz), string(leaves))
    end
    # Split by whether the fast axis moves — the two cases need different kernels
    # (blocked gather vs tiled staging transpose) and the counts decide the order.
    fast, moved, other = 0, 0, 0
    for ((_, _, leaves), n) in mixed
        p = nothing
        for l in leaves
            l isa Tuple && length(l) == 3 && l[2] === :Permuted && (p = l[3])
        end
        p === nothing ? (other += n) : (p[1] == 1 ? (fast += n) : (moved += n))
    end
    @printf("\nmixed dispatches: perm[1]==1 %d   perm[1]!=1 %d   not a permute %d\n",
            fast, moved, other)

    # The launches that cannot fill the device. `launch!` only records the
    # non-flat path, so this under-reports rather than over-reports.
    @printf("\n%-26s %-18s %8s %8s\n", "ndrange", "workgroup", "groups", "calls")
    rows = sort!(collect(probe); by = kv -> prod(kv.second[2]))
    for ((sz, wg), (count, grp)) in rows
        n = prod(grp)
        n >= 4 * SMS && continue          # fills the card; not the problem
        @printf("%-26s %-18s %8d %8d   %s\n", string(sz), string(wg), n, count,
                n < SMS ? "<- cannot fill 48 SMs" : "<- thin")
    end
    println("\n(`launch!` skips the probe when LAUNCH_FLAT routes a multi-dim ",
            "output through ndmap_flat!, so absent rows are not absent launches.)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
