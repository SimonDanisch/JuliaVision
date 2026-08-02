"""
The permuted copy, in isolation and at speed, so a kernel can be iterated in
seconds instead of one encode at a time.

    julia --project=. tools/permute_bench.jl

`dest .= PermutedDimsArray(src, perm)` is **203 of 203** dispatches on Lava's
`lava_broadcast_flat_mixed!` path for SAM 2's encoder — 40.1 ms moving ~2.4 GB at
**60 GB/s where a plain copy of the same bytes runs at 150-270**. Add
`lava_permutedims_kernel!` (11.95 ms, and the same naive per-element gather) and
it is **52 ms, 46% of the elementwise bucket** — the largest single opportunity
in `perf-plan.md`.

This measures three things per shape, which is what makes a result trustworthy:

  * the **floor** — a same-size linear `copyto!`, i.e. what the bytes cost when
    nothing is permuted. Any permuted kernel is judged against this, not against
    zero.
  * the **current** path, through the ordinary broadcast so it is what the model
    actually runs.
  * a **candidate**, once there is one — same call, compared bit-exactly, since a
    faster permute that moves different bytes is not a permute.

Warm-up is deliberately short. A 1.5 s warm-up once ran thousands of iterations,
each allocating reduction temporaries, and reported a kernel at 8x its honest
cost (see `perf-plan.md`); warm enough to raise the clock, not enough to move the
allocator's state. And the SM clock is printed, because this card idles at
210 MHz of 2265 and one measurement taken there reported a 9.99x win.
"""

using Lava, KernelAbstractions, Printf, Statistics
const KA = KernelAbstractions

# The two permutations the encoder actually runs, from the note above
# `BROADCAST_PROBE` in `Lava/src/array/gpuarrays.jl`: Hiera's window partition
# and attention's head/token swap. `perm[1] == 1` in the first — the fast axis
# does not move, which is the case that needs a blocked gather rather than a
# transpose — and the second is the minority that is a true transpose.
#
# The real shapes, from `tools/elementwise_profile.jl` on SAM 2's encoder — the
# source extents, so `PermutedDimsArray(src, perm)` reproduces the destination
# the model actually writes. `calls` is dispatches per encode.
#
# Two families, and the row length is what separates them: attention's head/token
# swap has `size(src,1) = 72`, i.e. **144-byte rows**, where scattering hurts
# most; Hiera's window partition has 576-1152, i.e. 1-2 KB rows that already
# stream reasonably. Both have `perm[1] == 1`. The genuine fast-axis transposes
# are 3 dispatches out of 203 and are here only so a candidate cannot quietly
# break them.
const CASES = [
    #  name              source shape                 perm             calls
    ("attn 72-row   ", (72, 8, 256, 16),        (1, 3, 2, 4)),        # 64
    ("attn 72 x1024 ", (72, 4, 16, 1024),       (1, 3, 2, 4)),        # 10
    ("hiera 576-row ", (576, 16, 4, 16, 4, 1),  (1, 2, 4, 3, 5, 6)),  # 33
    ("hiera 288-row ", (288, 4, 32, 4, 32, 1),  (1, 2, 4, 3, 5, 6)),  # 6
    ("true transpose", (256, 256, 144, 1),      (3, 1, 2, 4)),        # 1
]

"""
Independent buffer sets per variant, cycled by the repetition index.

8 sets of the largest case is ~600 MB, comfortably past any L2 on this class of
card, which is the point: one set measured the floor at 598 GB/s — above the
card's DRAM bandwidth — because 24 back-to-back launches never left cache.
"""
const NBUF = 8

gbps(bytes, secs) = bytes / secs / 2^30

"""
SM clock in MHz, or 9999 if it cannot be read.

Printed **per case**, not once at the end. A run that finished at 795 MHz told me
nothing about what the clock was during case 1 — and since the card drifts
*during* a measurement, an end-of-run reading is the one number guaranteed to be
taken while the GPU is idle. Ratios inside a case are still valid at any clock
because `timedall` interleaves; the absolute GB/s is not.
"""
smclock() = try
    parse(Int, first(split(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`,
                                String))))
catch
    9999
end

# ── candidate: same indexing, bigger per-workgroup footprint ──────────────────
#
# For `perm[1] == 1` — 88 of the encoder's 203 permuted copies, and the family
# with the worst ratio — nothing inside a row is reordered, so there is nothing
# to stage in shared memory. Each row is contiguous in *both* arrays; what costs
# is that the rows are scattered, so the hardware sees isolated 144-byte accesses
# (72 fp16). `staticgroup` hands the current kernel `(64,2,2,1)`, i.e. a 2×2 block
# of rows.
#
# Widening that block is the whole idea: a `B2 × B3` block of rows turns one
# scattered line into B streams of `B × rowbytes` contiguous on **both** sides at
# once. 8×8 rows of 144 bytes is 1.1 KB per stream instead of 144 B — the
# difference between random-line and streaming DRAM. A workgroup cannot cover
# that with one element per thread (8·8·72 = 4 608), so each thread loops.
#
# Rank-4 specialisation on purpose: prove the mechanism on the dominant family
# before generalising the index arithmetic to rank 6.
@kernel cpu=false function permute_blocked4!(dest, @Const(src), ::Val{IP},
                                             ::Val{B2}, ::Val{B3}) where {IP, B2, B3}
    i1, j2, j3, i4 = @index(Global, NTuple)
    d2 = size(dest, 2)
    d3 = size(dest, 3)
    @inbounds for a in 0:(B2 - 1)
        i2 = (j2 - 1) * B2 + 1 + a
        i2 > d2 && break
        for b in 0:(B3 - 1)
            i3 = (j3 - 1) * B3 + 1 + b
            i3 > d3 && break
            I = (i1, i2, i3, i4)
            dest[I...] = src[ntuple(d -> I[IP[d]], Val(4))...]
        end
    end
end

# ── candidate 2: tiled transpose, ported from llama.cpp's copy_transpose.comp ──
#
# For `perm[1] != 1` the fast axis genuinely moves and blocking is not enough —
# one side is strided whatever the block size. `reference/llama.cpp-vulkan/
# copy_transpose.comp` is the production answer and its comment names the trick:
#
#   "The workgroup does TILE_DIM x TILE_DIM, but swaps the LSBs of the src coords
#    to make memory accesses contiguous, dst has tid.x in i0, src has tid.x in i01"
#
# i.e. the thread index runs along the *source's* contiguous axis on the read and
# along the *destination's* on the write, and the shared tile absorbs the
# transpose — so both sides are coalesced. 32x32 tile carried by 32x8 threads,
# 4 rows each, `sh[TILE][TILE+1]` to break bank conflicts. `DNNKernels`'
# `toLE_tiled` is the same shape (33x32, 32x4 threads, 8 rows) and measured 7.1x,
# which is what makes this a port rather than a guess; it is modelled on that
# because the `@localmem` type must be a literal or the kernel silently writes
# nothing.
#
# Specialised to `dest[a,b,c,e] = src[b,c,a,e]` — perm (3,1,2,4), the encoder's
# only fast-axis transpose family.
@kernel cpu=false function transpose_tiled_312!(d, @Const(s), na::Int32, nb::Int32, nc::Int32)
    tile = @localmem Float16 (33, 32)
    tx, ty = @index(Local, NTuple)
    gx, gy, gz = @index(Group, NTuple)
    a0 = Int32(gx - 1) * Int32(32)
    b0 = Int32(gy - 1) * Int32(32)
    c  = Int32(gz)
    @inbounds begin
        # read: tx runs along b, which is src's unit-stride axis
        for j in Int32(0):Int32(7)
            a = a0 + Int32(ty) + Int32(4) * j
            b = b0 + Int32(tx)
            tile[Int32(ty) + Int32(4) * j, tx] =
                (a <= na && b <= nb) ? s[b, c, a] : zero(Float16)
        end
        @synchronize
        # write: tx runs along a, which is dest's unit-stride axis
        for j in Int32(0):Int32(7)
            a = a0 + Int32(tx)
            b = b0 + Int32(ty) + Int32(4) * j
            (a <= na && b <= nb) && (d[a, b, c] = tile[tx, Int32(ty) + Int32(4) * j])
        end
    end
end

"Run the ported tiled transpose; `nothing` unless this is the (3,1,2,4) family."
function tiled312(dest, src, perm, backend)
    (length(perm) == 4 && perm == (3, 1, 2, 4) && size(src, 4) == 1) || return nothing
    n1, n2, n3, _ = size(src)          # dest is (n3, n1, n2, 1)
    d3 = reshape(dest, n3, n1, n2)
    s3 = reshape(src, n1, n2, n3)
    gx, gy = cld(n3, 32), cld(n1, 32)
    () -> transpose_tiled_312!(backend, (32, 4, 1))(d3, s3, Int32(n3), Int32(n1), Int32(n2);
                                                    ndrange = (32gx, 4gy, n2))
end

"Run the blocked candidate; `nothing` when the shape is not the rank-4 case it covers."
function blocked(dest, src, perm, backend; B2 = 4, B3 = 2, wg = (64, 2, 2, 1))
    (length(perm) == 4 && perm[1] == 1) || return nothing
    d1, d2, d3, d4 = size(dest)
    nd = (d1, cld(d2, B2), cld(d3, B3), d4)
    IP = Tuple(invperm(collect(perm)))
    () -> permute_blocked4!(backend, wg)(dest, src, Val(IP), Val(B2), Val(B3); ndrange = nd)
end

"""
Median of `n` timed samples per variant, **interleaved**: every variant is timed
once per round, round-robin, so they all see the same clock history.

Measuring them one after another does not work on this card and the harness said
so out loud — a run that warmed to 2175 MHz and finished at 1290 reported the
blocked kernel at **254% of the linear-copy floor**, which is impossible. The
clock drifts *during* a run, so a variant measured late is measured slower, and
sequential A-then-B silently compares two different machines.

Two things that are not cosmetic:

  * **A sample is `reps` back-to-back launches with one sync, not one launch.**
    Syncing after every launch leaves the card idle for the host's `time_ns` and
    `push!`, and this GPU downclocks on that: the same file measured 660-1110 MHz
    with a per-launch sync and 1350-2000 with this. That matters because the
    ranking is not clock-invariant — the blocked kernel measured **0.75x** the
    broadcast at 795 MHz and **1.4-1.6x** at 1350. Back-to-back is also how the
    encoder issues them.
  * **`heat` runs between rounds**, on unrelated buffers, because `nvidia-smi
    -lgc` needs root here and the clock otherwise decays during the measurement
    itself.

Each variant is called as `f(r)` with the repetition index, and **must use it to
pick a buffer set**. Back-to-back launches over one 9-19 MB pair sit entirely in
this card's L2: with a single pair the floor read **598 GB/s**, above the card's
DRAM bandwidth, so it was measuring cache. The encoder's 203 dispatches each
touch a different tensor, and `NBUF` sets are chosen to exceed any L2 here.
"""
function timedall(fs, backend; n = 20, reps = 24, heat = nothing)
    for f in fs; f(1); end; KA.synchronize(backend)          # compile
    for r in 1:5, f in fs; f(r); end; KA.synchronize(backend) # settle
    ts = [Float64[] for _ in fs]
    for _ in 1:n
        heat === nothing || heat()
        for (i, f) in enumerate(fs)
            KA.synchronize(backend)
            t0 = time_ns()
            for r in 1:reps; f(r); end
            KA.synchronize(backend)
            push!(ts[i], (time_ns() - t0) / 1e9 / reps)
        end
    end
    map(median, ts)
end

function main()
    backend = LavaBackend()
    # Raise the SM clock before measuring anything. The first run of this file
    # reported 495 MHz of 2265 and numbers to match — this card idles at 210 and
    # a measurement taken there is worthless (see `perf-plan.md`). Enough work to
    # wake it, on a buffer unrelated to the cases below so the allocator state
    # the cases see is not the one this created.
    # 200 iterations reached only 780 MHz on one run and 2175 on another, so this
    # keeps going until the card actually reports a high clock rather than
    # trusting a fixed iteration count. Absolute GB/s is meaningless below ~2 GHz
    # here and the floors move with it (42.0 vs 50.6 for the same shape).
    # The same work is reused between measurement rounds (`heat` below): the clock
    # decays during the measurement too, and locking it needs root on this card.
    w = KA.allocate(backend, Float32, 1 << 22)
    v = KA.allocate(backend, Float32, 1 << 22)
    heat(k = 200) = (for _ in 1:k; w .= v .* 1.0001f0 .+ 0.5f0; end)
    let
        for round in 1:20
            heat(); KA.synchronize(backend)
            smclock() >= 1800 && break
            round == 20 && @warn "SM clock still low after warm-up" clock = smclock()
        end
    end
    # Both current paths, because they are two spellings of one unoptimised
    # operation and the plan counts them together: `.= PermutedDimsArray` goes
    # through `lava_broadcast_flat_mixed!` (40.1 ms in the encode) and
    # `permutedims!` through `lava_permutedims_kernel!` (11.95 ms). If they read
    # the same here, one kernel fixes both.
    @printf("%-16s %-22s %9s %9s %9s %9s %9s %6s   %s\n",
            "case", "shape", "copy GB/s", "bcast", "permdims!", "blocked", "of floor",
            "MHz", "perm")
    for (name, sz, perm) in CASES
        dsz  = ntuple(d -> sz[perm[d]], length(sz))
        # NBUF independent buffer sets, cycled by the repetition index, so
        # consecutive launches do not re-read the previous one's L2 lines.
        host = rand(Float16, sz...)
        srcs  = [KA.allocate(backend, Float16, sz...)  for _ in 1:NBUF]
        dests = [KA.allocate(backend, Float16, dsz...) for _ in 1:NBUF]
        foreach(s -> copyto!(s, host), srcs)
        # The floor copies between two FLAT allocations, not through
        # `reshape(src, prod(sz))`. Copying out of a reshape read 50.6 GB/s on
        # (72,8,256,16) — 9.4 MB in 186 µs — while (256,256,144,1) moved four
        # times the bytes in 96 µs. A smaller transfer taking twice as long is not
        # a bandwidth result, and 186 µs is far too slow to be launch overhead, so
        # the reshape was in the way. The floor has to be the honest cost of the
        # bytes or every "% of floor" below it is meaningless.
        flats = [KA.allocate(backend, Float16, prod(sz)) for _ in 1:NBUF]
        srcfs = [KA.allocate(backend, Float16, prod(sz)) for _ in 1:NBUF]
        foreach(s -> copyto!(s, reshape(host, prod(sz))), srcfs)
        bytes = 2 * prod(sz) * 2                 # read + write, 2 bytes each

        ref = permutedims(host, perm)
        pick(v, r) = @inbounds v[mod1(r, NBUF)]

        fblks = [blocked(dests[i], srcs[i], perm, backend) for i in 1:NBUF]
        fblks[1] === nothing &&
            (fblks = [tiled312(dests[i], srcs[i], perm, backend) for i in 1:NBUF])
        fs = Any[r -> copyto!(pick(flats, r), pick(srcfs, r)),
                 r -> (pick(dests, r) .= PermutedDimsArray(pick(srcs, r), perm)),
                 r -> permutedims!(pick(dests, r), pick(srcs, r), perm)]
        fblks[1] === nothing || push!(fs, r -> pick(fblks, r)())
        times = timedall(fs, backend; heat = () -> heat(60))
        mhz = smclock()
        tcopy, tbc, tpd = times[1], times[2], times[3]
        tblk = length(times) >= 4 ? times[4] : NaN

        # Correctness per path, checked SEPARATELY: one flag cannot say which is
        # wrong, and "the broadcast is right but `permutedims!` is not" is a very
        # different bug from the reverse. Run each once more on its own so the
        # result in `dest` belongs to a known path.
        dest = dests[1]
        fill!(dest, zero(Float16)); fs[2](1); KA.synchronize(backend)
        okbc = Array(dest) == ref
        fill!(dest, zero(Float16)); fs[3](1); KA.synchronize(backend)
        okpd = Array(dest) == ref
        okblk = if fblks[1] === nothing
            true
        else
            fill!(dest, zero(Float16)); fblks[1](); KA.synchronize(backend)
            Array(dest) == ref
        end

        bad = okbc && okpd && okblk ? "" :
              "  << WRONG: " * (okbc ? "" : "broadcast ") * (okpd ? "" : "permutedims! ") *
              (okblk ? "" : "blocked")
        @printf("%-16s %-22s %9.1f %9.1f %9.1f %9s %8.0f%% %6d   %s%s\n",
                name, string(sz), gbps(bytes, tcopy), gbps(bytes, tbc), gbps(bytes, tpd),
                isnan(tblk) ? "-" : @sprintf("%.1f", gbps(bytes, tblk)),
                100 * tcopy / min(tbc, tpd, isnan(tblk) ? Inf : tblk), mhz,
                string(perm), bad)
        # ~600 MB of buffer sets per case, and this card is shared with a running
        # editor session — hand them back before the next case allocates its own.
        srcs = dests = flats = srcfs = fblks = nothing
        GC.gc()
    end
    println("\nSM clock: ", try
        strip(read(`nvidia-smi --query-gpu=clocks.sm --format=csv,noheader`, String))
    catch; "?" end, "  (idles at 210 MHz of 2265 — a low reading invalidates the run)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
