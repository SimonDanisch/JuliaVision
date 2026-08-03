"""
Why loading a large model reserves several times its own weights.

    julia --project=. tools/pool_fragmentation_probe.jl

Standalone: allocates through `KernelAbstractions` and needs no model, no
weights and no export. It exists because "SAM 2 reserves 4 032 MiB to hold
941 MiB of weights" is a symptom, and this is the shape underneath it.

Three allocation patterns over the *same* set of buffer sizes — SAM 2.1's actual
checkpoint distribution by default: 909 tensors, 941 MiB, median 2 KiB, largest
exactly 64 MiB (which is `POOL_LARGE_THRESHOLD`, so it does not bypass the pool).

  1. **hold all live** — every buffer kept. This is the control, and it is the
     allocator working correctly: 16 blocks against a 15-block minimum, 1.1x.
     A 2 KiB median packs fine.
  2. **allocate and drop each immediately** — pure transient churn. Grows the
     pool to the *whole* working set and leaves **every block empty**. Nothing
     returns them: `reclaim_empty_pool_blocks!` is documented as running "only on
     the OOM retry path", and past `POOL_SOFT_CAP` the allocator calls
     `collect_for_pool!` — a garbage collection, which cannot help when the
     memory is already unreferenced and merely unreclaimed — and then cuts a new
     block anyway.
  3. **interleave one transient with one resident** — a weight upload's shape,
     and the damaging case. The pool grows to **2x** the resident set, and almost
     none of it is reclaimable, because a block is only freeable when *every*
     tenant is dead and the resident tensors are scattered one per block.

Pattern 3 is the residual. `reclaim_empty_pool_blocks!` recovers the wholly
transient blocks — 51 of 63 on SAM 2, taking it from **7.9x the resident set to
1.50x for 52 ms**, once, at load — and can do nothing about the rest, because the
pool has no notion of separating a short-lived allocation from one that lives as
long as the model.

**Two reorderings were proposed to remove that residual and both are disproved**
(`small-models` REPORT, 2026-08-03); do not spend an afternoon on either:

  * *fold constants before uploading the resident weights* — impossible.
    `constops` seeds its known set from `kind === :weight`, so a constant subgraph
    is by definition one that reads only weights: folding consumes what it would
    have to precede.
  * *two-phase upload, folding weights first* — computable but pointless. On
    SAM 2's encoder the foldable ops consume 896 of 941 MiB, **95% of the bytes**,
    so phase one uploads nearly everything and the interleaving happens inside it.

The two populations are interleaved in the *graph*, not in the upload order, so
no `Model`-level ordering separates them. Removing the residual means the
allocator learning about lifetimes — a separate pool, or an arena for load-time
transients.

**What this is not.** Not fragmentation by size: pattern 1 has the identical size
distribution and wastes 6%. Not a leak: every buffer here is properly freed, and
`GC.gc(true)` plus `drain_deferred_frees!` run before each count. It is lifetime
mixing in a bump allocator whose reclaim is on the failure path.

Filed by `small-models` (`plans/projects/small-models/REPORT.md`, 2026-08-03) for
`lava-core`, whose report owns the allocator worklist. The three small ports do
not reproduce it — they upload 21–239 entries and strand nothing — so the models
to check are the large ones: SAM 2, MatAnyone, Whisper's 2.55 GB fp32 encoder.
"""

using KernelAbstractions, Lava, Printf
const KA = KernelAbstractions

const MIB = 1024 * 1024
const BLOCK_MIB = 64        # Lava's POOL_BLOCK_SIZE

"""
SAM 2.1's checkpoint distribution, or a synthetic stand-in with the same
character — a few large tensors and a long tail of tiny ones. The tail is the
point: a median of 2 KiB is what makes a bump allocator's behaviour interesting.
"""
function sizes_from(path)
    if path !== nothing && isfile(path)
        return parse.(Int, readlines(path))
    end
    s = Int[]
    push!(s, 64 * MIB)
    append!(s, fill(20 * MIB, 12))
    append!(s, fill(2 * MIB, 60))
    append!(s, fill(256 * 1024, 120))
    append!(s, fill(2048, 715))
    return s
end

backend = LavaBackend()
ctx = Lava.vk_context()
bq = ctx.default_bq
blocks() = length(Lava.pool(ctx).blocks)

sizes = sizes_from(get(ENV, "POOL_PROBE_SIZES", "/tmp/sam2_sizes.txt"))
total = sum(sizes)
minblocks = ceil(Int, total / (BLOCK_MIB * MIB))

@printf("%d buffers, %.0f MiB total, median %.1f KiB — minimum %d blocks\n\n",
        length(sizes), total / MIB, sort(sizes)[end ÷ 2] / 1024, minblocks)

"""
Run one pattern with the pool quiesced before and after, so the numbers describe
the pattern rather than whatever ran before it.
"""
function trial(label, f)
    Lava.reclaim_empty_pool_blocks!(bq)
    before = blocks()
    keep = f()
    KA.synchronize(backend)
    GC.gc(true)
    Lava.drain_deferred_frees!(bq)
    grew = blocks() - before
    n, bytes = Lava.reclaim_empty_pool_blocks!(bq)
    @printf("%-38s grew %3d blocks (%5d MiB), %3d empty (%5.0f MiB), %3d PINNED\n",
            label, grew, grew * BLOCK_MIB, n, bytes / MIB, grew - n)
    keep === nothing || empty!(keep)
    GC.gc(true)
    Lava.drain_deferred_frees!(bq)
    Lava.reclaim_empty_pool_blocks!(bq)
    return grew - n
end

trial("1. hold all live (control)",
      () -> Any[KA.allocate(backend, UInt8, s) for s in sizes])

trial("2. allocate and drop each immediately",
      () -> (for s in sizes; KA.allocate(backend, UInt8, s); end; nothing))

pinned = trial("3. interleave transient + resident", () -> begin
    kept = Any[]
    for s in sizes
        KA.allocate(backend, UInt8, s)               # transient
        push!(kept, KA.allocate(backend, UInt8, s))  # resident
    end
    kept
end)

println()
println("Pattern 3 is a weight upload's shape. Blocks marked PINNED hold at least")
println("one live tensor, so no reclaim can return them however empty they are.")
println("Reclaiming after a load still recovers pattern 2 — on SAM 2 that is 7.9x")
println("the resident set down to 1.50x, for 52 ms once. The residual needs the")
println("allocator to separate lifetimes; two Model-level reorderings were tried")
println("and both are disproved. See this file's docstring.")

# Non-zero when pattern 3 pins more than a perfectly packed resident set would,
# so this can drop into a suite once the behaviour is decided. Verified firing:
# with the synthetic sizes it pins 14 against a minimum of 8 and exits 1.
exit(pinned > minblocks ? 1 : 0)
