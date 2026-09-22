"""
Convolution as an explicit im2col followed by a tensor-core GEMM.

`conv_implicit.jl` never materialises the im2col matrix, which is the right call
for a scalar kernel: it trades memory traffic for index arithmetic that the ALUs
would otherwise be idle for.

**The reason this one materialises it is NOT that a cooperative matrix cannot be
loaded from shared memory.** It said so here until 2026-09-22 — "Lava lowers a
`OpCooperativeMatrixLoadKHR` from a `PhysicalStorageBuffer` address, there is no
workgroup-storage load" — and that stopped being true a long time before anyone
noticed: `flash.jl` declares `kvs = @localmem Float16` and builds its `MatrixB`
operands straight out of it.

The real reason is the one below, and it was measured by writing the other
kernel. A gathered-A convolution — the staged GEMM with its `A` staging replaced
by the im2col expression, so the matrix never exists — re-does the gather **once
per `BN`-wide output-channel block**, where this path computes it once and pays
a memory round trip for it. Same tiling as the staged GEMM's shipped
`64 x 128 x 32`, output correct to 1e-4 against the definition:

    Cin -> Cout   spatial     im2col     gathered
     144 -> 144   1024x1024    84.0 ms    75.7 ms   1.11x   2 column blocks
     288 -> 144   1024x1024   145.2      169.7      0.86x   2
     288 -> 288   1024x1024   181.7      285.3      0.64x   3
     576 -> 576     512x512   131.1      271.9      0.48x   5
    1152 -> 1152    256x256    90.0      279.0      0.32x   9
    1152 -> 1152    128x128    32.2       63.8      0.50x   9

**Not because of the repetition**, which is what the column figures look like
they say and what this note claimed for half a day. Doubling `BN` halves the
column-block count, and on the same shapes it changes the time by -0.7% to
+21%: `1152 -> 1152` goes from nine blocks to five and from 276.0 ms to 274.1.
The apparent correlation was a confound — shapes with more column blocks also
have deeper reductions.

What it is, from the same kernel with the gather deliberately broken (the
`LDOFF` trick, one step at a time):

    shape        real   no bounds test   no channel stride
    144 -> 144   5.5 TFLOP/s   6.6           7.4
    1152 ->      5.3           6.1           7.5
    288 ->       5.6           9.5          10.0

So the bounds predicate is worth 1.2-1.7x and the address arithmetic another
1.05-1.24x — and with **both** gone the kernel still only reaches 7.4-10.0
against this path's staged GEMM at 18-19 on the same tiling. Most of the gap is
not the gather: it is that the experiment was the PLAIN staged kernel, and
`Mantle` ships `vec2`, `vec4`, double-buffered and prefetching variants of that
kernel which are where its rate comes from.

A gathered-A convolution is therefore not refuted — but it has to be the gather
ported into those tuned variants rather than a fresh basic kernel, and on this
evidence it would still only win where `Cout` is small. `plans/perf-plan.md`
has the full table and the four hypotheses tested along the way.

Materialising it is cheap here. The reduction extent `CRS = Cin*KH*KW` is large
and the pixel count `NPQ = N*OH*OW` small (the dominant layer is 15x8), so the
im2col matrix for that layer is 553 KB; writing and re-reading it costs a few
microseconds against a multiply that drops from 0.93 ms to 0.027 ms once it runs
on tensor cores.

The orientation matters and it is the *transposed* one:

    out[NPQ, Cout] = col[NPQ, CRS] * w[CRS, Cout]

Column-major, `out`'s first axis is the pixel index — which is exactly the
memory order of `(OW, OH, Cout, N)` — and `w`'s first axis is `CRS`, which is
exactly the memory order of `(KW, KH, Cin, Cout)`. So the weights are already
the B operand with no transpose, and the result lands in the destination's own
layout. Writing it the other way round (`Cout x NPQ`) would need a permute on
both ends.
"""

"""Round up to the cooperative-matrix tile."""
const VULKAN_GEMM_TILE = 16
padtile(n::Int) = cld(n, VULKAN_GEMM_TILE) * VULKAN_GEMM_TILE

"""
`K` rounded to the staged kernel's `bk`, which is what a TILING needs — the
16-wide `padtile` only makes the cooperative-matrix INSTRUCTION legal.

`gemm_divides` tests all three axes and every tiling in `GEMM_TILINGS` has
`bk = 32`, so a `CRS` padded only to 16 can still land on a K no tiling divides:
SAM 2's `CRS = 144` pads to 144, `144 % 32 = 16`, and its GEMM drops to the
register-blocked kernel although M and N are both fine. Same mistake `GEMM_BLOCK`
exists to fix one axis over — "padtile pads to the cooperative-matrix tile, 16,
and that is not enough" was true of K as well.
"""
const GEMM_BK = 32
padbk(n::Int) = cld(n, GEMM_BK) * GEMM_BK

"""
The reduction extent this convolution will actually run at.

**Adaptive, not simply the wider pad.** Widening every `CRS` to `bk`
unconditionally is a REGRESSION for shapes whose waste budget the wider pad
blows: scanning all 1476 convolutions across every export in `gen/graphs`, two
(`CRS = 48`, wanvae) pass `crswaste` at 48/48 = 1.00 and fail it at 64/48 = 1.33,
and NO shape is admitted by the wider pad that the narrower one refused. Losing
`conv_coopmat` entirely to buy a tiling the shape cannot afford is a bad trade —
and a `CRS` that cannot afford `bk` was never going to get a tiling anyway.

So: take `bk` when it fits the budget, else fall back to the tile. `crswaste`
then vetoes on whichever was chosen.
"""
@inline function crsextent(crs::Int, crspad::Float64)
    p = padbk(crs)
    p <= crs * crspad ? p : padtile(crs)
end



"""
    conv_coopmat_plan(dev, out, x, w; crspad = 1.25) -> ConvCoopMatPlan | Decline

Whether the tensor-core path can take this convolution.

`NPQ` is padded internally (the im2col kernel simply writes zeros past the last
pixel) and `CRS` is padded when the waste is small enough, which `crspad`
decides. `Cout` is refused off the tile: it is the weight's own extent and
padding it would widen the *output*, not just the reduction, so a convolution
whose output channels do not land on the tile falls back to the implicit-GEMM
kernel. In this model that is the stem (`7x7x3`), the 1x1 layers with a
concatenated scalar channel (`Cin` 17, 257, 769) and the single-channel alpha
heads, all of them small.
"""
function conv_coopmat_plan(dev::M.DeviceCaps, out, x, w; crspad::Float64 = 1.25,
                           im2colcap::Int = im2colbudget(x))
    # The operands have to be *on the Lava device*, not merely of a type
    # cooperative matrices could hold: `coopmat_gemm_available` asks the Vulkan
    # context, which answers yes whenever Lava is loaded, so without this the CPU
    # verification run took this path and handed host `Array`s to the SPIR-V
    # compiler.
    islavaarray(x) && islavaarray(w) || return Decline(:host)
    return conv_coopmat_plan(dev, eltype(x), eltype(w), size(out), size(w);
                             crspad, im2colcap)
end

"""
    conv_coopmat_plan(dev, Tx, Tw, outsize, wsize; kw...) -> ConvCoopMatPlan | Decline

The same question asked of extents and element types alone, for a caller whose
operands are graph resources rather than arrays.

Everything above reads `size` and `eltype` and nothing else, so this is the whole
decision and the array method is a wrapper on it. Split because a DECLARED
convolution has no `LavaArray` to hand over — the `:host` guard there is about a
host array reaching a SPIR-V compiler, which cannot happen to a resource — and
because `mmplan` already answers its half of the same question this way.

`im2colcap` defaults to the ceiling rather than to the driver's free-memory
share: a declared im2col is a transient the placer aliases against the whole
graph, so what it costs is decided by the plan and not by what happens to be free
when the graph is built.
"""
function conv_coopmat_plan(dev::M.DeviceCaps, ::Type{Tx}, ::Type{Tw},
                           outsize::Dims, wsize::Dims; crspad::Float64 = 1.25,
                           im2colcap::Int = IM2COL_CAP[]) where {Tx,Tw}
    Tx === Float16 && Tw === Float16 || return Decline(:eltype)
    out, w = outsize, wsize
    KW, KH, Cin, Cout = wsize
    CRS = Cin * KH * KW
    # Before any test that divides by `dev.tile`: a device with no matrix
    # hardware reports no tile, and those would throw instead of declining.
    #
    # `coopmatkernels` and not `dev.coopmat`: what this plan commits to is
    # `coopmat_gemm_dispatch!` and the `GEMM_TILINGS` table, which exist at one
    # tile width. A device with 8-wide cooperative matrices passes `dev.coopmat`
    # and has no staged kernel, and admitting it here reached an undefined name
    # instead of the implicit-GEMM path below.
    coopmatkernels(dev) || return Decline(:nocoopmat)
    Cout % dev.tile == 0 || return Decline(:cout)
    #
    # How much padding of the reduction axis a convolution may buy its way onto the
    # tensor cores with.
    #
    # `CRS` is the *weight's* extent, so padding it means padding the weight.
    # `convolution_coopmat!` zero-fills both halves, and the padding is paid
    # for: every padded column is written by im2col, read by the GEMM and
    # multiplied by a zero, so the cost is proportional to the waste.
    #
    # At `1.25` the line falls between the two cases this repo has:
    #
    #     SAM 2 stem      7x7x3      CRS 147 -> 160    +8.8%    admitted   (padbk)
    #     MatAnyone       1x1x17     CRS  17 ->  32   +88.2%    refused    (padtile)
    #     MatAnyone       1x1x257    CRS 257 -> 288   +12.1%    admitted   (padbk)
    #     MatAnyone       1x1x769    CRS 769 -> 800    +4.0%    admitted   (padbk)
    #
    # The rows are `crsextent` results, `padbk` or `padtile` as it chooses, and
    # this table is what a reader uses to set `crspad`.
    #
    # The stem is what the budget buys: **2.800 -> 1.147 ms, 2.44x**, 0.99 to
    # 2.42 TFLOP/s, and SAM 2's encode 102.65 -> 100.91. A keyword rather than a
    # constant so the two sides can be measured in one session, which is the
    # only comparison this project trusts; `1.0` refuses any padding.
    CRSP = crsextent(CRS, crspad)
    CRSP <= CRS * crspad || return Decline(:crswaste)

    # Materialising im2col is only worth it when the GEMM reuses each element
    # enough to pay for writing and re-reading it. Every column is read once per
    # output channel, so `Cout` *is* the reuse factor: at `Cout = 16` the
    # 240x128 layer writes a 35 MB matrix to do 0.57 GFLOP, and the traffic
    # (~0.2 ms) costs more than the arithmetic it enables (~0.035 ms) — the
    # implicit-GEMM kernel, which never materialises it, wins outright.
    #
    # The size cap is the same judgement from the other side: that 35 MB is also
    # what makes the workspace OOM when anything else is on the card.
    # `Cout = dev.tile` is admitted, not excluded, because the fallback is the
    # more expensive side for the full-resolution layers. Measured in situ
    # (`Diagnostics.opdoublefilter` + capture/replay): `3x3 64->16 @240x128` on
    # implicit GEMM costs **1.411 ms at 0.40 TFLOP/s**, 21% of the step's entire
    # convolution budget, while coopmat reaches 5.1-5.6 TFLOP/s on `@120x64`
    # shapes with 4x LESS tile parallelism. On the tensor cores that
    # convolution is **0.442 ms** (3.2x) and the step goes **11.08 -> 9.78 ms,
    # 90.2 -> 102.3 steps/s**. Cout=16 is a legal single N-tile, and the cap has
    # to clear the 35 MB those layers ask for.
    #
    # **The reuse rule is right and the route still loses on narrow `Cin`, and
    # that is arithmetic rather than tuning.** At `144 -> 144 @ 1024²` the
    # matrix is `H*W` by `Cin*9` = 1048576 x 1296 fp16 = **2.72 GB**. Writing it
    # once and reading it once, at the 213 GB/s this device moves under
    # `copyto!`, is **25.5 ms** — a lower bound, and torch does the WHOLE
    # convolution in 19.32. No GEMM gets under the vendor from there. Ours is
    # 87.6 ms, so the matrix is about 29% of it and the skinny `N = 144`
    # product, nine 16-wide column tiles, is most of the rest. The gap closes as
    # `Cin` grows for the same reason: at `Cin = 1152` the matrix is 8x larger
    # but the arithmetic is 64x, and that shape measures 1.80x rather than 5.02x.
    #
    # What would fix it is the trick `GLOBALKV` plays in `flash.jl`: read the
    # A tile with `OpCooperativeMatrixLoadKHR` straight from the input and never
    # write the matrix. A 16x16 tile whose rows are consecutive `p` (stride 1 in
    # `x`) and whose columns are 16 consecutive `c` (stride `W*H`) is a
    # constant-2D-stride tile, which is exactly what the load takes, and
    # `Cin % 16 == 0` holds for every VAE layer here.
    #
    # **But only after the reduction axis is reordered.** `im2col_kernel!` lays
    # the columns out `k = kx + KW*ky + KW*KH*c`, `kx` fastest, because that is
    # the order the weight already has — so sixteen consecutive columns span
    # several kernel offsets, each a different spatial shift, and there is no
    # single stride. Channel-fastest, `k = c + Cin*(kx + KW*ky)`, puts sixteen
    # consecutive columns inside one `(kx, ky)` and makes the tile addressable.
    # That is a permutation of the WEIGHT, done once where `Bp` is built rather
    # than per step.
    #
    # **It was built, and it loses.** The addressing is right — a spike computing
    # one 16x16 output tile that way agrees with the host to 4.5e-06 — and it is
    # still the wrong kernel:
    #
    #     Cin   shape    im2col+GEMM   direct 4x1   direct 4x4
    #      144  1024²        85.3 ms     110.4 ms     133.5 ms
    #      288  1024²       191.3        867.1            -
    #     1152   256²        93.3        428.3        714.6
    #
    # The reason is arithmetic intensity, and it is the thing materialising the
    # matrix BUYS. A direct-load kernel has no reuse: each subgroup loads its own
    # A and B tiles from global every k-step, 5 loads per 4 muladds in the 4x1
    # form. Raising the per-subgroup tile to 4x4 improves the ratio to 8 loads
    # per 16 muladds and measures WORSE still, because sixteen accumulators plus
    # eight operands do not fit the register budget. im2col exists so that a
    # blocked GEMM can stage A once into LDS and reuse it across the whole
    # column strip; nothing here does that.
    #
    # So the design that could win is not this one. It is to fuse the im2col
    # ADDRESSING into a blocked GEMM's LDS staging — stage A tiles into shared
    # straight from `x`, keep the blocking — which saves writing and re-reading
    # the 2.72 GB matrix (~26 ms of the 85) and keeps the reuse. That needs a
    # blocked cooperative-matrix GEMM in this package, because the one in use is
    # `Mantle.coopmat_gemm!` and Mantle is a pinned dependency.
    #
    # The other half of the gap is in the same place: `Cout = 144` is padded to
    # 256 because Mantle's narrow-`N` path measures 14.418 ms against 3.985 for
    # the padded shape, so the product does 1.8x the necessary arithmetic and
    # the fix is a 16-wide `N` tile that is not slow.
    #
    # (The border cases the direct path would also have to solve — a tile
    # touching the zero pad, and a 16-row tile crossing a `W` boundary — are
    # confined to a few tiles per thousand at 1024², so they were never the
    # obstacle. The spike above clamps rather than masks them and is a speed
    # measurement only.)
    Cout >= dev.tile || return Decline(:reuse)
    NPQ = out[4] * out[2] * out[1]
    # `CRSP`, not `padtile(CRS)` — the scratch is allocated at the extent the plan
    # chose, so budgeting the narrower one would under-count the allocation.
    #
    # The budget no longer REFUSES: it sizes a chunk. One im2col row is `CRSP`
    # halves, so the cap says how many pixels may be in flight, and the rest of
    # them go round the same buffer again. A convolution too big to materialise
    # at once used to fall to the implicit-GEMM kernel at ~1 TFLOP/s; the
    # Qwen-Image 2.1 VAE has twenty-seven of those and they are 12.96 s of a
    # 16.0 s decode, against 19-22 TFLOP/s for the eighteen that fit.
    rows = im2colcap ÷ (CRSP * sizeof(Float16))
    rows = min(NPQ, (rows ÷ GEMM_BLOCK) * GEMM_BLOCK)
    # One M-block is the floor: below it there is no chunk to take, and a
    # reduction axis that wide is beyond anything this can serve.
    rows >= GEMM_BLOCK || return Decline(:im2colsize)
    # ── Spread the pixels EVENLY over the chunks the cap forces.
    #
    # Taking the largest chunk that fits leaves the remainder in the last one,
    # and the last one still costs a full `MP`: `im2col` writes every row of the
    # buffer and the GEMM multiplies every row of it, because `MP` is the
    # leading dimension and there is nowhere to say "only this many are real".
    # At the VAE's `144 -> 144 @ 1024²` that was 1048576 pixels in chunks of
    # 102144 — ten full and a last one of 27136 doing 102144 rows of work, 74%
    # of it on padding, and 6.5% of the whole convolution.
    #
    # Same chunk COUNT, so the cap still holds and nothing else about the plan
    # moves; the chunk is just the smallest block-aligned size that still needs
    # only that many. Rounding up can only lower the count, never raise it, and
    # the guard keeps it inside the budget when the rounding would not fit.
    nchunk = cld(NPQ, rows)
    even = cld(cld(NPQ, nchunk), GEMM_BLOCK) * GEMM_BLOCK
    even <= rows && cld(NPQ, even) == nchunk && (rows = even)
    ConvCoopMatPlan(CRS, CRSP, Cout, convcoutpad(padgemm(rows), Cout, CRSP), NPQ, rows)
end

"""
    convcoutpad(MP, Cout, CRSP) -> Int

`Cout` widened to a column tile the staged GEMM actually has, or `Cout` where
no widening is worth it.

**The staged GEMM's rate is decided by which `bn` its tiling gets, and the fall
off the edge is a factor of thirteen.** Measured at `M = 65536, K = 2592`,
sweeping the column count alone:

    N        16    32    48    64    96   128   144   160   192   256   288   384
    TFLOP/s 0.83  2.28  0.70  8.09  1.24 16.40  0.47  1.26  9.32 16.15  1.27 17.29

Every `N % 128 == 0` is 16-17, every other `N % 64 == 0` is 8-9, and everything
else is around one. A convolution's `Cout` is the weight's own extent and is
under no obligation to be either — the Qwen-Image 2.1 VAE runs 288, 576 and 144
— so it is worth PADDING the weight with zero columns and discarding the extra
output, exactly as the reduction axis is already padded.

Which pad: the wider tiling is worth about 1.8x the narrower one per column, so
the score is `bn / np` and the budget is generous. `Cout = 288` takes 384 (1.33x
the columns at 17.29, i.e. 12.97 effective) over 320 (1.11x at 9.32, i.e. 8.39);
`Cout = 144` takes 256 (1.78x at 16.15, 9.08 effective) over 192 (1.33x at 9.32,
7.0). Both are what the sweep says.

A candidate only counts if some staged tiling divides `MP`, `CRSP` and it, which
is the same test [`Mantle.gemm_padn`](@ref) makes — this differs from that one
only in scoring by tile WIDTH rather than by least padding, because the matmul
path's alternative is another GEMM and a convolution's is the implicit-GEMM
kernel at one TFLOP/s.
"""
function convcoutpad(MP::Int, Cout::Int, CRSP::Int)
    best, bestscore = Cout, 0.0
    for c in Mantle.GEMM_TILINGS
        haskey(Mantle.GEMM_STAGED_KERNELS, c) || continue
        MP % Mantle.gemm_bm(c) == 0 && CRSP % Mantle.gemm_bk(c) == 0 || continue
        Mantle.gemm_aliasing(c, CRSP) && continue
        bn = Mantle.gemm_bn(c)
        np = cld(Cout, bn) * bn
        np <= Cout * CONV_COUT_PAD || continue
        score = bn / np
        score > bestscore && ((best, bestscore) = (np, score))
    end
    best
end

"""
How many columns of zero a convolution may pad its output channels with to reach
a wider GEMM tiling. `2.0`: `Cout = 144` wants 256, which is 1.78x, and nothing
in `gen/graphs` asks for more than that.
"""
const CONV_COUT_PAD = 2.0

"""
    IM2COL_CAP
    im2colbudget(x) -> Int

How much im2col matrix one chunk may hold, in bytes.

**This used to decide whether a convolution ran on the tensor cores at all**,
and it no longer does: `conv_coopmat_plan` divides the pixel axis by it instead
of refusing, so a convolution whose whole im2col would be gigabytes runs in
pieces through a buffer this size. What the number now chooses is the size of
that piece, which is a speed question and a much easier one.

**256 MiB**, which is the smallest chunk that saturates. Swept end to end on
the Qwen-Image 2.1 VAE decode, everything else equal:

    cap      decode (min / median)
     32 MiB   6.08 / 6.19 s
     64       6.08 / 6.26
    128       5.82 / 5.91
    256       5.34 / 5.39          <- here
    512       5.37 / 5.39          saturated

The ISOLATED kernel prefers a far smaller chunk — the VAE's widest
convolution, `288 -> 288` over 1024x1024, measures 201.6 ms at 32 MiB against
234.7 at 512, because the GEMM reads the chunk back while it is still in cache
— and the decode disagrees because it runs interpreted, where each of the three
passes a chunk costs is a host launch and there are forty-five convolutions
paying it.

**Not uniformly, though, and the exception is the shape that costs the most.**
Re-measured on RDNA 3.5 in 2026-09 with 16 MiB against 256, interleaved, five
rounds, minima: `1152 -> 1152 @ 256²` 87.8 against 90.9 (1.04x for the small
chunk), `576 -> 576 @ 512²` 104.1 against 118.7 (1.14x), but `288 -> 288 @
1024²` **193.2 against 163.1 — 0.84x**, and that one is the single most
expensive convolution in the decode. The staged GEMM re-reads the whole chunk
once per `bn`-wide column block (measured: ~1.74 ms per block at 145 GB/s,
independent of `N`), so the shapes with many column blocks are the ones a
cache-resident chunk pays off for — 9 blocks at `Cout = 1152`, 5 at 576, 3 at
288. A per-shape chunk would beat one number for everything; a different global
number would not. A recorded graph records them once and would rather have the small
chunk and the small arena; 256 MiB is within noise of the best for both.

For reference, that convolution on the implicit-GEMM kernel is **1482 ms at
1.06 TFLOP/s** against 218 ms here, and the whole decode goes **12.24 s to
5.34**.

`im2colbudget` is the smaller of this and a quarter of the VRAM the driver says
is free, and it is what keeps a busy card from being asked for a chunk it
cannot give — a smaller chunk is a slower convolution now, not a refused one.
When the driver has no `VK_EXT_memory_budget` (`budget == 0`), it is this.
"""
const IM2COL_CAP = Ref(256 << 20)

function im2colbudget(x)
    # The context comes from the OPERAND, not from a backend or a `DeviceCaps` —
    # the same reason `coopmat_gemm!` uses `get_backend(C)`: an unpinned backend
    # resolves through the global context, so on a second device the budget read
    # would describe the wrong GPU.
    ctx = islavaarray(x) ? Mantle.vk_context(x) : nothing
    ctx === nothing && return IM2COL_CAP[]
    free = 0
    for h in Mantle.probe_device_memory_budget(ctx)
        h.device_local || continue
        h.budget == 0 && return IM2COL_CAP[]      # extension absent
        free = max(free, h.budget - h.usage)
    end
    min(IM2COL_CAP[], free ÷ 4)
end

"""
    GEMM_BLOCK

The row count the im2col matrix is padded to: `lcm` of every `GEMM_TILINGS` block
height, so **some** tiling always divides it.

`padtile` pads to the cooperative-matrix tile, 16, and that is not enough.
`Mantle.gemm_tiling` takes the first tiling whose block DIVIDES the shape exactly
(`gemm.jl:185`) and the blocks are 96/64/32 — so a `padtile`d 3400 becomes 3408,
which none of them divide, and the GEMM inside the convolution silently drops to
the register-blocked kernel.

Measured on the shape that exposed it, `L=3400 C=256 k=7`: the lifted coopmat
path was **1.12 TF/s against the direct kernel's 0.88** — a 1.27x that looked
like "im2col traffic eats the gain" and was nothing of the sort. The same GEMM
with `M` padded onto a tiling runs at 19.7 TF/s.

`padtile(20401) = 20416` happens to be a multiple of 64, which is why the longest
convolutions looked fine and the mid-size ones did not; the difference was luck,
not shape.

At most 127 extra rows against thousands, and the im2col kernel already zero-fills
past `NPQ`, so the padding costs a fraction of a percent and needs no new code.

**128 and not 192, since `GEMM_TILINGS` gained a 128-row block.** This was the
LCM of the blocks a padded `M` could land on when those were 96/64/32.
llama.cpp's AMD cooperative-matrix `l_warptile` put a 128-row entry in the
table; `192 % 128 == 64`, so every convolution in this tree was locked out of
the faster block by THIS number. Forced per tiling on the VAE's four GEMM
shapes, with `M` rounded so both blocks apply (TFLOP/s):

    conv            M       N      K    128x128   96x128   64x128
    144 -> 144   95232    256   1312     15.60    15.61    14.55
    288 -> 288   50304    384   2592     18.92    17.81    14.65
    576 -> 576   24192    640   5184     21.86    17.48    15.48
    1152 ->      10752   1152  10368     20.08    18.93    16.52

**Not the LCM, which is 384 and is wrong here.** 128 is enough because 64 and 32
divide it, so every smaller block in the table is still reachable as a fallback;
only 96 is not, and 96 is dominated by 128 on all four shapes above. Taking the
LCM instead makes the pad coarser than small convolutions can afford — a
`16x16` output is `NPQ = 256`, which cannot round to 384 without blowing the
im2col budget, and those convolutions fell off this path to the scalar kernel
at about one TFLOP/s. `test_conv_coopmat_chunk.jl` caught it by losing twenty
assertions to `Decline(:im2colsize)` while still reporting green.
"""
const GEMM_BLOCK = 128

"Round `n` up to a multiple of [`GEMM_BLOCK`](@ref)."
@inline padgemm(n::Integer) = cld(n, GEMM_BLOCK) * GEMM_BLOCK

"""
The im2col matrix, `MP x CRS` column-major, zero past `NPQ` and outside the
input. `MP` is static so the destination stride folds into the address.

The index arithmetic is deliberately `Int32`. Julia hands out `Int64` indices
and Lava emits them as-is, but NVIDIA has no native 64-bit integer unit — adds
and compares are emulated and *division* especially so. This kernel does four
divisions and two remainders per element, which made it the worst case in the
model: 0.038 ms to write 590 KB. Measured on a tight loop, narrowing the
counter alone was worth 1.56x (2100 -> 3282 GFLOP/s). Every extent here is far
inside `Int32`; `MP * CRS` for the largest convolution we take is 17.7M.
"""
# `p0` is the first pixel this chunk covers and `NPQ` how many of them it has,
# so the whole matrix is the single chunk `p0 = 0, NPQ = plan.NPQ`. See
# `ConvCoopMatPlan.rows`.
@kernel function im2col_kernel!(col, @Const(x), ::Val{MP}, ::Val{VEC},
                                ::Val{KW}, ::Val{KH}, ::Val{SX}, ::Val{SY},
                                ::Val{PX}, ::Val{PY}, ::Val{DX}, ::Val{DY},
                                ::Val{OW}, ::Val{OH},
                                Wid, Hei, NPQ, ntot,
                                Cin, p0) where {MP,VEC,KW,KH,SX,SY,PX,PY,DX,DY,OW,OH}
    # Flat launch, like `conv_epilogue_kernel!`: a 2-D `ndrange` is partitioned
    # into 2-D workgroups, so a warp spans only a handful of consecutive `m` and
    # the writes to `col` (which is `m`-major) are fragmented.
    #
    # `OW` and `OH` are `Val`s and not arguments because the decomposition of a
    # pixel index below divides by `OW` and by `OH*OW`. As runtime values those
    # are two REAL integer divisions per element and this device has no integer
    # divide; as literals LLVM turns each into a multiply and a shift. Measured
    # against the identical kernel with the two as arguments, interleaved in one
    # process, one chunk each, **output compared and bit-identical**:
    #
    #     Cin    shape      Val      runtime
    #      144   1024²    3.506 ms   4.353 ms   -19.5%
    #      288   1024²    3.712      4.393      -15.5%
    #      576    512²    3.551      4.291      -17.3%
    #     1152    256²    3.639      4.327      -15.9%
    #     1152    128²    3.610      4.346      -16.9%
    #
    # RADV agrees on the mechanism: 239 instructions against 341, 50 scalar ALU
    # against 86. im2col is about half of this convolution, so it is worth
    # ~8-10% of the whole.
    #
    # It costs no extra specialisation: `MP` is already a `Val` and is derived
    # from `NPQ` and `CRSP`, so a shape differing in `OW` or `OH` differs in
    # `MP` too and was compiling its own kernel regardless.
    #
    # **Do not try to confirm this by timing `convolution!` across two
    # processes.** The whole convolution allocates a multi-GB im2col scratch and
    # its wall time on this machine swings ~40% run to run — the same shape
    # measured 35.48, 51.10, 37.17 and 36.89 ms in one session. A cross-process
    # A/B of the convolution reported this change as +30% on one shape and -8%
    # on another; both were noise. The table above is interleaved in one process
    # and is what the change is worth.
    lin = @index(Global, Linear)
    # `return` is not permitted in a KernelAbstractions kernel; guard instead.
    if (lin - 1) * VEC < ntot
        @inbounds begin
            T = eltype(col)
            q0 = (Int32(lin) - Int32(1)) * Int32(VEC)
            m0 = q0 % Int32(MP)
            c = q0 ÷ Int32(MP) + Int32(1)
            # The column decomposition is per COLUMN, so a thread that writes
            # `VEC` consecutive rows does it once rather than `VEC` times. That
            # is half of what the pairing buys; the other half is the store
            # width.
            crs = c - Int32(1)
            kw = crs % Int32(KW)
            t = crs ÷ Int32(KW)
            kh = t % Int32(KH)
            cin = t ÷ Int32(KH)
            # `col` has `CRS` rounded up to the tile, so the last few columns
            # have no input channel behind them. They stay zero, which makes
            # them contribute nothing to the product — the same trick the
            # `m > NPQ` rows already use. See `convolution_coopmat!`.
            inch = cin < Int32(Cin)
            base = Int32(MP) * (c - Int32(1))
            Base.Cartesian.@nexprs 2 j -> begin
                if j <= VEC
                    m_j = m0 + Int32(j - 1)
                    v_j = zero(T)
                    if m_j < Int32(NPQ) && inch
                        npq_j = m_j + Int32(p0)
                        n_j = npq_j ÷ Int32(OH * OW)
                        r_j = npq_j - n_j * Int32(OH * OW)
                        oh_j = r_j ÷ Int32(OW)
                        ow_j = r_j - oh_j * Int32(OW)
                        ix_j = ow_j * Int32(SX) - Int32(PX) + kw * Int32(DX)
                        iy_j = oh_j * Int32(SY) - Int32(PY) + kh * Int32(DY)
                        if Int32(0) <= ix_j < Int32(Wid) && Int32(0) <= iy_j < Int32(Hei)
                            v_j = T(x[ix_j + Int32(1), iy_j + Int32(1),
                                      cin + Int32(1), n_j + Int32(1)])
                        end
                    end
                    col[base + m_j + Int32(1)] = v_j
                end
            end
        end
    end
end

"""
How many consecutive rows of `col` one thread writes.

**Two.** `col` is `Float16`, so one element a thread is a 2-byte store — half
of what a lane can retire — and the column decomposition above it (four
divisions and two remainders) is paid per element rather than per column. A
pair is a 4-byte store and one decomposition. The unroll is two because four
and eight were measured and lose: at stride 1 a pair reads two adjacent `x`
elements, a quad reads four and starts crossing lines for no extra store width.

Interleaved over the WHOLE convolution, alternating the two forms in one
process, minimum of four rounds each:

    Cin -> Cout   spatial     scalar   paired
     288 -> 288   1024x1024   183.24   168.85   1.09x
     144 -> 144   1024x1024    79.33    69.12   1.15x
     288 -> 144   1024x1024   160.63   146.01   1.10x
     576 -> 576     512x512   128.48   117.66   1.09x
    1152 -> 1152    256x256   109.05   106.59   1.02x
    1152 -> 1152    128x128    27.22    25.66   1.06x

and `tools/gap_vs_rocm.jl` from a fresh process either way puts those six at
682.70 ms against 613.32, **1.11x** on the convolution family.

**The isolated kernel says 1.14x to 1.37x and that number is wrong.** Timing
`im2col_kernel!` on its own re-runs the SAME chunk, so `x` and much of `col`
stay resident and the pairing is measured against a warm cache; the real loop
advances `p0` and every chunk is cold, where the pass is much closer to what
DRAM will give and the pairing is worth roughly half as much. The first A/B
run this way reported the change as worth nothing at all for a different
reason — Revise had silently not applied the file — so both numbers here come
from the two kernels living side by side in one process, which is the only
form that was reproducible. Output is bit-identical either way, which
`test_conv_coopmat_chunk.jl` pins.

`padgemm` rounds `MP` to a multiple of `GEMM_BLOCK = 192`, so 2 always divides
it and a pair never straddles a column. The launcher asks anyway rather than
relying on it.
"""
const IM2COL_VEC = 2

"""
Scatter the `MP x Cout` GEMM result back into `(OW, OH, Cout, N)` and add the
bias. The GEMM's leading dimension is the padded pixel count, so this is not a
reshape — the rows past `NPQ` are dropped here.
"""
# One lane per output element, in the destination's own linear order.
#
# Launched flat rather than over `(OW, OH, Cout, N)`. A 4-D `ndrange` makes
# KernelAbstractions partition the index space into 4-D workgroups, and
# consecutive lanes then do not write consecutive memory, the same defect that
# costs Lava's broadcast a factor of six. Measured by running the phase twice
# and differencing: this kernel cost 2.9 ms a step, more than the tensor-core
# GEMM it follows (1.9 ms).
#
# Recovering `(pixel, channel, image)` from the linear index costs two integer
# divisions, done in `Int32` because NVIDIA emulates 64-bit integer division.
# A 3-D range over `(pixels in this chunk, channel, image)`, so nothing here
# divides by a runtime extent. The flat form did `% P`, `% Cout` and `÷ Cout`
# per thread over every output element, and a chunk's pixels are a range rather
# than the whole of `P` — which the flat index cannot say without another
# division. `p0` is the chunk's first pixel; `Cout` is the weight's own channel
# count and `MP * CoutP` the padded plane `C` was written at.
@kernel function conv_epilogue_kernel!(out, @Const(C), @Const(bias), ::Val{MP},
                                       ::Val{ACT}, ::Val{SPLITK}, ::Val{NIMG},
                                       P, Cout, npqc, p0, plane) where {MP,ACT,SPLITK,NIMG}
    pc, kc1 = @index(Global, NTuple)
    if pc <= npqc
        @inbounds begin
            pixc = Int32(pc) - Int32(1)     # 0-based pixel within the chunk
            kc = Int32(kc1) - Int32(1)      # channel, 0-based
            npq = Int32(p0) + pixc          # 0-based row of the whole im2col
            # `npq` runs over pixels AND images, so the image it belongs to is a
            # division — except at `N == 1`, which every graph here is, where it
            # is none.
            img = NIMG == 1 ? Int32(0) : npq ÷ Int32(P)
            pix = NIMG == 1 ? npq : npq - Int32(P) * img
            lin = Int32(1) + pix + Int32(P) * kc + Int32(P) * Int32(Cout) * img
            i = Int32(1) + pixc + Int32(MP) * kc
            # The split-K planes are summed here rather than by a separate
            # reduction pass: this kernel already touches every output element.
            v = C[i]
            for s in Int32(1):Int32(SPLITK - 1)
                v += C[i + s * Int32(plane)]
            end
            bias === nothing || (v += Float32(bias[kc + Int32(1)]))
            # Safe here in a way it is not in `conv2d_igemm!`: the splits have
            # been summed by the time the activation is applied.
            ACT === :relu && (v = max(v, zero(v)))
            out[lin] = eltype(out)(v)
        end
    end
end

"""
    convolution_coopmat!(ctx, out, x, w, bias, stride, padding, dilation) -> out

Tensor-core path. `conv_coopmat_plan` decides whether it can run, and hands the
padded extents over rather than leaving them to be recomputed here.
"""
function convolution_coopmat!(ctx, out, plan::ConvCoopMatPlan, x, w, bias, stride,
                              padding, dilation; act::Symbol=:none)
    KW, KH, Cin, Cout = size(w)
    Wid, Hei = size(x, 1), size(x, 2)
    OW, OH, _, N = size(out)
    CRS = plan.CRS
    NPQ = plan.NPQ
    # The reduction axis is padded to the tile the same way `NPQ` already is.
    # `CRS` is the weight's own extent, so refusing to pad it keeps SAM 2's stem
    # (`7x7x3`, `CRS = 147`) on the implicit-GEMM kernel at **1.01 TFLOP/s**.
    # The two halves of the pad:
    # `im2col` writes zeros for the columns with no input channel behind them,
    # and the weight gets a zeroed copy that is `CRSP` rows tall. Zero times
    # anything is zero, so the padded product is the real one — but the weight's
    # pad has to be *written*, not merely reserved: reading uninitialised memory
    # there would multiply a possible NaN by zero and give NaN.
    # `plan.CRSP`, never recomputed: the plan chooses between the `bk` pad and
    # the tile pad by waste budget (`crsextent`), so deriving it again here would
    # silently disagree with the shape the plan was accepted for.
    CRSP = plan.CRSP
    CoutP = plan.CoutP
    backend = ctx.backend
    # `rows` pixels at a time, which is `NPQ` whenever the whole matrix fits.
    ROWS = plan.rows
    MP = padgemm(ROWS)

    col = scratch!(ctx, Float16, MP, CRSP)

    # Untouched when `CRS` is already on the tile AND the channels need no
    # widening, which is every convolution that took this path before — same
    # buffer, no copy, no extra scratch.
    B = if CRSP == CRS && CoutP == Cout
        w
    else
        wp = scratch!(ctx, Float16, CRSP, CoutP)
        fill!(wp, zero(Float16))
        copyto!(view(wp, 1:CRS, 1:Cout), reshape(w, CRS, Cout))
        wp
    end

    _, splitk = Mantle.coopmat_gemm_shape(MP, CoutP, CRSP)
    # With a split there is no separate destination: the epilogue reads the
    # partial planes directly and sums them.
    C = scratch!(ctx, Float32, MP, CoutP, max(splitk, 1))

    for p0 in 0:ROWS:(NPQ - 1)
        npqc = min(ROWS, NPQ - p0)
        vec = MP % IM2COL_VEC == 0 ? IM2COL_VEC : 1
        im2col_kernel!(backend)(col, x, Val(MP), Val(vec),
                                Val(KW), Val(KH), Val(stride[1]), Val(stride[2]),
                                Val(padding[1]), Val(padding[2]),
                                Val(dilation[1]), Val(dilation[2]),
                                Val(OW), Val(OH),
                                Wid, Hei, npqc, MP * CRSP, Cin, p0;
                                ndrange = cld(MP * CRSP, vec))
        Mantle.coopmat_gemm!(C, col, B, MP, CoutP, CRSP; partials = C, reduce = false)
        conv_epilogue_kernel!(backend, (256, 1))(
            out, C, bias, Val(MP), Val(act), Val(splitk), Val(N),
            OW * OH, Cout, npqc, p0, MP * CoutP;
            ndrange = (cld(npqc, 256) * 256, Cout))
    end
    out
end
