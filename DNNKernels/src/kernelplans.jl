"""
The kernel plans: one type per implementation an entry point can choose.

`kernel-library-review.md` findings 1 and 2. A plan is the whole decision — "this
call takes the fused cooperative-matrix kernel, at 64x32 with eight subgroups" —
carried as a value instead of recomputed. The entry points dispatch on the type
(`sdpa!(ctx, ::FlashCMPlan, …)`, `matmul!(ctx, ::MMCoopMatPlan, …)`), so a new
path is a new type and a new method rather than another branch.

They live together in one file, above every kernel, for a plain reason: a method
signature naming a type needs that type to exist when the method is DEFINED, and
the entry points are spread across `attention.jl`, `matmul.jl` and `conv.jl`,
each of which is included before at least one of the files that would otherwise
own its plan. The *constructors* stay with their kernels, where the measurements
that justify each field are.

Each is built on a live `M.DeviceCaps` and holds nothing cached at module scope
(`GUARDRAILS.md` §8), so a plan is per call and per device by construction.
"""

"""
    Decline(reason)

A refused plan, and *why*.

`nothing` says only that something did not apply, and a caller that wants to
react — or a test that wants to assert *which* rule fired — cannot. Two of those
reactions are live already: `:wrapped` is recoverable by densifying the operands
(see [`sdpa`](@ref)), and a refusal that exists because of a **known bug** must
say so rather than looking like a design decision, which is what
`blockfor`'s refusal of non-square attention needs when it lands here.
"""
struct Decline
    reason::Symbol
end

"""
    coopmatkernels(dev::M.DeviceCaps) -> Bool

Whether a kernel in this tree may use Mantle's **cooperative-matrix intrinsics**
on this device, and whether Mantle's own staged cooperative-matrix GEMM
(`coopmat_gemm_shape`, `coopmat_gemm_dispatch!`, the `GEMM_TILINGS` table behind
them) exists for it.

One predicate for both, because they are one requirement. `coopmat_load` lowers
to an `llvmcall` whose external symbol is `_lava_coopmat_load_f16_16x16_a`: the
shape is in the NAME. Every kernel here that multiplies matrices — the staged
GEMM, `conv_coopmat`, `flashcm`, `coopmat_sdpa` — is emitted against that one
instruction.

NOT `dev.coopmat`. That flag is a fact about the *hardware*, and `DeviceCaps`
says so in its own words: it is "a floor, not a promise about every operation …
a kernel that wants [something narrower] asks separately and carries the
answer". This is that separate question. A device that multiplies 8x8 matrices
has cooperative matrices and none of these kernels, and `Mantle.coopmat_load` is
not merely unavailable there — it is not defined at all, because it is exported
from the tree that implements it.

The two were the same answer for as long as one backend existed, so the planners
read `dev.coopmat` and then divided by `dev.tile`. On a device with 8-wide
matrices that ADMITTED the plan and reached `Mantle.GEMM_TILINGS`, and a
convolution died with an `UndefVarError` rather than taking the implicit-GEMM
kernel beside it.

Pinning the tile also closes a hazard Mantle documents from the other side. Its
`caps` reads `tile` from the driver's shape table precisely because a constant 16
was wrong: "on a card reporting anything but 16 the kernels strided by 16 and
read another fragment's registers, and nothing would have crashed." The staged
kernels are still emitted at 16. So a device whose best square is not 16 has no
kernel here either, and this says so instead of striding.

`isdefined` is the third term, and is not reflection standing in for a capability.
The staged kernels are DEFINED in the tree that implements them, so on a build
without that tree `Mantle.GEMM_TILINGS` and `Mantle.coopmat_load` are not an empty
table and an unavailable intrinsic — they are not bindings at all. The tile alone
makes this a claim about the DEVICE that is only conditionally a claim about the
LIBRARY, and a hand-built 16-wide `DeviceCaps` then admits a plan that reaches an
undefined name two frames later. It is the guard the runners put on
`use_frozen_kernels` for the same reason: nothing to ask is not an error, it is no
kernel. Asked once per plan.
"""
coopmatkernels(dev::M.DeviceCaps) =
    dev.coopmat && dev.tile == COOPMAT_TILE && isdefined(M, :GEMM_TILINGS)


"""The one tile extent every cooperative-matrix kernel reachable from here is emitted
at, which is `Mantle.GEMM_TILE` and the `16x16` in `_lava_coopmat_load_f16_16x16_a`.
Restated rather than imported: the constant lives in the tree that defines those
kernels, and the whole point of [`coopmatkernels`](@ref) is to be answerable where
they do not exist."""
const COOPMAT_TILE = 16

"""
    FlashCMPlan

Everything the cooperative-matrix flash kernel needs to launch, decided once.

One plan, and not two predicates that have to agree: a `Bool` that runs
[`flashcm_tiling`](@ref) and throws the answer away can disagree with the tiling
the launch then computes. `sdpa`
then ran `flashcm_tiling` **again** for the tiling; and `sdpaflashcm!` re-checked
six more conditions and returned `false` from six places to mean "declined" —
*after* the caller had already allocated `out`. The symptom of the two drifting
apart is not an error, it is a **silent change of algorithm**, which is the kind
of bug this project has repeatedly spent days on.

Now there is one decision, one place, one object, and
[`sdpaflashcm!`](@ref) cannot decline: if you hold a `FlashCMPlan`, it runs.
"""
struct FlashCMPlan
    BR::Int
    BC::Int
    NW::Int
    # NW * dev.coopmatsubgroup — the width a cooperative-matrix module actually
    # runs at, which Lava pins to 32, NOT `dev.subgroup` (the device default, 64
    # on RDNA 3.5). Derived and never hardcoded to `NW * 32`: on wave64 hardware
    # the two disagree.
    NT::Int
    E::Int
    EP::Int          # E rounded up to the cooperative-matrix tile

    # ── Pad and mask a block the extents do not divide.
    #
    # Let a sequence length that does not divide the tile take the fused path, by
    # padding it and masking what is past the end.
    #
    # `flash_attn_cm1.comp` carries this as its `Clamp` spec constant and
    # `KV_bounds_check`; our port dropped it because SAM 2's **encoder** shapes all
    # land on 16. The **decoder's** do not: every one of its seven attentions has a 23
    # in it — the mask prompt's token count — and that is why they run the three-pass
    # path at 39.8 GFLOP/s, about 0.2% of what this kernel reaches on the encoder.
    #
    # Masking is not optional decoration. Staging already writes zeros past the end,
    # which gives a padded key a score of zero and therefore a weight of `exp(-m)` —
    # small, but not nothing, and it would land in `l` and in `O`. The softmax has to
    # exclude those keys from the maximum and write their weight as an explicit zero.
    #
    # **Off by default, and the default is a measurement not a preference.** Turning it
    # on lets the *encoder's* seven leftover calls onto the fused path too, and they
    # are tiny on **both** axes — `Lq` of 4 and 16 against `Lk` of 16 and 64 — so the
    # padding buys nothing and costs launches. Encode measured 133.7 -> 137.97 ms with
    # it on and no occupancy floor, and still 135.5 with a half-tile floor. The
    # decoder's shapes are the opposite: a 23 against a 4096, so the padded axis is
    # tiny beside the work it amortises.
    #
    # A single global cannot express "clamp the decoder, not the encoder", and a
    # heuristic tuned against two workloads is a heuristic that will be wrong on the
    # third. So this ships **off**, the capability tested, and turns on together with
    # `--decoder-precision autocast` — the change that makes the decoder's attention
    # fp16 and therefore eligible at all. Neither is any use without the other.
    #
    # Costs nothing when off: `CLAMP` is a `Val`, so the encoder compiles the same code
    # it did before.

    clamp::Bool

    # ── Keep `O` in registers rather than shared memory.
    #
    # Where `O` lives across key blocks: `true` puts it in registers, `false` in
    # shared memory as the value product's accumulator.
    #
    # **`false`, and that is the opposite of the reference.** `flash_attn_cm1.comp`
    # keeps `O` in registers and round-trips each `P·V` tile through shared memory to
    # add it. The shared form does twice the shared traffic on paper — a rescale pass
    # that reads and writes `BR x EP`, plus an accumulator load on top of the store,
    # 80 KB a key block against 40 — and loses anyway. See the measurement in
    # `FLASHCM_TILINGS`.
    #
    # Kept as a switch rather than deleted, because the paper argument for registers is
    # sound. Interleaved, one session, both forms of the same kernel:
    #
    #     tiling      floats/thread    shared      registers
    #     64x32/8w         20          5.032 ms     6.347 ms     <- shipped
    #     64x32/8w         20          0.457        0.537
    #     32x32/8w         10          7.284        6.696
    #     32x32/8w         10          0.567        0.560
    #
    # **The register form wins at `32x32` and loses at `64x32`, crossing where
    # `BR*EP/NT` goes from 10 floats a thread to 20.** The obvious reading is register
    # pressure, and the driver says it is not:
    # `VK_KHR_pipeline_executable_properties` reports **128 registers for the shared
    # form and 122 for the register form, stack size 0 in both** — the register form
    # uses *fewer* registers and neither spills.
    #
    # The `splitidx` per element in the sweep is not the cost: the row index is
    # loop-*invariant* whenever
    # `NT` is a multiple of `BR`, which every shipped tiling is — `idx = tid + (s-1)*NT`
    # gives `idx % BR == tid % BR` for every `s` — and `BR` is a power of two, so the
    # call was a mask, not a division, and the compiler was recomputing a constant.
    # Hoisting it out of the loop is free and changes **nothing**: the register form
    # still loses by 9.5% on the encoder (107.12 ms against 117.32, interleaved, one
    # session). An attribution that survived because nobody removed the thing it named.
    #
    # What it actually costs is a pass that the shared form does not have at all. With
    # `O` in shared, `pvs` **is** the cooperative-matrix accumulator: each key block
    # loads it, `MulAdd`s into it and stores it back, so accumulating across blocks is
    # free inside the store, and the rescale on top is *lazy* — skipped entirely on the
    # third of blocks where no row's maximum grew. With `O` in registers, `pvs` holds
    # one block's `P·V` and a separate, **unconditional** `BR x EP` sweep has to fold
    # it into the registers on every block, behind a barrier. The register form does
    # not save a pass, it adds one.
    #
    # So the reference is not wrong, it is answering a different constraint: in an LLM
    # decode `HSV` reaches 256 and shared memory is what binds, so `Of` has to live in
    # registers and the round-trip is the price of fitting. At 20 floats a thread
    # shared is not binding — 48 900 of ~100 KB per SM — and the accumulator is better
    # off where the cooperative-matrix store can reach it.
    #
    # **The crossing above is the rule, and `flashcm_plan` computes it**: `rego` is on
    # where `BR*EP/NT <= 10`. A device whose cooperative-matrix modules run narrower
    # than its wave takes a wider `NW` (see `flashcm_tiling`), which halves the floats
    # a thread at the same tile and puts `64x32` on the winning side of the same
    # crossing — measured there at 10.10 ms against 8.41. Nothing above is contradicted;
    # the tiling moved.

    rego::Bool

    # ── Hold `O` across the key loop and rescale it in place.
    #
    # Keep each subgroup's `O` tiles in cooperative-matrix accumulators for the whole
    # key loop, instead of loading and storing them through shared memory every block.
    #
    # `P·V` was a third of the kernel and almost none of that was arithmetic: it has
    # `RT*ET` accumulator tiles against the score product's `RT*CT`, each given only
    # `CT` muladds to amortise a load and a store — five times the accumulator traffic
    # for the same FLOPs. Held, the muladds add straight into the tile the subgroup has
    # been carrying, and the load and store are simply not issued.
    #
    # Three accumulators a subgroup, because `cld(RT*ET, NW)` is 3 at every tiling
    # [`flashcmfits`](@ref) admits (it refuses a fourth rather than silently dropping
    # tiles, since `@nexprs` needs a literal count).
    #
    # **Off, and the reason is worth more than the switch.** A version that never
    # rescales — `heldacc()` in `tools/attn_lab.jl` — measures **+31.1% / +21.4%**
    # (3.025 ms against 4.393, and 0.334 against 0.425), which is the prize this idea
    # is chasing and it is real. The version that does rescale loses:
    #
    #     shape        shared O   held O   registers (shared -> held)
    #     4096x4096    4.386 ms   4.950     123 -> 192
    #     256x256      0.423      0.506
    #
    # **The cause is register pressure**, from `Mantle.pipeline_exec_stats`: the
    # rescale costs **+69 registers**, 123 to 192. At 256 threads that is 49 152
    # of the SM's 65 536, so **one workgroup per SM instead of two**. Stack size
    # is 0 and local memory 16 bytes in both, so nothing spills — the component
    # access materialises a second copy of each held tile, and register
    # allocation is static.
    #
    # Neither the barriers nor the spills explain it. Scaling the tile where it
    # lives with `Lava.coopmat_setcomp`, no flush and no barrier, measures 4.950
    # against the flush-and-reload form's 5.183, and emitting one store for a
    # chain of component accesses instead of one per access reads 4.950 against
    # 4.955. How OFTEN the rescale fires does not explain it either (67.3% of
    # blocks at `BR = 64`, 31.7% at `BR = 16`): allocation being static, a
    # rarely-taken expensive path halves occupancy exactly as much as an
    # always-taken one, and a per-row-tile flag would not help.
    #
    # Swept over every tiling `flashcmfits` admits, held only wins where the tiling is
    # itself slow — `32x16x8` at -2.8% and `32x32x4` at -11.9%, both against a shared-O
    # baseline 40% worse than `64x32x8`'s. `REGO` is not the way round it either: it
    # costs *fewer* registers than shared `O` (119 against 123) and is 44% slower, so
    # whatever it pays is not occupancy.
    #
    # **And the way to the 31% is closed too — measured, on real activations.** The
    # 31% needs a rescale that is free in *registers*, and the only such rescale is one
    # that never runs: fix each row's reference once so the running maximum never
    # grows. `p` is fp16, so a reference works exactly while it stays within ~11.09
    # nats of the row's true maximum — in *either* direction, since scaling a row of
    # `P` by any constant cancels in `O/l`. Two candidates, both measured over SAM 2's
    # encoder run on the reference image, on the two shapes that are 95% of the work:
    #
    # | reference | `|reference − true row max|`, nats | rows outside fp16's window |
    # |---|---|---|
    # | `scale·‖q_r‖·max_k‖k_k‖` (Cauchy-Schwarz), L=4096 | median 7.16, p99 14.11, max 17.40 | **20.4%** |
    # | the same, L=256 | median 6.49, p99 11.94, max 13.00 | 4.7% |
    # | the first key block's own maximum, L=4096 | median 4.57, p99 19.98, max 24.23 | **21.1%** |
    # | the same, L=256 | median 2.57, p99 16.33, max 20.81 | 12.3% |
    #
    # A fifth of the rows drift out of representable range either way, and a row that
    # does returns zeros. The online rescale is not overhead on this data, it is
    # load-bearing, and both cheap references are dead. Note the shape dependence
    # before generalising: the same Cauchy-Schwarz bound on the `L = 64` windows is
    # tight — median 0.98 nats, max 5.56 — so a measurement on the small blocks alone
    # would have said yes.
    #
    # **The rescale primitive was never the problem, and the prize is bigger than the
    # 31% above.** Three ways of applying the factor were built — `:comp`
    # (`getcomp`/`setcomp`), `:perelem` (`OpCooperativeMatrixPerElementOpNV`), and
    # `:fmul` (a stride-0 factor matrix and one component-wise `OpFMul`, which names no
    # component at all). All three are bit-identical to shared `O`, and all three land
    # within 1.5% of each other: **the primitive does not matter.**
    #
    # What matters is one number. Pinned clock, interleaved, `nrsc` rescaling only the
    # first N of the three held tiles (numerically wrong below 3 — it prices registers,
    # not results):
    #
    # | variant | registers | workgroups/SM | 4096x4096 |
    # |:--|--:|--:|--:|
    # | shared `O` (shipped) | 123 | 2 | 5.087 ms |
    # | held, 0 rescales | 127 | 2 | 4.150 (**-18.4%**) |
    # | held, 1 rescale | 128 | 2 | 4.093 (**-19.5%**) |
    # | held, 2 rescales | 231 | 1 | 5.575 (+9.6%) |
    # | held, 3 rescales | 172 | 1 | 5.601 (+10.1%) |
    #
    # A step function at exactly 128 registers — `128 * 256 * 2 = 65536`, the file
    # size. Residency is *measured*, not derived: an atomic entry/exit probe at this
    # kernel's 48904-byte shared footprint reports 97 concurrent workgroups on 48 SMs,
    # so shared memory allows two and registers are what decide.
    #
    # So holding `O` is worth **-19.5%** for as long as the allocator stays under 128,
    # and the entire question is how to rescale three tiles without crossing it. Note
    # the count is not per-tile and not even monotone (128 -> 231 -> 172): this is an
    # allocation *decision*, not a countable set of live values, and nothing spills
    # (local memory 16 bytes, stack 0, in every row above).
    #
    # Two things that do NOT work, both measured: hoisting the factor matrix so one
    # serves all three tiles (correct — `t_j % RT == w % RT` for every `j` when `RT`
    # divides `NW` — but 220 registers, worse than reloading), and the register ballast
    # that would have priced this directly (the driver caps itself at 128 no matter how
    # many live values it is handed, which is itself the finding).
    #
    # Held `O` stays off until three rescales fit under 128. It is not a dead end; it
    # is the largest measured win left in this kernel.

    held::Bool

    # ── Which primitive applies the held rescale.
    #
    # How a held `O` gets its per-row factor: `:comp`, `:perelem` or `:fmul`.
    #
    # Inert unless [`FLASHCM_HELD`](@ref) is on, which it is not — see there for the
    # measured table, which is the reason. `:fmul` is the default of the three because
    # it is the only one that never names a component, needs no extra shared memory
    # (`cs` already holds the factors) and is plain SPV_KHR_cooperative_matrix rather
    # than `VK_NV_cooperative_matrix2`, so it is the one that also runs on AMD.
    #
    # `:perelem` requires [`flashcm_perelem_available`](@ref); selecting it on a device
    # without the extension will fail to compile rather than silently fall back.

    rescale::Symbol

    # ── One pass over the key block instead of two, and defer the
    # rescale to the write-out. Both won everywhere and are kept as
    # fields only because `test_flash.jl` A/Bs them.
    onepass::Bool
    lazyrescale::Bool

    # ── Splits of the KEY axis, i.e. flash-decoding. 1 disables it.
    #
    # `Lq = 23` against `Lk = 4096` is two query blocks, so the launch is
    # `2 * H*B` = 16 workgroups on 48 shader cores and the kernel is
    # latency-bound rather than anything else. Splitting the key axis gives every
    # split its own workgroup and merges the partial softmaxes afterwards.
    #
    # Measured on that shape, same arithmetic rearranged to the parallelism each
    # split count would produce (200 runs, min):
    #
    #     splits   Lk each   workgroups   min ms
    #       1       4096         16       0.4239
    #       4       1024         64       0.1589
    #       8        512         64       0.0948
    #      16        256        128       0.0721
    #
    # The count comes from llama.cpp's own rule (`ggml-vulkan.cpp`, the
    # flash-attention path), which aims at **two workgroups per shader core** —
    # not one, which is what our earlier grid floor assumed:
    #
    #     split = cores * 2 / workgroups_without_split
    #     chunk = roundup_pow2(Lk / split)      then re-derive split from chunk
    #
    # Re-deriving from the chunk is what keeps the key range a multiple of `BC`,
    # so the last split is not a ragged remainder.
    nsplit::Int

    # ── Two launches, because a partial last key block costs 40% of the kernel.
    #
    # Measured on an 8060S at `Lq = 4096`, `E = 128`, 32 heads, `BC = 16`: 50.8
    # ms at `Lk = 4096`, 53.5 at 4128, and 67-74 at EVERY length in between.
    # One masked column costs the same as fifteen, so it is not the masking
    # work: rewriting the mask as selects, clamping the load address so staging
    # is unconditional, and finally deleting every `KCLAMP` effect from the body
    # so that both cases compile the same program all left the cliff exactly
    # where it was. What it follows is `Val{KCLAMP}` itself.
    #
    # So the fix is to not ask for it on the bulk of the work: one launch over
    # the `div(Lk, BC)` blocks that fill a tile with `KCLAMP` off, one over the
    # ragged remainder with it on, and `attn_flash_cm_merge!` to combine them.
    # The second launch is one key block of 258 and can be as slow as it likes.
    #
    # Rides on `nsplit = 2` so that every caller's scratch allocation and the
    # merge dispatch already do the right thing; what differs is that the two
    # launches carry their own key ranges rather than halving one.
    tailsplit::Bool

    # ── Read K and V as cooperative matrices, rather than staging them.
    #
    # `OpCooperativeMatrixLoadKHR` takes a pointer in any storage class, so the
    # tile the tensor core wants can come straight from the tensor. What that
    # removes is a store to shared, a barrier, and a load, for both operands, on
    # every key block: **37.36 ms -> 25.56** at Qwen-Image 2.1's attention.
    #
    # It lives on the plan rather than being re-derived at launch because it also
    # decides the TILING. The table `flashcm_tiling` orders was measured with
    # staging, and staging is what made a wide workgroup pay: with the operands
    # read as tiles the same shape ranks the other way round, 128 threads over
    # 512 at every block width, and the two decisions cannot be allowed to
    # disagree about which kernel is being chosen for.
    globalkv::Bool
end

"""
    CoopMatSDPAPlan

The two-GEMM cooperative-matrix attention path: whether it applies, and with
what chunk. `kernel-library-review.md` finding 2, the same move `FlashCMPlan`
makes for the fused path — one decision, carried, instead of a `Bool` predicate
and a separate constant read six frames later.

## `chunk` — query rows per step

The score matrix is the whole memory cost of attention: `Lq x Lk` per (head,
batch), and this path needs it twice — fp32 out of the GEMM, fp16 into the next
one. At SAM 2's global blocks (Lq = Lk = 4096, 8 head-batches) that is 536 MB +
268 MB in one go, and it is why a single encode peaked over 11 GB where PyTorch
peaks at 1.6 (its SDPA dispatches to a fused kernel and never writes the matrix).

Chunking the QUERY axis costs nothing in arithmetic — same tiles, same tensor
cores, same total FLOPs — and divides the live score memory by `Lq / QCHUNK`.
Only the queries are chunked: the softmax reduces over KEYS, so a key-chunked
version would need the running-max rescaling that makes flash attention flash,
and flash measured 43.9 ms against this path's 4.08 on the same shape.

Rows are copied into a chunk-sized staging buffer rather than passed as a view,
because `coopmat_gemm!` conflates `M` with the leading dimension and a row slice
of a column-major matrix is not contiguous.

**2048 measured**, on SAM 2's global blocks (Lq = Lk = 4096, 8 head-batches):

    QCHUNK  chunks   p50 ms   min ms   S fp32 + P fp16
      512      8      400.1    398.5      101 MB
     1024      4      370.1    366.1      201 MB
     2048      2      352.4    350.0      403 MB
     4096      1      554.4    337.2      805 MB   <- OOMs mid-run and reclaims

Unchunked has the best floor and no ceiling: it needs 805 MB in one step, which
on a 20 GB card shared with a desktop pushes the pool into an OOM-retry that
dumps ~10 GB and costs more than chunking ever does. 2048 gives up ~13 ms
against that floor for 402 MB and, unlike it, is stable.

## `minl` — the sequence length below which this path is not worth taking

256. Below it the score matrix is small enough that the two extra passes over it
cost more than the tensor cores save, and the plain three-pass path wins.
"""
struct CoopMatSDPAPlan
    chunk::Int       # query rows per step, rounded onto the tile
    EP::Int          # head dimension padded onto the tile
    nbatch::Int
end

"""
    MMCoopMatPlan

The cooperative-matrix GEMM: whether it applies, and the padded `N` it runs at.
`kernel-library-review.md` finding 2, the same shape as `FlashCMPlan`.
""" 
struct MMCoopMatPlan
    NP::Int          # N padded onto the tile
    tile::Int
end

"""
    MMGemvPlan

The batch-1 matrix-vector path: `Mantle.gemv!` instead of a cooperative-matrix GEMM
with fifteen sixteenths of every tile empty.

Carries nothing but the marker. Unlike `MMCoopMatPlan` there is no padded extent
to record — a GEMV has no tile to land on — and the block/unroll choice is
`Mantle.gemv_ncontig_config`'s, made per device from the operands it is given.
"""
struct MMGemvPlan end

"""
    MMInt8Plan

The weight is a [`QInt8Matrix`](@ref). Carries nothing: `N == 1` goes to the
int8 GEMV and anything wider dequantises into the workspace and reuses the
fp16 GEMM, and both of those are decided from the operands.
"""
struct MMInt8Plan end

"""A ConvRot weight: rotate activations, then use the ordinary packed INT8 path."""
struct MMConvRotInt8Plan end

"""
    ConvCoopMatPlan

The tensor-core convolution: whether it applies, and the shape it runs at.

THREE extents are padded and each for its own reason. `CRSP` pads the reduction
so the staged GEMM has a `bk` that divides it. `CoutP` pads the output channels
onto a column tile, because the GEMM's rate falls off a cliff without one — see
[`convcoutpad`](@ref). `rows` is not padding at all: it is how many pixels one
im2col chunk covers, which is what lets a convolution whose whole im2col matrix
would be gigabytes run on the tensor cores anyway.

`gather` chooses between the two ways of producing the GEMM's A operand: write
the im2col matrix to memory, or compute each element where the GEMM reads it.
See [`ConvGather`](@ref) for what that is worth and when.
"""
struct ConvCoopMatPlan
    CRS::Int         # the weight's own reduction extent
    CRSP::Int        # …padded onto the tile
    Cout::Int        # the weight's own output channels
    CoutP::Int       # …padded onto a column tile the staged GEMM has
    NPQ::Int
    rows::Int        # pixels per im2col chunk; `NPQ` when it fits at once
    gather::Bool     # read the image in the GEMM instead of materialising im2col
end

"""
    StridedOperand{T,N,P}

A declared view as `(root, offset, strides)` rather than as storage of its own.

A kernel that takes each operand as a base plus per-axis strides reads a permuted
or sliced view IN PLACE. `attn_flash_cm!` is written that way — see the operand
paragraph in [`flashcm_plan`](@ref), which records what materialising k and v
instead cost — and so is `coopmat_gemm!` through `gemmstrides`.

`Mantle.ResourceView` cannot express this: it is `(parent, dims, offset)`, so it
names a contiguous window and nothing else. This is the strided form, and it is
DNNKernels' rather than Mantle's because only the kernels here consume it; a
resource still reaches every other dispatch dense.

`stridedroot` answers `(root, offset)` and `Base.strides` the strides, which is
exactly what `flashstrides` already asks of an array operand.
"""
struct StridedOperand{T,N,P}
    parent::P
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int                 # in ELEMENTS, from the root's first
end

StridedOperand(parent, dims::NTuple{N,Int}, strides::NTuple{N,Int},
               offset::Integer) where {N} =
    StridedOperand{eltype(parent),N,typeof(parent)}(parent, dims, strides, Int(offset))

Base.size(s::StridedOperand) = s.dims
Base.size(s::StridedOperand, d::Integer) = d <= length(s.dims) ? s.dims[d] : 1
Base.eltype(::StridedOperand{T}) where {T} = T
Base.ndims(::StridedOperand{T,N}) where {T,N} = N
Base.length(s::StridedOperand) = prod(s.dims)
Base.strides(s::StridedOperand) = s.strides
