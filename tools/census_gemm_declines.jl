# Which matmuls in the shipping models fall off the tensor-core path, and why.
#
# WHY THIS MATTERS. `mm_coopmat_plan` (DNNKernels/src/kernels/extern/matmul.jl)
# pads N internally via `Lava.gemm_padn`, but M and K are taken as-is:
#
#     size(A,1) % dev.tile == 0 && size(A,2) % dev.tile == 0 || return Decline(:extent)
#
# A ragged M or K therefore declines the WHOLE matmul to `Lava.mul!`, which has
# its own `% GEMM_TILE == 0` gate and drops to `gemmlaunch!`'s scalar kernel.
# Measured cost of being on that path, interleaved:
#
#     1280 x 1536 x 1280  (all %16)     0.134 ms   37.64 TF/s
#     1288 x 1544 x 1288  (none %16)    0.875 ms    5.86 TF/s      6.43x
#
# So a single ragged extent is worth ~6x on that matmul, silently. It does not
# come from transformer model dims — 768/1280/512 are all %16 — but from
# convolutions routed through im2col, where `K = Cin*kh*kw`. Three sources, and
# the census below found one I had not predicted:
#
#   * stems, where Cin is 3 or 4              (64,3,7,7) K=147
#   * one-channel heads, where Cout is 1      ragged M rather than K
#   * CONCAT-DERIVED channel counts           (1024,1090,1) K=1090
#
# The third is the one carrying real work. "K = Cin*kh*kw is a multiple of 16
# whenever Cin is" is true and led me to conclude only stems could be ragged; it
# misses that a concat makes Cin ITSELF 1090 or 514. One stem was already fixed
# by hand for 2.44x; this asks how many more there are.
#
# HOW IT COUNTS, and why it needs no instrumentation. Rather than patch
# `mm_coopmat_plan`, run a step and look at which GEMM kernels the device
# actually compiled. A scalar GEMM pipeline in `Lava.kernel_stats` after a step
# means something declined — the kernel cannot exist otherwise.
#
# THE NEGATIVE CONTROL, which this tool would be worthless without: a shape known
# to decline is pushed through first, and the census must SEE it. Without that,
# "no declines found" is indistinguishable from "the detector does not work" —
# the exact failure mode that made an earlier audit here report all-clean because
# its pattern matched every line.
using Lava, KernelAbstractions, Printf, LinearAlgebra
const KA = KernelAbstractions

"""Names of GEMM pipelines that only exist when the tensor-core path declined."""
const SCALAR_GEMM_KERNELS = ("strided_gemm_kernel", "scalar_gemm_staged_kernel")

"""
    gemm_kernel_census(ctx) -> (scalar, coopmat)

Compiled pipeline names split into the ones that mean "declined" and the rest.
Read after a step; `Lava.clear_kernel_cache!()` before it if you want the census
to describe that step alone.
"""
function gemm_kernel_census(ctx)
    scalar = String[]
    coopmat = String[]
    # `kernel_stats` takes a LavaLinkedKernel, NOT a context — the cache is
    # `linked_kernel_cache(ctx)`, a Dict of key => LavaLinkedKernel.
    #
    # And the field to read is `source`, NOT `name`: `name` is the SPIR-V entry
    # point, which is `"main"` for every kernel Lava emits. Reading `name` made
    # this census report zero declines for a matmul that had visibly declined —
    # caught only by the planted control below, which is why it exists.
    for (_, linked) in Lava.linked_kernel_cache(ctx)
        s = Lava.kernel_stats(linked)
        nm = isempty(s.source) ? s.name : s.source
        low = lowercase(nm)
        (occursin("gemm", low) || occursin("matmul", low)) || continue
        push!(any(t -> occursin(t, nm), SCALAR_GEMM_KERNELS) ? scalar : coopmat, nm)
    end
    (unique(scalar), unique(coopmat))
end

"""
    control_declines(back) -> Bool

Push a deliberately ragged fp16 matmul through `mul!` so the census has
something it MUST detect. Returns whether a scalar GEMM kernel appeared.
"""
function control_declines(back)
    ctx = Lava.vk_context()
    A = KA.allocate(back, Float16, 200, 184); fill!(A, Float16(0.01))
    B = KA.allocate(back, Float16, 184, 216); fill!(B, Float16(0.01))
    C = KA.allocate(back, Float32, 200, 216); fill!(C, 0f0)
    mul!(C, A, B)
    KA.synchronize(back)
    !isempty(first(gemm_kernel_census(ctx)))
end

"""
    report_census(label, ctx)

Print the split. A `Decline` shows up as a scalar GEMM pipeline; the shapes
themselves need `mm_coopmat_plan` instrumentation, which this deliberately does
not add — the first question is whether ANY model declines at all.
"""
function report_census(label, ctx)
    scalar, coopmat = gemm_kernel_census(ctx)
    @printf("%-16s coopmat: %d   SCALAR (declined): %d\n", label, length(coopmat), length(scalar))
    for s in scalar
        println("    declined -> ", s)
    end
end

# ── THE INSTRUMENT IS VALIDATED, both directions (2026-08-07):
#
#     ragged  200x184x216 fp16  ->  gpu_scalar_gemm_staged_kernel   (declined)
#     aligned 256x256x256 fp16  ->  gpu_coopmat_gemm_kernel         (tensor cores)
#
# Which also confirms by KERNEL IDENTITY, not just by timing, that a ragged fp16
# matmul lands on the scalar path — the 5.86 TF/s route measured against 37.64
# for the same work aligned.
#
# The first version of this file read `s.name` and reported ZERO declines for
# that same ragged matmul, because `name` is the SPIR-V entry point and is
# `"main"` for every kernel Lava emits. Only the planted control caught it. A
# census with no control is a way of writing "all clean" without checking.
#
# ── STATIC CENSUS (host-only, from the exported graph JSON — no GPU, no model
# load; the artifacts carry every weight shape):
#
#     147 of 1150 convolutions (12.8%) have Cout or K not a multiple of 16
#     basicvsrpp 46, readout_query 21, kokorovoc 20, MatAnyone family ~45,
#     rife 7, singles in neurallut / depthanything / sam2_encoder
#
# Largest by Cout*K are KOKORO's vocoder, not MatAnyone:
#     (1024, 1090, 3) K=3270 x3     (1024, 1090, 1) K=1090 x3   <- 1x1 -> matmul!
#     (1024,  514, 3) K=1542        (1024,  514, 1) K= 514      <- 1x1
# 1090 and 514 are concat-derived (a feature map with extra channels appended),
# which is the source I failed to predict — I had reasoned that K = Cin*kh*kw is
# a multiple of 16 whenever Cin is, which is true and misses that a concat makes
# Cin itself 1090.
#
# FIRST VERSION OF THAT CENSUS WAS WRONG: globbing `artifacts/*/graphs/**/*.json`
# matched only MatAnyone (the one model whose artifact uses a `graphs/` subdir),
# so "all the big ones are MatAnyone's" was circular. Detect graphs by STRUCTURE
# — a dict with both `ops` and `buffers` — not by path.
#
# WHICH OF THEM COSTS ANYTHING IS STILL UNMEASURED. 1x1 convolutions route to
# `matmul!` and so hit the gate; general convolutions use the conv planner, which
# already pads K. A static census sizes the SURFACE, not the time. That is what
# the runtime half of this file is for: run it per model with the GPU otherwise
# idle, `control_declines` first, and `Lava.clear_kernel_cache!()` between models
# so each census describes its own step.

# ── The STATIC half: which matmul weights land on the tile, read straight out of
# the exported graph JSON. Needs no GPU and no model load.
#
# `mm_coopmat_plan` gates on the weight's extents, and a weight almost never
# reaches its op directly — 1209 of 1211 arrive through a `permute` view — so the
# chain has to be followed. A `view` buffer carries `"of"`; addmm's weight is its
# LAST operand.

"""Follow `of` links until a non-view buffer; `nothing` if the chain breaks."""
function resolveview(buffers::Dict, id; maxdepth::Int = 12)
    b = get(buffers, id, nothing)
    depth = 0
    while b !== nothing && get(b, "kind", "") == "view" && depth < maxdepth
        parent = get(b, "of", nothing)
        parent === nothing && break
        b = get(buffers, parent, nothing)
        depth += 1
    end
    b
end

"""
    census_graph_matmuls(graph; tile = 16) -> Vector

Weights of `addmm`/`mm` ops in one parsed graph whose extents do not land on
`tile` — i.e. what `mm_coopmat_plan` will refuse with `Decline(:extent)`.

THE SELF-CHECK THIS NEEDS. Three earlier versions of this census were wrong in
three different ways (scoped to one model's directory layout; read the
ACTIVATION instead of the weight, so 1500 — a sequence length `gemm_padn`
already pads — looked like 667 declines; could not see through views). Each
produced a confident, well-formatted, wrong table.

What separates a right answer from those: **Whisper's ENCODER must come back
empty.** Its `addmm` measures ~27 TF/s in situ, so it is demonstrably on the
tensor-core path, and any census that reports encoder declines is misreading the
graph. Run `whisper.json` through this before believing it on anything else.
"""
function census_graph_matmuls(graph::Dict; tile::Int = 16)
    buffers = Dict(b["id"] => b for b in get(graph, "buffers", []))
    out = Tuple{String,Vector{Int}}[]
    for o in get(graph, "ops", [])
        a = get(o, "aten", "")
        (occursin("addmm", a) || a == "mm.default") || continue
        ins = get(o, "in", String[])
        isempty(ins) && continue
        root = resolveview(buffers, last(ins))
        root === nothing && continue
        # `weight` OR `transient`: under autocast the weight reaches the op
        # through a `_to_copy` dtype cast, whose output is a transient. That
        # accounts for 283 of 1103 matmuls — all of sam2_encoder's — and a cast
        # preserves the shape, which is what the gate actually tests. Accepting
        # only `weight` left 26% of the census unexamined.
        get(root, "kind", "") in ("weight", "transient") || continue
        shp = get(root, "shape", nothing)
        (shp isa Vector && all(x -> x isa Integer, shp)) || continue
        any(d -> d % tile != 0, shp) && push!(out, (get(o, "id", "?"), Vector{Int}(shp)))
    end
    out
end

# ── MEASURED 2026-08-07 across every exported graph: **1103 of 1103** matmul
# weights checked (no blind spot), 12 with a ragged extent, and whisper.json
# (the encoder) EMPTY as required.
#
#     whisperdec    x3   (51866, 1280)    vocabulary — 51866 % 16 = 10
#     kokorovoc     x4   ( 2180,  128)
#     kokorovoc     x1   ( 1028,  128)
#     kokorotext    x1   (   50,  512)
#     sam2_decoder  x2   (    2,  128)    tiny head, not worth padding
#     sam2_decoder  x1   (    4,  256)
#
# The decoder's logits projection is the interesting one: padding 51866 -> 51872
# is SIX columns, +0.01%, against a measured 6.43x penalty for being off the
# tile. Size it in situ before fixing — the decoder is dispatch-bound and this
# may not be where its time goes.
