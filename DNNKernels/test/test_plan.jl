"""
What a DECLARED plan is allowed to contain.

The placement is the one thing in this package where a mistake is silent: an
offset that overlaps a buffer still being read produces a *plausible* tensor, not
a crash, and it surfaces as a slightly wrong mask several graphs later. So the
invariant is asserted here rather than trusted.

This file used to test `planslab`/`checkslab`/`lifetimes` — a static slab this
package planned itself, per resolution — and those went with the interpreted run.
Mantle's `Place` does the placement now and Mantle tests that it does not
overlap, so what is left here is what only this package can say: WHICH buffers
may be placed at all. Two of its three assertions are still exactly the ones
worth making, and the third one — an escaping buffer must never be a transient —
is the bug `viewfor` shipped: two of SAM 2's decoder outputs were materialised
views, both transients, and the placer correctly put one on the other.

Over every exported graph, because it is shape-driven and one model would not
exercise it — after `dropdead`, which is what the driver plans. An op writing a
buffer nothing consumes has no destination (`declare!` gives storage to what is
CONSUMED), so a raw export is not a thing the driver ever asks for: SAM 2's
encoder has a dead `repeat_1`. `dropdead` is the one pass of the driver's that
needs no weights, which is why it is the one this file can run.

Run standalone:
    julia --project=. dev/JuliaVision/DNNKernels/test/test_plan.jl
"""

using Test
using DNNKernels
using DNNKernels: loadgraph, escaping, emitgraph, evalshape, dropdead, fuseops
import Mantle
include("fixtures.jl")

const MP = Mantle

# Nothing here names a directory. `Fixtures` owns the two layouts (SAM 2's JSONs
# are flat, MatAnyone's sit under `graphs/`), which is exactly the kind of detail
# a caller must not have to know.

"""
The driver's graph passes that need no weights, in its order.

`Model` runs a dozen more and all of them read weight VALUES — `foldbatchnorm`
folds the statistics into the convolution, `hoistcasts` evaluates a constant cast
— so a weight-free test cannot reproduce the graph the driver plans. These two
can, and they are the two whose absence changes what may be declared: a dead op
has no destination (`declare!` gives storage to what is CONSUMED), and an unfused
chain declares every intermediate it would otherwise not have.
"""
prepared(g) = first(fuseops(first(dropdead(g))))

"""Every graph we can find, with the dims it is planned at.

Named rather than globbed: the export drops `op_histogram.json` next to the
graphs and it is not one.
"""
function testgraphs()
    out = Tuple{String,Any,Any}[]
    for n in ("encode_image", "transform_key", "encode_mask_deep",
              "encode_mask_shallow", "pixel_fusion", "pred_uncertainty",
              "segment", "readout_query")
        Fixtures.have(n) &&
            push!(out, (n, prepared(Fixtures.matanyone(n)), (h = 32, w = 32)))
    end
    for n in ("sam2_encoder", "sam2_decoder")
        push!(out, (n, prepared(Fixtures.sam2(n)), (res = 1024,)))
    end
    out
end

const GRAPHS = testgraphs()
@assert !isempty(GRAPHS) "no exported graphs in the artifacts — nothing to plan"

"""
Graphs the declared path cannot emit yet, and the op each one stops at.

MatAnyone's ops, all four of them. The declared port was driven by SAM 2 and
these are what it does not reach; each is a `runop!` with no `emitop!`, and the
emit REFUSES by name rather than running as something else.

A list and not a skip, checked in both directions: a graph named here must still
refuse and must refuse on THIS op, so porting one fails this test and says so.
Three of the four are a gather the `mapbody!` pattern already covers
(`upsample_nearest2d` is the worked example beside them); the 1-D convolution is
the one with a design question in it, since lifting it to the 2-D implicit GEMM
means a degenerate axis carrying a trip count of 1 through `conv2d_igemm_ki!`.
"""
const UNPORTED = Dict(
    "encode_mask_deep"    => "convolution.default",       # the 1-D form
    "encode_mask_shallow" => "convolution.default",
    "pixel_fusion"        => "_adaptive_avg_pool2d.default",
    "pred_uncertainty"    => "_adaptive_avg_pool2d.default",
    "segment"             => "upsample_bilinear2d.vec",
    "readout_query"       => "slice_scatter.default",
)

"""
Zeroed device buffers of each weight's declared shape.

Nothing here runs, but the emit still has to see real operands: `mmplan` decides
from `Core.Typeof` and `size`, so a placeholder of the wrong rank or on the host
declines to a path that then refuses for being non-dense. The shape and dtype
come from the graph, which is where they are declared.

Every `:weight` buffer, not only the ones an op reads: `declare!` walks
`aten.order`, so it needs one per weight buffer the export carries. SAM 2's
decoder holds `model.maskmem_tpos_enc`, which no op in it reads.

Dropped and collected per graph — SAM 2's encoder alone is 1.8 GiB of weights,
and ten graphs' worth held at once is not a cost a unit test should pay.
`LavaArray` registers its own finalizer, so releasing the dict is the whole of
it.
"""
function stubweights(dev, g, dims)
    w = Dict{String,Any}()
    for (_, b) in g.buffers
        (b.kind === :weight && !isempty(b.key)) || continue
        haskey(w, b.key) && continue
        w[b.key] = MP.Buffer(dev, zeros(b.dtype, evalshape(b.shape, dims)))
    end
    return w
end

"""The resource that owns `r`'s bytes: a view's parent, transitively."""
ownerof(r) = r
ownerof(r::MP.ResourceView) = ownerof(r.parent)

function declaredplan(dev, label)
@testset "declared plan — $label" begin
    @testset "$name" for (name, g, dims) in GRAPHS
        weights = stubweights(dev, g, dims)
        if haskey(UNPORTED, name)
            err = try
                emitgraph(dev, g, weights, dims)
                nothing
            catch e
                e
            end
            @test err !== nothing
            @test occursin(UNPORTED[name], sprint(showerror, err))
            weights = nothing
            GC.gc(true)
            continue
        end
        mantlegraph, emitctx = emitgraph(dev, g, weights, dims)
        plan = MP.Plan(mantlegraph)

        # The old test also asserted that no buffer in `fusableset(g)` was
        # planned — 1334 MB of SAM 2's 2020 MB slab. That set named buffers the
        # interpreted run kept LAZY, deferring them as `Broadcasted` values, and
        # there is no such thing here: `fuseops` collapses a fusable chain into
        # one op before planning, so the intermediates are not in the graph to be
        # declared. The statement that survives is about the pass and not the
        # plan — `fusableset` is empty after `fuseops` — and it belongs with
        # `test_fusepass.jl`.

        # An ESCAPING value must not be a transient: a step chains graphs through
        # one placement, and an output the next graph reads cannot live in memory
        # that graph may overwrite. `make` is the one rule for this and `viewfor`
        # was not asking it.
        for id in Iterators.flatten((escaping(g), g.outputs))
            haskey(emitctx.res, id) || continue
            @test !(ownerof(emitctx.res[id]) isa MP.TransientBuffer)
        end

        # Reuse actually happens. `naivebytes` is every transient laid end to
        # end; equality would mean no two lifetimes are disjoint, which no real
        # graph manages once it has more than a handful of buffers.
        @test MP.peakbytes(plan) <= MP.naivebytes(plan)
        length(mantlegraph.transients) > 4 &&
            @test MP.peakbytes(plan) < MP.naivebytes(plan)

        MP.free!(plan)
        weights = nothing
        GC.gc(true)
    end

    # Nothing may declare a transient it does not use.
    #
    # `Liveness` refuses one by name — a transient's interval comes from its uses
    # and one with none has nothing to place — so this asserts the emit never
    # produces one. `keepall` is the mode that provokes it: there every result of
    # a multi-output op has a declared destination, so `destor` takes
    # `maybedest` on all of them, and an eagerly-evaluated `scratch` beside it
    # then belongs to no pass. `something(a, b)` evaluates BOTH arguments, which
    # is how that shipped.
    #
    # The decoder only. `keepall` gives every buffer storage of its own instead
    # of the placer's peak, which is 0.14 GiB here and 5.68 for the encoder.
    @testset "nothing declares a transient it does not use" begin
        g = prepared(Fixtures.sam2("sam2_decoder"))
        dims = (res = 1024,)
        mantlegraph, _ = emitgraph(dev, g, stubweights(dev, g, dims), dims;
                                   keepall = true)
        used = Set{Any}()
        for p in mantlegraph.passes, (rid, _) in MP.usages(p)
            push!(used, rid)
        end
        @test isempty([size(t) for (rid, t) in mantlegraph.transient_by_id
                       if !(rid in used)])
        # The same statement from the other side: a scratch that reached no
        # `dispatch!` never gets an id at all, so the two collections agree.
        @test length(mantlegraph.transients) == length(mantlegraph.transient_by_id)
        GC.gc(true)
    end
end
end

let bes = MP.eachbackend()
    isempty(bes) && @info "no usable backend; skipping the declared-plan checks"
    for be in bes
        declaredplan(MP.Device(be), string(nameof(typeof(be))))
    end
end

# ── Every generated block size is reachable ──────────────────────────────────
#
# `kernel-library-review.md` finding 5: `ATTN_BLOCKS` generates one kernel per
# entry, and the dispatchers used to be two hand-written `tk == 32 && return …`
# chains beside it. Adding a block size generated a kernel and silently did not
# dispatch to it — dead code that read as live, and nothing failed.
#
# The dispatchers are now folded out of the same tuple, so the two cannot
# disagree by construction. This asserts it anyway, because "by construction"
# is exactly the claim that rots when someone writes the next chain by hand.
@testset "every ATTN_BLOCKS entry has a kernel and a dispatch arm" begin
    for B in DNNKernels.ATTN_BLOCKS
        for prefix in ("attn_scores_b", "attn_apply_b")
            @test isdefined(DNNKernels, Symbol(prefix, B, "!"))
        end
    end

    # And the arms actually name those kernels. Reading the lowered source is the
    # only way to see a *missing* arm: a selector with a fallback answers every
    # `tk`, so calling it can never reveal that one size fell through to the
    # wrong kernel.
    #
    # The SELECTOR, not the launcher. `scoresblocked!` used to hold the `tk ==
    # 32 && return …` chain itself; the declared path needs to know which kernel
    # a shape takes without launching it, so the chain moved into
    # `scoresblocked!kernel` and the launcher now lowers to one call to that.
    # Reading the launcher found none of the eight names and said so ten times.
    for (f, prefix) in ((DNNKernels.var"scoresblocked!kernel", "attn_scores_b"),
                        (DNNKernels.var"applyblocked!kernel",  "attn_apply_b"))
        body = string(Base.uncompressed_ast(only(methods(f))).code)
        for B in DNNKernels.ATTN_BLOCKS
            @test occursin(string(prefix, B, "!"), body)
        end
    end
end
