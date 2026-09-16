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

Globbed, through `Fixtures.matanyonenames`, which is what knows that
`op_histogram.json` sits next to the graphs and is not one.
"""
function testgraphs()
    out = Tuple{String,Any,Any}[]
    # Every graph the artifact has, asked for by name rather than restated here:
    # a list of the eight expected ones goes out of date the moment the export
    # gains a ninth, and `matanyonenames` is what knows which files are graphs.
    for n in sort(Fixtures.matanyonenames())
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
Shape cases a DECLARED op refuses, per graph.

`convolution.default` has an `emitop!` and covers the dense forward 2-D case;
1-D lifts to it and 3-D has `convolution3d!`, and neither is written. So these
graphs refuse at an op the coverage table says is ported — which is the right
refusal and not a gap in the table, just one it cannot express.

MatAnyone's mask encoders are 1-D convolutions over a flattened mask. Lifting one
to the 2-D implicit GEMM is the port, and the design question in it is that the
degenerate axis carries a trip count of 1 through `conv2d_igemm_ki!`.
"""
const UNPORTED_SHAPES = Dict(
    "encode_mask_deep"    => "is 1-D",
    "encode_mask_shallow" => "is 1-D",
)

"""
    portedatens() -> Set{String}

Every aten op the declared path has an `emitop!` for.

Read off the method table rather than listed beside it: `emitop!` dispatches on
`Val(Symbol(aten))`, so its `Val` parameters ARE the ported set, and a second
copy of that set goes stale the first time one is added.
"""
function portedatens()
    out = Set{String}()
    for m in methods(DNNKernels.emitop!)
        p = Base.unwrap_unionall(m.sig).parameters
        length(p) >= 4 || continue
        T = p[4]
        T isa DataType && T <: Val || continue
        isempty(T.parameters) && continue
        T.parameters[1] isa Symbol || continue
        push!(out, String(T.parameters[1]))
    end
    return out
end

"""
Every aten op a graph needs that the declared path has no `emitop!` for.

MatAnyone's, and all of them. The declared port was driven by SAM 2; each of
these is a `runop!` with no `emitop!`, and the emit REFUSES by name rather than
running as something else.

**Complete, and that is the point.** This began as one op per graph, taken from
what `emitgraph` threw — but the emit stops at the FIRST op it cannot do, so a
list built that way named four ops where there are eleven. Comparing the SETS
instead is static, needs no device and no weights, and reports the whole gap at
once.

Checked with `==` in both directions, so porting an op fails this test with a
diff naming it. Most are a gather or a reduction the existing patterns cover
(`mapbody!` and `sumdims!`, with `upsample_nearest2d` as the worked example
beside them). `readout_query` is the outlier at nine, and `_softmax`/`max.dim`
are new shapes rather than transcriptions.

The 1-D `convolution.default` is NOT here: it is a shape case inside an op that
IS declared, so no coverage check can see it. `UNPORTED_SHAPES` records it
separately, because two graphs hit it BEFORE any of the ops above and a test that
insisted the refusal name a missing op would fail on them.
"""
const UNPORTED = Dict(
    "encode_mask_deep"    => ["_adaptive_avg_pool2d.default", "slice_scatter.default"],
    "encode_mask_shallow" => ["_adaptive_avg_pool2d.default", "slice_scatter.default"],
    "pixel_fusion"        => ["_adaptive_avg_pool2d.default"],
    "pred_uncertainty"    => ["_adaptive_avg_pool2d.default"],
    "segment"             => ["_adaptive_avg_pool2d.default", "prod.dim_int",
                              "upsample_bilinear2d.vec"],
    "readout_query"       => ["_softmax.default", "any.dim", "bitwise_and.Tensor",
                              "constant_pad_nd.default", "ge.Tensor", "max.dim",
                              "prod.dim_int", "scalar_tensor.default",
                              "slice_scatter.default"],
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
        gaps = sort(collect(setdiff(Set(o.aten for o in g.ops), portedatens())))
        @test gaps == get(UNPORTED, name, String[])
        if !isempty(gaps)
            # It must actually refuse, and on one of the ops the gap names —
            # a coverage table that agrees with nothing is not a check.
            err = try
                emitgraph(dev, g, weights, dims)
                nothing
            catch e
                e
            end
            @test err !== nothing
            msg = sprint(showerror, err)
            @test any(a -> occursin(a, msg), gaps) ||
                  occursin(get(UNPORTED_SHAPES, name, "\0"), msg)
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
