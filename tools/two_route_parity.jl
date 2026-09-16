# Declared against interpreted, per op, with no reference artifact.
#
#     julia --project=. dev/JuliaVision/tools/two_route_parity.jl
#
# `verifygraph` compares the declared path against PyTorch's own per-node values
# and needs a refs artifact — MatAnyone's is 978 MB and lazy. `runop!` is the
# OTHER route to the same answer and ships with the package: it writes each op as
# broadcasting and `launch!` over ordinary arrays and runs on the CPU backend, so
# this comparison needs only the weights. See `execute.jl`'s header for why that
# second implementation exists.
#
# ── What it established, 2026-09-17 ──────────────────────────────────────────
#
#   transform_key        9 ops   worst rel 1.5e-4
#   pred_uncertainty     7 ops   worst rel 0        (exact)
#   pixel_fusion        25 ops   worst rel 1.4e-3
#   segment             65 ops   worst rel 1.5e-3
#   encode_image        58 ops   51 over 1e-2, first at `add_94`
#   readout_query, encode_mask_deep, encode_mask_shallow
#                               the INTERPRETED arm cannot run: "This kernel is
#                               unavailable for backend CPU"
#
# 1.5e-3 is fp16 convolution precision and those four are agreement.
# `encode_image` is a real lead and NOT one of these: none of the ops ported on
# 2026-09-17 appears in it (its five kinds are `_to_copy`, `add.Tensor`,
# `convolution.default`, `fused.elementwise`, `max_pool2d_with_indices`), and
# `destor` — the one changed function it does reach — returns the same resource
# before and after.
#
# ── Two limits, and they are why this is a tool and not a test ───────────────
#
# **It cannot localise.** The interpreted arm keeps fusable values LAZY
# (`fusableset`), so an op whose result is deferred has nothing to compare and is
# skipped. The first divergence it can SEE in `encode_image` is `add_94`, whose
# two inputs both agree — so the actual first divergence is at one of the skipped
# ops before it. Settling that needs PyTorch's values as a third opinion, i.e.
# the refs artifact.
#
# **It is not a reference.** Both arms are ours. Agreement means the two routes
# do the same thing, which is what catches a wrong kernel; it does not mean
# either matches torch. `verifygraph` is for that.

ENV["DISPLAY"] = ":0"
ENV["XAUTHORITY"] = "/run/user/1000/.mutter-Xwaylandauth.G5CYV3"
using DNNKernels, MatAnyoneRunner, KernelAbstractions, Statistics, Printf
using Mantle: LavaBackend
import Mantle
const DK = DNNKernels
const KA = KernelAbstractions

back = LavaBackend()
model = MatAnyoneRunner.matanyonemodel(; backend = back)
# HOST copies of the MODEL's weights, and both of those matter.
#
# The model's, not `matanyoneweights()`: `hoistcasts` folded constant casts into
# new weights keyed `"<graph>|<op>|hoistcast"`, so the rewritten graphs name
# entries the raw checkpoint does not have.
#
# Host, because `runop!` on `KA.CPU()` indexes them elementwise and a device
# array refuses that ("Scalar indexing is disallowed"). The declared arm uploads
# what it needs itself, through `residentweights`.
hostweights = Dict{String,Any}(k => DK.tohost(v) for (k, v) in model.weights)
dims = (h = 6, w = 8)

hostref(x) = Float64.(vec(Array(DK.tohost(x))))

function compare(name)
    g = model.graphs[name]
    # Inputs: deterministic, small, and the DECLARED dtype and shape.
    ins = Dict{String,Any}()
    for (i, id) in enumerate(g.inputs)
        b = g.buffers[id]
        sz = DK.evalshape(b.shape, dims)
        T = b.dtype
        h = if T <: Integer
            reshape(T.(mod.(1:prod(sz), 7)), sz)
        else
            reshape(T.(0.3 .* sin.(range(i, i + 6, length = max(prod(sz), 1)))), sz)
        end
        ins[id] = h
    end
    interp = DK.execute!(g, Dict{String,Any}(ins), hostweights;
                         dims, backend = KA.CPU())
    decl = DK.declaredvalues(g, ins, hostweights; dims, backend = back)
    worst, worstid, checked, nbad = 0.0, "", 0, 0
    for op in g.ops
        haskey(interp, op.out) && haskey(decl, op.out) || continue
        a, b = interp[op.out], decl[op.out]
        # A multi-output op: the interpreted arm returns a tuple and the
        # declared one keys each element `"\$(id)#\$(i)"`. Skipping these was
        # the blind spot -- `max_pool2d_with_indices` is one and it feeds the
        # first divergence in `encode_image`.
        if a isa Tuple
            for (i, ai) in enumerate(a)
                key = "$(op.out)#$(i - 1)"
                haskey(decl, key) && decl[key] isa AbstractArray &&
                    ai isa AbstractArray || continue
                size(ai) == size(decl[key]) || continue
                eltype(ai) === Bool && continue
                av2, bv2 = hostref(ai), hostref(decl[key])
                all(isfinite, av2) && all(isfinite, bv2) || continue
                sc = max(maximum(abs, av2; init = 0.0), 1e-6)
                e2 = maximum(abs.(av2 .- bv2); init = 0.0) / sc
                checked += 1
                if e2 > 1e-2
                    nbad += 1
                    nbad <= 3 && @printf("    %-16s %-26s rel %.3g   |interp| max %.4g  |decl| max %.4g\n",
                                         key, op.aten, e2,
                                         maximum(abs, av2; init = 0.0),
                                         maximum(abs, bv2; init = 0.0))
                end
                e2 > worst && ((worst, worstid) = (e2, key))
            end
            continue
        end
        b isa Tuple && continue
        a isa AbstractArray && b isa AbstractArray || continue
        size(a) == size(b) || (println("  SHAPE ", op.out, " ", size(a), " vs ", size(b)); continue)
        eltype(a) === Bool && continue
        av, bv = hostref(a), hostref(b)
        all(isfinite, av) && all(isfinite, bv) || continue
        scale = max(maximum(abs, av; init = 0.0), 1e-6)
        e = maximum(abs.(av .- bv); init = 0.0) / scale
        checked += 1
        e > worst && ((worst, worstid) = (e, op.out))
        # The FIRST divergence is the one that says something; everything after
        # it inherits. Three, so a single flagged op can be read in context.
        if e > 1e-2
            nbad += 1
            nbad <= 3 && @printf("    %-16s %-26s rel %.3g   |interp| max %.4g  |decl| max %.4g\n",
                                 op.out, op.aten, e,
                                 maximum(abs, av; init = 0.0), maximum(abs, bv; init = 0.0))
        end
    end
    @printf("%-22s %4d ops checked   %d over 1e-2   worst rel %.3g at %s\n",
            name, checked, nbad, worst, worstid)
    return worst
end

for n in ("transform_key", "pred_uncertainty", "pixel_fusion", "segment",
          "readout_query", "encode_mask_shallow", "encode_mask_deep", "encode_image")
    try
        compare(n)
    catch e
        println(rpad(n, 22), " FAILED: ", first(sprint(showerror, e), 150))
    end
end
