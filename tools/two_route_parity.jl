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
# On SYNTHETIC inputs it found two silent bugs in the declared path, both
# pre-existing and both invisible to every suite:
#
#   1. A folded activation was DROPPED on `add.Tensor`. `foldrelu` records the
#      relu it deleted as `attrs["act"]` on the op before it — a convolution or
#      the add that ends a residual block — and only the convolution and `addmm`
#      emits read it. 25 relus across three of MatAnyone's graphs computed the
#      plain sum: finite, the right shape and dtype, wrong only where the sum was
#      negative. `encode_image` went from 51 of 59 ops disagreeing (worst rel
#      3.67e4) to 0 and worst 3.6e-3.
#
#   2. A NEGATIVE view index read out of bounds. `select.int(0, -1)` is the last
#      element; `viewstrides` took it literally, so the offset was `-1 * stride`
#      and `stridedcopy!` read one element before the parent. The value was
#      garbage that changed between runs and between graphs (-4.29e37 in
#      `readout_query`, 1.49e-5 in `encode_mask_shallow`).
#
# Then, on the inputs a REAL step produced — `runmatanyone` once, and each
# graph re-compared on exactly the buffers its plan read — all eight agree:
#
#   encode_image        59 ops   worst rel 5.2e-3
#   encode_mask_deep    87        1.9e-3
#   encode_mask_shallow 82        1.4e-3
#   pixel_fusion        25        1.2e-3
#   pred_uncertainty     7        7.9e-4
#   readout_query      266        4.4e-3
#   segment             65        5.9e-3
#   transform_key        9        1.2e-3
#
# 600 ops, none over 1e-2, which is fp16 convolution precision.
#
# SAM 2 was never affected by either: it folds only into `addmm`, 48 times, and
# that emit reads `act`.
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

"""
Both arms, with the reference arm's backend chosen.

`KA.CPU()` is the honest reference — no device, no plan, plain arrays — but three
of MatAnyone's graphs reach a kernel it has no method for ("This kernel is
unavailable for backend CPU"). Running `runop!` on Lava instead still compares
two different code paths (broadcasting and `launch!` per op, against `emitgraph`
and one recorded plan), which is what catches a wrong kernel; it just shares the
device with the arm under test.
"""
function compare(name, refbackend = KA.CPU())
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
    refw = refbackend isa KA.CPU ? hostweights :
           Dict{String,Any}(k => DK.toback(refbackend, v) for (k, v) in hostweights)
    refins = refbackend isa KA.CPU ? Dict{String,Any}(ins) :
             Dict{String,Any}(k => DK.toback(refbackend, v) for (k, v) in ins)
    interp = DK.execute!(g, refins, refw; dims, backend = refbackend)
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

# CPU first: it is the reference with no device in it. Where it has no method
# for a kernel, fall back to the same `runop!` on Lava rather than reporting
# nothing — two of those three graphs are ones the dropped-activation bug hit.
for n in ("transform_key", "pred_uncertainty", "pixel_fusion", "segment",
          "readout_query", "encode_mask_shallow", "encode_mask_deep", "encode_image")
    try
        compare(n)
    catch e
        msg = sprint(showerror, e)
        if occursin("unavailable for backend CPU", msg)
            try
                print("  [ref on Lava] ")
                compare(n, back)
            catch e2
                println(rpad(n, 22), " FAILED on Lava too: ",
                        first(sprint(showerror, e2), 140))
            end
        else
            println(rpad(n, 22), " FAILED: ", first(msg, 150))
        end
    end
end
