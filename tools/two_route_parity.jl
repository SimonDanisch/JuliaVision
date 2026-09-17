# Declared against interpreted, per op, for every model whose artifact is here.
#
#     julia --project=. dev/JuliaVision/tools/two_route_parity.jl
#
# `verifygraph` compares the declared path against PyTorch's own per-node values
# and needs a refs artifact; most models here have none installed. `runop!` is
# the OTHER route to the same answer and ships with the package — it writes each
# op as broadcasting and `launch!` over ordinary arrays — so comparing the two
# needs only the weights. See `execute.jl`'s header for why that second
# implementation exists.
#
# ── What it established, 2026-09-17 ──────────────────────────────────────────
#
# It found two silent bugs in the declared path, both pre-existing and both
# invisible to every suite:
#
#   1. A folded activation was DROPPED on `add.Tensor`. `foldrelu` records the
#      relu it deleted as `attrs["act"]` on the op before it — a convolution or
#      the add that ends a residual block — and only the convolution and `addmm`
#      emits read it. 25 relus across three of MatAnyone's graphs computed the
#      plain sum: finite, right shape, right dtype, wrong only where the sum was
#      negative. `encode_image` went from 51 of 59 ops disagreeing (worst rel
#      3.67e4) to 0 and worst 3.6e-3.
#
#   2. A NEGATIVE view index read out of bounds. `select.int(0, -1)` is the last
#      element; `viewstrides` took it literally, so the offset was `-1 * stride`
#      and `stridedcopy!` read one element before the parent. The value was
#      garbage that changed between runs and between graphs (-4.29e37 in
#      `readout_query`, 1.49e-5 in `encode_mask_shallow`).
#
# After both, every model the declared path covers:
#
#   neurallut       22 ops   5.9e-5   clean
#   basicvsrpp    2282 ops   1.2e-4   clean, and it exercises `deform_conv2d`,
#                                     `flip`, `avg_pool2d` and `grid_sampler_2d`
#   whispercross    10 ops   3.1e-6   clean
#   whisperdec      74 ops   1.0e-6   clean — and it exercises `index_put`
#   matanyone      600 ops   9.8e-3   clean, on the inputs a real step produced
#   sam2                              covered better, by `verifygraph`
#
# Re-measured 2026-09-17 after `folddims!` took a `post` map and `_to_copy` took
# `castfn`, both of which every model reads: the four clean models are clean to
# the same figures and the three residuals below are unchanged. MatAnyone's
# eight graphs on SYNTHETIC inputs read 9, 87, 59, 266, 82, 65, 25 and 7 ops,
# all clean but `readout_query`, whose `_softmax_2` is 1.03e-2. That is the
# case the first limit below names, not a change: the graph consumes a memory
# bank readout and synthetic inputs are not a legitimate one for it.
#
# `kokorotext` is NOT in the loop below and cannot be: its weights live in the
# `kokoro-ckpt` artifact, which is not installed here, and `KokoroRunner.ready()`
# is false without it. It was verified the same way with STUB weights (zeros and
# small normals of each declared shape, identical on both arms) over 199 ops at
# t = 9, 17, 23, 30 and 41, worst rel 8.6e-4 and nothing over 1e-2. Stub weights
# are enough for what this tool looks for, since both arms read the same ones;
# what they cannot check is a value-dependent path, and kokorotext has one
# (`gather` indexes by value), so its index operands come from the graph's own
# constants rather than from the weights. The LSTM is checked against torch's
# definition instead, in `DNNKernels/test/test_lstm.jl`, because both routes
# share `lstm_kernel!` and this tool would pass with it wrong.
#
# Three models sit above 1e-2 and NONE of them is a defect this tool can pin on
# the declared path. Each is the ill-conditioned-op case `verifygraph`'s own
# docstring describes: "this criterion will report a correct half-precision port
# as broken at whichever op the model is worst conditioned at."
#
#   depthanything  first at `_softmax`, rel 4.4e-2. Ops 1-13 agree to 4.8e-3
#                  (fp16 convolution and two GEMMs) and the softmax amplifies
#                  that. Checked against a Float64 HOST softmax over the same
#                  operand: the declared arm is exact to 3.4e-7 and the
#                  interpreted arm is off by 3.6e-2, so if either is the less
#                  accurate one it is the reference.
#   rife           first at `grid_sampler_2d_2`, rel 4.9e-2. The first two
#                  samplers ATTENUATE (operands 1.2e-4 -> output 6.0e-5); the
#                  third and fourth amplify ~80x and ~270x. Same attributes on
#                  all four; what differs is that the last two sample feature
#                  maps rather than a smooth frame, and a bilinear tap times a
#                  large local gradient is a large output change. The kernel is
#                  the one BasicVSR++ runs clean over 2282 ops.
#   whisper        first at `addmm_32`, rel 1.1e-2, worst 1.49. This encoder is
#                  the documented case: `verifygraph`'s docstring measures 1576x
#                  amplification at `add_42`, block 20's feed-forward, where
#                  PyTorch's own fp16 amplifies 415x against its own fp32.
#
# ── Two limits, and they are why this is a tool and not a test ───────────────
#
# **It cannot always localise.** An op whose result the interpreted arm defers
# (`fusableset`) has nothing to compare and is skipped, so the first divergence
# it can SEE may be downstream of the real one. On MatAnyone that was `add_94`,
# whose two inputs both agreed; the fix was found by reading the op's attributes,
# not by narrowing further.
#
# **It is not a reference.** Both arms are ours. Agreement means the two routes
# do the same thing, which is what catches a wrong kernel; it does not mean
# either matches torch. `verifygraph` is for that, and where a residual is
# amplification rather than a defect, only a third precision settles it.
ENV["DISPLAY"] = get(ENV, "DISPLAY", ":0")
using DNNKernels, KernelAbstractions, Printf
using Mantle: LavaBackend
import Mantle
const DK = DNNKernels
const KA = KernelAbstractions
const BACK = LavaBackend()

tohost(x) = Array(DK.tohost(x))
flat(x) = Float64.(vec(tohost(x)))

"""
Deterministic synthetic inputs of each graph's declared shape and dtype.

Synthetic is enough to find a wrong kernel — both bugs above showed on these —
but NOT enough for a graph that indexes by value: `readout_query` and the mask
encoders consume a memory-bank readout, and garbage there sends index arithmetic
anywhere. For those, re-compare on what a real step produced; `mp.inputs` of the
plan `call` cached IS the buffers it read.
"""
function synthinputs(g, dims)
    ins = Dict{String,Any}()
    for (i, id) in enumerate(g.inputs)
        b = get(g.buffers, id, nothing)
        b === nothing && continue
        sz = DK.evalshape(b.shape, dims)
        T = b.dtype
        n = max(prod(sz), 1)
        # `range(a, b; length = 1)` throws when the endpoints differ, and a
        # length-1 input is ordinary — RIFE's `timestep` is one scalar.
        vals = n == 1 ? [Float64(i)] : collect(range(i, i + 6, length = n))
        ins[id] = T <: Integer ? reshape(T.(mod.(1:n, 5)), sz) :
                                 reshape(T.(0.3 .* sin.(vals)), sz)
    end
    return ins
end

"""One graph, both routes. `refbackend` is which backend `runop!` runs on."""
function compare(name, g, weights, dims, ins; refbackend = KA.CPU())
    hw = Dict{String,Any}(k => tohost(v) for (k, v) in weights)
    onref(d) = refbackend isa KA.CPU ? Dict{String,Any}(d) :
               Dict{String,Any}(k => DK.toback(refbackend, v) for (k, v) in d)
    interp = DK.execute!(g, onref(ins), onref(hw); dims, backend = refbackend)
    decl = DK.declaredvalues(g, ins, hw; dims, backend = BACK)
    worst, worstid, checked, nbad, firstid = 0.0, "", 0, 0, ""
    for op in g.ops
        a = get(interp, op.out, nothing)
        b = get(decl, op.out, nothing)
        a isa AbstractArray && b isa AbstractArray || continue
        size(a) == size(b) && eltype(a) !== Bool || continue
        av, bv = flat(a), flat(b)
        all(isfinite, av) && all(isfinite, bv) || continue
        sc = max(maximum(abs, av; init = 0.0), 1e-6)
        e = maximum(abs.(av .- bv); init = 0.0) / sc
        checked += 1
        if e > 1e-2
            nbad += 1
            isempty(firstid) &&
                (firstid = "$(op.out) ($(op.aten)) rel $(round(e, sigdigits = 3))")
        end
        e > worst && ((worst, worstid) = (e, op.out))
    end
    @printf("%-26s %5d ops  %4d over 1e-2  worst rel %-10.3g %s\n",
            name, checked, nbad, worst,
            isempty(firstid) ? "" : "first: " * firstid)
end

"""
CPU first, then the same `runop!` on Lava.

The CPU is the honest reference — no device, no plan, plain arrays — but it
cannot run a `cpu=false` kernel (`__index_Global_Linear` has no method for its
metadata), cannot index a device array elementwise, and has no method for
several of the extern kernels. Which of those it is does not change what to do,
and on Lava `runop!` still exercises a different code path from `emitgraph`.
"""
function bothways(name, g, weights, dims = (;); ins = synthinputs(g, dims))
    try
        compare(name, g, weights, dims, ins)
    catch e
        print("  [ref on Lava] ")
        try
            compare(name, g, weights, dims, ins; refbackend = BACK)
        catch e2
            println(rpad(name, 26), " FAILED: ",
                    first(sprint(showerror, e2), 130))
        end
    end
end

# Each runner exports `<model>graph` and `<model>weights`; only the ones whose
# artifact is installed are asked, and an absent one says so rather than
# throwing.
for (pkg, gf, wf) in (("NeuralLUTRunner", :neurallutgraph, :neurallutweights),
                      ("BasicVSRRunner", :basicvsrppgraph, :basicvsrppweights),
                      ("RIFERunner", :rifegraph, :rifeweights),
                      ("DepthAnythingRunner", :depthanythinggraph,
                       :depthanythingweights),
                      ("WhisperRunner", :whispergraph, :whisperweights))
    m = try
        Base.require(Main, Symbol(pkg))
    catch e
        println(rpad(pkg, 26), " not loadable: ", first(sprint(showerror, e), 90))
        continue
    end
    (isdefined(m, gf) && isdefined(m, wf)) || continue
    try
        # `invokelatest`: `Base.require` defines these in a world NEWER than
        # this statement, so a direct call is a `MethodError` saying the
        # function exists but has no method for no arguments.
        bothways(lowercase(replace(pkg, "Runner" => "")),
                 Base.invokelatest(getfield(m, gf)),
                 Base.invokelatest(getfield(m, wf)))
    catch e
        println(rpad(pkg, 26), " no assets: ", first(sprint(showerror, e), 90))
    end
end

# Whisper's DECODER graphs, which sit in their own artifact as JSON beside the
# weights rather than behind a `<model>graph` accessor.
try
    WR = Base.require(Main, :WhisperRunner)
    dir = Base.invokelatest(WR.decoderdir)
    w = Base.invokelatest(DK.readsafetensors, joinpath(dir, "weights.safetensors"))
    for n in ("whispercross", "whisperdec")
        p = joinpath(dir, n * ".json")
        isfile(p) || continue
        bothways(n, DK.loadgraph(p), w)
    end
catch e
    println("whisper-decoder", " " ^ 11, " skipped: ",
            first(sprint(showerror, e), 90))
end

# MatAnyone, on the inputs a real step produced — see `synthinputs`.
try
    MA = Base.require(Main, :MatAnyoneRunner)
    model = Base.invokelatest(MA.matanyonemodel; backend = BACK)
    W, H = 128, 96
    img = DK.toback(BACK, fill(0.5f0, W, H, 3, 1))
    hm = zeros(Float32, W, H); hm[(W ÷ 4):(3W ÷ 4), (H ÷ 4):(3H ÷ 4)] .= 255.0f0
    Base.invokelatest(MA.runmatanyone, model, img, DK.toback(BACK, hm))
    KA.synchronize(BACK)
    for (k, v) in model.scratch
        k isa Tuple && length(k) >= 3 && k[1] === :plan || continue
        name, dims = k[2], k[3]
        g = model.graphs[name]
        ins = Dict{String,Any}(id => tohost(b)
                               for (id, b) in zip(g.inputs, v.inputs))
        bothways("matanyone/$name", g, model.weights, dims; ins)
    end
catch e
    println("MatAnyoneRunner", " " ^ 11, " skipped: ",
            first(sprint(showerror, e), 90))
end
nothing
