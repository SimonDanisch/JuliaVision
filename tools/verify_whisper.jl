"""
Whisper large-v3-turbo's encoder on Lava, against the PyTorch reference.

    julia --project=envs/whisper tools/verify_whisper.jl [gpu|cpu] [fp32|fp16]

Reads what `tools/dump_whisper_refs.py` wrote.

**This runs on the GPU by default, unlike `verify_sam2.jl`.** The encoder is 617
ops over a 1500x1280 residual stream with 32 blocks of (1, 20, 1500, 1500)
attention — roughly 2.3 TFLOP for one 30 s window. A KA CPU backend does not
finish that in a working session, so "compare against the host reference" and
"run on the device" cannot be separated here the way they were for SAM 2's
decoder; `verifygraph` downloads each result instead. `cpu` is still useful on
the two-block prefix, which covers all eleven op types.

Three checks, because they exercise **different code** and any one can fail
alone:

  * *prefix* — the first two encoder blocks, node by node, under the
    amplification criterion in `verify.jl`. Every op type the encoder uses
    appears here, for a sixteenth of the arithmetic, so this is the one to run
    while iterating.
  * *layers* — all 617 ops, node by node. Unplanned memory: each op keeps its
    own output, so the whole graph's intermediates stay resident (~7 GB). That
    is the price of being able to say *which* op is wrong.
  * *e2e* — the output only, through `Model`/`call`. **Not a cheaper version of
    the above.** `Model` rewrites the graph before it runs it — `hoistpermutes`
    turns every permuted weight into a dense one, and that is what decides
    whether a matmul reaches the cooperative-matrix path. In fp16 the two paths
    differ by two orders of magnitude in accuracy, so a graph verified only by
    `verifygraph` is not a graph verified. It also poisons the scratch slab with
    a NaN bit pattern first, so an op that skips part of its output shows up as
    NaN rather than as a plausible number left over from the last call
    (GUARDRAILS 3).

    That difference is also why the fp16 diagnosis went wrong once: the scalar
    GEMM's fp16 accumulator was real, 234x on one op, and it is on the path
    `verifygraph` runs and *not* on the one `Model` runs. It moved the node table
    and left the e2e number where it was.

**In fp16 the e2e gate is a three-way comparison** and needs the fp32 references
as well (`--precision fp32` of the same dump). Two fp16 runs of this model differ
by more than either differs from fp32 — see the block of comment at the gate.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics
using LinearAlgebra: dot
using DNNKernels: readsafetensors, verifygraph, coverage, toback, Model, call, loadgraph,
                  Workspace
const KA = KernelAbstractions

mode = "gpu"
prec = "fp32"
for a in ARGS
    a in ("gpu", "cpu") && (global mode = a)
    a in ("fp32", "fp16") && (global prec = a)
end
const DIR = "/sim/Programmieren/VideoEdit/gen/graphs/" * (prec == "fp32" ? "whisper" : "whisper-$prec")

isfile(joinpath(DIR, "refs.safetensors")) || error(
    "no refs at $DIR — run `uv run tools/dump_whisper_refs.py --precision $prec` first")

backend = if mode == "gpu"
    using Lava
    LavaBackend()
else
    KA.CPU()
end

vram() = try
    parse(Int, first(split(read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader`,
                                String))))
catch
    nothing
end

@info "Whisper large-v3-turbo encoder on $mode, $prec"
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
refs = readsafetensors(joinpath(DIR, "refs.safetensors"))
g = loadgraph(joinpath(DIR, "whisper.json"))
@printf("  %d ops, %d weight tensors (%.3f GiB), %d reference tensors\n",
        length(g.ops), length(weights), sum(sizeof, values(weights)) / 2^30, length(refs))

_, missingops = coverage(joinpath(DIR, "whisper.json"))
isempty(missingops) || error("unimplemented ops: " * join(sort(missingops), ", "))

bad = String[]

"""Ops 1..`n` as a graph of their own, so a prefix can be checked in isolation."""
prefix(g, upto) = (i = findfirst(o -> o.out == upto, g.ops);
                   DNNKernels.Graph(g.name, g.symbols, g.inputs, [upto],
                                    g.buffers, g.order, g.ops[1:i]))

"""
    torchamplifies(id, prev) -> Float64 | nothing

How much **PyTorch's own fp16** amplifies at node `id`, measured against its own
fp32: the ratio of its error there to its error at `prev`.

`verify.jl`'s criterion is that "drift only ever carries error forward, a bug
creates it out of nothing". At an op that is genuinely ill-conditioned that
premise is false for *every* implementation, and reporting the op as a mismatch
says nothing about whose implementation it is. This is the control: if PyTorch's
own half precision amplifies at the same node, the node is the model.

`nothing` when the fp32 dump does not carry both nodes — the two exports
decompose attention differently, so only the residual chain is named in both.
"""
function torchamplifies(id, prev)
    prec == "fp16" || return nothing
    isdefined(Main, :fp32refs) || return nothing
    k, kp = "whisper/node/$id", "whisper/node/$prev"
    all(haskey(fp32refs, x) && haskey(refs, x) for x in (k, kp)) || return nothing
    e(x) = maximum(abs.(Float64.(refs[x]) .- Float64.(fp32refs[x])))
    ep = e(kp)
    ep > 0 ? e(k) / ep : nothing
end

# Loaded up front, not at the gate: the node-by-node pass below needs it too.
prec == "fp16" && (fp32refs = readsafetensors(
    "/sim/Programmieren/VideoEdit/gen/graphs/whisper/refs.safetensors"))

for (name, gg) in (("prefix (blocks 0-1)", prefix(g, "add_4")), ("all 617 ops", g))
    println("\n=== $name, node by node ===")
    t0 = time()
    ok, diffs, _ = verifygraph(gg, refs, weights; dims = (;), backend, verbose = true)
    @printf("  %.1f s\n", time() - t0)
    if !ok
        f = first(diffs)
        println("  FIRST MISMATCH at op $(f.index): $(f.id) ($(f.aten)) " *
                "max|Δ| $(f.maxabs) rel $(f.relative) inflow $(f.inflow) $(f.shape)")
        # `add_N` is the residual chain; its predecessor is `add_{N-1}`, which is
        # the only pair the fp32 dump names as well.
        n = match(r"^add_(\d+)$", f.id)
        amp = n === nothing ? nothing :
              torchamplifies(f.id, parse(Int, n[1]) == 1 ? "add" : "add_$(parse(Int,n[1])-1)")
        if amp !== nothing && amp >= 4
            # One literal, not a concatenation: `@printf` needs its format string
            # to BE a literal and rejects `"a" * "b"` with an error that names the
            # macro rather than the splice.
            @printf("  ...and PyTorch's own fp16 amplifies %.1fx at the same node against its own fp32,\n  so this is the MODEL, not the port. The gate for fp16 is the three-way below.\n", amp)
        else
            push!(bad, name)
        end
    end
    mode == "gpu" && (GC.gc(true); Lava.trim_gpu_pool!())
end

println("\n=== end to end, through Model (rewritten graph, planned slab) ===")
m = Model(DIR, joinpath(DIR, "weights.safetensors"); names = ["whisper"], backend)
@printf("  %d ops after rewrite, %.3f GiB resident\n",
        length(m.graphs["whisper"].ops), sum(sizeof, values(m.weights)) / 2^30)
mel = toback(backend, refs["whisper/in0"])
slab, = DNNKernels.scratchfor(m, (;))
@printf("  scratch slab %.1f MB\n", length(slab) / 2^20)
fill!(slab, 0xff)                       # GUARDRAILS 3: coverage, not just values
out, = call(m, "whisper", mel; dims = (;))
KA.synchronize(backend)
got = Array(out)
want = refs["whisper/node/" * g.outputs[1]]
size(got) == size(want) || error("output $(size(got)) vs reference $(size(want))")
nan = count(isnan, got)
nan == 0 || (println("  $nan of $(length(got)) outputs are NaN — the graph left "
                     * "part of its result unwritten"); push!(bad, "coverage"))
d = maximum(abs.(Float64.(got) .- Float64.(want)))
scale = maximum(abs.(Float64.(want)))
relrms = sqrt(mean(abs2, Float64.(got) .- Float64.(want))) / sqrt(mean(abs2, Float64.(want)))
cs = dot(vec(Float64.(got)), vec(Float64.(want))) /
     (sqrt(sum(abs2, Float64.(got))) * sqrt(sum(abs2, Float64.(want))))
@printf("  output %s: max|Δ| %.4g on scale %.4g (rel %.3g), rel rms %.4g, cosine %.10f\n",
        size(got), d, scale, d / scale, relrms, cs)
prec == "fp32" && relrms > 1e-3 && push!(bad, "e2e")

# ── the fp16 gate is a THREE-way comparison, and it has to be ────────────────
#
# Comparing two fp16 runs of this model measures their *difference*, which can
# be larger than either one's distance from the truth. Eight of the 1500 frames
# are ill-conditioned in half precision — at block 20's feed-forward the error
# there jumps ~30x in a single op — and PyTorch's own fp16 has the same cliff at
# the same frames. So `ours vs torch-fp16` is not a measure of us; the question
# is how much further from fp32 we are than PyTorch's own fp16 is.
#
# Measured 2026-08-05: at the last residual, ours 2.028e-2 against PyTorch's own
# fp16 at 1.816e-2 — 1.12x. The old gate (`ours vs torch-fp16 < 3e-2`) read
# 2.968e-2 before the padded matmul changed its last ulp and 4.078e-2 after,
# while the distance from the *truth* went the other way, 2.093e-2 -> 2.028e-2.
# A gate that moves in the opposite direction to the accuracy it is meant to
# protect is the wrong gate.
#
# `add_64` is the last residual and the one node both exports name identically —
# their attention decompositions differ, so the graph's own output node exists in
# only one of them.
if prec == "fp16"
    ws = Workspace(backend)
    gg = m.graphs["whisper"]
    vals = DNNKernels.execute!(gg, Dict{String,Any}(gg.inputs[1] => mel), m.weights;
                               dims = (;), backend, ws)
    o   = Float64.(Array(vals["add_64"]))
    t16 = Float64.(refs["whisper/node/add_64"])
    t32 = Float64.(fp32refs["whisper/node/add_64"])
    vals = nothing; GC.gc(true)
    rr(a, b) = sqrt(mean(abs2, a .- b)) / sqrt(mean(abs2, b))
    ours, torch = rr(o, t32), rr(t16, t32)
    @printf("  at the last residual: ours vs torch-fp32 %.4e, torch-fp16 vs torch-fp32 %.4e",
            ours, torch)
    @printf("  ->  %.2fx PyTorch's own fp16 drift\n", ours / torch)
    # 1.5x, not 1.0: two fp16 implementations of the same graph reassociate
    # differently and neither is the truth, so parity is not the same as
    # identity. Far enough above the measured 1.12x to pin behaviour, far enough
    # below the 5x this started at to fail loudly if the width regresses.
    ours > 1.5 * torch && push!(bad, "fp16 drift")
end

if isempty(bad)
    println("\nwhisper encoder ($prec) matches the PyTorch reference")
else
    println("\nMISMATCH in: ", join(bad, ", "))
    exit(1)
end
