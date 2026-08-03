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
"""

using DNNKernels, KernelAbstractions, Printf, Statistics
using LinearAlgebra: dot
using DNNKernels: readsafetensors, verifygraph, coverage, toback, Model, call, loadgraph
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

for (name, gg) in (("prefix (blocks 0-1)", prefix(g, "add_4")), ("all 617 ops", g))
    println("\n=== $name, node by node ===")
    t0 = time()
    ok, diffs, _ = verifygraph(gg, refs, weights; dims = (;), backend, verbose = true)
    @printf("  %.1f s\n", time() - t0)
    ok || (f = first(diffs);
           println("  FIRST MISMATCH at op $(f.index): $(f.id) ($(f.aten)) " *
                   "max|Δ| $(f.maxabs) rel $(f.relative) inflow $(f.inflow) $(f.shape)");
           push!(bad, name))
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
# fp16 carries an order of magnitude more of its own rounding, and the reference
# it is compared against is itself fp16, so the gate is the dtype's not a fixed one.
relrms > (prec == "fp16" ? 3e-2 : 1e-3) && push!(bad, "e2e")

if isempty(bad)
    println("\nwhisper encoder ($prec) matches the PyTorch reference")
else
    println("\nMISMATCH in: ", join(bad, ", "))
    exit(1)
end
