"""
Check SAM 2.1's two graphs against the PyTorch references, layer by layer.

    julia --project=. tools/verify_sam2.jl

Reads what `tools/dump_sam2_refs.py` wrote. The decoder is recorded per node, so
a mismatch there names the op that caused it; the encoder is recorded at its six
outputs only, so it answers yes or no, and `--nodes all` at a smaller size is the
next step when the answer is no.

This runs on the CPU backend, and deliberately so: `verifygraph` compares against
host references element by element, and a wrong answer here is a bug in an op's
arithmetic. Whether the GPU agrees with the CPU is a different question with a
different answer — see `tools/bench_sam2.jl`, which checks that end to end.
"""

using DNNKernels, KernelAbstractions
using DNNKernels: readsafetensors, verifygraph, coverage
const KA = KernelAbstractions

const DIR = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "sam2-large"))
const RES = 1024

isfile(joinpath(DIR, "refs.safetensors")) ||
    error("no refs at $DIR — run `uv run tools/dump_sam2_refs.py --size large` first")

@info "loading weights and references"
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
refs = readsafetensors(joinpath(DIR, "refs.safetensors"))

# Which graphs to check; the encoder alone is 1493 ops at 1024x1024 on a CPU
# backend, so being able to iterate on the decoder by itself matters.
wanted = isempty(ARGS) ? ["sam2_decoder", "sam2_encoder"] : ["sam2_" * a for a in ARGS]

bad = String[]
for name in wanted
    path = joinpath(DIR, "$name.json")
    println("\n=== $name ===")

    _, missingops = coverage(path)
    if !isempty(missingops)
        println("  unimplemented ops: ", join(sort(collect(missingops)), ", "))
        push!(bad, "$name (missing ops)")
        continue
    end

    ok, diffs, ties = verifygraph(path, refs, weights; dims=(res=RES,), verbose=false)
    if ok
        println("  matches the reference", isempty(ties) ? "" :
                "  ($(length(ties)) tie-sensitive predicate(s), expected)")
    else
        f = first(diffs)
        println("  FIRST MISMATCH at op $(f.index): $(f.id) ($(f.aten))")
        println("    max abs $(f.maxabs), relative $(f.relative), " *
                "error already on its inputs $(f.inflow), shape $(f.shape)")
        push!(bad, name)
    end
end

if isempty(bad)
    println("\n$(join(wanted, " and ")) match the PyTorch reference")
else
    println("\nMISMATCH in: ", join(bad, ", "))
    exit(1)
end
