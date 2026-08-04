"""
Whisper large-v3-turbo's DECODER on Lava, against the PyTorch reference.

    julia --project=. tools/verify_whisper_decoder.jl [gpu|cpu]

Reads what `tools/dump_whisper_decoder_refs.py` wrote.

Four checks, in the order that a failure is cheapest to read:

  * *cross* — the 10-op `whispercross` graph, node by node. If this is wrong
    every token is wrong, and it is the smaller graph.
  * *step* — the 96-op `whisperdec` graph, node by node, at cache position 3.
    Position 3 and not 0 on purpose: at position 0 a mask that admits one slot
    too many, an `index_put` that writes the wrong slot, and an attention that
    ignores the cache entirely all agree. See the dumper's docstring.
  * *e2e* — both graphs through `Model`/`call`, which is **not** a cheaper
    version of the above: `Model` rewrites the graph before running it, and
    `hoistpermutes` in particular decides whether a matmul reaches the
    cooperative-matrix path. A graph verified only node by node is not verified.
  * *tokens* — `greedy` against a pure argmax loop over the reference step,
    exactly. This is the acceptance criterion for the decoding path as opposed
    to the arithmetic: every per-node tolerance can pass while the loop feeds
    back the wrong id, forgets to advance the cache, or stops on the wrong
    token. Deliberately NOT `generate`, which applies logits processors nobody
    has implemented yet — see the comment at that check.
"""

using DNNKernels, KernelAbstractions, Printf, Statistics, JSON3
using DNNKernels: readsafetensors, verifygraph, coverage, Model, call, loadgraph
const KA = KernelAbstractions

mode = isempty(ARGS) ? "gpu" : first(ARGS)
const DIR = "/sim/Programmieren/VideoEdit/gen/graphs/whisperdec"
const ENCDIR = "/sim/Programmieren/VideoEdit/gen/graphs/whisper"

isfile(joinpath(DIR, "refs.safetensors")) || error(
    "no refs at $DIR — run `uv run tools/dump_whisper_decoder_refs.py` first")

backend = if mode == "gpu"
    using Lava
    LavaBackend()
else
    KA.CPU()
end

@info "Whisper large-v3-turbo decoder on $mode"
weights = readsafetensors(joinpath(DIR, "weights.safetensors"))
refs = readsafetensors(joinpath(DIR, "refs.safetensors"))
bad = String[]

for name in ("whispercross", "whisperdec")
    g = loadgraph(joinpath(DIR, "$name.json"))
    _, missingops = coverage(g)
    isempty(missingops) && continue
    push!(bad, "$name: unimplemented ops " * join(sort(missingops), ", "))
end
isempty(bad) || error(join(bad, "\n"))

for name in ("whispercross", "whisperdec")
    g = loadgraph(joinpath(DIR, "$name.json"))
    @printf("\n=== %s: %d ops, node by node ===\n", name, length(g.ops))
    t0 = time()
    ok, diffs, _ = verifygraph(g, refs, weights; dims = (;), backend, verbose = true)
    @printf("  %.1f s\n", time() - t0)
    if !ok
        f = first(diffs)
        println("  FIRST MISMATCH at op $(f.index): $(f.id) ($(f.aten))")
        push!(bad, "$name node-by-node: first mismatch $(f.id) ($(f.aten))")
    end
end

# ---------------------------------------------------------------- e2e + tokens
#
# Through `WhisperRunner`, not through a hand-rolled loop here: the thing under
# test is the shipped path, cache copy-back and all, and a second implementation
# in the verifier would be a second thing that can be wrong.
using WhisperRunner
using WhisperRunner: whisper, fillcross!, greedy, whispermodel, tokenizer

w = whisper(; backend, decdir = DIR)

# `readsafetensors` already reverses torch's order, so the (1, 128, 3000) mel
# arrives as the (3000, 128, 1) the encoder wants — no permute here.
mel = Array(refs["whisperdec/mel"])
m = KA.allocate(backend, Float32, 3000, 128, 1)
copyto!(m, mel)
hid, = call(w.encoder, "whisper", m; dims = (;))

enc_ref = Array(refs["whispercross/in0"])           # (1, 1500, 1280)
enc_got = Array(hid)
rel(a, b) = sqrt(mean(abs2, Float64.(a) .- Float64.(b))) / sqrt(mean(abs2, Float64.(b)))
e = rel(vec(enc_got), vec(enc_ref))
@printf("\n=== e2e ===\n  encoder hidden: rel rms %.3e\n", e)
e < 1e-3 || push!(bad, @sprintf("encoder hidden rel rms %.3e", e))

gcrossout = loadgraph(joinpath(DIR, "whispercross.json")).outputs
fillcross!(w, hid)
# The graph's outputs are the `view`s over the stacks, which is what it names
# them; `whispercross/node/stack` is the pre-view fx node and is not recorded.
for (nm, got, key) in (("cross_k", w.cache.cross_k, "whispercross/node/" * gcrossout[1]),
                       ("cross_v", w.cache.cross_v, "whispercross/node/" * gcrossout[2]))
    local d = rel(vec(Array(got)), vec(Array(refs[key])))
    @printf("  %s: rel rms %.3e\n", nm, d)
    d < 1e-4 || push!(bad, @sprintf("%s rel rms %.3e", nm, d))
end

# Against the PURE ARGMAX loop, not against `generate`. `generate` runs Whisper's
# logits processors — suppression lists, begin-suppression, the timestamp rules —
# which change which token wins, and none of those are implemented yet
# (`plans/whisper-decoder.md` D6). Comparing against it here would report a
# missing *policy* as a broken *cache*. `whisperdec/generate_tokens` holds that
# target for when D6 lands.
prompt = Int.(Array(refs["whisperdec/prompt"]))
gen = Int.(Array(refs["whisperdec/tokens"]))
# `maxtokens = length(gen)` exactly: the reference stopped at ITS cap, not at
# an end-of-text token, so allowing more here would report a length difference
# that says nothing.
got = greedy(w, prompt; maxtokens = length(gen))
n = min(length(got), length(gen))
firstdiff = findfirst(i -> got[i] != gen[i], 1:n)
@printf("\n=== tokens ===\n  reference %d, greedy %d, first difference %s\n",
        length(gen), length(got), firstdiff === nothing ? "none" : string(firstdiff))
if got != gen
    println("  want ", gen[1:min(end, 16)])
    println("  got  ", got[1:min(end, 16)])
    push!(bad, "greedy token sequence differs from the argmax reference" *
               (firstdiff === nothing ? " (length only)" : " at $firstdiff"))
end

# ------------------------------------------------------------------- tokenizer
#
# Against HuggingFace's own `WhisperTokenizer`, on strings picked to break the
# merge order rather than to be representative: leading spaces, contractions,
# digits, non-ASCII, emoji, whitespace runs. A BPE encoder that is subtly wrong
# still produces plausible ids, and nothing downstream would notice — the ids go
# into a prompt, and a wrong prompt degrades the transcript without failing.
tokrefpath = joinpath(DIR, "tokenizer_refs.json")
if isfile(tokrefpath)
    tr = JSON3.read(read(tokrefpath, String))
    tk = tokenizer(; dir = DIR)
    nenc = nbad = 0
    for (text, want) in pairs(tr[:encode])
        got = WhisperRunner.encode(tk, String(text))
        nenc += 1
        if got != Int.(collect(want))
            nbad += 1
            nbad <= 3 && println("  encode mismatch ", repr(String(text)),
                                 "\n    want ", Int.(collect(want)), "\n    got  ", got)
        end
    end
    @printf("\n=== tokenizer ===\n  encode: %d/%d probe strings exact\n", nenc - nbad, nenc)
    nbad == 0 || push!(bad, "tokenizer encode: $nbad of $nenc probes differ")

    gottext = WhisperRunner.decode(tk, gen)
    wanttext = String(tr[:decode_tokens])
    println("  decode: ", gottext == wanttext ? "exact" : "DIFFERS")
    if gottext != wanttext
        println("    want ", repr(first(wanttext, 90)))
        println("    got  ", repr(first(gottext, 90)))
        push!(bad, "tokenizer decode differs from WhisperTokenizer")
    else
        println("  transcript of the reference tokens: ", repr(first(gottext, 90)))
    end
else
    println("\n=== tokenizer ===\n  no tokenizer_refs.json — re-run the dumper")
end

println()
if isempty(bad)
    println("ALL CHECKS PASSED")
else
    for b in bad
        println("FAILED: ", b)
    end
    exit(1)
end
