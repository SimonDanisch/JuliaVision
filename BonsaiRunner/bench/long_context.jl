using BonsaiRunner, KernelAbstractions, LinearAlgebra, Mantle

path = get(ENV, "BONSAI_GGUF", "")
isempty(path) && error("set BONSAI_GGUF to Ternary-Bonsai-2-27B-PTQ1_0.gguf")
selector = get(ENV, "BONSAI_DEVICE", "")
dev = isempty(selector) ? Mantle.device() : Mantle.device(selector)
chunk = parse(Int, get(ENV, "BONSAI_PREFILL_CHUNK", "8"))
ntokens = parse(Int, get(ENV, "BONSAI_PREFILL_TOKENS", "32"))

model = Bonsai2(path; device=dev, progress=true)
ids = encode(model.tokenizer, repeat("Vulkan barriers order memory accesses. ", 32))[1:ntokens]

# Run each pipeline shape once before measuring it. State is session-local, so
# the measured sessions below still start from the same zero state.
warmseq = session(model; kv_page=max(chunk, 4))
step!(warmseq, first(ids))
KernelAbstractions.synchronize(model.backend)
warmbatch = session(model; kv_page=max(chunk, 4))
prefill!(warmbatch, ids[1:min(chunk, ntokens)]; chunk=min(chunk, ntokens))
KernelAbstractions.synchronize(model.backend)
warmseq = warmbatch = nothing
GC.gc()

sequential = session(model; kv_page=max(ntokens, 4))
chunked = session(model; kv_page=max(ntokens, 4))

tseq = @elapsed begin
    global seq_logits = nothing
    for id in ids
        global seq_logits = step!(sequential, id)
    end
    KernelAbstractions.synchronize(model.backend)
end
tchunk = @elapsed begin
    global chunk_logits = prefill!(chunked, ids; chunk)
    KernelAbstractions.synchronize(model.backend)
end

a = Array(seq_logits)
b = Array(chunk_logits)
argmax_match = argmax(a) == argmax(b)
max_abs = maximum(abs, a .- b)
cosine = dot(a, b) / sqrt(dot(a, a) * dot(b, b))

println("native context:       ", model.maxcontext)
println("KV bytes/token:       ", BonsaiRunner.kv_bytes(1))
println("KV at 250k:           ", round(BonsaiRunner.kv_bytes(250_000) / 2.0^30; digits=3), " GiB")
println("KV capacity:           ", BonsaiRunner.kv_capacity(chunked))
println("sequential prefill:    ", round(ntokens / tseq; digits=3), " tok/s")
println("chunked prefill:       ", round(ntokens / tchunk; digits=3), " tok/s")
println("speedup:               ", round(tseq / tchunk; digits=3), "x")
println("last-logit max abs:    ", max_abs)
println("last-logit cosine:     ", cosine)
println("last-token argmax same:", argmax_match)

argmax_match || error("chunked and sequential prefill chose different next tokens")
cosine > 0.999f0 || error("chunked prefill logits diverged")
