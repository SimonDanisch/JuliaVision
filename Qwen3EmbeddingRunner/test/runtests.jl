using Test, Artifacts, Qwen3EmbeddingRunner
using Qwen3EmbeddingRunner: assetdir, refsdir, ready, tokens, bucket, DNNKernels, JSON3
using .DNNKernels: readsafetensors

@testset "Qwen3EmbeddingRunner" begin
    @test bucket(1, 512) == 32 && bucket(33, 512) == 64 && bucket(512, 512) == 512
    bindings = joinpath(@__DIR__, "..", "Artifacts.toml")
    @test basename(assetdir()) == string(artifact_hash("qwen3embedding", bindings))
    @test basename(refsdir()) == string(artifact_hash("qwen3embedding-refs", bindings))
    if !ready()
        error("The Qwen3-Embedding artifact is incomplete")
    else
        dir = assetdir()
        refs = refsdir()
        @test all(f -> isfile(joinpath(dir, f)), ["qwen3embedding.json", "weights.safetensors",
            "embed_tokens.safetensors", "export.json", "vocab.json", "merges.txt", "added_tokens.json"])
        @test !isfile(joinpath(dir, "reference.safetensors"))
        ref = readsafetensors(joinpath(refs, "reference.safetensors"))
        texts = String.(JSON3.read(read(joinpath(refs, "reference.json"), String)).texts)
        lengths = Int.(ref["lengths"])
        offsets = cumsum([0; lengths])
        m = qwen3embedding()
        # The reference ran the model card's recipe in fp32, its texts through
        # the Hugging Face tokenizer (which appends `<|endoftext|>` itself).
        for (i, text) in enumerate(texts)
            @test tokens(m, text) == Int.(ref["ids"][offsets[i]+1:offsets[i+1]])
        end
        # (1024, n): readsafetensors reverses PyTorch's row-major axes.
        expected = ref["embeddings"]
        got = reduce(hcat, [embed(m, t) for t in texts])
        cosines = [sum(got[:, j] .* expected[:, j]) for j in axes(got, 2)]
        @info "cosine to the fp32 reference" cosines
        @test all(>(0.999), cosines)
        @test all(j -> abs(sum(abs2, got[:, j]) - 1) < 1e-5, axes(got, 2))
        # Retrieval agrees with the reference: each query's best document is the same.
        nq = JSON3.read(read(joinpath(refs, "reference.json"), String)).queries
        best(e) = [argmax(e[:, nq+1:end]' * e[:, q]) for q in 1:nq]
        @test best(got) == best(expected)
        # The instruction is the export's, and the same query embedded through
        # `embed_query` is the instructed text's embedding.
        query = last(split(texts[1], "Query:"))
        @test embed_query(m, query) ≈ got[:, 1]
    end
end
