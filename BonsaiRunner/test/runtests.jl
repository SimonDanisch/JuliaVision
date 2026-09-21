using Test, BonsaiRunner, DNNKernels, Artifacts

@testset "checkpoint artifact bindings" begin
    toml = joinpath(pkgdir(BonsaiRunner), "Artifacts.toml")
    @test isfile(toml)
    @test BonsaiRunner.CHECKPOINT_BYTES == 5_946_648_928
    @test BonsaiRunner.CHECKPOINT_SHA256 ==
          "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3"
    for name in BonsaiRunner.CHECKPOINT_ARTIFACTS
        @test artifact_hash(name, toml) !== nothing
    end
end

@testset "Qwen35 GGUF tokenizer" begin
    enc, _ = BonsaiRunner._bytemap()
    tokens = string.(enc)
    push!(tokens, string(enc[Int(UInt8('h'))+1], enc[Int(UInt8('i'))+1]))
    md = Dict{String,Any}(
        "tokenizer.ggml.model"=>"gpt2",
        "tokenizer.ggml.tokens"=>tokens,
        "tokenizer.ggml.token_type"=>ones(Int,length(tokens)),
        "tokenizer.ggml.merges"=>[string(enc[Int(UInt8('h'))+1], ' ', enc[Int(UInt8('i'))+1])],
        "tokenizer.ggml.bos_token_id"=>0,
        "tokenizer.ggml.eos_token_id"=>1)
    file = GGUFFile("",UInt32(3),md,Dict{String,GGUFTensor}(),UInt64(0))
    tk = BonsaiTokenizer(file)
    ids = encode(tk,"hi")
    @test ids == [256]
    @test decode(tk,ids) == "hi"
    @test occursin("<|im_start|>user\nhello<|im_end|>", chatprompt(["user"=>"hello"]))
end


@testset "long-context KV sizing" begin
    @test BonsaiRunner.kv_bytes(1) == 33_280
    @test BonsaiRunner.kv_bytes(250_000) == 8_320_000_000
    @test BonsaiRunner.kv_bytes(262_144) == 8_724_152_320
end
