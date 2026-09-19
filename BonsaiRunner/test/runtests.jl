using Test, BonsaiRunner, DNNKernels

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
