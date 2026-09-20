using Test, QwenImageRunner

@testset "Qwen-Image 2.1 architecture" begin
    c = QWEN_IMAGE_21
    @test c.transformer_layers == 32
    @test c.hidden_size == c.attention_heads * c.attention_head_dim == 4096
    @test c.latent_channels == 64
    @test c.vae_scale_factor == 16
    @test c.patch_size == 1
    @test image_sequence_length(1024, 1024) == 4096
    @test image_sequence_length(2048, 2048) == 16384
    @test_throws ArgumentError image_sequence_length(1000, 1024)
end

@testset "latent packing matches the Diffusers order" begin
    x = reshape(Float32.(1:(4 * 6 * 3 * 2)), 4, 6, 3, 2)
    p = packlatents(x)
    @test size(p) == (3, 24, 2)
    @test unpacklatents(p, 4, 6) == x
    @test p[:, 1, 1] == Float32[1, 25, 49]
    @test p[:, 2, 1] == Float32[2, 26, 50]
    @test_throws DimensionMismatch unpacklatents(zeros(Float32, 8, 3, 1), 4, 4)
end

@testset "Qwen dynamic FlowMatch schedule" begin
    s = qwen_schedule(1024, 1024; steps=25)
    @test length(s.timesteps) == 25
    @test length(s.sigmas) == 26
    @test s.sigmas[1] == 1f0
    @test s.sigmas[end-1] ≈ 0.02f0 atol=2f-7
    @test s.sigmas[end] == 0f0
    @test all(diff(s.sigmas) .< 0)
    @test s.mu ≈ 0.6935484f0 atol=2f-7
    @test s.sigmas[2] ≈ 0.97834116f0 atol=2f-7
    @test s.sigmas[end-2] ≈ 0.09564310f0 atol=2f-7

    x = ones(Float32, 2, 3)
    prediction = fill(2f0, 2, 3)
    euler_step!(x, prediction, 0.8f0, 0.5f0)
    @test x ≈ fill(0.4f0, 2, 3)
end

@testset "export discovery" begin
    @test !ready(dir="")
    withenv("JULIA_QWENIMAGE21_ASSETS" => nothing) do
        @test_throws ErrorException assetdir()
    end
    @test_throws ArgumentError qwenimagegraph(:not_a_component; dir=".")
    mktempdir() do dir
        touch(joinpath(dir, "qwenimage21_transformer.json"))
        touch(joinpath(dir, "transformer.safetensors"))
        @test ready(:transformer; dir)
        @test !ready(; dir)
    end
end

@testset "ConvRot is its own inverse" begin
    # The embedding table is stored in the rotated basis and nothing rotates a
    # lookup on the way in, so `token_embeddings` recovers the row by applying
    # the SAME transform again. That only works because each radix-4 stage is a
    # symmetric orthogonal matrix — if it were merely orthogonal this would be a
    # silently wrong embedding, and the image would still look like an image.
    x = reshape(Float32.(sinpi.((1:512) ./ 37)), 256, 2)
    rotated = QwenImageRunner.convrot!(copy(x), 256)
    @test rotated != x
    # Orthogonal: the norm of every group is preserved.
    @test sum(abs2, rotated) ≈ sum(abs2, x) rtol=1f-5
    @test QwenImageRunner.convrot!(copy(rotated), 256) ≈ x rtol=1f-5
    @test_throws DimensionMismatch QwenImageRunner.convrot!(zeros(Float32, 100, 1), 256)
    @test_throws ArgumentError QwenImageRunner.convrot!(zeros(Float32, 128, 1), 32)
end

@testset "the prompt template is the one the checkpoint was trained on" begin
    # A literal chat string, not `apply_chat_template`'s rendering of one: the
    # two tokenize differently and the checkpoint expects this one.
    template = qwen_prompt_template("a fox")
    @test startswith(template, "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n")
    @test occursin("<|im_start|>user\na fox<|im_end|>\n", template)
    @test endswith(template, "<|im_start|>assistant\n")
    @test startswith(template, QwenImageRunner.qwen_system_prefix())

    dir = get(ENV, "JULIA_QWENIMAGE21_PROCESSOR", "")
    if isfile(joinpath(dir, "vocab.json"))
        tk = QwenTokenizer(dir)
        # Against `AutoTokenizer` on the same string, which is where these came
        # from. The leading 14 are the system turn the pipeline drops.
        ids = QwenImageRunner.encode(tk, template)
        @test ids[1:3] == [151644, 8948, 198]
        @test length(QwenImageRunner.encode(tk, QwenImageRunner.qwen_system_prefix())) == 14
        @test QwenImageRunner.decode(tk, ids) == template
        @test QwenImageRunner.encode(tk,
            "a red fox sitting in a snowy forest at sunrise, photorealistic") ==
            [64, 2518, 38835, 11699, 304, 264, 89773, 13638, 518, 63819, 11, 4503, 89768, 4532]
    else
        @test_skip false
    end
end
