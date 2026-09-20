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
