using Test, QwenImageRunner
using DNNKernels: eulerstep!
using Artifacts: artifact_hash, artifact_exists, artifact_path

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
    eulerstep!(x, prediction, 0.8f0, 0.5f0)
    @test x ≈ fill(0.4f0, 2, 3)
end

@testset "every artifact the loaders ask for is bound" begin
    toml = QwenImageRunner.ARTIFACTS_TOML
    @test isfile(toml)

    # Scanned out of the source rather than retyped here. The failure this
    # catches is a shard added to `compact_denoiser` and not to
    # `Artifacts.toml`: the package installs, `ready()` says yes, and the first
    # generation dies inside `@artifact_str` on a name nobody bound.
    asked = Set{String}()
    for f in readdir(dirname(pathof(QwenImageRunner)); join=true)
        endswith(f, ".jl") || continue
        for m in eachmatch(r"@artifact_str\(\"([^\"]+)\"\)", read(f, String))
            push!(asked, m.captures[1])
        end
    end
    @test length(asked) == 13
    for n in sort!(collect(asked))
        @test artifact_hash(n, toml) !== nothing
    end

    # `ready` is the question asked to decide whether to download, so it has to
    # cover every name a loader will reach for. One missing and it answers yes
    # to a pipeline that then fetches 6.8 GB.
    @test asked == Set(reduce(vcat, values(QwenImageRunner.COMPONENT_ARTIFACTS)))
    @test ready() isa Bool
    @test ready(:vae_decoder) isa Bool
    @test_throws ArgumentError ready(:not_a_component)
    @test_throws ArgumentError qwenimagegraph(:not_a_component; dir=".")
end

@testset "each artifact carries the licence it is redistributed under" begin
    # Section 3 of the Qwen Research License Agreement permits redistribution
    # only if each recipient gets a copy of the Agreement and the attribution
    # notice from 3(c). Each artifact is separately downloadable, so each has to
    # carry both — this asserts the condition on whichever are present rather
    # than downloading 14 GB to check.
    toml = QwenImageRunner.ARTIFACTS_TOML
    names = sort!(unique(reduce(vcat, values(QwenImageRunner.COMPONENT_ARTIFACTS))))
    present = filter(names) do n
        h = artifact_hash(n, toml)
        h !== nothing && artifact_exists(h)
    end
    for n in present
        dir = artifact_path(artifact_hash(n, toml))
        @test isfile(joinpath(dir, "LICENSE"))
        notice = joinpath(dir, "NOTICE")
        @test isfile(notice)
        @test occursin("Qwen RESEARCH LICENSE AGREEMENT", read(notice, String))
    end
    isempty(present) && @test_skip false
end

@testset "the downloaded trees have the files the loaders open" begin
    if ready()
        @test isfile(joinpath(assetdir(), "qwenimage21_transformer.json"))
        @test isfile(joinpath(assetdir(), "qwenimage21_text_encoder.json"))
        @test isfile(joinpath(assetdir(), "transformer_constants.safetensors"))
        @test isfile(joinpath(assetdir(), "text_encoder_constants.safetensors"))
        # The three `QwenTokenizer` opens, and only those: `tokenizer.json` is
        # 11 MB of BPE table this package never reads.
        @test isfile(joinpath(processordir(), "vocab.json"))
        @test isfile(joinpath(processordir(), "merges.txt"))
        @test isfile(joinpath(processordir(), "added_tokens.json"))
        @test isfile(joinpath(vaedir(), "qwenimage21_vae_decoder.json"))
        @test isfile(joinpath(vaedir(), "vae.safetensors"))

        toml = QwenImageRunner.ARTIFACTS_TOML
        for (shards, stem) in ((QwenImageRunner.DIT_SHARDS, "qwenimage21-dit-w"),
                               (QwenImageRunner.ENC_SHARDS, "qwenimage21-enc-w"))
            for (i, file) in enumerate(shards)
                dir = artifact_path(artifact_hash(stem * string(i), toml))
                @test isfile(joinpath(dir, file))
            end
        end
    else
        @test_skip false
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

    if ready()
        tk = QwenTokenizer(processordir())
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

# The decoder holds a Model and plans per grid; there is no unplanned path left.
include(joinpath(@__DIR__, "test_vae_plans.jl"))
