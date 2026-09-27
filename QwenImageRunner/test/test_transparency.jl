"""
Transparent backgrounds: the decoder's fourth channel is a real alpha matte.

This is one of the main reasons to use Qwen-Image 2.1, and for the first months
of this port it did not come through. The VAE was exported in fp16 while the
checkpoint is fp32, and the decoder's intermediates leave fp16's 65504 range:
seeded `randn` latents decoded to 80.9% NaN, and a "transparent background"
prompt measured 2026-09-23 decoded to 60.5% NaN, the NaN being exactly the
transparent part. Opaque backgrounds stay in range, and the old decode test
scaled its latents by 0.3 "because raw N(0,1) overflows fp16", so nothing
noticed. The exporter also loaded the fp32 checkpoint as bf16 before casting.

Fixed 2026-09-26: the VAE ships at fp32 with the checkpoint's own weights.
Against diffusers at fp32 on the latents below, mean |alpha error| is 1.4e-6.

Since 2026-09-27 the default decoder runs its convolutions on fp16 operands
(`halfconvs`) and both decoders are pinned here. What overflowed is the residual
stream, which reaches 3.5e5 on the `randn` latents below; a plain fp16 cast of
the one convolution that reads it at that size gave 0.37% Inf there. ROCm then
decoded 70% NaN, and Vulkan decoded no NaN at all but a wrong picture: mean
|difference| from the fp32 decoder 0.60, against 2.0e-4 for `halfconvs`. So the
`randn` check compares with the fp32 decoder and does not stop at NaN.
"""

using Test
using QwenImageRunner
using QwenImageRunner: qwenimagevae, decode!, latenttype, unpacklatents, ready, vaedir, refsdir
using DNNKernels
using Mantle
using Random

# On every backend this session has, not only Vulkan: ROCm's attention computed
# one head in 32 until 2026-09-27 (a launch-size bug in KernelInterface), and
# nothing here ran there to see it.
function transparencycheck(backend)
    @testset "transparent backgrounds come through the decoder — $(nameof(typeof(backend)))" begin
        vae32 = qwenimagevae(; backend, halfconvs = false)
        vae16 = qwenimagevae(; backend)
        try
            # Unscaled N(0,1), handed over in fp16 as the denoiser hands its
            # latents over: `decode!` converts. The fp16 decoder returned 80.9%
            # NaN for exactly this input.
            Random.seed!(0)
            lat = DNNKernels.toback(backend, Float16.(randn(Float32, 16, 16, 1, 64, 1)))
            ref = Array(decode!(vae32, lat))
            @testset "the fp32 decoder" begin
                @test latenttype(vae32) === Float32
                @test size(ref) == (256, 256, 4, 1)
                @test count(isnan, ref) == 0
                @test all(x -> -1.001f0 <= x <= 1.001f0, ref)
            end
            @testset "fp16 convolution operands keep fp32's range" begin
                @test latenttype(vae16) === Float32
                img = Array(decode!(vae16, lat))
                @test count(isnan, img) == 0
                d = abs.(img .- ref)
                # Measured 2.0e-4 mean and 48 of 262144 values past 0.05 on
                # Vulkan and ROCm alike; the plain cast was 0.60 and 140059.
                @test sum(d) / length(d) < 1f-3
                @test count(>(5f-2), d) < length(d) ÷ 1000
            end

            refs = DNNKernels.readsafetensors(joinpath(refsdir(), "transparent.safetensors"))
            lat = refs["latents"]                     # (64, 4096, 1), fp16, from the denoiser
            want = refs["alpha"]                      # (1024, 1024), diffusers at fp32
            grid = DNNKernels.toback(backend, reshape(unpacklatents(lat, 64, 64), 64, 64, 1, 64, 1))
            img32 = Array(decode!(vae32, grid))
            @testset "a transparent-background prompt decodes to its matte" begin
                alpha = Float32.(img32[:, :, 4, 1])
                @test count(isnan, img32) == 0
                # Half the picture is background, and it is transparent.
                @test 0.50 < count(<(0), alpha) / length(alpha) < 0.52
                @test maximum(abs.(alpha .- want)) < 1f-3
            end
            @testset "and to the same matte with fp16 convolution operands" begin
                img = Array(decode!(vae16, grid))
                alpha = Float32.(img[:, :, 4, 1])
                @test count(isnan, img) == 0
                @test 0.50 < count(<(0), alpha) / length(alpha) < 0.52
                # 3.1e-3 and 3.2e-3 at most, 1.0e-4 on average, on Vulkan and
                # ROCm. One step of an 8-bit channel is 7.8e-3 on this scale.
                @test maximum(abs.(alpha .- want)) < 5f-3
                @test sum(abs.(alpha .- want)) / length(alpha) < 2f-4
                # The colour where the picture is opaque; under a transparent
                # pixel it is arbitrary and differs by up to 0.07.
                opaque = alpha .> 0.9f0
                @test maximum(abs.(img[:, :, 1:3, 1][opaque, :] .- img32[:, :, 1:3, 1][opaque, :])) < 1f-2
            end
        finally
            Mantle.release!(vae16)
            Mantle.release!(vae32)
        end
    end
end

if !(ready(:vae_decoder) && isfile(joinpath(vaedir(), "vae.safetensors")))
    @info "no Qwen-Image VAE checkpoint; the transparency test is SKIPPED, not passing"
    @testset "transparent backgrounds come through the decoder" begin
        @test_skip ready(:vae_decoder)
    end
else
    foreach(transparencycheck, Mantle.eachbackend())
end
