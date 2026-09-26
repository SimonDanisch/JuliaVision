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
"""

using Test
using QwenImageRunner
using QwenImageRunner: qwenimagevae, decode!, latenttype, unpacklatents, ready, vaedir, refsdir
using DNNKernels
using Mantle
using Random

@testset "transparent backgrounds come through the decoder" begin
    if !(ready(:vae_decoder) && isfile(joinpath(vaedir(), "vae.safetensors")))
        @info "no Qwen-Image VAE checkpoint; the transparency test is SKIPPED, not passing"
        @test_skip ready(:vae_decoder)
    else
        backend = Mantle.LavaBackend()
        vae = qwenimagevae(; backend)
        try
            @testset "the decoder runs with fp32's range" begin
                @test latenttype(vae) === Float32
                # Unscaled N(0,1), handed over in fp16 as the denoiser hands its
                # latents over: `decode!` converts. The fp16 decoder returned
                # 80.9% NaN for exactly this input.
                Random.seed!(0)
                lat = Float16.(randn(Float32, 16, 16, 1, 64, 1))
                img = Array(decode!(vae, DNNKernels.toback(backend, lat)))
                @test size(img) == (256, 256, 4, 1)
                @test count(isnan, img) == 0
                @test all(x -> -1.001f0 <= x <= 1.001f0, img)
            end

            @testset "a transparent-background prompt decodes to its matte" begin
                refs = DNNKernels.readsafetensors(joinpath(refsdir(), "transparent.safetensors"))
                lat = refs["latents"]                     # (64, 4096, 1), fp16, from the denoiser
                want = refs["alpha"]                      # (1024, 1024), diffusers at fp32
                grid = unpacklatents(lat, 64, 64)
                img = Array(decode!(vae, DNNKernels.toback(backend, reshape(grid, 64, 64, 1, 64, 1))))
                alpha = Float32.(img[:, :, 4, 1])
                @test count(isnan, img) == 0
                # Half the picture is background, and it is transparent.
                @test 0.50 < count(<(0), alpha) / length(alpha) < 0.52
                @test maximum(abs.(alpha .- want)) < 1f-3
            end
        finally
            Mantle.release!(vae)
        end
    end
end
