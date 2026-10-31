"""
The host-side half of the pipeline, against numbers taken from Diffusers and
transformers themselves (`diffusers 0.39`, `transformers 5.17`, 2026-10-12):
the timestep projection, the schedule, both rotary tables, the encoder mask, the
bf16 view and the reference sizing. None of it needs a GPU or the weights.

The device half is checked by `tools/check_fluxklein_smoke.jl` (each exported
graph against its torch reference) and end to end by comparing a generation with
the Diffusers pipeline run from the same noise and prompt embeddings (38.5 dB
PSNR for the 4B with one reference image, 36.4 dB text to image).
"""

using Test, FluxKleinRunner
const FK = FluxKleinRunner

@testset "FluxKleinRunner host side" begin
    @testset "timestep projection" begin
        # `Timesteps(256, flip_sin_to_cos = true, downscale_freq_shift = 0)` of 750.
        want = Dict(0 => -0.6669830083847046, 1 => 0.8797562718391418, 127 => 0.9967539310455322,
                    128 => 0.7450729608535767, 129 => 0.47542497515678406, 255 => 0.08050834387540817)
        got = FK.timestepproj(0.75)
        @test size(got) == (256, 1)
        for (i, v) in want
            @test Float32(got[i + 1, 1]) ≈ v atol = 2e-3     # fp16 output
        end
    end

    @testset "schedule" begin
        @test FK.empiricalmu(1024, 4) ≈ 2.0306897079499455
        @test FK.empiricalmu(3952, 4) ≈ 2.278969416639057
        @test FK.empiricalmu(5000, 4) ≈ 1.30301666
        @test FK.fluxschedule(3952, 4).sigmas ≈ Float32[1.0, 0.9669965505599976, 0.9071202278137207,
                                                         0.7650120258331299, 0.0] atol = 1e-6
    end

    @testset "FLUX rotary: text, target and reference ids" begin
        # `Flux2PosEmbed(2000, [32, 32, 32, 32])` of the ids (0,0,0,3), (0,2,5,0) and (10,1,1,0):
        # text token 4 of 4, target row 2 column 5 of a 3x6 grid, reference row 1 column 1 of a 2x2 grid.
        cols = (0, 1, 2, 33, 70, 100, 127)
        wantcos = [[1.0, 1.0, 1.0, 1.0, 1.0, 0.39925500750541687, 0.999997079372406],
                   [1.0, 1.0, 1.0, -0.416146844625473, 0.36017686128616333, 1.0, 1.0],
                   [-0.83907151222229, -0.83907151222229, 0.9979086518287659, 0.5403022766113281,
                    0.971226692199707, 1.0, 1.0]]
        wantsin = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.9168398976325989, 0.0024121550377458334],
                   [0.0, 0.0, 0.0, 0.9092974066734314, 0.9328840374946594, 0.0, 0.0],
                   [-0.5440211296081543, -0.5440211296081543, -0.06464039534330368, 0.8414709568023682,
                    0.23815695941448212, 0.0, 0.0]]
        cos, sin = FK.fluxrope(4, [(0, 3, 6), (10, 2, 2)]; axes = (32, 32, 32, 32), theta = 2000)
        @test size(cos) == (128, 4 + 18 + 4)
        for (r, token) in enumerate((4, 4 + 18, 4 + 18 + 4))
            @test Float32.(cos[collect(cols) .+ 1, token]) ≈ wantcos[r] atol = 1e-3
            @test Float32.(sin[collect(cols) .+ 1, token]) ≈ wantsin[r] atol = 1e-3
        end
    end

    @testset "Qwen3 rotary" begin
        # `Qwen3RotaryEmbedding(head_dim = 128, rope_theta = 1e6)` at positions 0, 5, 7.
        dims = (0, 1, 63, 64, 127)
        wantcos = [[1.0, 1.0, 1.0, 1.0, 1.0],
                   [0.28366219997406006, -0.6312609314918518, 1.0, 0.28366219997406006, 1.0],
                   [0.7539022564888, 0.8007261157035828, 1.0, 0.7539022564888, 1.0]]
        wantsin = [[0.0, 0.0, 0.0, 0.0, 0.0],
                   [-0.9589242935180664, -0.7755705118179321, 6.2046888160693925e-06, -0.9589242935180664,
                    6.2046888160693925e-06],
                   [0.6569865942001343, -0.599030613899231, 8.686563887749799e-06, 0.6569865942001343,
                    8.686563887749799e-06]]
        cos, sin = FK.qwen3rope(8, 128, 1e6)
        for (r, p) in enumerate((0, 5, 7))
            @test Float32.(cos[collect(dims) .+ 1, p + 1, 1]) ≈ wantcos[r] atol = 1e-3
            @test Float32.(sin[collect(dims) .+ 1, p + 1, 1]) ≈ wantsin[r] atol = 1e-3
        end
    end

    @testset "encoder mask: causal, right padding masked" begin
        m = FK.encodermask(3, 5)
        @test size(m) == (5, 5, 1, 1)
        for q in 1:5, k in 1:5
            @test (m[k, q, 1, 1] == 0) == (k <= q && k <= 3)
        end
        @test all(isinf, m[4:5, :, 1, 1])
    end

    @testset "bf16 view" begin
        bits = UInt16[0x3fc0, 0xc000, 0x0000, 0x4049]         # 1.5, -2.0, 0.0, 3.140625
        @test FK.BF16As{Float16}(bits) == Float16[1.5, -2.0, 0.0, 3.140625]
        @test FK.BF16As{Float32}(reshape(bits, 2, 2)) == Float32[1.5 0.0; -2.0 3.140625]
    end

    @testset "reference sizing" begin
        # The pipeline's: 2105x768 is shrunk to 1695x618 and cropped to 1680x608.
        img = rand(Float32, 2105, 768, 3)
        @test size(FK.fitreference(img)) == (1680, 608, 3)
        # Within the area and on the grid: untouched.
        small = rand(Float32, 832, 608, 3)
        @test FK.fitreference(small) == small
        # An area average keeps a constant picture constant.
        @test all(≈(0.25f0), FK.arearesize(fill(0.25f0, 100, 60, 3), 37, 23))
    end
end
