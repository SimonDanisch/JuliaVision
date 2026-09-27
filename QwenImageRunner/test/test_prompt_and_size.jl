"""
How long a prompt may be and how large a picture.

**The prompt.** The text encoder was exported static at 64 tokens until
2026-09-27. After the 14-token system turn the pipeline drops and the 8 tokens
of template around the prompt, that left 42 tokens, about thirty words, and a
longer prompt threw. The reference pipeline truncates nothing. The encoder's
length is now its `t` symbol, bounded by the system turn plus the 1024 prompt
tokens the denoiser takes, and a prompt is right-padded to a bucket so a new
length does not compile a new plan. Padding is exact only because the encoder
is causal, which is checked below rather than assumed.

The first check here is the one that failed before this file existed in a
different form: an exporter change reaches nobody until the artifact is rebound,
so the bound tree itself is asked whether its encoder is symbolic.

**The picture.** `generate` takes a width and a height, floors each to a multiple
of 32 as the reference does, and refuses a grid outside the denoiser's exported
token bounds before anything is built.
"""

using Test
using QwenImageRunner
using QwenImageRunner: encodertokens, encoderbucket, ENCODER_BUCKETS, qwen_resolution,
                       qwenimagetextencoder, encode_prompt, qwen_prompt_template, token_embeddings,
                       ENCODERGRAPH, ready, assetdir
using DNNKernels
using Mantle

@testset "a prompt goes to the first bucket it fits" begin
    @test encoderbucket(1, 1038) == 64
    @test encoderbucket(64, 1038) == 64
    @test encoderbucket(65, 1038) == 128
    @test encoderbucket(512, 1038) == 512
    @test encoderbucket(513, 1038) == 1038
    @test encoderbucket(1038, 1038) == 1038
    # A graph exported with a smaller bound than a bucket takes its bound.
    @test encoderbucket(100, 96) == 96
    @test issorted(ENCODER_BUCKETS)
end

@testset "sizes are floored to 32 and bounded by the denoiser" begin
    if ready(:transformer)
        @test qwen_resolution(1024, 1024) == (1024, 1024)
        @test qwen_resolution(1664, 928) == (1664, 928)
        @test (@test_logs (:warn,) qwen_resolution(1000, 1030)) == (992, 1024)
        @test_throws ArgumentError qwen_resolution(16, 16)       # under 32
        @test_throws ArgumentError qwen_resolution(32, 32)       # 4 tokens, under 16
        @test_throws ArgumentError qwen_resolution(4096, 2048)   # 32768 tokens, over 16384
    else
        @test_skip ready(:transformer)
    end
end

@testset "the bound text encoder takes a prompt of any length" begin
    if ready(:text_encoder)
        maxtokens, symbolic = encodertokens(assetdir())
        @test symbolic
        @test maxtokens == 14 + QWEN_IMAGE_21.max_prompt_tokens
    else
        @test_skip ready(:text_encoder)
    end
end

const LONG_PROMPT = "A sprawling steampunk harbor city at golden hour, seen from a hillside: " *
    "brass airships with patched canvas balloons moor at iron towers, cobblestone streets " *
    "wind between crooked timber houses with copper roofs gone green, a crowded fish market " *
    "spills onto the docks, gulls circle above tall ships whose sails glow orange in the low " *
    "sun, steam vents from chimneys and drifts across the water, and in the foreground a " *
    "young mechanic in goggles and a leather apron repairs a small clockwork bird on a " *
    "workbench covered in gears, highly detailed, cinematic lighting, shallow depth of field."

function encodercheck(backend)
    @testset "long prompts and exact padding — $(nameof(typeof(backend)))" begin
        enc = qwenimagetextencoder(; backend)
        try
            # 129 tokens with its template: three times what the static graph held.
            long = encode_prompt(enc, LONG_PROMPT)
            @test size(long) == (4096, 129, 1)
            @test all(isfinite, long)
            @test size(encode_prompt(enc, "a red fox")) == (4096, 11, 1)

            # Right padding is exact because the encoder is causal: at one bucket,
            # what the padding CONTAINS must not reach a kept row.
            ids = QwenImageRunner.encode(enc.tokenizer, qwen_prompt_template(LONG_PROMPT))
            kept = (enc.drop + 1):length(ids)
            function rows(padid)
                emb = token_embeddings(vcat(ids, fill(padid, 256 - length(ids))); compact = enc.compact)
                x = DNNKernels.toback(backend, reshape(emb, size(emb, 1), size(emb, 2), 1))
                Array(first(DNNKernels.call(enc.model, ENCODERGRAPH, x; dims = (; t = 256))))[:, kept]
            end
            @test rows(151643) == rows(1000)

            too_long = join(fill("gear", 1100), " ")
            @test_throws ArgumentError encode_prompt(enc, too_long)
        finally
            Mantle.release!(enc)
        end
    end
end

if ready(:text_encoder)
    foreach(encodercheck, Mantle.eachbackend())
else
    @testset "long prompts and exact padding" begin
        @test_skip ready(:text_encoder)
    end
end

# End to end at a small, non-square size: the latent grid's two extents reach the
# rotary tables and the unpacking separately, so a swapped width and height would
# still run at a square size and not here.
if ready()
    @testset "generate at a non-square size" begin
        img = generate("a red fox"; width = 256, height = 128, steps = 2)
        @test size(img) == (256, 128, 4)
        @test count(isnan, img) == 0
        @test all(x -> 0f0 <= x <= 1f0, img)
    end
else
    @testset "generate at a non-square size" begin
        @test_skip ready()
    end
end
