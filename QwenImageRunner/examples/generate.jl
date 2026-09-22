"""
One Qwen-Image 2.1 text-to-image generation, end to end on Lava.

    julia --project=. QwenImageRunner/examples/generate.jl \\
        "a red fox sitting in a snowy forest at sunrise, photorealistic" fox.ppm

Three components run in sequence, and the sequence is the point: the denoiser
decodes to 7.26 GB of INT8 on the device and the Qwen3-VL conditioner to another
6.9 GB, which do not fit at once on an 8060S. Each is released as soon as its
output is in hand — the prompt embeddings are 360 KB and the latents 512 KB.

Everything downloads itself: thirteen artifacts, 13.4 GB, fetched on the first
call and cached in the depot.

The latent resolution is bound at export time; the prompt length is not. The
denoiser graph carries a `t` symbol and the plan binds it to this prompt's
length, which is why nothing here has to match a number chosen by the exporter.
The encoder is the one limit: it is exported for prompts up to 64 tokens.

Writes a binary PPM, which needs no image package in this environment.
"""

using QwenImageRunner, DNNKernels, Mantle, Random
using QwenImageRunner: encode, qwen_prompt_template, qwen_system_prefix

const PROMPT = length(ARGS) >= 1 ? ARGS[1] :
    "a red fox sitting in a snowy forest at sunrise, photorealistic"
const OUTPUT = length(ARGS) >= 2 ? ARGS[2] : "qwenimage21.ppm"
const STEPS = parse(Int, get(ENV, "QWENIMAGE21_STEPS", "20"))
const SIZE = parse(Int, get(ENV, "QWENIMAGE21_SIZE", "1024"))
const SEED = parse(Int, get(ENV, "QWENIMAGE21_SEED", "42"))

"""Write `(w, h, 3)` Float32 in [0,1] as a binary PPM."""
function writeppm(path, rgb)
    w, h = size(rgb, 1), size(rgb, 2)
    open(path, "w") do io
        write(io, "P6\n$w $h\n255\n")
        bytes = Vector{UInt8}(undef, 3w * h)
        i = 1
        for y in 1:h, x in 1:w, c in 1:3
            bytes[i] = round(UInt8, clamp(rgb[x, y, c], 0f0, 1f0) * 255)
            i += 1
        end
        write(io, bytes)
    end
    path
end

backend = Mantle.LavaBackend()
total = time()

t0 = time()
encoder = qwenimagetextencoder(; backend)
prompt_embeds = encode_prompt(encoder, PROMPT)
release!(encoder)
println("prompt: $(size(prompt_embeds, 2)) embeddings in $(round(time() - t0, digits=1)) s")

t0 = time()
# The graph is generic in prompt length; the recorded plan is bound to this
# prompt's length here, which is the only place it has to be known.
transformer = qwenimagetransformer(; backend, context_tokens = size(prompt_embeds, 2))
println("denoiser ready in $(round(time() - t0, digits=1)) s")

Random.seed!(SEED)
tokens = image_sequence_length(SIZE, SIZE)
channels = QWEN_IMAGE_21.latent_channels
latents = DNNKernels.toback(backend, Float16.(randn(Float32, channels, tokens, 1)))
embeds = DNNKernels.toback(backend, Float16.(prompt_embeds))
timestep = DNNKernels.toback(backend, fill(Float16(0), 1))
schedule = qwen_schedule(SIZE, SIZE; steps=STEPS)

t0 = time()
for i in 1:STEPS
    fill!(timestep, convert(Float16, schedule.timesteps[i] / 1000f0))
    euler_step!(latents, denoise!(transformer, latents, embeds, timestep),
                schedule.sigmas[i], schedule.sigmas[i + 1])
end
Mantle.waitidle(Mantle.todevice(backend))
denoised = Array(latents)
println("denoise: $(round(time() - t0, digits=1)) s for $STEPS steps " *
        "($(round((time() - t0) / STEPS, digits=2)) s/step)")
latents = embeds = timestep = nothing
release!(transformer)

t0 = time()
vae = qwenimagevae(; backend)
side = SIZE ÷ QWEN_IMAGE_21.vae_scale_factor
grid = unpacklatents(denoised, side, side)
image = Array(decode!(vae, DNNKernels.toback(backend,
    reshape(Float16.(grid), side, side, 1, channels, 1))))
println("decode: $(round(time() - t0, digits=1)) s")

# The decoder returns RGBA in [-1, 1]; the alpha channel is opaque for a
# text-to-image generation and the PPM has nowhere to put it.
rgb = Float32.(image[:, :, 1, 1:3, 1]) ./ 2f0 .+ 0.5f0
writeppm(OUTPUT, rgb)
println("wrote $OUTPUT in $(round(time() - total, digits=1)) s total")
