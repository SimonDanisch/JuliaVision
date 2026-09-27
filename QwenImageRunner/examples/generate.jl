"""
One Qwen-Image 2.1 text-to-image generation, end to end on Lava.

    julia --project=. QwenImageRunner/examples/generate.jl \\
        "a red fox sitting in a snowy forest at sunrise, photorealistic" fox.ppm

This is [`generate`](@ref) and a file writer; from a session, call `generate`
directly:

    img = generate("a red fox ..."; width = 1664, height = 928)   # (w, h, 4), RGBA in [0, 1]

Three environment variables, all optional:

  * `QWENIMAGE21_SIZE`: `1024` for a square or `WIDTHxHEIGHT`, e.g. `1664x928`.
    Sides are floored to multiples of 32; anything from 64x64 to 2048x2048 in
    image tokens runs, and the checkpoint is at its best around 1024x1024.
  * `QWENIMAGE21_STEPS`: denoising steps, 40 by default as in the reference.
  * `QWENIMAGE21_SEED`: the noise seed, 42 by default.

A prompt may be up to 1024 tokens, the reference pipeline's own limit.

Everything downloads itself: thirteen artifacts, 13.4 GB, fetched on the first
call and cached in the depot.

**Transparent backgrounds work:** ask for one in the prompt and the output is
RGBA with a real alpha matte, written as a PAM:

    julia --project=. QwenImageRunner/examples/generate.jl \\
        "A glossy red apple, isolated on a transparent background." apple.ppm

Writes a binary PPM, or a binary PAM when the decoder's alpha channel is not
opaque — netpbm both ways, so this needs no image package in this environment.
"""

using QwenImageRunner, Mantle

const PROMPT = length(ARGS) >= 1 ? ARGS[1] :
    "a red fox sitting in a snowy forest at sunrise, photorealistic"
const OUTPUT = length(ARGS) >= 2 ? ARGS[2] : "qwenimage21.ppm"

"""`QWENIMAGE21_SIZE` as `(width, height)`: one number is a square."""
function parsesize(s::AbstractString)
    parts = split(lowercase(s), 'x')
    length(parts) in (1, 2) || throw(ArgumentError("QWENIMAGE21_SIZE is `1024` or `WIDTHxHEIGHT`, got $s"))
    w = parse(Int, parts[1])
    return (w, length(parts) == 2 ? parse(Int, parts[2]) : w)
end

"""Write `(w, h, >=3)` Float32 in [0,1] as a binary PPM, RGB only."""
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

"""Write `(w, h, 4)` Float32 in [0,1] as a binary PAM — netpbm's RGBA, which
needs no image package here and which every viewer and ImageMagick reads."""
function writepam(path, rgba)
    w, h = size(rgba, 1), size(rgba, 2)
    open(path, "w") do io
        write(io, "P7\nWIDTH $w\nHEIGHT $h\nDEPTH 4\nMAXVAL 255\n" *
                  "TUPLTYPE RGB_ALPHA\nENDHDR\n")
        bytes = Vector{UInt8}(undef, 4w * h)
        i = 1
        for y in 1:h, x in 1:w, c in 1:4
            bytes[i] = round(UInt8, clamp(rgba[x, y, c], 0f0, 1f0) * 255)
            i += 1
        end
        write(io, bytes)
    end
    path
end

const WIDTH, HEIGHT = parsesize(get(ENV, "QWENIMAGE21_SIZE", "1024"))
const STEPS = parse(Int, get(ENV, "QWENIMAGE21_STEPS", "40"))
const SEED = parse(Int, get(ENV, "QWENIMAGE21_SEED", "42"))

t0 = time()
rgba = generate(PROMPT; width = WIDTH, height = HEIGHT, steps = STEPS, seed = SEED,
                backend = Mantle.LavaBackend())

# The fourth channel is a REAL matte, not a formality: ask for a transparent
# background and it comes back soft-edged with the background at 0, which is the
# model doing the cut-out for you. Ask for an opaque one — "plain white seamless
# background" — and it is 1.0 everywhere. So the image is written as RGBA
# whenever the alpha says anything, and as RGB when it does not.
alpha = @view rgba[:, :, 4]
if all(>=(0.999f0), alpha)
    writeppm(OUTPUT, rgba)
    println("wrote $OUTPUT ($(size(rgba, 1))x$(size(rgba, 2))) in $(round(time() - t0, digits=1)) s")
else
    out = replace(OUTPUT, r"\.ppm$" => "") * ".pam"
    writepam(out, rgba)
    println("alpha is not opaque (min $(round(minimum(alpha), digits=3))): wrote $out, RGBA, " *
            "$(size(rgba, 1))x$(size(rgba, 2)), in $(round(time() - t0, digits=1)) s")
end
