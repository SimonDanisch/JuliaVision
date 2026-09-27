"""
One whole Qwen-Image 2.1 generation, timed per stage, on each backend asked for.

    julia --project=<repo-root> JuliaVision/QwenImageRunner/benchmark/generate.jl vulkan rocm

or from a session, which is how it is meant to be run while working:

    include("JuliaVision/QwenImageRunner/benchmark/generate.jl")
    generationtimes(Mantle.LavaBackend())

The stages are the three components of `examples/generate.jl`, in its order and
with its staging: the encoder is released before the denoiser loads and the
denoiser before the decoder, because they do not fit on the card together. Every
stage ends with a device sync, so a stage is charged for its own GPU work and not
for the next one's.

A load is timed apart from a first call, and a first call apart from a warm one:
the first denoising step and the first decode compile and record their plans, and
a number that folds that in describes a cold start rather than the model.
"""

using QwenImageRunner, DNNKernels, Mantle, Random, Printf

const DEFAULT_PROMPT = "a red fox sitting in a snowy forest at sunrise, photorealistic"

sync(backend) = Mantle.waitidle(Mantle.todevice(backend))

"""Seconds `f()` takes, with the device idle at both ends."""
function timed(f, backend)
    sync(backend)
    t = time()
    x = f()
    sync(backend)
    return x, time() - t
end

"""
    generationtimes(backend; prompt, size = 1024, steps = 20, seed = 42, decodes = 3)

Per-stage seconds of one generation on `backend`, as a NamedTuple:

  * `encoder_load`, `encode`: build Qwen3-VL, then encode `prompt`;
  * `denoiser_load`: build the transformer for this prompt length and grid;
  * `first_step`, `step`: the first denoising step (it records the plan), then
    the mean over the remaining `steps - 1`;
  * `vae_load`, `first_decode`, `decode`: build the decoder, decode once, then the
    fastest of `decodes` more;
  * `total`: wall time from the first load to the first decoded image, which is
    what a user of `examples/generate.jl` waits for.

Also returns the decoded image, `(width, height, 4)` RGBA in [0, 1], and its NaN
count: a fast number from a broken generation must not pass for a result, and
the two backends' images side by side are the check that they generate the
same picture.
"""
function generationtimes(backend; prompt::AbstractString = DEFAULT_PROMPT,
                         size::Integer = 1024, steps::Integer = 20, seed::Integer = 42,
                         decodes::Integer = 3)
    side = size ÷ QWEN_IMAGE_21.vae_scale_factor
    t_start = time()

    encoder, encoder_load = timed(() -> qwenimagetextencoder(; backend), backend)
    embeds_host, encode = timed(() -> encode_prompt(encoder, prompt), backend)
    Mantle.release!(encoder)
    encoder = nothing

    transformer, denoiser_load = timed(backend) do
        qwenimagetransformer(; backend, context_tokens = Base.size(embeds_host, 2),
                             latent = (side, side))
    end
    Random.seed!(seed)
    tokens = image_sequence_length(size, size)
    channels = QWEN_IMAGE_21.latent_channels
    latents = DNNKernels.toback(backend, Float16.(randn(Float32, channels, tokens, 1)))
    embeds = DNNKernels.toback(backend, Float16.(embeds_host))
    timestep = DNNKernels.toback(backend, fill(Float16(0), 1))
    schedule = qwen_schedule(size, size; steps)
    step!(i) = begin
        fill!(timestep, convert(Float16, schedule.timesteps[i] / 1000f0))
        DNNKernels.eulerstep!(latents, denoise!(transformer, latents, embeds, timestep),
                              schedule.sigmas[i], schedule.sigmas[i + 1])
    end
    _, first_step = timed(() -> step!(1), backend)
    _, rest = timed(() -> foreach(step!, 2:steps), backend)
    denoised = Array(latents)
    latents = embeds = timestep = nothing
    Mantle.release!(transformer)
    transformer = nothing

    vae, vae_load = timed(() -> qwenimagevae(; backend), backend)
    grid = unpacklatents(denoised, side, side)
    lat = DNNKernels.toback(backend, reshape(Float16.(grid), side, side, 1, channels, 1))
    image, first_decode = timed(() -> Array(decode!(vae, lat)), backend)
    total = time() - t_start
    decode = minimum(last(timed(() -> decode!(vae, lat), backend)) for _ in 1:decodes)
    Mantle.release!(vae)

    return (; encoder_load, encode, denoiser_load, first_step,
            step = rest / (steps - 1), vae_load, first_decode, decode, total,
            steps, size, nan = count(isnan, image),
            image = Float32.(image[:, :, 1:4, 1]) ./ 2f0 .+ 0.5f0)
end

"""Print `generationtimes` results side by side, one column per backend."""
function printtimes(io::IO, results::AbstractVector{<:Pair})
    rows = (:encoder_load => "encoder load", :encode => "encode prompt",
            :denoiser_load => "denoiser load", :first_step => "first step (records)",
            :step => "step (mean of the rest)", :vae_load => "VAE load",
            :first_decode => "first decode (records)", :decode => "decode (warm)",
            :total => "total, first image")
    @printf(io, "%-26s", "stage")
    foreach(r -> @printf(io, "%14s", first(r)), results)
    println(io)
    for (key, label) in rows
        @printf(io, "%-26s", label)
        foreach(r -> @printf(io, "%12.2f s", getfield(last(r), key)), results)
        println(io)
    end
    @printf(io, "%-26s", "NaN in the image")
    foreach(r -> @printf(io, "%14d", last(r).nan), results)
    println(io)
    return nothing
end

"""The loaded backend called `name` (`vulkan` or `rocm`), from Mantle's registry."""
function backendnamed(name::AbstractString)
    want = Dict("vulkan" => :LavaBackend, "rocm" => :ROCBackend)[lowercase(name)]
    for b in Mantle.eachbackend()
        nameof(typeof(b)) === want && return b
    end
    error("no $name backend is loaded; for rocm, `using AMDGPU` first")
end

if abspath(PROGRAM_FILE) == @__FILE__
    names = isempty(ARGS) ? ["vulkan"] : ARGS
    "rocm" in lowercase.(names) && @eval using AMDGPU
    results = Pair{String,Any}[]
    for n in names
        b = Base.invokelatest(backendnamed, n)
        push!(results, n => Base.invokelatest(generationtimes, b))
        GC.gc(true); Mantle.trim_gpu_pool!()
    end
    printtimes(stdout, results)
end
