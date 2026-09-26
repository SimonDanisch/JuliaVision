# Qwen-Image 2.1 VAE decoder: this tree against PyTorch, on latents the model
# actually produces.
#
#     julia --project=<repo-root> tools/parity_qwenimage_vae.jl [assetdir]
#     benchmark/torch.sh tools/parity_qwenimage_vae.py
#
# Two processes and two files between them, the same shape as
# `gap_vs_rocm.jl`/`.py`: they need different runtimes, and the INPUT is stated
# once — here — so both sides decode the identical latents.
#
# ## Latents the model actually produces
#
# The latents come out of twenty real denoise steps rather than from the
# exporter's `vae_reference.safetensors`, whose `randn` latents are far outside
# anything the denoiser emits. Until 2026-09-26 that reference could not be
# reproduced at all: the VAE was exported in fp16, its intermediates left fp16's
# 65504 range and 81% of the output came back NaN — the same failure that turned
# a transparent background into NaN. The VAE ships in fp32 now and matches that
# reference too (mean |diff| 1.1e-6), but real latents remain the fairer test.
#
# Measured 2026-09-26 at 1024², the fp32 decoder against diffusers at fp32 on the
# latents of a transparent-background prompt: mean |alpha diff| 1.4e-6, max
# 8.5e-5. The fp16 decoder it replaced was at 1.8e-3 against a bf16 reference.

using QwenImageRunner, DNNKernels, Mantle, KernelAbstractions, Random, Printf
const QIR = QwenImageRunner
const KA = KernelAbstractions

const PROMPT = "a red apple on a wooden table, studio photo"
const W = 256
const H = 256
const STEPS = 20
const SEED = 7

dir = isempty(ARGS) ? QIR.assetdir() : ARGS[1]
out = joinpath(@__DIR__, "..", "tmp")
mkpath(out)
backend = Mantle.LavaBackend()

lw, lh = W ÷ 16, H ÷ 16
@printf("prompt %s\n  %dx%d, latent %dx%d, %d steps, graph from %s\n",
        repr(PROMPT), W, H, lh, lw, STEPS, dir)

encoder = QIR.qwenimagetextencoder(; backend)
emb = QIR.encode_prompt(encoder, PROMPT)
Mantle.release!(encoder)
encoder = nothing
GC.gc(true); Mantle.trim_gpu_pool!()

transformer = QIR.qwenimagetransformer(; context_tokens = size(emb, 2), dir, latent = (lh, lw))
schedule = QIR.qwen_schedule(W, H; steps = STEPS)
Random.seed!(SEED)
latents = DNNKernels.toback(backend, Float16.(randn(Float32, 64, lh * lw, 1)))
timestep = similar(latents, eltype(latents), size(latents, 3))
embeddings = DNNKernels.toback(backend, emb)
for i in eachindex(schedule.timesteps)
    fill!(timestep, convert(eltype(timestep), schedule.timesteps[i] / 1000.0f0))
    prediction = QIR.denoise!(transformer, latents, embeddings, timestep)
    DNNKernels.eulerstep!(latents, prediction, schedule.sigmas[i], schedule.sigmas[i + 1])
end
KA.synchronize(backend)
Mantle.release!(transformer)
transformer = nothing
GC.gc(true); Mantle.trim_gpu_pool!()

grid = QIR.unpacklatents(Array(latents), lw, lh)
grid5 = reshape(grid, lw, lh, 1, size(grid, 3), size(grid, 4))

vae = QIR.qwenimagevae(; backend, dir)
image = Float32.(Array(QIR.decode!(vae, DNNKernels.toback(backend, Float16.(grid5)))))
KA.synchronize(backend)

# Raw Float32, no header. A Julia array of `(w, h, 1, c, b)` in column-major
# order is byte-identical to a C-order `(b, c, 1, h, w)` — torch's layout — so
# the reader needs no transpose and there is no axis convention to get wrong.
write(joinpath(out, "parity_latents.bin"), Float32.(vec(grid5)))
write(joinpath(out, "parity_ours.bin"), Float32.(vec(image)))
@printf("latents %s, image %s, %d NaN\n  wrote tmp/parity_latents.bin and tmp/parity_ours.bin\n",
        size(grid5), size(image), count(isnan, image))
@printf("now: benchmark/torch.sh tools/parity_qwenimage_vae.py\n")
