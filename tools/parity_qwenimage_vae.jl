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
# ## Why not the exporter's own reference
#
# `export_qwenimage21.py --component vae` writes `vae_reference.safetensors`
# from a `torch.randn` latent, and that reference cannot be reproduced in fp16 by
# anyone: the decoder's intermediates leave float16's 65504 range and 81% of the
# output comes back NaN. It is not a regression and not this tree's fault — the
# SHIPPED pre-symbolic decoder returns NaN on the same input, and PyTorch only
# escapes it because bfloat16 carries float32's exponent. A reference built from
# data the model never emits tests the wrong thing.
#
# So the latents here come out of twenty real denoise steps. Measured
# 2026-09-22, a 16x16 latent decoded to 256x256, ours fp16 on an 8060S against
# PyTorch bfloat16 on the host:
#
#     correlation   0.9999861
#     mean |diff|   0.00215     on a [-1, 1] output
#     RMS           0.00357
#     max |diff|    0.10352     isolated pixels, no structure
#
# The companion script renders ours, PyTorch and a 20x difference side by side;
# the difference is high-frequency speckle on the apple's skin and nothing else.

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
    QIR.euler_step!(latents, prediction, schedule.sigmas[i], schedule.sigmas[i + 1])
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
