# Reproduce the MatAnyone device loss in ~20 seconds.
#
#     julia --project=. tools/repro_matanyone_deviceloss.jl 128 96    # FAULTS
#     julia --project=. tools/repro_matanyone_deviceloss.jl 256 144   # fine
#     julia --project=. tools/repro_matanyone_deviceloss.jl 512 288   # fine
#
# See task #13. The fault is **deterministic** at a given size — same timeline
# every run — which is why it kept being attributed to whatever was edited last.
#
# 128x96 is the only size tested whose deepest encoder feature map is 8x6, and
# therefore the only one that feeds `M = 48` to the 1x1-convolution -> `matmul!`
# route. Gating that route off, or skipping it for `M < 192`, removes the fault;
# eight other hypotheses did not (frozen cache, padgemm, the 1-D lift, the slab,
# the workspace rewind, the retired-buffer drop, split-K, the submit threshold).
#
# Per-op synchronisation also removes it, so it needs dispatches in flight — but
# submitting more often does not help, only waiting does.
#
# `matmul!` at those exact shapes does NOT fault standalone, with either native
# 2-D operands or 4-D ones reshaped as the route builds them.

using MatAnyoneRunner, Lava, DNNKernels, KernelAbstractions
using DNNKernels: toback
const KA = KernelAbstractions
backend = LavaBackend()
model = MatAnyoneRunner.matanyonemodel(; backend)
W, H = parse(Int, ARGS[1]), parse(Int, ARGS[2])
image = toback(backend, fill(0.5f0, W, H, 3, 1))
host = zeros(Float32, W, H); host[(W÷4):(3W÷4), (H÷4):(3H÷4)] .= 255.0f0
mask = toback(backend, host)
a = MatAnyoneRunner.runmatanyone(model, image, mask)
ah = Array(a); KA.synchronize(backend)
println("OK $(W)x$(H) finite=$(all(isfinite, ah)) max=$(maximum(ah))")
