"""
Which of the two changes to the padded matmul moved Whisper's fp16 accuracy?

    julia --project=. tools/bench_padded_dst.jl

`matmul_coopmat!` changed twice at once: `N` is now padded to the staged kernel's
block instead of the tile, **and** the padded case now writes a destination of
`out`'s own type with the bias and activation fused into the GEMM's store,
instead of an fp32 scratch plus `mm_epilogue_kernel!`.

End to end the encoder's rel rms against PyTorch's *fp16* went 2.968e-2 ->
4.078e-2 while its distance from PyTorch's *fp32* went 2.093e-2 -> 2.028e-2 —
i.e. closer to the truth and further from the other fp16 implementation, which is
what an ill-conditioned model does when the last ulp moves. This attributes that
last ulp, on one op, against an fp64 reference over the same fp16 inputs, so the
claim rests on a measurement rather than on the shape of the story.

Four arms on Whisper's `fc2` (K = 5120), all against the same Float64 answer:

  * `tile+fp32`   — what shipped: NP = 1504, fp32 scratch, epilogue.
  * `block+fp32`  — NP = 1536, same fp32 route. Isolates the padding.
  * `block+fp16`  — NP = 1536, fp16 destination, bias in the accumulator.
                    Isolates the destination.
  * `exact->fp16` — the Float64 answer rounded once. The floor.

## The answer, 2026-08-05

    tile+fp32   (shipped)  rel rms 2.1263e-04
    block+fp32  (padding)  rel rms 2.1263e-04   0 of 1920000 elements differ
    block+fp16  (now)      rel rms 2.1261e-04   21758 of 1920000 differ
    exact -> fp16 (floor)  rel rms 2.0629e-04
    PyTorch's own fp16     rel rms 3.9635e-04

**The padding is bit-exact** — as it has to be, since the extra columns of `B`
are zeros and each output column depends only on its own column of `B`. All of
the movement is the destination, it is ~1 ulp on 1.1% of elements, and it lands
*marginally closer* to the exact answer, not further. Both routes sit within 3%
of what fp16 can represent at all and at half PyTorch's own error.

So the encoder's 2.968e-2 -> 4.078e-2 against PyTorch's *fp16* is one ulp on one
percent of elements, amplified over 32 blocks by the eight frames the model is
ill-conditioned at — and the distance from *fp32* went the other way.
"""

using Lava, DNNKernels, KernelAbstractions, Printf, Statistics, LinearAlgebra
using DNNKernels: Ctx, Workspace, scratch!, matmul_coopmat!, MMCoopMatPlan,
                  mm_epilogue_kernel!, padcols_kernel!, readsafetensors, toback
const KA = KernelAbstractions

relrms(a, b) = sqrt(sum(abs2, Float64.(a) .- Float64.(b)) / sum(abs2, Float64.(b)))

backend = LavaBackend()
ctx = Ctx(backend; ws = Workspace(backend))
const D16 = normpath(joinpath(@__DIR__, "..", "gen", "graphs", "whisper-fp16"))
W = readsafetensors(joinpath(D16, "weights.safetensors"))
R = readsafetensors(joinpath(D16, "refs.safetensors"))

Wj = W["encoder.layers.1.fc2.weight"]                 # (5120, 1280) Julia
bj = W["encoder.layers.1.fc2.bias"]
X  = reshape(R["whisper/node/clone_13"], 5120, 1500)  # PyTorch's own fp16 input
M, N, K = 1280, 1500, 5120
Y64 = permutedims(Float64.(Wj), (2, 1)) * Float64.(X) .+ Float64.(bj)

A = toback(backend, permutedims(Wj, (2, 1)))
B = toback(backend, X)
bias = toback(backend, bj)
out = KA.allocate(backend, Float16, M, N)

"The route that shipped, at whichever NP it is given."
function oldroute!(ctx, out, A, B, bias, M, N, K, NP)
    Bp = B
    if NP != N
        Bp = scratch!(ctx, Float16, K, NP)
        padcols_kernel!(ctx.backend)(Bp, B, Val(K), N; ndrange = (K, NP))
    end
    bs = Lava.coopmat_gemm_shape(M, NP, K)
    C = scratch!(ctx, Float32, M, NP, max(bs[2], 1))
    Lava.coopmat_gemm!(C, A, Bp, M, NP, K; blk_split = bs, partials = C, reduce = false)
    mm_epilogue_kernel!(ctx.backend)(out, C, bias, identity, Val(M), Val(bs[2]),
                                     M * NP, M * N; ndrange = M * N)
    out
end

# NOT `run` — `Base.run` is imported here and redefining a function name that
# already resolves to Base is an error, not a shadow.
once(f) = (DNNKernels.reset!(ctx.ws); fill!(out, Float16(0)); f();
           KA.synchronize(backend); Float64.(Array(out)))

results = [
    ("tile+fp32   (shipped)", once(() -> oldroute!(ctx, out, A, B, bias, M, N, K, 1504))),
    ("block+fp32  (padding)", once(() -> oldroute!(ctx, out, A, B, bias, M, N, K, 1536))),
    ("block+fp16  (now)", once(() -> matmul_coopmat!(ctx, out, MMCoopMatPlan(1536, 16),
                                                    A, B, bias, identity))),
    ("exact -> fp16 (floor)", Float64.(Float16.(Y64))),
    ("PyTorch's own fp16", Float64.(R["whisper/node/addmm_9"])),
]
println("fc2, M=$M N=$N K=$K, against the same Float64 answer over the same fp16 inputs:")
for (label, got) in results
    @printf("  %-22s rel rms %.4e   max|Δ| %.4e\n", label, relrms(got, Y64),
            maximum(abs.(got .- Y64)))
end
ref = results[1][2]
println("\n...and against what shipped:")
for (label, got) in results[2:3]
    @printf("  %-22s rel rms %.4e   elements differing: %d of %d\n", label,
            relrms(got, ref), count(!=(0), got .- ref), length(ref))
end
