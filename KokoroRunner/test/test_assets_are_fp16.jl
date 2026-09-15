using Test, KokoroRunner, DNNKernels, Lava
using Mantle: LavaBackend

# The published artifact went stale and nothing noticed for however long.
#
# `tools/export_kokoro.py` produces a mixed-precision export (88% of weights
# fp16, `predictor.F0` and `bert` held exact) and task #71 recorded it as
# shipped. The artifact `KokoroRunner.assetdir()` actually resolves contains the
# fp32 export: 2456 fp32 buffers and ZERO fp16 ones, against the fp16 export's
# 1595/1276. Measured cost of that, same code, only the asset tree differing:
#
#     artifact (90 fp32-operand convs)   min 380.9 ms
#     fp16 export (9 fp32-operand)       min 227.1 ms      1.68x
#
# The mechanism: `conv_coopmat_plan` gates on both operands being fp16, and this
# device has NO fp32 tensor-core path (fp32 measured 60x on the GEMM), so every
# convolution declined the cooperative-matrix kernel. An `opdouble` ablation puts
# the 90 convolutions at +341 ms of a 477 ms baseline — about 72% of the model.
#
# Nothing failed. The suite was green, the audio was correct, and the only symptom
# was a number in a benchmark table that had already been dismissed as GC noise.
# A stale publish is invisible because the runner resolves an artifact hash and
# loads whatever is behind it — so assert the PROPERTY the export promises.
@testset "shipped assets are the mixed-precision export" begin
    if !KokoroRunner.ready()
        @info "KokoroRunner: assets not installed — skipping" dir = KokoroRunner.assetdir()
    else
        k = KokoroRunner.Kokoro(; backend = LavaBackend())
        g = k.model.graphs["kokorovoc"]

        nfp16 = count(b -> b.dtype === Float16, values(g.buffers))
        convs = [op for op in g.ops if occursin("convolution", op.aten)]
        fp16convs = count(convs) do op
            b = get(g.buffers, op.ins[2], nothing)
            b !== nothing && b.dtype === Float16
        end

        # The fp16 export has 1276 fp16 buffers and 81 of 90 convolutions on fp16
        # operands; the fp32 export has 0 and 0. The thresholds are deliberately
        # loose — this catches "the wrong export shipped", not a small drift.
        #
        # These were `@test_broken` while the published artifact was the fp32
        # export. Rebound to `kokoro-fp16` (tree e72003e9) and uploaded to
        # assets-v1 on 2026-08-12, at which point both reported "Unexpectedly
        # Passing" — the signal this comment asked for — and became plain tests.
        # Measured then: 1278 fp16 buffers, 81 fp16 convolutions, and the
        # utterance 658.8 -> 285.7 ms.
        @info "shipped Kokoro assets" fp16_buffers = nfp16 fp16_convs = fp16convs
        @test nfp16 > 500
        @test fp16convs >= 70

        # The nine that stay fp32 are `predictor.F0`'s convolutions, held exact
        # because its error accumulates phase through sin(cumsum(F0)). If this
        # ever reaches 90, someone made F0 fp16 and the long-utterance cosine
        # check in tools/precision_blame.py has to be re-run.
        @test fp16convs <= 81
    end
end
