"""
The encoder, end to end, on whatever device is here.

The artifact carries the graph and the weights but **not** the references — those
are ~1 GB of developer material and belong in a `whisper-refs` artifact beside it
(the shape `sam2-large-refs` and `matanyone-refs` already set), which does not
exist yet. So this file cannot assert *numerical parity*; that lives in
`tools/verify_whisper.jl gpu fp32`, which reads `gen/graphs/whisper/` directly and
measured rel rms 6.30e-5 / cosine 0.999999998.

What it can assert, and does, is everything parity would be worthless without:
the artifact resolves, the model builds, one real 30 s window runs, the output has
the right shape, and **every element of it was written** — the encoder's output is
checked for NaN, which is what a partially-written result looks like when the
scratch slab is poisoned (GUARDRAILS 3). A graph that silently skips part of its
output passes a shape check and fails this one.

Still to write: `frozen_stats().misses == 0` in a fresh process, which is the
latency claim the package exists for. It has to run in a subprocess because
Julia's compile-time counter is per-process; see SAM2Runner/test for the shape.
"""

using Test, WhisperRunner, KernelAbstractions, Lava
const KA = KernelAbstractions

@testset "WhisperRunner" begin
    # No `assetdir()` in the assertions. It is internal — it names where the
    # artifact happens to put things, so a test that calls it has to know the
    # layout and a re-export that moves a file breaks a test that never knew it
    # depended on that. Ask for the graph and the weights instead.
    if WhisperRunner.ready()
        @info "Whisper large-v3-turbo: artifact present"
        g = WhisperRunner.whispergraph()
        @test g !== nothing
        @test length(g.ops) == 617          # the fp32 export; fp16 traces 681

        # A device is not guaranteed on every machine that runs this suite, and
        # loading 2.37 GiB of weights onto one that has none should skip rather
        # than fail.
        #
        # Catching only `LavaError` on purpose. A bare `catch` here already ate an
        # `UndefVarError` — `LavaBackend` was not in scope, the whole encode was
        # skipped, and the file reported 2 green assertions instead of 5. A guard
        # that swallows bugs in the test itself is worse than no guard.
        backend = try
            b = LavaBackend(); KA.synchronize(b); b
        catch err
            err isa Lava.LavaError || rethrow()
            @info "no working device; skipping the encode" exception = err
            nothing
        end

        if backend !== nothing
            m = WhisperRunner.whispermodel(; backend)
            mel = KA.allocate(backend, Float32, 3000, 128, 1)
            fill!(mel, 0.0f0)
            h = WhisperRunner.encode(m, mel)
            KA.synchronize(backend)
            got = Array(h)
            @test size(got) == (1280, 1500, 1)
            # The coverage assertion. An op that writes part of its output leaves
            # the rest as whatever was in the slab; a NaN here says so, where a
            # plausible number would not.
            @test count(isnan, got) == 0
            # A zero mel is not a zero hidden state — the encoder has biases and
            # a learned positional embedding. All-zero output would mean nothing
            # ran at all, which is the other way this can look green and be dead.
            @test any(!iszero, got)
        end
    else
        @info "Whisper large-v3-turbo: no export; run tools/export_whisper.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError WhisperRunner.whispergraph()
    end
end
