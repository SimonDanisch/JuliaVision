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
        # 681, not 617: the shipped artifact is the **fp16** export, and the two
        # traces differ in more than dtype — fp16 gets 32 `clamp`s from
        # `WhisperEncoderLayer.forward`'s half-precision guard and
        # `_scaled_dot_product_flash_attention` where fp32 gets `_efficient_`.
        #
        # This assertion is why it is here rather than a shape check: it read 617
        # for as long as fp32 shipped, and the day the binding moved it was the
        # one thing that noticed. Bind a different export and this fails first.
        @test length(g.ops) == 681

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

            # ── the decoder half, which nothing here used to touch ────────────
            #
            # This exists because of a bug it would have caught immediately and
            # did not: `whisper-decoder.tar.gz` was bound, committed, and NEVER
            # UPLOADED. `decoderdir()` resolved to a release asset that did not
            # exist, so `whisper()` and `transcribe` — the package's headline
            # feature — worked only on the machine that had built the tree, and
            # 404'd for everyone else. It went unnoticed from the day the decoder
            # was bound until 2026-08-05, because every assertion above stops at
            # the encoder and every measurement was taken here.
            #
            # Resolving the artifact IS the test. It downloads 386 MiB on a cold
            # machine, which is the cost of asserting that the download works.
            @test WhisperRunner.decoderready()
            w = whisper(; backend)
            @test w.layers == 4 && w.heads == 20 && w.headdim == 64

            # ── and what it SAYS, against known speech ────────────────────────
            #
            # `fixtures/kokoro_pangrams_16k.wav` is 8.2 s of speech synthesised by
            # `KokoroRunner` (`af_heart`, `noise = false`), resampled to 16 kHz and
            # stored as Int16 — 257 KiB. Generated rather than recorded so the
            # ground truth is the sentence that was spoken, and chosen to contain
            # no digits and no -ise/-ize word: either would make this assert
            # Whisper's *normalisation* rather than its transcription. As it is,
            # the expected output is exactly the input, character for character.
            #
            # Regenerating it will not give byte-identical audio even at
            # `noise = false` — Kokoro's vocoder ends in an atomic scatter-add
            # whose summation order is not fixed, which moves samples by ~2e-7,
            # under one Int16 step but enough to flip a few on the rounding
            # boundary. The file is a recording, not a reproducible derivation.
            #
            # This is the assertion the suite lacked, and its absence is why a
            # decoder artifact that 404'd for everyone went unnoticed: everything
            # else here stops at the encoder.
            wavpath = joinpath(@__DIR__, "fixtures", "kokoro_pangrams_16k.wav")
            @test isfile(wavpath)
            raw = read(wavpath)
            @test raw[1:4] == b"RIFF" && raw[9:12] == b"WAVE"     # not a git-lfs stub
            pcm = Float32.(reinterpret(Int16, raw[45:end])) ./ 32767f0
            @test length(pcm) == 131416                            # 8.21 s at 16 kHz

            text, segs = transcribe(w, pcm)
            @test segs isa AbstractVector{<:Segment}
            @test strip(text) == "The quick brown fox jumps over the lazy dog. " *
                                 "Pack my box with five dozen liquor jugs. " *
                                 "How vexingly quick daft zebras jump."
        end
    else
        @info "Whisper large-v3-turbo: no export; run tools/export_whisper.py"
        # The error has to name the path — a caller who has not run the exporter
        # should be told where to put it, not handed a MethodError later.
        @test_throws ArgumentError WhisperRunner.whispergraph()
    end
end
