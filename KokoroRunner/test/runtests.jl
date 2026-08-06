"""
Kokoro-82M, ported.

The G2P and the number readings are pure host code and run unconditionally —
they need no device and no artifact beyond the lexicon. Everything below
`ready()` needs the export.

The parity assertions are the point of the file. Kokoro decides its own output
length, so **the frame count is checked before the audio**: if the predicted
durations disagree with PyTorch, every sample after the divergence is being
compared against the wrong instant and a correlation on it means nothing.
"""

using Test
using KokoroRunner
using KokoroRunner: Kokoro, speak, phonemize, voices, tokenize, align, SAMPLERATE,
                    Lexicon, applystress, numwords, ordinalwords, yearwords, tokens,
                    assetdir, refsdir, ready
using DNNKernels: readsafetensors
using JSON3

@testset "KokoroRunner" begin

@testset "number readings" begin
    @test numwords(0) == "zero"
    @test numwords(42) == "forty two"
    @test numwords(100) == "one hundred"
    @test numwords(1250000) == "one million two hundred fifty thousand"
    @test ordinalwords(1) == "first"
    @test ordinalwords(23) == "twenty third"
    @test ordinalwords(40) == "fortieth"
    # The year rule, and the three places the paired reading does NOT apply.
    @test yearwords(1984) == "nineteen eighty four"
    @test yearwords(1905) == "nineteen oh five"
    @test yearwords(1900) == "nineteen hundred"
    @test yearwords(2000) == "two thousand"          # round century -> cardinal
    @test yearwords(2007) == "two thousand seven"    # and its remainder
    @test yearwords(2024) == "twenty twenty four"    # but 24 >= 10, so paired
end

@testset "tokenisation" begin
    @test tokens("Hello, world!") == ["Hello", ",", "world", "!"]
    @test tokens("don't stop") == ["don't", "stop"]
    @test tokens("well-known") == ["well-known"]
    # A comma inside digits is part of the number; anywhere else it separates.
    @test tokens("1,234.50 apples") == ["1,234.50", "apples"]
    @test tokens("Dr. Smith") == ["Dr", "Smith"]          # the abbreviation's stop is eaten
    @test tokens("stop. Go") == ["stop", ".", "Go"]       # a sentence's is not
end

@testset "stress placement" begin
    # The mark lands before its vowel, not at the front of the word.
    @test applystress("kat", 2) == "kˈat"
    @test applystress("ˈkat", -1) == "ˌkat"
    @test applystress("ˈkat", -2) == "kat"
    # A word with no vowel takes no mark rather than a stranded one.
    @test applystress("pst", 2) == "pst"
end

if !ready()
    @info "KokoroRunner: assets not installed — device tests skipped" dir = assetdir()
else

lex = Lexicon(joinpath(assetdir(), "lexicon.json"))

@testset "G2P against misaki" begin
    p = joinpath(refsdir(), "g2p_reference.json")
    if !isfile(p)
        @info "no g2p_reference.json in kokoro-refs — parity test skipped"
    else
        ref = JSON3.read(read(p, String))
        nok = ntot = 0
        for row in ref.sentences
            got = split(phonemize(lex, String(row.text)))
            want = split(String(row.phonemes))
            length(got) == length(want) || (ntot += max(length(got), length(want)); continue)
            ntot += length(want)
            nok += count(got .== want)
        end
        # 99.4% measured. The residue is the POS-conditioned heteronyms, which
        # need a part-of-speech tagger this package does not have; a drop below
        # this threshold means something other than tagging changed.
        @test nok / ntot > 0.98
        @info "G2P token agreement with misaki" agreement = nok / ntot tokens = ntot
    end

    # Every four-digit year, against `num2words` — the rule has three branches
    # and a hand-picked sample is how a wrong one survives.
    p2 = joinpath(refsdir(), "g2p_reference.json")
    if isfile(p2)
        nums = JSON3.read(read(p2, String)).numbers
        for (kind, fn) in (("cardinal", numwords), ("ordinal", ordinalwords),
                           ("year", yearwords))
            tbl = nums[Symbol(kind)]
            bad = [String(k) for (k, v) in pairs(tbl)
                   if split(fn(parse(Int, String(k)))) != String.(v)]
            @test isempty(bad)
            isempty(bad) || @info "$kind mismatches" first(bad, 5)
        end
    end
end

k = Kokoro()

@testset "assets" begin
    @test length(voices(k)) == 54
    @test "af_heart" in voices(k)
    # The boundary token at both ends is what the model was trained with.
    ids = tokenize(k, "hˈɛlO")
    @test first(ids) == 0 && last(ids) == 0
    @test length(ids) == 7
    # A character outside the vocabulary is dropped, not substituted.
    @test length(tokenize(k, "hˈɛlO☃")) == length(ids)
end

@testset "alignment" begin
    # `pred_dur[i]` frames of token `i`, and the clamp keeps a token that rounds
    # to zero from vanishing out of the alignment entirely.
    @test align([2.0, 1.0, 3.0], 1.0) == Int32[1, 1, 2, 3, 3, 3]
    @test align([0.1, 0.1], 1.0) == Int32[1, 2]
    @test length(align([4.0], 2.0)) == 2       # speed divides
end

@testset "parity with PyTorch" begin
    p = joinpath(refsdir(), "refs_speak.safetensors")
    mp = joinpath(refsdir(), "refs_speak.json")
    if !(isfile(p) && isfile(mp))
        @info "no refs_speak in kokoro-refs — parity test skipped"
    else
        refs = readsafetensors(p)
        meta = JSON3.read(read(mp, String))
        for key in sort(collect(String.(keys(meta))))
            m = meta[Symbol(key)]
            voice = String(split(key, "/")[2])
            # `noise = false` on BOTH sides: the vocoder's excitation noise is a
            # different random stream here than in PyTorch, so leaving it in
            # measures the streams and nothing about the model.
            got = speak(k; phonemes = String(m.phonemes), voice, noise = false, trim = false)
            want = vec(refs["$key/audio"])

            # The model's own prediction of how long the audio should be. First,
            # because everything after it depends on it.
            @test length(got) ÷ 600 == m.nframes

            n = min(length(got), length(want))
            c = sum(got[1:n] .* want[1:n]) /
                (sqrt(sum(abs2, got[1:n])) * sqrt(sum(abs2, want[1:n])))
            @test c > 0.95
        end
    end
end

@testset "length generalisation" begin
    # The export was done at 30 tokens. None of these is 30, and all of them go
    # through the same two graphs — no bucket, no re-export.
    for ps in ("hˈɛlO", "hˈɛlO wˈɜɹld",
               "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.")
        audio = speak(k; phonemes = ps, noise = false)
        @test length(audio) > 0
        @test all(isfinite, audio)
        @test maximum(abs, audio) < 1.5           # not clipping into nonsense
    end
end

@testset "determinism" begin
    rel(x, y) = sqrt(sum(abs2, x .- y) / length(x)) /
                sqrt(sum(abs2, x) / length(x))

    # **Not bit-equality, and the reason is a real property of the model.**
    # `index_put(accumulate=true)` — the iSTFT's overlap-add — is a scatter-add
    # over repeated indices, and it runs as an atomic fp32 add. Atomics complete
    # in whatever order the scheduler gives them and floating-point addition is
    # not associative, so two runs differ in the last one or two ULPs. PyTorch's
    # `index_put_(accumulate=True)` on CUDA has exactly the same property; it is
    # one of the ops `torch.use_deterministic_algorithms` refuses.
    #
    # Asserting a tolerance rather than equality is what makes the claim true.
    # The test's PURPOSE is unchanged and in fact sharper: what it exists to
    # catch is a `noise = false` that does not actually zero the excitation, and
    # the two scales are five orders of magnitude apart — so the gap is asserted
    # directly rather than left implicit in `!=`.
    # `trim = false` throughout: `trimsilence` picks its boundary from the audio
    # VALUES, so with the noise on the two runs trim to different lengths and a
    # sample-by-sample comparison has nothing to align. The frame count fixes the
    # length instead.
    a = speak(k; phonemes = "hˈɛlO", noise = false, trim = false)
    b = speak(k; phonemes = "hˈɛlO", noise = false, trim = false)
    @test length(a) == length(b)
    @test rel(a, b) < 1e-4

    c = speak(k; phonemes = "hˈɛlO", noise = true, trim = false)
    d = speak(k; phonemes = "hˈɛlO", noise = true, trim = false)
    @test c != d
    # The noise has to move the output by far more than atomic reordering does,
    # or `noise = false` is measuring nothing.
    @test rel(c, d) > 100 * rel(a, b)
end

@testset "speed" begin
    slow = speak(k; phonemes = "hˈɛlO wˈɜɹld", speed = 0.5, noise = false)
    fast = speak(k; phonemes = "hˈɛlO wˈɜɹld", speed = 2.0, noise = false)
    @test length(slow) > length(fast)
end

# Guards that the SHIPPED assets are the mixed-precision export. The published
# artifact is currently the fp32 one — 1.68x slower, silently — so its two key
# assertions are `@test_broken` until it is republished.
include(joinpath(@__DIR__, "test_assets_are_fp16.jl"))

end  # ready()
end  # KokoroRunner
