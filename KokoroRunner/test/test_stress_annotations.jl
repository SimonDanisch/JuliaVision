using Test
using KokoroRunner: Lexicon, phonemize, stress_tokens, applystress

@testset "inline stress annotations" begin
    # Synthetic lexicon keeps these parser/context tests independent of assets
    # and of the GPU. Normal phoneme generation itself is unchanged.
    lex = Lexicon(Dict("or" => "ɔɹ", "opening" => "ˈOpənɪŋ", "bigger" => "bˈɪɡəɹ",
                       "apple" => "ˈæpəl", "lens" => "lˈɛnz", "two" => "tu"), Dict{String,String}())
    # Only punctuation INSIDE the label carries its metadata.
    @test stress_tokens("[or](+2), [opening](-1)!") ==
        [("or", 2.0), (",", nothing), ("opening", -1.0), ("!", nothing)]
    @test phonemize(lex, "[or](+2)") == "ˈɔɹ"
    @test phonemize(lex, "[opening](-1)") == "ˌOpənɪŋ"
    @test phonemize(lex, "[opening](-2)") == "Opənɪŋ"
    @test phonemize(lex, "[or](+1)") == "ˌɔɹ"
    @test phonemize(lex, "[2](+2)") == "tˈu"
    @test phonemize(lex, "[or](+0.5)") == "ˌɔɹ"
    @test phonemize(lex, "[opening](-0.5)") == "ˌOpənɪŋ"
    @test phonemize(lex, "[BIGGER](-2)") == "bɪɡəɹ"
    @test phonemize(lex, "[bigger opening](-2)!") == "bɪɡəɹ Opənɪŋ!"
    @test phonemize(lex, "[or](+2)[lens](-2)") == "ˈɔɹ lɛnz"
    @test phonemize(lex, "the [apple](-2)") == "ði æpəl"
    @test phonemize(lex, "the [lens](-2)") == "ðə lɛnz"
    @test phonemize(lex, "bigger opening!") == "bˈɪɡəɹ ˈOpənɪŋ!"
    @test isempty(stress_tokens(""))
    @test phonemize(lex, "") == ""
    @test stress_tokens("“[or](+2)”") == [("\"", nothing), ("or", 2.0), ("\"", nothing)]
    for level in ("-2", "-1", "0", "1", "+1", "+2", "0.5", "+0.5", "-0.5")
        @test last(only(stress_tokens("[or]($level)"))) == parse(Float64, level)
    end
    for level in ("NaN", "Inf", "+1.5", "loud", "", "/ɔɹ/")
        @test_throws ArgumentError phonemize(lex, "[or]($level)")
    end
end
