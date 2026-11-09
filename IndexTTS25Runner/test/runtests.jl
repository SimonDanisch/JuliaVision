using Test, IndexTTS25Runner, DNNKernels, JSON3, Mantle
using DNNKernels: readsafetensors
const R = IndexTTS25Runner

@testset "IndexTTS25Runner host rules" begin
    @test R.numbertowords(big(0)) == "null"
    @test R.numbertowords(big(21)) == "einundzwanzig"
    @test R.numbertowords(big(123)) == "einhundertdreiundzwanzig"
    @test R.numbertowords(big(10)^12) == "eine Billion"
    @test R.ordinaltowords(12) == "zwölfte"
    @test R.preparegerman("Dazu kommen 3 Gigabyte Speicher.") == "dazu kommen drei gigabyte speicher."
    @test R.annotate("<上手|じょうず>") == " じょうず "
    @test R.langtoken("DE") == 2
    @test R.langtoken("unknown") == 105
    @test R.silence(200) == 4410
    t, dt = R.eulerschedule(25)
    @test length(t) == length(dt) == 25
    @test t[1] == 0f0 && sum(dt) ≈ 1f0
    @test R.nearestindex(3, 6) == [0, 0, 1, 1, 2, 2]
    scores = Float32[3, -2, 6, -4]
    @test R.penalize!(scores, Set([0, 1]), 2f0) == Float32[1.5, -4, 6, -4]
    @test R.emotionweights("\"angry\"", "") == [0, 1, 0, 0, 0, 0, 0, 0]
    @test R.emotionweights("{}", "") == [0, 0, 0, 0, 0, 0, 0, 1]
    @test R.emotionweights("{\"悲伤\":0.9,\"低落\":0.1}", "gloomy") == [0, 0, 0.1, 0, 0, 0.9, 0, 0]
    for row in JSON3.read(read(joinpath(@__DIR__, "fixtures", "german_frontend.json"), String))
        @test R.preparegerman(String(row.text)) == row.prepared
    end
end

const ARTIFACTS = joinpath(@__DIR__, "..", "Artifacts.toml")
if !isfile(ARTIFACTS)
    @info "IndexTTS25Runner artifacts not bound; fixture tests skipped"
else
    include("parity.jl")
end
