include("scalar_format_followup.jl")
using Test
function test_scalar_format_followup()
    tests=@testset "Scalar format followup registration only" begin
        mktempdir() do directory
            p=freeze_scalar_format_followup(;directory)
            @test length(p["jobs"])==4
            @test length(unique(j.id for j in p["jobs"]))==4
            @test all(j->j.arm=="scalar_program",p["jobs"])
            @test all(j->occursin("FINAL OUTPUT CONTRACT",last(j.messages).second),p["jobs"])
            @test all(j->occursin("Never output function",first(j.messages).second),p["jobs"])
            @test laptop_shared_source_state(directory).packet.decoding.max_tokens==2048
            @test_throws ErrorException freeze_scalar_format_followup(;directory)
            @test !score_scalar_format_followup(;directory).complete
        end
    end
    c=Test.get_test_counts(tests)
    report=(;complete=true,passed=c.passes+c.cumulative_passes,scope="CPU protocol only, not model performance",
        source_sha256=Dict(f=>scalar_format_digest(joinpath(@__DIR__,f)) for f in ("scalar_format_followup.jl","test_scalar_format_followup.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"scalar_format_followup_validation.json"),"w");report
end
