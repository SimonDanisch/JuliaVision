include("freeze_pubmedqa_reading.jl")
using Test
function test_freeze_pubmedqa_reading()
    @testset "Source-reading protocol freezes all48 before model use" begin
        mktempdir() do directory
            p=freeze_pubmedqa_reading(;directory)
            @test p.ready_for_generation
            @test length(p.jobs)==48
            @test length(unique(j.task for j in p.jobs))==16
            @test all(j->length(j.messages)==2,p.jobs)
            @test all(j->j.input_tokens+160<=8192,p.jobs)
            @test count(j->j.arm=="corresponding_context",p.jobs)==16
            @test all(j->j.arm!="mismatched_context"||j.context_pmid!=j.task,p.jobs)
            @test !haskey(p.source_sha256,"score_pubmedqa_reading.jl")
            parent=JSON3.read(read(joinpath(@__DIR__,"alias_transfer3_results/conditions.json"),String))
            @test Set(keys(p.source_sha256))==union(Set(String.(keys(parent.source_sha256))),Set(["run_pubmedqa_reading.jl"]))
            @test_throws ErrorException freeze_pubmedqa_reading(;directory)
        end
    end
end
