include("freeze_pubmedqa_retrieval.jl")
using Test
function test_freeze_pubmedqa_retrieval()
    @testset "Automatic retrieval comparison freezes all64 with private grading excluded" begin
        mktempdir() do directory
            p=freeze_pubmedqa_retrieval(;directory)
            @test p.ready_for_generation&&length(p.jobs)==64
            @test length(unique(j.task for j in p.jobs))==16
            @test all(j->j.input_tokens+160<=8192,p.jobs)
            @test all(j->length(j.messages)==2,p.jobs)
            @test count(j->j.arm=="retrieved_context",p.jobs)==16
            @test !haskey(p.source_sha256,"score_pubmedqa_retrieval.jl")
            @test !haskey(p.source_sha256,"pubmedqa_metrics.jl")
            @test p.decoding.max_tokens==128&&!p.decoding.thinking
            @test_throws ErrorException freeze_pubmedqa_retrieval(;directory)
        end
    end
end
