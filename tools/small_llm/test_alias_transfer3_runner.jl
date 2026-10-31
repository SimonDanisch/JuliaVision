include("run_alias_transfer3.jl")
using Test
function test_alias_transfer3_runner()
    @testset "External transfer registration and CPU-only paired skip" begin
        mktempdir() do directory
            packet=freeze_alias_transfer3(;directory)
            @test length(packet.jobs)==16
            @test all(j->j.eligible,packet.jobs)
            @test packet.decoding.thinking===false&&packet.decoding.max_tokens==4096
            @test length(unique(j.id for j in packet.jobs))==16
            @test !haskey(packet.source_sha256,"score_alias_transfer3.py")
            @test !haskey(packet.source_sha256,"julia_public_binding_docs.json") # excerpts frozen in prompts
            @test_throws ErrorException freeze_alias_transfer3(;directory)
        end
        mktempdir() do directory
            packet=(;source_sha256=Dict{String,String}(),jobs=[(;id="alias3_mock_direct",eligible=false)])
            open(io->JSON3.pretty(io,packet),joinpath(directory,"conditions.json"),"w")
            result=run_alias_transfer3(;directory)
            @test result.not_generated
            receipt=JSON3.read(read(joinpath(directory,"alias3_mock_direct.json"),String))
            attempt=joinpath(directory,"alias3_mock_direct_attempt.json")
            @test receipt.not_generated&&receipt.gpu_setup===false
            @test receipt.reason=="paired_context_capacity_failure"
            @test receipt.attempt_sha256==alias3_digest(attempt)
            @test run_alias_transfer3(;directory).complete
        end
    end
end
