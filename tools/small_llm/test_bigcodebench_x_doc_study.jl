include("bigcodebench_x_doc_study.jl")
using Test

function test_bigcodebench_x_doc_study()
    @testset "Prospective document-study registration; CPU only" begin
        mktempdir() do directory
            packet=freeze_bcbx_doc_study(;directory)
            @test length(packet.jobs)==16
            @test length(unique(j.task for j in packet.jobs))==8
            @test count(j->j.arm=="direct",packet.jobs)==8
            @test count(j->j.arm=="public_docs",packet.jobs)==8
            @test packet.decoding.max_tokens==4096
            @test packet.thinking_budget==2048
            @test readdir(directory)==["conditions.json"] # no attempt or Device
            prior=Set(String.(JSON3.read(read(joinpath(BCBX_SNAPSHOT,"manifest.json"),String)).selected_ids))
            @test all(!(String(j.task) in prior) for j in packet.jobs)
            for position in 1:2:16
                a,b=packet.jobs[position:position+1]
                @test a.task==b.task && a.eligible==b.eligible
                @test Set((a.arm,b.arm))==Set(("direct","public_docs"))
                @test all(length(j.messages)==2 for j in (a,b))
            end
            @test all(isempty(j.retrieved_names) for j in packet.jobs if j.arm=="direct")
            @test all(!isempty(j.retrieved_names) for j in packet.jobs if j.arm=="public_docs")
            @test all(j.prompt_tokens+packet.decoding.max_tokens+32<=8192 for j in packet.jobs if j.eligible)
            @test_throws ErrorException freeze_bcbx_doc_study(;directory)
        end
    end
end
