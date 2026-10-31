include("score_laptop_native_memory.jl")
using Test

function test_laptop_native_memory_registration()
    checks=@testset "Native-memory quality protocol CPU (no generations)" begin
        mktempdir() do root
            directory=joinpath(root,"study");p=freeze_laptop_native_memory_study(;directory)
            @test length(p.jobs)==10
            @test length(unique(j.id for j in p.jobs))==10
            @test p.records==254 && p.width==2560
            @test !p.thinking && p.budget_tokens==512
            @test count(j->j.weight==0,p.jobs)==4
            @test count(j->j.shuffled,p.jobs)==2
            @test count(j->occursin("Published installed method source",last(last(j.messages))),p.jobs)==2
            @test_throws ErrorException freeze_laptop_native_memory_study(;directory)
            @test_throws ArgumentError run_laptop_native_memory_study(;directory,seconds=0)
            @test length(laptop_native_memory_state(directory).packet.jobs)==10
            summary=score_laptop_native_memory(;directory)
            @test !summary.complete && all(r->r.status=="pending",summary.rows)
            output=joinpath(directory,String(first(p.jobs).id)*".json")
            write(output,"{}")
            @test_throws Exception score_laptop_native_memory(;directory) # refuses outputs without the zero-mixture gate
        end
    end
    n=Test.get_test_counts(checks)
    report=(;complete=true,scope="CPU registration/integrity checks only; not a task-quality result",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("test_laptop_native_memory_registration.jl","laptop_native_memory_study.jl","score_laptop_native_memory.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"laptop_native_memory_protocol_validation.json"),"w");report
end
