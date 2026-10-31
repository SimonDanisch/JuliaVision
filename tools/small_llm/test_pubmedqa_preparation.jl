include("pubmedqa_reading_study.jl")
using Test
function test_pubmedqa_preparation()
    @testset "Public preparation is capacity-checked before registration" begin
        mktempdir() do root
            public=mkpath(joinpath(root,"public"))
            tasks=[(;pmid=string(i),question="Synthetic question?",abstract_context=["Public context"]) for i in 1:16]
            source=joinpath(public,"tasks.json");open(io->JSON3.write(io,tasks),source,"w")
            manifest=(;complete=true,count=16,public_sha256=bytes2hex(sha256(read(source))),
                snapshotter_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"snapshot_pubmedqa_pilot.py")))))
            open(io->JSON3.write(io,manifest),joinpath(root,"manifest.json"),"w")
            destination=joinpath(root,"preparation.json")
            result=prepare_pubmedqa_reading(;snapshot=root,destination,token_count=_ -> 200)
            @test result.complete&&!result.ready_for_generation
            @test length(result.jobs)==48
            @test all(j->j.input_tokens==200,result.jobs)
            @test all(j->length(j.messages)==2,result.jobs)
            @test result.decoding.max_tokens==128&&!result.decoding.thinking
            @test_throws ErrorException prepare_pubmedqa_reading(;snapshot=root,destination,token_count=_ -> 200)
            oversized=joinpath(root,"oversized.json")
            @test_throws ErrorException prepare_pubmedqa_reading(;snapshot=root,destination=oversized,token_count=_ -> 8192)
            @test !ispath(oversized)
            for count in (0,-1,true,2.5)
                @test_throws ErrorException prepare_pubmedqa_reading(;snapshot=root,destination=oversized,token_count=_ -> count)
                @test !ispath(oversized)
            end
        end
    end
end
