include("laptop_native_datastore.jl")
using Test

function test_laptop_native_datastore_registration()
    checks=@testset "Laptop source-only datastore registration CPU" begin
        mktempdir() do root
            directory=joinpath(root,"data");p=freeze_laptop_native_datastore(;directory)
            @test length(p.rows)==2
            @test sum(length(r.token_ids)-1 for r in p.rows)==254
            @test p.width==2560
            @test p.vocabulary>maximum(maximum(r.token_ids) for r in p.rows)
            @test !p.thinking
            @test length(laptop_native_datastore_state(directory).packet.rows)==2
            @test_throws ErrorException freeze_laptop_native_datastore(;directory)
            @test_throws ArgumentError capture_laptop_native_datastore(;directory,seconds=0)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer("gen/qwen35-4b")
            @test all(r->SmallLLMChecks.QwenText.encode(tokenizer,String(r.source))==r.token_ids,p.rows)
        end
    end
    n=Test.get_test_counts(checks)
    packet=(;complete=true,scope="CPU source/registration checks only; not device capture or model quality",passed=n.passes+n.cumulative_passes,
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in
            ("laptop_native_datastore.jl","test_laptop_native_datastore_registration.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(@__DIR__,"laptop_native_registration_validation.json"),"w");packet
end
