include("laptop_public_source_transfer.jl")
using Test
function test_laptop_public_source_transfer()
    tests=@testset "Public-only deployment manifest, mocked transport" begin
        mktempdir() do directory
            file="public_program_diagnostics.jl";policy="bigcodebench_x_revision_policy.json"
            p=Dict("source_sha256"=>Dict(file=>bytes2hex(open(sha256,joinpath(@__DIR__,file)))),
                "policy_sha256"=>bytes2hex(open(sha256,joinpath(@__DIR__,policy))))
            path=joinpath(directory,"conditions.json");open(io->JSON3.pretty(io,p),path,"w")
            payload=laptop_public_revision_payload(path)
            @test length(payload.updates)==3
            @test all(!occursin("_score",key)&&!occursin("PRIVATE",key) for key in keys(payload.updates))
            mocked=(f,packet)->Dict(name=>bytes2hex(sha256(data)) for (name,data) in packet.updates)
            @test deploy_public_revision_payload(mocked,payload).complete
            wrong=(f,packet)->Dict("wrong"=>"bad")
            @test_throws ErrorException deploy_public_revision_payload(wrong,payload)
            p["source_sha256"]=Dict("../PRIVATE_tests.json"=>repeat("0",64))
            open(io->JSON3.pretty(io,p),path,"w")
            @test_throws ErrorException laptop_public_revision_payload(path)
            p["source_sha256"]=Dict("hidden_score_validation.json"=>repeat("0",64))
            open(io->JSON3.pretty(io,p),path,"w")
            @test_throws ErrorException laptop_public_revision_payload(path)
        end
    end
    c=Test.get_test_counts(tests);report=(;complete=true,passed=c.passes+c.cumulative_passes,
        scope="Public deployment selection/mock byte-integrity tests, not GPU/model quality",
        source_sha256=Dict(f=>bytes2hex(open(sha256,joinpath(@__DIR__,f))) for f in ("laptop_public_source_transfer.jl","test_laptop_public_source_transfer.jl")))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"laptop_public_source_transfer_validation.json"),"w");report
end
