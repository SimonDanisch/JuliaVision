include("alias_transfer3_deploy.jl")
using Test
function test_alias_transfer3_deploy()
    path=joinpath(@__DIR__,"alias_transfer3_results/conditions.json")
    packet=JSON3.read(read(path,String),Dict{String,Any})
    @testset "External study deployment excludes private grading and answers" begin
        payload=alias_transfer3_payload(path)
        @test !haskey(payload.updates,"score_alias_transfer3.py")
        @test !haskey(payload.updates,"julia_public_binding_docs.json")
        @test all(name->basename(name)==name,keys(payload.updates))
        @test haskey(payload.updates,"run_alias_transfer3.jl")
        for forbidden in ("score_alias_transfer3.py","PRIVATE_selected_reference_tests.jsonl","../secret.jl")
            mktempdir() do directory
                changed=deepcopy(packet);changed["source_sha256"][forbidden]="wrong"
                mock=joinpath(directory,"conditions.json");open(io->JSON3.pretty(io,changed),mock,"w")
                @test_throws ErrorException alias_transfer3_payload(mock)
            end
        end
        mktempdir() do directory
            cp(path,joinpath(directory,"conditions.json"))
            captured=Ref(String[])
            empty_reader=(f,ids)->begin captured[]=ids;Dict{String,Vector{UInt8}}() end
            @test collect_alias_transfer3(empty_reader;directory)==0
            @test captured[]==String[j["id"] for j in packet["jobs"]]
            id=first(captured[])
            files=Dict(id*".json"=>UInt8[1],id*"_attempt.json"=>UInt8[2])
            @test collect_alias_transfer3((f,ids)->files;directory)==1
            @test collect_alias_transfer3((f,ids)->files;directory)==1
            changed=merge(files,Dict(id*".json"=>UInt8[3]))
            @test_throws ErrorException collect_alias_transfer3((f,ids)->changed;directory)
        end
    end
end
