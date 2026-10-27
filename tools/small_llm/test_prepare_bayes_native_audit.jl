include("prepare_bayes_native_audit.jl")
include("bayes_extraction_references.jl")
using Test
function test_prepare_bayes_native_audit(device)
    @testset "Private parity packet uses generated inputs, not answer references" begin
        mktempdir() do directory
            source=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
            cp(source,joinpath(directory,"conditions.json"));p=JSON3.read(read(source,String))
            references=bayes_extraction_references();tasks=unique(String(j.task) for j in p.jobs)
            lookup=Dict(tasks[i]=>references[i] for i in 1:8);rows=Any[]
            save(path,x)=open(io->JSON3.write(io,x),path,"w");digest(path)=bytes2hex(sha256(read(path)))
            for (i,job) in enumerate(p.jobs)
                output=joinpath(directory,job.id*".json")
                answer=job.arm=="bayes_network" ? JSON3.write(lookup[job.task].network) : "{\"probability\":0}"
                save(output,(;answer,normal_final_waitidle=true,conditions_sha256=digest(joinpath(directory,"conditions.json"))))
                save(joinpath(directory,job.id*"_attempt.json"),(;pid=1000+i))
                push!(rows,(;id=job.id,arm=job.arm,admitted=true,generation_sha256=digest(output)))
            end
            summary=joinpath(directory,"summary.json");save(summary,(;complete=true,conditions_sha256=digest(joinpath(directory,"conditions.json")),rows))
            destination=joinpath(directory,"private.json")
            result=prepare_bayes_native_audit(device;directory,destination)
            @test result.complete&&result.cases==8
            @test_throws ErrorException prepare_bayes_native_audit(device;directory,destination)
            bytes=read(destination);packet=validate_bayes_native_payload(bytes,result.input_sha256)
            @test length(packet.generation_pids)==16
            @test all(r->r.expected.status==0,packet.rows)
            @test_throws ErrorException validate_bayes_native_payload(bytes,"wrong_hash")
            altered=JSON3.read(bytes,Dict{String,Any});altered["rows"][1]["network"]="{}"
            changed=Vector{UInt8}(codeunits(JSON3.write(altered)))
            @test_throws ErrorException validate_bayes_native_payload(changed,bytes2hex(sha256(changed)))
            altered=JSON3.read(bytes,Dict{String,Any});altered["generation_pids"][1]=0
            changed=Vector{UInt8}(codeunits(JSON3.write(altered)))
            @test_throws ErrorException validate_bayes_native_payload(changed,bytes2hex(sha256(changed)))
            save(summary,(;complete=false,conditions_sha256=digest(joinpath(directory,"conditions.json")),rows))
            @test_throws ErrorException prepare_bayes_native_audit(device;directory,destination=joinpath(directory,"other.json"))
        end
    end
end
