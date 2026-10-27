include("prepare_bayes_thinking_native_audit.jl")
include("bayes_extraction_references.jl")
using Test

function test_prepare_bayes_thinking_native_audit(device)
    @testset "Thinking parity uses generated networks, not gold" begin
        mktempdir() do directory
            cp(joinpath(@__DIR__,"bayes_thinking_results/conditions.json"),joinpath(directory,"conditions.json"))
            p=JSON3.read(read(joinpath(directory,"conditions.json"),String));rows=Any[]
            references=Dict("bayes_extract_"*lpad(i,2,'0')=>r for (i,r) in enumerate(bayes_extraction_references()))
            digest(path)=bytes2hex(sha256(read(path)));save(path,x)=open(io->JSON3.pretty(io,x),path,"w")
            for (i,job) in enumerate(p.jobs)
                network=JSON3.read(JSON3.write(references[job.task].network),Dict{String,Any})
                if job.task=="bayes_extract_01"
                    network["nodes"][1]["rows"][1]["p"]=[0.877,0.123]
                end
                path=joinpath(directory,job.id*".json");attempt=joinpath(directory,job.id*"_attempt.json")
                save(attempt,(;pid=1000+i));save(path,(;answer="thought</think>"*JSON3.write(network),normal_final_waitidle=true,
                    truncated=false,finished_thinking=true,forced_closures=0,thinking_budget=1024,
                    conditions_sha256=digest(joinpath(directory,"conditions.json")),attempt_sha256=digest(attempt)))
                push!(rows,(;id=job.id,arm=job.arm,admitted=job.arm=="bayes_network",generation_sha256=digest(path)))
            end
            summarypath=joinpath(directory,"summary.json")
            save(summarypath,(;complete=true,conditions_sha256=digest(joinpath(directory,"conditions.json")),rows))
            destination=joinpath(directory,"private.json")
            result=prepare_bayes_thinking_native_audit(device;directory,destination)
            @test result.complete&&result.cases==8
            packet=validate_bayes_native_payload(read(destination),result.input_sha256)
            @test length(packet.generation_pids)==16
            firstnetwork=JSON3.read(first(packet.rows).network)
            @test firstnetwork.nodes[1].rows[1].p[2]==0.123
            # A changed prior must not be silently repaired to the task's gold.
            @test abs(Float64(first(packet.rows).expected.posterior[2])-Float64(first(bayes_extraction_references()).answer))>0.01
            @test_throws ErrorException prepare_bayes_thinking_native_audit(device;directory,destination)
            save(summarypath,(;complete=false,conditions_sha256=digest(joinpath(directory,"conditions.json")),rows))
            @test_throws ErrorException prepare_bayes_thinking_native_audit(device;directory,destination=joinpath(directory,"other.json"))
        end
    end
end
