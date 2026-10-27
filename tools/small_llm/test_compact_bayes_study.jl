include("compact_bayes_study.jl")
include("score_compact_bayes.jl")
using Test

function test_compact_bayes_study(device)
    @testset "Compact workflow mocks, never model-quality claims" begin
        mktempdir() do directory
            p=freeze_compact_bayes(;token_count=m->100,directory)
            @test length(p.jobs)==8&&p.decoding.thinking===false&&p.decoding.max_tokens==1024
            @test_throws ErrorException freeze_compact_bayes(;token_count=m->100,directory)
            @test_throws ErrorException score_compact_bayes(device;directory)
            references=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()))
            save(path,x)=open(io->JSON3.pretty(io,x),path,"w")
            for (i,job) in enumerate(p.jobs)
                r=references[job.task]
                nodes=map(r.network.nodes) do n
                    n.states==["no","yes"] ? (;name=n.name,parents=n.parents,table=[row.p[2] for row in n.rows]) :
                        (;name=n.name,states=n.states,parents=n.parents,table=[row.p for row in n.rows])
                end
                attempt=joinpath(directory,job.id*"_attempt.json");save(attempt,(;pid=1000+i,id=job.id))
                answer=JSON3.write((;nodes,query=r.network.query,observed=r.network.observed))
                receipt=(;id=job.id,messages=job.messages,answer,conditions_sha256=bytes2hex(sha256(read(joinpath(directory,"conditions.json")))),
                    attempt_sha256=bytes2hex(sha256(read(attempt))),normal_final_waitidle=true,
                    input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),budget_tokens=1024,decode_steps_executed=64,decode_submissions=64,
                    eos_output_poll_chunk=64,truncated=false,elapsed_seconds=0.,sampling=(;pairs(p.decoding)...,draws=65))
                save(joinpath(directory,job.id*".json"),receipt)
            end
            result=score_compact_bayes(device;directory)
            @test result.passed==result.admitted==8
            @test all(r->r.solver_submissions==1,result.rows)
            output=joinpath(directory,first(p.jobs).id*".json");original=JSON3.read(read(output,String),Dict{String,Any})
            changed=copy(original);changed["answer"]*= " trailing text";save(output,changed)
            @test !first(score_compact_bayes(device;directory).rows).admitted
            changed=copy(original);network=JSON3.read(changed["answer"],Dict{String,Any});network["nodes"][1]["table"][1]=0.02
            changed["answer"]=JSON3.write(network);save(output,changed)
            row=first(score_compact_bayes(device;directory).rows)
            @test row.admitted&&row.network_faithful===false&&row.status=="failed"
            changed=copy(original);changed["messages"]=[];save(output,changed)
            @test_throws ErrorException score_compact_bayes(device;directory)
        end
        mktempdir() do directory
            @test_throws ErrorException freeze_compact_bayes(;token_count=m->true,directory)
            @test !isfile(joinpath(directory,"conditions.json"))
        end
    end
end
