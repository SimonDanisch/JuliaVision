include("compact_bayes_demo_study.jl")
include("score_compact_bayes.jl")
using Test

function test_compact_bayes_demo_study(device)
    @testset "Demonstration study orchestration, not neural quality" begin
        mktempdir() do directory
            p=freeze_compact_bayes_demo(token_count=m->100,directory=directory)
            @test length(p.jobs)==8
            @test p.decoding.max_tokens==1024&&!p.decoding.thinking
            @test all(j->endswith(j.id,"_demonstrations"),p.jobs)
            @test all(j->occursin("Public format examples",last(last(j.messages))),p.jobs)
            @test_throws ErrorException freeze_compact_bayes_demo(token_count=m->100,directory=directory)
            @test_throws ErrorException score_compact_bayes(device;directory)
            save(path,x)=open(io->JSON3.pretty(io,x),path,"w")
            references=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()))
            for (i,job) in enumerate(p.jobs)
                r=references[job.task]
                nodes=map(r.network.nodes) do n
                    n.states==["no","yes"] ? (;name=n.name,parents=n.parents,table=[row.p[2] for row in n.rows]) :
                        (;name=n.name,states=n.states,parents=n.parents,table=[row.p for row in n.rows])
                end
                attempt=joinpath(directory,job.id*"_attempt.json");save(attempt,(;pid=2000+i,id=job.id))
                receipt=(;id=job.id,messages=job.messages,answer=JSON3.write((;nodes,query=r.network.query,observed=r.network.observed)),
                    conditions_sha256=bytes2hex(sha256(read(joinpath(directory,"conditions.json")))),
                    attempt_sha256=bytes2hex(sha256(read(attempt))),normal_final_waitidle=true,
                    input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),budget_tokens=1024,decode_steps_executed=64,decode_submissions=64,
                    eos_output_poll_chunk=64,truncated=false,elapsed_seconds=0.,sampling=(;pairs(p.decoding)...,draws=65))
                save(joinpath(directory,job.id*".json"),receipt)
            end
            result=score_compact_bayes(device;directory)
            @test result.passed==result.admitted==8
            @test all(r->r.solver_submissions==1,result.rows)
            output=joinpath(directory,first(p.jobs).id*".json");changed=JSON3.read(read(output,String),Dict{String,Any})
            changed["messages"]=[];save(output,changed)
            @test_throws ErrorException score_compact_bayes(device;directory)
        end
        mktempdir() do directory
            @test_throws ErrorException freeze_compact_bayes_demo(token_count=m->true,directory=directory)
            @test !isfile(joinpath(directory,"conditions.json"))
        end
    end
end
