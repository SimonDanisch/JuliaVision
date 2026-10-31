include("bayes_thinking_study.jl")
include("score_bayes_thinking.jl")
using Test

function test_bayes_thinking_study(device)
    @testset "Frozen thinking workflow, mock answers only" begin
        mktempdir() do directory
            p=freeze_bayes_thinking(;token_count=m->100,directory)
            @test length(p.jobs)==16
            @test p.decoding.thinking&&p.decoding.max_tokens==2048&&p.decoding.thinking_budget==1024
            @test_throws ErrorException freeze_bayes_thinking(;token_count=m->100,directory)
            refs=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()))
            save(path,value)=open(io->JSON3.pretty(io,value),path,"w")
            conditions=joinpath(directory,"conditions.json")
            for (i,job) in enumerate(p.jobs)
                attempt=joinpath(directory,job.id*"_attempt.json")
                save(attempt,(;id=job.id,pid=1000+i))
                r=refs[job.task]
                payload=JSON3.write(job.arm=="direct_probability" ? (;probability=Float64(r.answer)) : r.network)
                row=(;id=job.id,messages=job.messages,answer="reasoning</think>\n"*payload,
                    conditions_sha256=bayes_thinking_digest(conditions),attempt_sha256=bayes_thinking_digest(attempt),
                    normal_final_waitidle=true,input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
                    budget_tokens=2048,decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,
                    truncated=false,finished_thinking=true,thinking_budget=1024,forced_closures=0,elapsed_seconds=0.,
                    sampling=merge(p.decoding,(;draws=65)))
                save(joinpath(directory,job.id*".json"),row)
            end
            result=score_bayes_thinking(device;directory)
            @test result.complete&&all(a->a.passed==8,result.arms)
            @test all(r->r.solver_submissions==1,filter(r->r.arm=="bayes_network",result.rows))
            output=joinpath(directory,first(p.jobs).id*".json")
            original=JSON3.read(read(output,String),Dict{String,Any})
            changed=copy(original);changed["answer"]*=" trailing ignored by permissive JSON parser"
            save(output,changed)
            @test first(score_bayes_thinking(device;directory).rows).status=="failed"
            changed=copy(original);changed["forced_closures"]=2;save(output,changed)
            @test_throws ErrorException score_bayes_thinking(device;directory)
            save(output,original);rm(output)
            @test_throws ErrorException score_bayes_thinking(device;directory)
        end
        mktempdir() do directory
            @test_throws ErrorException freeze_bayes_thinking(;token_count=m->true,directory)
            @test !ispath(joinpath(directory,"conditions.json"))
        end
    end
end
