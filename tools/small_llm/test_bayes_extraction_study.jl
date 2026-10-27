include("bayes_extraction_study.jl")
include("score_bayes_extraction.jl")
using Test
function test_bayes_extraction_study(device)
    @testset "Mock full Bayesian extraction study, not neural quality" begin
        mktempdir() do directory
            p=freeze_bayes_extraction(;token_count=_ -> 1000,directory)
            @test length(p.jobs)==16
            @test p.decoding.max_tokens==1024
            @test_throws ErrorException freeze_bayes_extraction(;token_count=_ -> 1000,directory)
            @test_throws ErrorException score_bayes_extraction(device;directory)
            references=bayes_extraction_references();tasks=bayes_extraction_tasks()
            lookup=Dict(t.id=>r for (t,r) in zip(tasks,references))
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            digest(path)=bytes2hex(sha256(read(path)))
            conditions=joinpath(directory,"conditions.json")
            for job in p.jobs
                reference=lookup[job.task];messages=JSON3.read(JSON3.write(job.messages))
                answer=JSON3.write(job.arm=="direct_probability" ? (;probability=Float64(reference.answer)) : reference.network)
                attempt=joinpath(directory,job.id*"_attempt.json");save(attempt,(;id=job.id))
                receipt=(;id=job.id,messages,answer,normal_final_waitidle=true,conditions_sha256=digest(conditions),attempt_sha256=digest(attempt),
                    input_sha256=bytes2hex(sha256(JSON3.write(messages))),budget_tokens=1024,decode_steps_executed=64,
                    decode_submissions=64,eos_output_poll_chunk=64,truncated=false,
                    sampling=(;seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,draws=65))
                save(joinpath(directory,job.id*".json"),receipt)
            end
            result=score_bayes_extraction(device;directory)
            @test result.complete
            @test all(a->a.passed==a.numeric_correct==a.total==8,result.arms)
            @test result.arms[2].faithful_networks==8
            @test all(r->r.solver_submissions==1,filter(r->r.arm=="bayes_network",result.rows))
            reference=JSON3.read(JSON3.write(first(references).network),Dict{String,Any})
            for node in reference["nodes"]
                reverse!(node["states"])
                foreach(row->reverse!(row["p"]),node["rows"])
            end
            @test bayes_extraction_faithful(JSON3.read(JSON3.write(reference)),JSON3.read(JSON3.write(first(references).network)))
            @test admit_bayes_extraction("```json\n{\"probability\":0.25}\n```","direct_probability").accepted
            @test !admit_bayes_extraction("prefix\n```json\n{\"probability\":0.25}\n```","direct_probability").accepted
        end
    end
end
