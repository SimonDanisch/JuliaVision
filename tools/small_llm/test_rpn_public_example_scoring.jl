isdefined(@__MODULE__,:score_rpn_public_example_study)||include("score_rpn_public_example_study.jl")
using Test

function test_rpn_public_example_scoring(device)
    @testset "PRIVATE mock scoring, cache and semantic dead ends" begin
        mktempdir() do directory
            task=first(rpn_public_example_tasks());fixture=first(private_public_example_fixtures())
            jobs=[(;id="fixture_"*string(i),task=task.id,arm=i==2 ? "disabled" : "enabled",enabled=i!=2,
                public_cases=task.public_cases,messages=rpn_public_example_messages(task)) for i in 1:3]
            gate=joinpath(@__DIR__,"laptop_example_identity_gate")
            p=(;scope="PRIVATE synthetic scorer fixture, NOT model attempts",policy="mock",
                jobs,source_sha256=Dict("rpn_public_example_study.jl"=>example_study_digest(joinpath(@__DIR__,"rpn_public_example_study.jl"))),
                identity_conditions_sha256=example_study_digest(joinpath(gate,"conditions.json")),identity_receipt_sha256=example_study_digest(joinpath(gate,"result.json")),
                private_scorer_sha256=example_study_digest(joinpath(@__DIR__,"score_rpn_public_example_study.jl")),
                private_reference_sha256=example_study_digest(joinpath(@__DIR__,"rpn_public_example_reference_validation.json")),
                grammar=(;operation_budget=32,max_depth=8),public_example_tolerance=(;atol=0f0,rtol=0f0),
                decoding=(;max_tokens=512,output_chunk=64,seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0))
            writejson(path,data)=open(io->JSON3.pretty(io,data),path,"w")
            conditions=joinpath(directory,"conditions.json");writejson(conditions,p)
            for (i,job) in enumerate(jobs)
                path=joinpath(directory,job.id*".json");attempt=replace(path,".json"=>"_attempt.json")
                writejson(attempt,(;scope="PRIVATE mock",id=job.id))
                failed=i==3;program=i==1 ? fixture.program : i==2 ? "zero\nfinish" : "x0\n"
                k=(;enabled=job.enabled,accepted=!failed,raw_status=0,sampler_status=failed ? 1 : 0,rejected=failed ? 1 : 0,done=failed ? 0 : 1,
                    public_examples_matched=i==1,operation_budget=32,max_depth=8,
                    examples_sha256=public_example_bank_digest(public_example_bank(job,p)),reason=failed ? "no_admissible_token_or_sampler_rejection" : "completed")
                row=(;id=job.id,task=job.task,arm=job.arm,messages=job.messages,normal_final_waitidle=true,
                    conditions_sha256=example_study_digest(conditions),attempt_sha256=example_study_digest(attempt),input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),
                    answer=program,budget_tokens=512,decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,
                    sampling=(;seed=20261024,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,draws=65),constraint=k,truncated=false)
                writejson(path,row)
            end
            result=score_rpn_public_example_study(device;directory)
            @test result.complete
            @test [r.status for r in result.rows]==["passed","failed","failed"]
            @test [r.syntax_accepted for r in result.rows]==[true,true,false]
            @test [r.public_examples_matched for r in result.rows[1:2]]==[true,false]
            @test result.rows[3].stage=="constraint_completion_or_syntax"
            @test !isfile(joinpath(directory,"fixture_3_score.json"))
            before=Mantle.submissions(device);second=score_rpn_public_example_study(device;directory)
            @test Mantle.submissions(device)==before
            @test [r.status for r in second.rows]==[r.status for r in result.rows]
            path=joinpath(directory,"fixture_1.json");row=JSON3.read(read(path,String),Dict{String,Any})
            row["constraint"]["sampler_status"]=1;writejson(path,row)
            @test_throws ErrorException score_rpn_public_example_study(device;directory)
        end
    end
end
