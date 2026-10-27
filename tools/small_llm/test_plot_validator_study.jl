include("plot_validator_study.jl")
include("score_plot_validator_study.jl")
using Test
function test_plot_validator_study()
    @testset "Conditional plot repair keeps admitted-but-wrong originals" begin
        mktempdir() do root
            parent=joinpath(root,"parent");mkpath(parent)
            cp(joinpath(@__DIR__,"plot_interface_results/conditions.json"),joinpath(parent,"conditions.json"))
            original=JSON3.read(read(joinpath(parent,"conditions.json"),String))
            tasks=unique(String(j.task) for j in original.jobs);references=plot_interface_references()
            lookup=Dict(tasks[i]=>references[i] for i in 1:8)
            digest(path)=bytes2hex(sha256(read(path)))
            save(path,value)=open(io->JSON3.write(io,value),path,"w")
            function receipt!(directory,job,answer)
                conditions=joinpath(directory,"conditions.json");attempt=joinpath(directory,job.id*"_attempt.json")
                save(attempt,(;id=job.id));messages=JSON3.read(JSON3.write(job.messages))
                value=(;id=job.id,messages,answer,normal_final_waitidle=true,conditions_sha256=digest(conditions),attempt_sha256=digest(attempt),
                    input_sha256=bytes2hex(sha256(JSON3.write(messages))),budget_tokens=1024,decode_steps_executed=64,
                    decode_submissions=64,eos_output_poll_chunk=64,truncated=false,
                    sampling=(;seed=20261025,topk=20,temperature=0.6f0,top_p=0.95f0,presence=0f0,draws=65))
                save(joinpath(directory,job.id*".json"),value)
            end
            for job in original.jobs
                reference=lookup[job.task]
                answer=job.arm=="plot_spec" ? JSON3.write(reference) : plot_interface_native_reference(SafePlotSpec.parse_spec(JSON3.write(reference)))
                if job.arm=="plot_spec"&&job.task==tasks[1];answer="{}"
                elseif job.arm=="plot_spec"&&job.task==tasks[2]
                    wrong=JSON3.read(answer,Dict{String,Any});wrong["axes"][1]["title"]="WRONG but admitted";answer=JSON3.write(wrong)
                end
                receipt!(parent,job,answer)
            end
            save(joinpath(parent,"summary.json"),(;not_read_for_routing=true))
            preparation=joinpath(root,"preparation.json")
            prepared=prepare_plot_validator_policy(;token_count=_ -> 1000,parent,destination=preparation)
            @test length(prepared.decisions)==8
            @test count(d->d.triggered,prepared.decisions)==1
            @test length(prepared.jobs)==2
            directory=joinpath(root,"revisions");p=freeze_plot_validator_study(;preparation,directory,parent)
            @test_throws ErrorException score_plot_validator_study(;directory,parent)
            for job in p.jobs;receipt!(directory,job,JSON3.write(lookup[job.task]));end
            result=score_plot_validator_study(;directory,parent)
            @test result.complete&&length(result.rows)==16
            @test all(a->a.passed==7&&a.total==8&&a.revised==1&&a.unchanged==7,result.arms)
            @test all(r->r.retained_original&&r.status=="failed",filter(r->r.task==tasks[2],result.rows))
            job=only(filter(j->j.arm=="self_review",p.jobs));receipt!(directory,job,"{}")
            changed=score_plot_validator_study(;directory,parent)
            @test changed.arms[1].passed==6&&changed.arms[2].passed==7
            @test_throws ErrorException freeze_plot_validator_study(;preparation,directory,parent)
            counter=Ref(0)
            function rejects_tampering(change)
                altered=JSON3.read(JSON3.write(prepared),Dict{String,Any});change(altered)
                counter[]+=1;bad=joinpath(root,"bad_preparation_$(counter[]).json");save(bad,altered)
                @test_throws ErrorException freeze_plot_validator_study(;preparation=bad,directory=joinpath(root,"bad_result_$(counter[])"),parent)
            end
            rejects_tampering(x->(x["jobs"][1]["messages"][end]["user"]*=" Hidden expected plan"))
            rejects_tampering(x->(x["jobs"][2]["id"]=x["jobs"][1]["id"]))
            rejects_tampering(x->(x["decisions"][1]["triggered"]=false))
            rejects_tampering(x->(x["jobs"][1]["task"]="unknown_task"))
        end
    end
end
