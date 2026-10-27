include("compact_bayes_demo_study.jl")
include("score_compact_bayes_demo_resume.jl")
include("summarize_compact_bayes_demo_resume.jl")
using Test

function test_compact_bayes_demo_resume(device)
    @testset "Interrupted cohort score preserves missing outcome" begin
        mktempdir() do directory
            p=freeze_compact_bayes_demo(token_count=m->100,directory=directory)
            digest(path)=bytes2hex(sha256(read(path)))
            save(path,x)=open(io->JSON3.pretty(io,x),path,"w")
            hash=digest(joinpath(directory,"conditions.json"))
            save(joinpath(directory,"interruption_amendment.json"),(;conditions_sha256=hash,
                grader_source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in
                    ("score_compact_bayes_demo_resume.jl","compact_bayes_demo_interruption.jl","summarize_compact_bayes_demo_resume.jl"))))
            refs=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()))
            for (i,job) in enumerate(p.jobs)
                attempt=(;id=job.id,pid=4000+i,conditions_sha256=hash)
                path=joinpath(directory,job.id*"_attempt.json");save(path,attempt)
                if i==2
                    save(joinpath(directory,job.id*"_lost.json"),(;attempt...,terminal_status="lost_process",
                        attempt_sha256=digest(path),original_process_absent=true,shared_lock_observed_free=true,
                        normal_final_waitidle=false,regeneration_permitted=false));continue
                end
                r=refs[job.task]
                nodes=map(r.network.nodes) do n
                    n.states==["no","yes"] ? (;name=n.name,parents=n.parents,table=[row.p[2] for row in n.rows]) :
                        (;name=n.name,states=n.states,parents=n.parents,table=[row.p for row in n.rows])
                end
                save(joinpath(directory,job.id*".json"),(;id=job.id,messages=job.messages,
                    answer=JSON3.write((;nodes,query=r.network.query,observed=r.network.observed)),
                    conditions_sha256=hash,attempt_sha256=digest(path),normal_final_waitidle=true,
                    input_sha256=bytes2hex(sha256(JSON3.write(job.messages))),budget_tokens=1024,
                    decode_steps_executed=64,decode_submissions=64,eos_output_poll_chunk=64,truncated=false,
                    elapsed_seconds=1.,input_tokens=100,generated_tokens=10,sampling=(;pairs(p.decoding)...,draws=65)))
            end
            s=score_compact_bayes_demo_resume(device;directory)
            @test s.complete&&!s.generation_complete&&s.all_slots_terminal
            @test s.completed_generations==7&&s.lost_attempts==1&&s.passed==s.admitted==7
            @test s.rows[2].status=="inference_error"&&!s.rows[2].normal_final_waitidle
            @test s.rows[2].numeric_correct===nothing&&s.rows[2].expected_probability===nothing
            @test !isfile(joinpath(directory,p.jobs[2].id*".json"))
            @test_throws ErrorException score_compact_bayes_demo_resume(device;directory)
            a=summarize_compact_bayes_demo_resume(;directory)
            @test length(a.complete_case_tasks)==7&&a.lost_attempts==1
            @test length(a.completed_generation_costs.rows)==7
            @test only(a.completed_generation_costs.arms).generation_timer_seconds==7.
            @test_throws ErrorException summarize_compact_bayes_demo_resume(;directory)
        end
    end
end
