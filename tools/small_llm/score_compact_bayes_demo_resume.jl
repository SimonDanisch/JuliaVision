include("score_compact_bayes.jl")
include("compact_bayes_demo_interruption.jl")

function score_compact_bayes_demo_resume(device;directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Explicit software grading device")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String));condition_hash=digest(path)
    amendmentpath=joinpath(directory,"interruption_amendment.json");amendment=JSON3.read(read(amendmentpath,String))
    amendment.conditions_sha256==condition_hash||error("Amendment drift")
    for registry in (p.private_source_sha256,p.source_sha256,amendment.grader_source_sha256),(f,h) in pairs(registry)
        digest(joinpath(@__DIR__,String(f)))==h||error("Source drift")
    end
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
    digest(parentpath)==p.parent_conditions_sha256&&digest(joinpath(dirname(parentpath),"summary.json"))==p.parent_summary_sha256||error("Original study drift")
    isempty(compact_bayes_demo_pending(directory,p,condition_hash))||error("Finish every unattempted slot before grading")
    references=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()));tasks=Dict(t.id=>t for t in bayes_extraction_tasks())
    rows=map(p.jobs) do job
        digest(joinpath(dirname(parentpath),String(job.task)*"_bayes_network.json"))==job.original_generation_sha256||error("Original generation drift")
        if compact_bayes_demo_lost(directory,job,condition_hash)
            return (;id=job.id,task=job.task,arm=job.arm,status="inference_error",admitted=false,
                reason="lost_process_no_output",network_faithful=nothing,numeric_correct=nothing,probability=nothing,
                expected_probability=nothing,solver_status=nothing,solver_submissions=0,solver_seconds=0.,truncated=nothing,
                generation_sha256=nothing,loss_sha256=digest(joinpath(directory,job.id*"_lost.json")),normal_final_waitidle=false)
        end
        output=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(output,String))
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=condition_hash,attempt_sha256=digest(joinpath(directory,job.id*"_attempt.json")))
        failed=hasproperty(m,:inference_error)
        a=failed||m.truncated ? (;accepted=false,reason=failed ? "inference_error" : "truncated",network=nothing,packed=nothing) : admit_compact_bayes(String(m.answer))
        r=references[job.task];task=tasks[job.task];faithful=nothing;probability=nothing;solver_status=nothing;submissions=0;solver_seconds=0.
        if a.accepted
            faithful=bayes_extraction_faithful(a.network,JSON3.read(JSON3.write(r.network)))
            target=findfirst(==(task.target),a.packed.query_states)
            if a.packed.query==Symbol(task.query)&&target!==nothing
                started=time();e=PortableEvidence.prepare_bayes(device,a.packed.factors,a.packed.query,a.packed.observed)
                try
                    before=Mantle.submissions(device);PortableEvidence.run!(e);Mantle.waitidle(device)
                    submissions=Mantle.submissions(device)-before;solver_status=only(Array(e.outputs.status))
                    probability=solver_status==0 ? Float64(Array(e.outputs.posterior)[target]) : nothing
                finally close(e) end
                solver_seconds=time()-started
            end
        end
        numeric_correct=probability!==nothing&&abs(probability-Float64(r.answer))<=1e-5
        correct=faithful===true&&numeric_correct
        (;id=job.id,task=job.task,arm=job.arm,status=failed ? "inference_error" : correct ? "passed" : "failed",
            admitted=a.accepted,reason=a.reason,network_faithful=faithful,numeric_correct,probability,expected_probability=Float64(r.answer),
            solver_status,solver_submissions=submissions,solver_seconds,truncated=failed ? nothing : m.truncated,
            generation_sha256=digest(output),loss_sha256=nothing,normal_final_waitidle=true)
    end
    losses=count(r->r.reason=="lost_process_no_output",rows)
    result=(;complete=true,generation_complete=losses==0,all_slots_terminal=true,scope=p.scope,policy=p.policy,rows,total=8,
        completed_generations=8-losses,lost_attempts=losses,passed=count(r->r.status=="passed",rows),admitted=count(r->r.admitted,rows),
        conditions_sha256=condition_hash,interruption_amendment_sha256=digest(amendmentpath),
        attrition="Interrupted attempt retained as unavailable, not a semantic model failure or normal waitidle. No regeneration. Report complete-case results and conservative registered-cohort denominator separately.")
    output=joinpath(directory,"summary.json");ispath(output)&&error("Preserve first score")
    open(io->JSON3.pretty(io,result),output,"w");result
end
