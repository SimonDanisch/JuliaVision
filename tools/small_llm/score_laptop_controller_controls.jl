include("score_laptop_pending_study.jl")
include("laptop_controller_controls.jl")

"""Score posthoc controls offline without changing original pilot scores."""
function score_laptop_controller_controls(;directory=joinpath(@__DIR__,"laptop_controller_controls_results"))
    s=laptop_control_state(directory);p=s.packet;rows=Any[];tasks=kernel_transfer_tasks()
    digest(path)=bytes2hex(open(sha256,path))
    for job in p.jobs
        output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
        if !isfile(output)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.attempt_sha256==digest(attempt) || error("attempt drift")
        receipt=JSON3.read(read(attempt,String))
        receipt.conditions_sha256==digest(s.path) || error("attempt protocol drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        candidate.conditions_sha256==digest(s.path) || error("generation protocol drift")
        candidate.messages==job.messages || error("prompt drift")
        candidate.thinking_budget==p.thinking_limit && candidate.budget_tokens==p.decoding.max_tokens || error("budget drift")
        candidate.forced_closures==0 || error("unexpected closure")
        candidate.decode_steps_executed==candidate.decode_submissions || error("submission invariant")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
            (field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) : a==b) || error("sampling drift")
        end
        pilot=JSON3.read(read(joinpath(@__DIR__,"laptop_pending_results",String(job.id)*".json"),String))
        candidate.input_sha256==pilot.input_sha256 || error("tokenized prompt drift")
        prefix=first(split(String(pilot.answer),"</think>";limit=2))
        same_prefix=startswith(String(candidate.answer),prefix)
        code=String(candidate.code);lowered=CheckedPointwiseLowering.lower(code)
        intermediate=lowered.accepted ? lowered.code : code
        tail=KernelVoidTail.normalize(intermediate);compiled=tail.accepted ? tail.code : intermediate
        task=only(filter(t->t.id==job.task,tasks))
        for (variant,selected) in (("raw",code),("compiler",compiled))
            scorepath=replace(output,".json"=>"_"*variant*"_score.json")
            score=if isfile(scorepath)
                old=JSON3.read(read(scorepath,String));old.code_sha256==bytes2hex(sha256(selected)) || error("score drift");old
            else
                JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,variant,status=score.passed ? "passed" : "failed",
                truncated=candidate.truncated,pre_closure_prefix_matches_pilot=same_prefix,generation_sha256=digest(output)))
        end
    end
    summary=(;scope=p.scope,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=digest(s.path),scorer_sha256=digest(@__FILE__),execution="Original sandboxed software-Vulkan evaluator only")
    open(io->JSON3.pretty(io,summary),joinpath(directory,"summary.json"),"w");summary
end
