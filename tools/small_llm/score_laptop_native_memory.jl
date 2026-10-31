include("laptop_native_memory_study.jl")

function score_laptop_native_memory(;directory=joinpath(@__DIR__,"laptop_native_memory_results"))
    s=laptop_native_memory_state(directory);p=s.packet;rows=Any[]
    digest(path)=bytes2hex(open(sha256,path))
    if any(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)
        gate=JSON3.read(read(joinpath(directory,"zero_gate.json"),String))
        gate.complete && gate.checks==4 && gate.conditions_sha256==digest(s.path)||error("zero-mixture identity gate missing")
    end
    for job in p.jobs
        path=joinpath(directory,String(job.id)*".json");attempt=replace(path,".json"=>"_attempt.json")
        if !isfile(path)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(s.path)&&candidate.attempt_sha256==digest(attempt)||error("generation receipt drift")
        JSON3.read(read(attempt,String)).conditions_sha256==digest(s.path)||error("attempt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        task=only(filter(t->t.id==job.task,JULIA_ADVANCED_TASKS))
        bytes2hex(sha256(task.tests))==job.contract_sha256==candidate.contract_sha256||error("grading contract drift")
        candidate.messages==job.messages&&candidate.budget_tokens==p.budget_tokens||error("prompt/budget drift")
        Float32(candidate.memory_weight)==Float32(job.weight)&&candidate.shuffled_labels==job.shuffled||error("memory intervention drift")
        Float32(candidate.memory_temperature)==Float32(p.temperature)&&candidate.memory_topk==p.topk||error("retrieval settings drift")
        candidate.final_memory_status==candidate.worst_memory_status==0||error("invalid retrieval query")
        candidate.decode_submissions==candidate.decode_steps_executed||error("decode submission invariant")
        candidate.datastore_registration_sha256==p.datastore_registration_sha256||error("wrong datastore")
        selected=String(candidate.code);scorepath=replace(path,".json"=>"_score.json")
        score=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(selected))||error("score drift");prior
        else
            JuliaTaskEval.evaluate(task,selected;output=scorepath)
        end
        push!(rows,(;id=job.id,task=job.task,arm=job.arm,status=score.passed ? "passed" : "failed",
            truncated=candidate.truncated,generated_tokens=candidate.generated_tokens,generation_sha256=digest(path)))
    end
    packet=(;scope=p.scope,design=p.design,rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=digest(s.path),scorer_sha256=digest(@__FILE__),execution="Only original sandboxed CPU/Makie evaluator, no generated privileged code")
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
