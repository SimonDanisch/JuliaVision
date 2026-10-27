include("run_projection_memory_pilot.jl")

function score_projection_memory_pilot()
    directory=joinpath(@__DIR__,"projection_memory_results")
    path=joinpath(directory,"conditions.json");packet=JSON3.read(read(path,String))
    conditions_sha=bytes2hex(sha256(read(path)));rows=Any[]
    qualification=JSON3.read(read(joinpath(@__DIR__,"api_generalization_results","harness_validation.json"),String))
    for registration in packet.tasks,arm in packet.arms
        task=only(filter(task->task.id==registration.id,JULIA_ADVANCED_TASKS))
        only(filter(reference->reference.id==task.id,qualification.references)).passed || error("unqualified reference")
        bytes2hex(sha256(task.tests))==registration.contract_sha256 || error("task contract drift")
        output=joinpath(directory,String(task.id)*"_"*String(arm.id)*".json")
        if !isfile(output)
            push!(rows,(;task=task.id,arm=arm.id,status="pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.task==task.id && candidate.arm==arm.id || error("candidate identity drift")
        candidate.conditions_sha256==conditions_sha && candidate.contract_sha256==registration.contract_sha256 || error("candidate registration drift")
        result=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(output,".json"=>"_score.json"))
        push!(rows,(;task=task.id,arm=arm.id,status=result.passed ? "passed" : "failed",
            truncated=candidate.truncated,memory_status=candidate.memory_status,
            generation_seconds=candidate.elapsed_seconds,decode_submissions=candidate.decode_submissions))
    end
    report=(;scope=packet.scope,policy=packet.policy,
        limitations=["Two previously examined authored tasks, not held-out transfer",
            "Report every registered weight/control, not the best arm only",
            "memory_status records the final query only, not a history of all queries",
            "generation_seconds includes graph preparation and prefill, not steady-state token throughput"],rows,
        totals=Dict(String(arm.id)=>(;passed=count(row->row.arm==arm.id&&row.status=="passed",rows),
            pending=count(row->row.arm==arm.id&&row.status=="pending",rows)) for arm in packet.arms))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
