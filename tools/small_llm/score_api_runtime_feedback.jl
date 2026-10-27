include("api_runtime_feedback.jl")

function score_api_runtime_feedback()
    directory=joinpath(@__DIR__,"api_runtime_feedback_results")
    conditions=joinpath(directory,"conditions.json");packet=JSON3.read(read(conditions,String))
    conditions_sha=bytes2hex(sha256(read(conditions)));rows=Any[]
    qualification=JSON3.read(read(joinpath(@__DIR__,"api_generalization_results","harness_validation.json"),String))
    for registration in packet.rows
        task=only(filter(task->task.id==registration.id,JULIA_ADVANCED_TASKS))
        only(filter(reference->reference.id==task.id,qualification.references)).passed || error("unqualified reference")
        registration.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        baseline=joinpath(@__DIR__,"projection_memory_results",task.id*"_zero.json")
        diagnostic=joinpath(directory,task.id*"_diagnostic.json")
        if !isfile(baseline)||!isfile(diagnostic)
            for arm in packet.arms
                push!(rows,(;task=task.id,arm,status="pending"))
            end
            continue
        end
        original=JSON3.read(read(baseline,String));observation=JSON3.read(read(diagnostic,String))
        original.conditions_sha256==packet.baseline_conditions_sha256 || error("baseline drift")
        observation.code_sha256==bytes2hex(sha256(String(original.code))) || error("diagnostic drift")
        # Scores are computed here only; NEVER passed into the job selector.
        baseline_score=JuliaTaskEval.evaluate(task,String(original.code))
        positive=observation.analysis_completed&&!observation.payload.completed
        for arm in packet.arms
            output=joinpath(directory,task.id*"_"*String(arm)*".json")
            if !positive
                push!(rows,(;task=task.id,arm,status=baseline_score.passed ? "passed" : "failed",action="retain_baseline"))
            elseif !isfile(output)
                push!(rows,(;task=task.id,arm,status="pending"))
            else
                candidate=JSON3.read(read(output,String))
                candidate.task==task.id && candidate.arm==arm || error("candidate identity drift")
                candidate.contract_sha256==registration.contract_sha256 && candidate.conditions_sha256==conditions_sha || error("revision drift")
                candidate.diagnostic_sha256==bytes2hex(sha256(read(diagnostic))) || error("revision diagnostic drift")
                score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(output,".json"=>"_score.json"))
                push!(rows,(;task=task.id,arm,status=score.passed ? "passed" : "failed",action="single_revision",
                    baseline_passed=baseline_score.passed,truncated=candidate.truncated))
            end
        end
    end
    report=(;scope=packet.scope,policy=packet.trigger,rows,
        totals=Dict(String(arm)=>(;passed=count(row->row.arm==arm&&row.status=="passed",rows),
            pending=count(row->row.arm==arm&&row.status=="pending",rows)) for arm in packet.arms))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
