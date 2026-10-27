include("source_channel_control.jl")

function score_source_channel_control()
    directory=joinpath(@__DIR__,"source_channel_results");path=joinpath(directory,"conditions.json")
    packet=JSON3.read(read(path,String));conditions_sha=bytes2hex(sha256(read(path)));rows=Any[]
    qualification=JSON3.read(read(joinpath(@__DIR__,"api_generalization_results","harness_validation.json"),String))
    for row in packet.rows
        task=only(filter(task->task.id==row.id,JULIA_ADVANCED_TASKS))
        only(filter(reference->reference.id==task.id,qualification.references)).passed || error("unqualified reference")
        row.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        output=joinpath(directory,String(row.id)*"_source_context.json")
        if !isfile(output)
            push!(rows,(;task=row.id,status="pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.task==row.id && candidate.arm=="source_context" || error("candidate identity drift")
        candidate.conditions_sha256==conditions_sha && candidate.contract_sha256==row.contract_sha256 || error("candidate contract drift")
        result=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(output,".json"=>"_score.json"))
        push!(rows,(;task=row.id,status=result.passed ? "passed" : "failed",truncated=candidate.truncated,
            generation_seconds=candidate.elapsed_seconds))
    end
    report=(;scope=packet.scope,rows,passed=count(row->row.status=="passed",rows),pending=count(row->row.status=="pending",rows),
        limitations="Same source content/output cap, but input lengths and decode-graph cost differ; two development tasks only")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
