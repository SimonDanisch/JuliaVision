include("plot_interface_references.jl")
isdefined(@__MODULE__,:plot_validator_trigger)||include("plot_validator_policy.jl")
isdefined(@__MODULE__,:validate_pubmedqa_receipt)||include("score_pubmedqa_reading.jl")

function score_plot_validator_study(;directory=joinpath(@__DIR__,"plot_validator_results"),parent=joinpath(@__DIR__,"plot_interface_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    digest(@__FILE__)==p.private_scorer_sha256||error("Scorer drift")
    for (file,hash) in pairs(p.private_sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Source/reference drift")
    end
    parentpath=joinpath(parent,"conditions.json");original=JSON3.read(read(parentpath,String))
    digest(parentpath)==p.parent_runtime_conditions_sha256||error("Parent drift")
    all(isfile(joinpath(directory,j.id*".json")) for j in p.jobs)||error("Finish ALL triggered paired revisions BEFORE grading")
    length(p.decisions)==8&&length(unique(d.task for d in p.decisions))==8||error("Complete original cohort required")
    references=plot_interface_references();tasks=unique(String(j.task) for j in original.jobs)
    expected=Dict(tasks[i]=>SafePlotSpec.parse_spec(JSON3.write(references[i])) for i in 1:8)
    rows=Any[]
    for decision in p.decisions
        baseline=only(filter(j->j.id==decision.baseline_id,original.jobs));basepath=joinpath(parent,baseline.id*".json")
        digest(basepath)==decision.baseline_generation_sha256||error("Original generation drift")
        receipt=JSON3.read(read(basepath,String));trigger=plot_validator_trigger(receipt)
        trigger.triggered==decision.triggered&&trigger.reason==decision.public_reason||error("PUBLIC trigger drift")
        for arm in ("self_review","validator_feedback")
            id=String(decision.task)*"_"*arm
            job=decision.triggered ? only(filter(j->j.id==id,p.jobs)) : baseline
            folder=decision.triggered ? directory : parent
            selected=joinpath(folder,job.id*".json");answer=JSON3.read(read(selected,String))
            packet=decision.triggered ? p : original;conditions=decision.triggered ? conditionpath : parentpath
            validate_pubmedqa_receipt(answer,job,packet;condition_hash=digest(conditions),attempt_hash=digest(joinpath(folder,job.id*"_attempt.json")))
            failed=hasproperty(answer,:inference_error)
            admitted=failed||answer.truncated ? (;accepted=false,reason=failed ? "inference_error" : "truncated",spec=nothing) : admit_plot_interface(String(answer.answer),"plot_spec")
            correct=admitted.accepted&&plot_specs_equivalent(admitted.spec,expected[String(decision.task)])
            push!(rows,(;id,task=decision.task,arm,status=failed ? "inference_error" : correct ? "passed" : "failed",
                triggered=decision.triggered,retained_original=!decision.triggered,selected_generation_id=job.id,
                generation_sha256=digest(selected),original_generation_sha256=digest(basepath),
                admitted=admitted.accepted,admission_reason=admitted.reason,specification_correct=correct,
                truncated=failed ? nothing : answer.truncated,actual_render="not_evaluated_in_this_revision_study",aesthetic_score=nothing))
        end
    end
    arms=map(("self_review","validator_feedback")) do arm
        selected=filter(r->r.arm==arm,rows)
        (;arm,total=length(selected),passed=count(r->r.status=="passed",selected),admitted=count(r->r.admitted,selected),
            revised=count(r->r.triggered,selected),unchanged=count(r->r.retained_original,selected))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,rows,arms,conditions_sha256=digest(conditionpath),scorer_sha256=digest(@__FILE__),
        caveat="SAME8 posthoc development workflow, extra generation/tool compute, not independent transfer. Public schema cannot detect all semantic errors. Original primary/render results retained; new revisions are not assumed rendered or aesthetically good.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
