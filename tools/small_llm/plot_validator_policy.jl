isdefined(@__MODULE__,:admit_plot_interface)||include("plot_interface_admission.jl")
using JSON3,SHA

"""PUBLIC schema/output-budget trigger ONLY; NEVER consult a semantic grade."""
function plot_validator_trigger(receipt)
    hasproperty(receipt,:inference_error)&&throw(ArgumentError("Do not convert inference infrastructure errors into revision attempts"))
    receipt.truncated&&return (;triggered=true,reason="Output reached its registered token allowance without a complete final answer.")
    admission=admit_plot_interface(String(receipt.answer),"plot_spec")
    (;triggered=!admission.accepted,reason=admission.accepted ? "Public schema admits the plan; this says NOTHING about task correctness." : admission.reason)
end

function plot_validator_revision_messages(original,receipt,arm)
    arm in ("self_review","validator_feedback")||throw(ArgumentError("Unknown paired condition"))
    trigger=plot_validator_trigger(receipt);trigger.triggered||throw(ArgumentError("Admitted plans must remain unchanged"))
    messages=[String(first(keys(m)))=>String(first(values(m))) for m in original]
    length(messages)==2&&first(messages[1])=="system"&&first(messages[2])=="user"||throw(ArgumentError("Original task-only context required"))
    push!(messages,"assistant"=>String(receipt.answer))
    request="Review and revise the previous answer against the ORIGINAL public plot request and JSON schema. Return ONLY ONE COMPLETE JSON plot object, no Julia, markdown or explanation. Do not transform data or invent fields. No private tests, expected plans or semantic grading feedback are supplied.\n\n"
    request*=arm=="self_review" ? "Self-review control: no validator observations are supplied." :
        "Public output/schema validator observation (NOT a task-correctness judgment): "*trigger.reason
    push!(messages,"user"=>request)
    messages
end

"""Posthoc SAME8 workflow after the original pilot; not independent transfer.
All original JSON answers retained unless the PUBLIC trigger fires. Both arms
get exactly one revision from the SAME original rejected answer when triggered.
Admitted-but-wrong plans are deliberately left alone; no hidden-score routing.
"""
function prepare_plot_validator_policy(;token_count,parent=joinpath(@__DIR__,"plot_interface_results"),destination=joinpath(@__DIR__,"plot_validator_preparation.json"))
    isfile(destination)&&error("Preserve preparation")
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(parent,"conditions.json");p=JSON3.read(read(conditionpath,String))
    length(p.jobs)==16&&all(isfile(joinpath(parent,j.id*".json")) for j in p.jobs)||error("Finish ALL original generations first")
    baseline=filter(j->j.arm=="plot_spec",p.jobs);length(baseline)==8||error("Keep ALL8 original JSON tasks")
    jobs=Any[];decisions=Any[]
    for (i,job) in enumerate(baseline)
        output=joinpath(parent,job.id*".json");receipt=JSON3.read(read(output,String))
        receipt.conditions_sha256==digest(conditionpath)&&receipt.normal_final_waitidle||error("Baseline provenance/cleanup")
        trigger=plot_validator_trigger(receipt)
        push!(decisions,(;task=job.task,baseline_id=job.id,baseline_generation_sha256=digest(output),triggered=trigger.triggered,public_reason=trigger.reason))
        trigger.triggered||continue
        for arm in (isodd(i) ? ("self_review","validator_feedback") : ("validator_feedback","self_review"))
            messages=plot_validator_revision_messages(job.messages,receipt,arm);tokens=token_count(messages)
            tokens isa Integer&&!(tokens isa Bool)&&0<tokens&&tokens+1024+32<=8192||error("Capacity failure: no filtering/retry")
            push!(jobs,(;id=job.task*"_"*arm,task=job.task,arm,messages,input_tokens=tokens))
        end
    end
    result=(;complete=true,posthoc=true,parent_conditions_sha256=digest(conditionpath),decisions,jobs,
        policy="SAME8 authored JSON tasks, original answers fixed. PUBLIC schema/truncation failure triggers both one self-review and one validator-feedback revision from SAME original answer, each1024 nonthinking tokens and same seed. ALL admitted originals kept even if semantically wrong. Grade ALL8 workflow outcomes per arm only after all triggered revisions/worker retirements. No hidden correctness-based routing, retries or reference corrections.",
        parent_scope=p.scope,source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,result),destination,"w");result
end
