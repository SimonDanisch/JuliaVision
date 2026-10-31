isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
using JSON3,SHA

"""Separate literal-prompt audit; never mutates the frozen primary score."""
function audit_rpn_prompt_budget(;directory=joinpath(@__DIR__,"rpn_stencil_results"))
    digest(path)=bytes2hex(open(sha256,path))
    conditions=joinpath(directory,"conditions.json");p=JSON3.read(read(conditions,String))
    summarypath=joinpath(directory,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.conditions_sha256==digest(conditions)||error("summary drift")
    [r.id for r in summary.rows]==[j.id for j in p.jobs]||error("cohort drift")
    rows=Any[]
    for (job,primary) in zip(p.jobs,summary.rows)
        path=joinpath(directory,String(job.id)*".json")
        if !isfile(path)||!hasproperty(JSON3.read(read(path,String)),:answer)
            push!(rows,(;id=job.id,arm=job.arm,status="not_auditable",primary_status=primary.status));continue
        end
        candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(conditions)||error("candidate drift")
        hasproperty(primary,:generation_sha256)&&primary.generation_sha256!=digest(path)&&error("primary generation receipt drift")
        parsed=RPNPointwiseInterface.parse_program(String(candidate.answer);budget=Int(p.grammar.operation_budget),max_depth=Int(p.grammar.max_depth))
        operations=parsed.accepted ? length(parsed.actions)-1 : nothing
        literal_compliance=parsed.accepted&&operations<=p.grammar.operation_budget
        push!(rows,(;id=job.id,arm=job.arm,status="audited",primary_status=primary.status,
            grammar_admitted=parsed.accepted,operations,literal_compliance,
            strict_success=primary.status=="passed"&&literal_compliance,generation_sha256=digest(path)))
    end
    packet=(;scope="Separate literal32-operation prompt-compliance audit. Primary frozen scores retained; NOT a new generation or corrected prompt experiment.",
        deviation="Public prompt says maximum32 non-finish operations; implemented parser/device grammar allows up to7 final binary stack reductions beyond nominal32. Same parser/mask policy for both arms, but not the literal published limit.",
        rule="Strict success requires original mathematical pass AND original parser admission AND at most32 non-finish actions. No repairs, retries, changed tests or generation feedback.",
        complete=summary.complete,conditions_sha256=digest(conditions),primary_summary_sha256=digest(summarypath),
        auditor_sha256=digest(@__FILE__),rows)
    open(io->JSON3.pretty(io,packet),joinpath(directory,"literal_prompt_budget_audit.json"),"w");packet
end
