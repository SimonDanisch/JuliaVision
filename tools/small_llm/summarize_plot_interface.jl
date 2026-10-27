include("paired_workflow_uncertainty.jl")
using JSON3,SHA

function plot_interface_interpretation(primary,render=nothing)
    primary.complete&&length(primary.rows)==16||throw(ArgumentError("Complete frozen16 required"))
    tasks=unique(String(r.task) for r in primary.rows)
    length(tasks)==8||throw(ArgumentError("Eight paired requests required"))
    all(r->r.specification_correct==(r.status=="passed")&&(!r.specification_correct||r.admitted),primary.rows)||throw(ArgumentError("Score consistency"))
    comparison=paired_summary_uncertainty(primary,"julia_source","plot_spec")
    arms=map(("julia_source","plot_spec")) do arm
        rows=filter(r->r.arm==arm,primary.rows)
        (;arm,total=length(rows),admitted=count(r->r.admitted,rows),
            specification_correct=count(r->r.specification_correct,rows),
            admission_reasons=Dict(reason=>count(r->r.admission_reason==reason,rows) for reason in unique(String(r.admission_reason) for r in rows)),
            truncated=count(r->r.truncated===true,rows),inference_errors=count(r->r.status=="inference_error",rows))
    end
    render_state=render===nothing ? (;status="not_attempted",aesthetic_preference_tested=false) :
        (;status=render.complete ? "resolved" : "incomplete",
            counts=Dict(status=>count(r->r.status==status,render.rows) for status in unique(String(r.status) for r in render.rows)),
            aesthetic_preference_tested=false)
    (;complete=true,arms,comparison,render=render_state,
        interpretation="Two output interfaces to the SAME restricted drawing IR/backend, same public requirements. Better admission/specification accuracy would be a SYSTEM/interface benefit, not proof that model weights improved. Native Julia arm is a published constructor subset, NOT unrestricted Julia or arbitrary existing Makie code. JSON is not automatically correct merely because it parses.",
        limits=["Eight authored related-family requests, one seed; not public held-out or contamination-free evidence",
            "A successful/unclipped image does not establish aesthetic preference, correct scientific interpretation or accessibility",
            "Renderer was a separate PRE-GRADE development registration after generation began, not original prospective primary score",
            "No regeneration, hidden-feedback revision, arbitrary candidate-code execution or frontier-level capability claim"])
end

function summarize_plot_interface(;directory=joinpath(@__DIR__,"plot_interface_results"))
    primarypath=joinpath(directory,"summary.json");renderpath=joinpath(directory,"render_audit_summary.json")
    primary=JSON3.read(read(primarypath,String));render=isfile(renderpath) ? JSON3.read(read(renderpath,String)) : nothing
    digest(path)=bytes2hex(sha256(read(path)))
    render===nothing||render.primary_summary_sha256==digest(primarypath)||error("Render primary drift")
    result=(;plot_interface_interpretation(primary,render)...,primary_summary_sha256=digest(primarypath),
        render_summary_sha256=render===nothing ? nothing : digest(renderpath),source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_interpretation.json"),"w");result
end
