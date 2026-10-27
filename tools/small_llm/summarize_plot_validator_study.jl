include("paired_workflow_uncertainty.jl")
include("workflow_cost_summary.jl")

function plot_validator_summary_statistics(primary,baseline)
    primary.complete&&baseline.complete&&length(primary.rows)==16||throw(ArgumentError("Complete paired workflow required"))
    originals=filter(r->r.arm=="plot_spec",baseline.rows)
    length(originals)==8&&length(unique(r.task for r in originals))==8||throw(ArgumentError("ALL8 original JSON outcomes required"))
    before=Dict(String(r.task)=>r.status=="passed" for r in originals)
    tasks=sort!(collect(keys(before)))
    comparisons=map(("self_review","validator_feedback")) do arm
        selected=filter(r->r.arm==arm,primary.rows)
        length(selected)==8&&Set(String(r.task) for r in selected)==Set(tasks)||throw(ArgumentError("Matched ALL8 per arm"))
        for row in selected
            row.retained_original&&(row.status=="passed")!=before[String(row.task)]&&throw(ArgumentError("Retained original cannot change correctness"))
        end
        after=Dict(String(r.task)=>r.status=="passed" for r in selected)
        (;arm,relative_to_original=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]))
    end
    (;baseline_correct=count(identity,values(before)),total=8,comparisons,
        feedback_vs_review=paired_summary_uncertainty(primary,"self_review","validator_feedback"),
        scope="POSTHOC SAME8 conditional workflows. Original grades never routed revisions. No independent transfer, model-weight/frontier gain, or aesthetic score.")
end

function summarize_plot_validator_study(;directory=joinpath(@__DIR__,"plot_validator_results"),parent=joinpath(@__DIR__,"plot_interface_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    primarypath=joinpath(directory,"summary.json");baselinepath=joinpath(parent,"summary.json")
    p=JSON3.read(read(joinpath(directory,"conditions.json"),String));primary=JSON3.read(read(primarypath,String));baseline=JSON3.read(read(baselinepath,String))
    primary.conditions_sha256==digest(joinpath(directory,"conditions.json"))||error("Score provenance")
    p.observed_parent_summary_sha256==digest(baselinepath)||error("Original score drift")
    receipts=Dict(String(j.id)=>JSON3.read(read(joinpath(directory,j.id*".json"),String)) for j in p.jobs)
    for row in filter(r->r.triggered,primary.rows)
        digest(joinpath(directory,row.id*".json"))==row.generation_sha256||error("Generation drift")
    end
    # Score coverage is16 workflow outcomes but only SIX new inference jobs.
    # Never invent decode work for the ten retained-original workflow choices.
    revision_scores=(;complete=primary.complete,rows=filter(r->r.triggered,primary.rows))
    costs=workflow_cost_totals(p,revision_scores,receipts)
    result=(;complete=true,plot_validator_summary_statistics(primary,baseline)...,incremental_generation_costs=costs,
        primary_summary_sha256=digest(primarypath),baseline_summary_sha256=digest(baselinepath),source_sha256=digest(@__FILE__),
        caveat="Both branches share the original8 JSON generations. Costs here are ONLY additional triggered revisions, excluding original generation/model loads/worker startup/validator/render time. Timer is NOT end-to-end latency or steady-state decode throughput. Rendering of original admitted plans does NOT establish new revision renderability/aesthetics.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_interpretation.json"),"w");result
end
