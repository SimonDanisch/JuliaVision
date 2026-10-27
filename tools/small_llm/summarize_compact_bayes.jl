include("paired_workflow_uncertainty.jl")
include("workflow_cost_summary.jl")

function compact_bayes_summary_statistics(primary,baseline)
    primary.complete&&baseline.complete&&length(primary.rows)==8&&length(baseline.rows)==16||throw(ArgumentError("Complete ALL8/ALL16 results required"))
    originals=filter(r->r.arm=="bayes_network",baseline.rows)
    length(originals)==8&&length(unique(String(r.task) for r in originals))==8||throw(ArgumentError("Unique ALL8 original networks required"))
    length(unique(String(r.task) for r in primary.rows))==8&&all(r->r.arm=="compact_network",primary.rows)||throw(ArgumentError("Unique ALL8 compact rows required"))
    all(r->r.status in ("passed","failed","inference_error"),vcat(collect(originals),collect(primary.rows)))||throw(ArgumentError("Unresolved status"))
    before=Dict(String(r.task)=>r.status=="passed" for r in originals)
    after=Dict(String(r.task)=>r.status=="passed" for r in primary.rows)
    Set(keys(before))==Set(keys(after))||throw(ArgumentError("Cohort mismatch"))
    tasks=sort!(collect(keys(before)))
    (;tasks,relative_to_full_schema=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]),
        scope="POSTHOC same8 fictional problems. Changed public interface/prompt/output lengths, original sampler/model; not independent transfer, equal compute or isolated model-weight improvement.")
end

function summarize_compact_bayes(;directory=joinpath(@__DIR__,"compact_bayes_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    baselinepath=joinpath(@__DIR__,"bayes_extraction_results/summary.json");baseline=JSON3.read(read(baselinepath,String))
    primary.conditions_sha256==digest(path)&&p.parent_summary_sha256==digest(baselinepath)||error("Score provenance drift")
    receipts=Dict(String(j.id)=>JSON3.read(read(joinpath(directory,j.id*".json"),String)) for j in p.jobs)
    for row in primary.rows
        digest(joinpath(directory,row.id*".json"))==row.generation_sha256||error("Generated answer drift")
    end
    result=(;complete=true,compact_bayes_summary_statistics(primary,baseline)...,
        incremental_generation_costs=workflow_cost_totals(p,primary,receipts),
        solver_seconds=sum(r.solver_seconds for r in primary.rows),
        primary_summary_sha256=digest(primarypath),baseline_summary_sha256=digest(baselinepath),source_sha256=digest(@__FILE__),
        caveat="Model generation timer excludes model/session loading, worker startup/retirement and external solver; not end-to-end speed or decode throughput. No semantic correction or reference network supplied to solver.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_interpretation.json"),"w");result
end
