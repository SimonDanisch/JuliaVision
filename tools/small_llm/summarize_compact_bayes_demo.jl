using JSON3,SHA
include("paired_workflow_uncertainty.jl")
include("workflow_cost_summary.jl")

function compact_bayes_demo_statistics(primary,baseline)
    primary.complete&&baseline.complete&&length(primary.rows)==length(baseline.rows)==8||throw(ArgumentError("Complete ALL8 required"))
    for result in (primary,baseline)
        length(unique(String(r.task) for r in result.rows))==8||throw(ArgumentError("Duplicate tasks"))
        all(r->r.arm=="compact_network"&&r.status in ("passed","failed","inference_error"),result.rows)||throw(ArgumentError("Invalid arm/status"))
    end
    before=Dict(String(r.task)=>r.status=="passed" for r in baseline.rows)
    after=Dict(String(r.task)=>r.status=="passed" for r in primary.rows)
    Set(keys(before))==Set(keys(after))||throw(ArgumentError("Cohort mismatch"))
    tasks=sort!(collect(keys(before)))
    (;tasks,relative_to_bare_compact=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]),
        scope="POSTHOC SAME8 with two public unrelated format examples. Extra context/compute, not independent transfer, weight improvement or frontier evidence.")
end

function summarize_compact_bayes_demo(;directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    barepath=joinpath(@__DIR__,"compact_bayes_results/conditions.json")
    baselinepath=joinpath(dirname(barepath),"summary.json");baseline=JSON3.read(read(baselinepath,String))
    p.bare_conditions_sha256==digest(barepath)&&p.bare_summary_sha256==digest(baselinepath)||error("Baseline drift")
    primary.conditions_sha256==digest(path)&&baseline.conditions_sha256==digest(barepath)||error("Score provenance drift")
    receipts=Dict(String(j.id)=>JSON3.read(read(joinpath(directory,j.id*".json"),String)) for j in p.jobs)
    for job in p.jobs
        original=joinpath(dirname(barepath),String(job.task)*"_compact_network.json")
        digest(original)==job.bare_generation_sha256||error("Baseline answer drift")
    end
    for row in primary.rows
        digest(joinpath(directory,row.id*".json"))==row.generation_sha256||error("Generated answer drift")
    end
    result=(;complete=true,compact_bayes_demo_statistics(primary,baseline)...,
        incremental_generation_costs=workflow_cost_totals(p,primary,receipts),
        solver_seconds=sum(r.solver_seconds for r in primary.rows),
        primary_summary_sha256=digest(primarypath),baseline_summary_sha256=digest(baselinepath),source_sha256=digest(@__FILE__),
        caveat="Generation timer excludes loading/startup/retirement and external solver; not end-to-end speed or tokens/s. Wrong generated networks are not corrected.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_interpretation.json"),"w");result
end
