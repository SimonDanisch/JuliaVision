include("summarize_compact_bayes_demo.jl")

function summarize_compact_bayes_demo_resume(;directory=joinpath(@__DIR__,"compact_bayes_demo_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    barepath=joinpath(@__DIR__,"compact_bayes_results/conditions.json")
    baselinepath=joinpath(dirname(barepath),"summary.json");baseline=JSON3.read(read(baselinepath,String))
    p.bare_conditions_sha256==digest(barepath)&&p.bare_summary_sha256==digest(baselinepath)||error("Baseline drift")
    primary.conditions_sha256==digest(path)&&baseline.conditions_sha256==digest(barepath)||error("Score drift")
    generated=filter(r->r.generation_sha256!==nothing,primary.rows)
    generated_ids=Set(String(r.id) for r in generated)
    receipts=Dict(String(j.id)=>JSON3.read(read(joinpath(directory,j.id*".json"),String)) for j in p.jobs if j.id in generated_ids)
    for job in p.jobs
        digest(joinpath(dirname(barepath),String(job.task)*"_compact_network.json"))==job.bare_generation_sha256||error("Baseline answer drift")
    end
    for row in primary.rows
        row.generation_sha256===nothing ?
            digest(joinpath(directory,row.id*"_lost.json"))==row.loss_sha256||error("Loss drift") :
            digest(joinpath(directory,row.id*".json"))==row.generation_sha256||error("Generated answer drift")
    end
    completed_tasks=Set(String(r.task) for r in generated)
    before=Dict(String(r.task)=>r.status=="passed" for r in baseline.rows if r.task in completed_tasks)
    after=Dict(String(r.task)=>r.status=="passed" for r in generated)
    tasks=sort!(collect(completed_tasks))
    costs=workflow_cost_totals((;p...,jobs=filter(j->j.id in generated_ids,p.jobs)),(;complete=true,rows=generated),receipts)
    result=(;complete=true,compact_bayes_demo_statistics(primary,baseline)...,
        complete_case_tasks=tasks,complete_case_paired=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]),
        attrition=primary.attrition,lost_attempts=primary.lost_attempts,completed_generations=primary.completed_generations,
        completed_generation_costs=costs,lost_generation_cost="Unknown; NOT zero or included in completed timer totals",
        solver_seconds=sum(r.solver_seconds for r in primary.rows),primary_summary_sha256=digest(primarypath),
        baseline_summary_sha256=digest(baselinepath),source_sha256=digest(@__FILE__),
        caveat="Registered8 conservative comparison counts missing outcome as not passed; complete-case7 is descriptive attrition-affected. Timers exclude load/startup/retirement/solver; not throughput or equal-compute.")
    output=joinpath(directory,"paired_interpretation.json");ispath(output)&&error("Preserve first interpretation")
    open(io->JSON3.pretty(io,result),output,"w");result
end
