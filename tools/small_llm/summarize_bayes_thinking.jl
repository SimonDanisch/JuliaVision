include("paired_workflow_uncertainty.jl")
include("workflow_cost_summary.jl")

function bayes_thinking_summary_statistics(primary,baseline)
    primary.complete&&baseline.complete&&length(primary.rows)==length(baseline.rows)==16||throw(ArgumentError("Complete ALL16 paired results required"))
    arms=("direct_probability","bayes_network")
    all(r->r.arm in arms&&r.status in ("passed","failed","inference_error"),vcat(collect(primary.rows),collect(baseline.rows)))||throw(ArgumentError("Unknown or incomplete outcome"))
    comparisons=map(arms) do arm
        function outcomes(summary)
            selected=filter(r->r.arm==arm,summary.rows)
            length(selected)==8&&length(unique(String(r.task) for r in selected))==8||throw(ArgumentError("ALL8 unique tasks per arm"))
            Dict(String(r.task)=>r.status=="passed" for r in selected)
        end
        before,after=outcomes(baseline),outcomes(primary)
        Set(keys(before))==Set(keys(after))||throw(ArgumentError("Cohort drift"))
        tasks=sort!(collect(keys(before)))
        (;arm,relative_to_nonthinking=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]))
    end
    (;comparisons,thinking_network_vs_direct=paired_summary_uncertainty(primary,"direct_probability","bayes_network"),
        scope="POSTHOC SAME8 development problems; thinking/template, allowance and controller change together. Extra inference/solver work, not equal compute, isolated mode/weight effect, independent transfer, clinical or frontier intelligence.")
end

function summarize_bayes_thinking(;directory=joinpath(@__DIR__,"bayes_thinking_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    primarypath=joinpath(directory,"summary.json");primary=JSON3.read(read(primarypath,String))
    baselinepath=joinpath(@__DIR__,"bayes_extraction_results/summary.json");baseline=JSON3.read(read(baselinepath,String))
    primary.conditions_sha256==digest(path)&&p.parent_summary_sha256==digest(baselinepath)||error("Score provenance drift")
    receipts=Dict(String(j.id)=>JSON3.read(read(joinpath(directory,j.id*".json"),String)) for j in p.jobs)
    result=(;complete=true,bayes_thinking_summary_statistics(primary,baseline)...,
        incremental_generation_costs=workflow_cost_totals(p,primary,receipts),
        primary_summary_sha256=digest(primarypath),baseline_summary_sha256=digest(baselinepath),source_sha256=digest(@__FILE__),
        caveat="Generation timers include prefill/JIT/decode but exclude model load/startup and solver. NOT tokens/s or end-to-end latency; preserve nonthinking and native-FP32/greedy controls separately.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"paired_interpretation.json"),"w");result
end
