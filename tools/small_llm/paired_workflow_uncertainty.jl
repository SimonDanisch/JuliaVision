"""Descriptive binomial interval; not a population-generalization guarantee."""
function workflow_wilson(successes::Integer,total::Integer)
    0<=successes<=total<=10000||throw(ArgumentError("Bounded counts required"))
    total==0&&return (;successes,total,estimate=nothing,lower=nothing,upper=nothing)
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2total))/den
    radius=z*sqrt(p*(1-p)/total+z*z/(4total*total))/den
    (;successes,total,estimate=p,lower=successes==0 ? 0. : max(0.,center-radius),
        upper=successes==total ? 1. : min(1.,center+radius))
end

"""Sidecar only: leaves frozen primary scores/conditions unchanged."""
function paired_summary_uncertainty(summary,before_arm,after_arm)
    hasproperty(summary,:complete)&&!summary.complete&&throw(ArgumentError("Incomplete summary"))
    function arm_rows(arm)
        rows=filter(r->String(r.arm)==arm,summary.rows)
        isempty(rows)&&throw(ArgumentError("Missing arm"))
        all(r->r.status in ("passed","failed","inference_error"),rows)||throw(ArgumentError("Unresolved outcome"))
        length(unique(String(r.task) for r in rows))==length(rows)||throw(ArgumentError("Duplicate task"))
        Dict(String(r.task)=>r.status=="passed" for r in rows)
    end
    before,after=arm_rows(before_arm),arm_rows(after_arm)
    Set(keys(before))==Set(keys(after))||throw(ArgumentError("Different cohorts"))
    tasks=sort!(collect(keys(before)))
    (;before_arm,after_arm,tasks,statistics=paired_workflow_uncertainty([before[t] for t in tasks],[after[t] for t in tasks]))
end

"""Exact two-sided paired sign test on discordant binary outcomes ONLY.
No independence/representativeness claim is supplied by this arithmetic.
"""
function paired_workflow_uncertainty(before,after)
    length(before)==length(after)<=10000||throw(ArgumentError("Equal bounded cohorts required"))
    all(x->x isa Bool,before)&&all(x->x isa Bool,after)||throw(ArgumentError("Explicit binary outcomes required"))
    repairs=count(i->!before[i]&&after[i],eachindex(before))
    regressions=count(i->before[i]&&!after[i],eachindex(before));discordant=repairs+regressions
    numerator=2sum(binomial(big(discordant),k) for k in 0:min(repairs,regressions))
    probability=min(1.,Float64(BigFloat(numerator)/BigFloat(big(2)^discordant)))
    (;total=length(before),repairs,regressions,discordant,
        unchanged_correct=count(i->before[i]&&after[i],eachindex(before)),
        unchanged_incorrect=count(i->!before[i]&&!after[i],eachindex(before)),
        before=workflow_wilson(count(identity,before),length(before)),
        after=workflow_wilson(count(identity,after),length(after)),
        exact_two_sided_discordance_p=probability,
        caveat="POSTHOC descriptive arithmetic, unadjusted for MANY experiments. Authored/public fixed cohorts need not be representative or independent; intervals/tests cannot establish frontier, clinical or held-out generalization. No optional stopping or output selection is authorized.")
end
