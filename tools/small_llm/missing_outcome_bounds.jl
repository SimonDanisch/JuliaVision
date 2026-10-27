"""Identification bounds for missing outcomes, NOT sampling confidence intervals."""
function missing_outcome_bounds(passed,completed,total)
    all(x->x isa Integer&&!(x isa Bool),(passed,completed,total))||throw(ArgumentError("Integer counts"))
    0<=passed<=completed<=total&&total>0||throw(ArgumentError("Count order"))
    missing=total-completed
    (;passed,completed,total,missing,complete_case_rate=completed==0 ? nothing : passed/completed,
        registered_cohort_lower=passed/total,registered_cohort_upper=(passed+missing)/total,
        scope="All missing outcomes might fail or pass. Identification bounds, not confidence intervals or a claim that process loss is a reasoning failure.")
end
