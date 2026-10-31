include("summarize_pubmedqa_reading.jl")
using JSON3,SHA
function summarize_pubmedqa_retrieval(;directory=joinpath(@__DIR__,"pubmedqa_retrieval_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"summary.json");s=JSON3.read(read(path,String))
    conditions=joinpath(directory,"conditions.json");p=JSON3.read(read(conditions,String))
    s.complete&&s.conditions_sha256==digest(conditions)||error("Finish all64 frozen conditions")
    identitypath=joinpath(directory,"identical_prompt_audit.json");identity=JSON3.read(read(identitypath,String))
    identity.complete&&identity.all_agree&&identity.conditions_sha256==digest(conditions)&&identity.summary_sha256==digest(path)||error("Repeated-prompt identity audit missing/drifted")
    oracle=only(filter(a->a.arm=="corresponding_context",s.arms))
    class_counts=Dict(String(label)=>sum(values(counts)) for (label,counts) in pairs(oracle.confusion))
    retrieval_jobs=filter(j->j.arm=="retrieved_context",p.jobs)
    result=(;complete=true,scope=s.scope,conditions_sha256=s.conditions_sha256,summary_sha256=digest(path),
        retrieval_source_hits=count(j->j.context_pmid==j.task,retrieval_jobs),retrieval_total=length(retrieval_jobs),
        question_vs_retrieval=pubmedqa_paired_changes(s.rows,"question_only","retrieved_context"),
        mismatch_vs_retrieval=pubmedqa_paired_changes(s.rows,"mismatched_context","retrieved_context"),
        oracle_vs_retrieval=pubmedqa_paired_changes(s.rows,"corresponding_context","retrieved_context"),
        identical_prompt_groups=identity.group_count,all_identical_prompts_agree=identity.all_agree,
        class_counts,best_constant_accuracy=maximum(values(class_counts))/sum(values(class_counts)),
        interpretation="NEW16 same-seed source-reading/retrieval workflow comparison, separate from prior16. Baseline/control all-no behavior is weak and not evidence of substantive medical knowledge. Compare also with the best constant label. Body-only1000-abstract retrieval is not general web search. All maybe items must be reported; no calibrated uncertainty, clinical, frontier, model-weight or pooled-cohort claim.",
        source_sha256=digest(@__FILE__))
    destination=joinpath(directory,"paired_interpretation.json");isfile(destination)&&error("Preserve final interpretation")
    open(io->JSON3.pretty(io,result),destination,"w");result
end
