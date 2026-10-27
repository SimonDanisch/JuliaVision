using JSON3,SHA
function pubmedqa_paired_changes(rows,control,treatment)
    tasks=unique(String(r.task) for r in rows)
    pairs=map(tasks) do task
        a=only(filter(r->r.task==task&&r.arm==control,rows))
        b=only(filter(r->r.task==task&&r.arm==treatment,rows))
        (;task,control_correct=a.status=="passed",treatment_correct=b.status=="passed",
            prediction_changed=a.prediction!=b.prediction)
    end
    (;control,treatment,total=length(pairs),repairs=count(r->!r.control_correct&&r.treatment_correct,pairs),
        regressions=count(r->r.control_correct&&!r.treatment_correct,pairs),
        prediction_changes=count(r->r.prediction_changed,pairs),pairs)
end

function summarize_pubmedqa_reading(;directory=joinpath(@__DIR__,"pubmedqa_reading_results"))
    path=joinpath(directory,"summary.json");s=JSON3.read(read(path,String))
    s.complete||error("Finish ALL48 before final interpretation")
    conditionpath=joinpath(directory,"conditions.json")
    s.conditions_sha256==bytes2hex(sha256(read(conditionpath)))||error("Score drift")
    context=only(filter(a->a.arm=="corresponding_context",s.arms))
    class_counts=Dict(String(label)=>sum(values(counts)) for (label,counts) in pairs(context.confusion))
    result=(;complete=true,scope=s.scope,conditions_sha256=s.conditions_sha256,
        summary_sha256=bytes2hex(sha256(read(path))),
        question_vs_corresponding=pubmedqa_paired_changes(s.rows,"question_only","corresponding_context"),
        mismatched_vs_corresponding=pubmedqa_paired_changes(s.rows,"mismatched_context","corresponding_context"),
        question_mismatched_same_prediction=count(unique(String(r.task) for r in s.rows)) do task
            only(filter(r->r.task==task&&r.arm=="question_only",s.rows)).prediction==
                only(filter(r->r.task==task&&r.arm=="mismatched_context",s.rows)).prediction
        end,
        class_counts,best_constant_accuracy=maximum(values(class_counts))/sum(values(class_counts)),
        uncertainty_warning="maybe is a dataset conclusion label, not calibrated confidence or abstention; report its recall separately",
        interpretation="Small one-seed oracle-source reading workflow effect, not a trained-weight improvement, full retriever, independent unseen-paper proof, current medical consensus or clinical advice. Question-only is a memorization/missing-information diagnostic; all-no behavior is NOT substantive medical reasoning. Gold/conclusions never entered prompts. No significance or frontier claim.",
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    destination=joinpath(directory,"paired_interpretation.json")
    isfile(destination)&&error("Preserve final interpretation")
    open(io->JSON3.pretty(io,result),destination,"w");result
end
