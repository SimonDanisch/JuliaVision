module PubMedQAMetrics
const CLASSES=("yes","no","maybe")
"""Exact labels, full denominators, explicit invalids. No confidence estimate."""
function summarize(rows,gold;arms=("question_only","corresponding_context","mismatched_context"))
    !isempty(gold)&&!isempty(arms)&&length(unique(arms))==length(arms)||throw(ArgumentError("Nonempty cohort and unique arms"))
    all(label->label in CLASSES,values(gold))||throw(ArgumentError("Gold labels"))
    length(rows)==length(gold)*length(arms)||throw(ArgumentError("Complete all conditions"))
    length(unique((String(r.task),String(r.arm)) for r in rows))==length(rows)||throw(ArgumentError("Duplicate condition"))
    all(r->String(r.task) in keys(gold)&&String(r.arm) in arms&&String(r.prediction) in (CLASSES...,"invalid"),rows)||throw(ArgumentError("Unknown row"))
    map(collect(arms)) do arm
        subset=filter(r->r.arm==arm,rows)
        length(subset)==length(gold)||throw(ArgumentError("Arm coverage"))
        confusion=Dict(actual=>Dict(pred=>count(r->gold[String(r.task)]==actual&&r.prediction==pred,subset)
            for pred in (CLASSES...,"invalid")) for actual in CLASSES)
        per_class=map(CLASSES) do label
            tp=confusion[label][label]
            fp=count(r->r.prediction==label&&gold[String(r.task)]!=label,subset)
            fn=count(r->gold[String(r.task)]==label&&r.prediction!=label,subset)
            denominator=2tp+fp+fn
            (;label,tp,fp,fn,f1=denominator==0 ? 0.0 : 2tp/denominator)
        end
        correct=sum(confusion[label][label] for label in CLASSES)
        (;arm,total=length(subset),correct,accuracy=correct/length(subset),
            macro_f1=sum(r.f1 for r in per_class)/3,
            invalid=count(r->r.prediction=="invalid",subset),per_class,confusion)
    end
end
end
