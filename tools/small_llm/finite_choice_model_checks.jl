"""Integration checks only; accepting a legal name is not task correctness."""
function finite_choice_model_checks(baseline,disabled,constrained,sequences,eos;steps=8)
    observations=(baseline,disabled,constrained)
    accounting=all(o->o.status==0&&o.draws==steps+1&&o.submissions==steps&&length(o.tokens)==steps,observations)
    healthy=all(o->eltype(o.raw_logits)==Float32&&!any(isnan,o.raw_logits)&&!any(==(Inf32),o.raw_logits)&&any(isfinite,o.raw_logits),observations)
    identity=baseline.tokens==disabled.tokens&&baseline.seen==disabled.seen&&
        reinterpret(UInt32,baseline.raw_logits)==reinterpret(UInt32,disabled.raw_logits)&&
        disabled.node==1&&disabled.done==0&&disabled.rejected==0
    stop=findfirst(==(eos),constrained.tokens)
    support=stop!==nothing&&any(path->constrained.tokens[1:stop]==path,sequences)&&
        all(==(eos),constrained.tokens[stop:end])&&constrained.done==1&&constrained.rejected==0
    (;passed=accounting&&healthy&&identity&&support,accounting,healthy,disabled_identity=identity,finite_support=support,
        scope="Same model/sampler identity and canonical support only; no correct API/function selection or coding accuracy claim")
end
