isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
import Mantle

"""Owned-by-caller destination; wraps but never owns/changes the head weights."""
struct HeadInputTap{W,Q}
    weight::W
    query::Q
end

# Extend the existing interface on OUR type, not an existing weight method.
# Both sequential decode and batched prefill use this same final-head entry.
function SmallLLMChecks.Qwen35Experiment.linear!(graph,output,tap::HeadInputTap,input;name)
    tap.query isa Mantle.Buffer || throw(ArgumentError("device-owned query buffer required"))
    tap.query.dev===graph.dev || throw(ArgumentError("head query belongs to another device"))
    length(tap.query)==length(input) && eltype(tap.query)===eltype(input)===Float32 ||
        throw(DimensionMismatch("native head-input capture"))
    SmallLLMChecks.Qwen35Experiment.dispatch(graph,SmallLLMChecks.Qwen35Experiment.BR.copy_kernel!,
        (tap.query,input,Int32(length(input))),length(input);group=256,name=name*".capture_input")
    SmallLLMChecks.Qwen35Experiment.linear!(graph,output,tap.weight,input;name)
end
# The tap is not a weight; the head it wraps is.
SmallLLMChecks.Qwen35Experiment.semanticweights(key,tap::HeadInputTap)=
    SmallLLMChecks.Qwen35Experiment.semanticweights(key,tap.weight)

"""Scope a read-only native head-input capture around ALL session work.

Keep the wrapper installed through session creation, prefill, decode and session
close: prefill plans can be created lazily. The caller owns query and must not
close it while recorded plans still reference it. No concurrent model mutation.
"""
function with_head_input_tap(f,model,query)
    length(query)==Int(model.config.hidden_size) && eltype(query)===Float32 ||
        throw(ArgumentError("head query must be model-width Float32"))
    original=model.weights["lm_head"]
    original isa HeadInputTap && throw(ArgumentError("nested head taps are not supported"))
    model.weights["lm_head"]=HeadInputTap(original,query)
    try
        f()
    finally
        # Do not resurrect dangling weights if the caller closed the model.
        haskey(model.weights,"lm_head") && (model.weights["lm_head"]=original)
    end
end
