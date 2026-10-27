isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
module StencilBoundaryContract
import ..RPNPointwiseInterface

"""Narrow affirmative phrase recognizer, NOT general language understanding.
Ambiguity/negation means unknown. No task IDs, answers or private tests.
"""
function recognize(specification::AbstractString)
    text=lowercase(specification)
    occursin(r"\b(not|never|without|except|nonperiodic)\b",text)&&return :unknown
    periodic=occursin(r"\b(previous|next|previous and next) periodic neighbors?\b",text)
    clamped=occursin(r"\b(previous|next|previous and next) clamped neighbors?\b",text)
    periodic==clamped ? :unknown : periodic ? :periodic : :clamped
end

allowed_push(action,mode)=mode==:periodic ? !(action in (4,5)) : mode==:clamped ? !(action in (2,3)) : true

"""Conservative syntactic feature contract, not proof of wrong arithmetic.
Even a cancelling use of the other boundary mode is excluded from this subset.
"""
function accepts(program::AbstractString,mode::Symbol)
    mode in (:periodic,:clamped,:unknown)||throw(ArgumentError("Unknown boundary contract"))
    parsed=RPNPointwiseInterface.parse_program(program)
    parsed.accepted&&all(action->allowed_push(action,mode),parsed.actions)
end
end
