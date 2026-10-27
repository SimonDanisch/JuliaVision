isdefined(@__MODULE__,:StencilBoundaryContract)||include("stencil_boundary_contract.jl")
module StencilBoundaryMaxProof
import ..StencilBoundaryContract
import ..RPNPointwiseInterface

"""Narrow symbolic boundary proof, not task correctness or general equivalence.

For every valid index, clamped previous is either current or periodic previous;
clamped next is either current or periodic next. In a pure max expression a
clamped feature is redundant when BOTH members of its covering pair are already
present. No numeric evaluator is introduced and no program is rewritten.
This supplementary posthoc proof never changes frozen benchmark decisions.
"""
function certify(program::AbstractString,mode::Symbol)
    mode in (:periodic,:clamped,:unknown)||throw(ArgumentError("Invalid mode"))
    parsed=RPNPointwiseInterface.parse_program(program)
    parsed.accepted||return (;accepted=false,reason="invalid_program",rules=String[])
    StencilBoundaryContract.accepts(program,mode)&&return (;accepted=true,reason="existing_syntactic_contract",rules=String[])
    mode==:periodic||return (;accepted=false,reason="no_supported_boundary_proof",rules=String[])
    all(a->a<=9||a==16||a==17,parsed.actions)||return (;accepted=false,reason="not_a_pure_max_expression",rules=String[])
    leaves=Set(a for a in parsed.actions if a<=9);rules=String[]
    if 4 in leaves
        1 in leaves&&2 in leaves||return (;accepted=false,reason="uncovered_clamped_previous",rules=String[])
        push!(rules,"clamped_previous belongs to {current, periodic_previous}; max already covers both")
    end
    if 5 in leaves
        1 in leaves&&3 in leaves||return (;accepted=false,reason="uncovered_clamped_next",rules=String[])
        push!(rules,"clamped_next belongs to {current, periodic_next}; max already covers both")
    end
    (;accepted=true,reason="redundant_clamped_leaves_in_periodic_max",rules)
end
end
