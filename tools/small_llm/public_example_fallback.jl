isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
isdefined(@__MODULE__,:RPNExampleConstraint)||include("rpn_example_constraint.jl")

"""Public-example-only workflow, never a hidden-test or task-ID selector."""
module PublicExampleFallback
import ..RPNPointwiseInterface
import ..RPNPrefixGrammar
import ..RPNExampleSemantics
import ..RPNExampleConstraint

function matches(program::AbstractString,bank::RPNExampleSemantics.Bank)
    parsed=RPNPointwiseInterface.parse_program(program;max_depth=Int(bank.max_depth))
    parsed.accepted||return false
    n=length(bank.expected);stack=zeros(Float32,n*Int(bank.max_depth));status=zeros(Int32,n)
    depth=Int32(0);operations=Int32(0)
    for action in parsed.actions
        for example in 1:n
            status[example]==0&&(status[example]=RPNExampleSemantics.step!(stack,bank.features,Int32(example),action,depth,bank.max_depth))
        end
        depth,operations,_,invalid=RPNPrefixGrammar.action_state(RPNPrefixGrammar.ARITIES[action],depth,operations,bank.max_depth,Int32(32))
        invalid==0||return false
    end
    RPNExampleConstraint.finish_allowed(stack,bank.expected,status,depth,bank.max_depth,Int32(n),bank.atol,bank.rtol)
end

function choose(neural_program::AbstractString,bank::RPNExampleSemantics.Bank,symbolic;infrastructure_ok=true)
    infrastructure_ok isa Bool||throw(ArgumentError("Explicit Boolean infrastructure flag"))
    infrastructure_ok||return (;selected="none",program="",reason="infrastructure_failure")
    matches(neural_program,bank)&&return (;selected="neural_baseline",program=String(neural_program),reason="public_examples_match")
    symbolic.found isa Bool||throw(ArgumentError("Boolean symbolic result required"))
    symbolic.found||return (;selected="none",program="",reason="no_public_matching_proposal")
    program=String(symbolic.program)
    matches(program,bank)||return (;selected="none",program="",reason="symbolic_proposal_does_not_fit_public_examples")
    (;selected="symbolic_public_examples",program,reason="neural_public_example_failure")
end
end
