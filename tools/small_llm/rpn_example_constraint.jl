isdefined(@__MODULE__,:RPNExampleSemantics)||include("rpn_example_semantics.jl")
isdefined(@__MODULE__,:RPNPrefixGrammar)||include("rpn_prefix_grammar.jl")
module RPNExampleConstraint
import ..RPNExampleSemantics
import ..RPNPrefixGrammar
import ..TokenPrefixTrie

"""Nodes with some descendant action other than finish. No semantic labels."""
function nonfinish_nodes(grammar::RPNPrefixGrammar.Grammar)
    result=zeros(Int32,length(grammar.terminal_action))
    # Trie edges always point to higher-numbered nodes within each root group.
    for node in length(result):-1:1
        terminal=grammar.terminal_action[node]
        possible=terminal!=0&&terminal!=RPNPrefixGrammar.FINISH
        for edge in grammar.offsets[node]:grammar.offsets[node+1]-Int32(1)
            target=grammar.targets[edge]
            target>node||throw(ArgumentError("non-forward trie edge"))
            possible|=result[target]!=0
        end
        result[node]=Int32(possible)
    end
    result
end

@inline function finish_allowed(stack,expected,status,depth::Int32,max_depth::Int32,n::Int32,atol::Float32,rtol::Float32)
    n>0&&n<=length(status)&&n<=length(expected)&&depth==1||return false
    for example in Int32(1):n
        @inbounds status[example]==0||return false
        RPNExampleSemantics.matches(stack,expected,example,depth,max_depth,atol,rtol)||return false
    end
    true
end

"""Prune only finish-only subtrees; shared BPE prefixes remain available.

When finish is the sole grammar continuation and examples fail, there is no
admissible token. Callers must report a failed/incomplete attempt, NOT repair it
or force an incorrect finish. This does not establish prefix completable-ness.
"""
@inline function token_allowed(offsets,labels,targets,nonfinish,node::Int32,token::Int32,allow_finish::Bool)
    node==0&&return true # completed/unconstrained sentinel, like the base grammar
    target=TokenPrefixTrie.edge_target(offsets,labels,targets,node,token)
    target>0||return false
    @inbounds allow_finish||nonfinish[target]!=0
end

@inline function committed_action(offsets,labels,targets,terminal,node::Int32,token::Int32)
    node==0&&return Int32(0)
    target=TokenPrefixTrie.edge_target(offsets,labels,targets,node,token)
    target>0||return Int32(-1)
    @inbounds terminal[target]
end
end
