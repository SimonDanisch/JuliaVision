isdefined(@__MODULE__,:TokenPrefixTrie)||include("token_prefix_trie.jl")
"""Finite stack-syntax constraints, NOT a solver for a requested computation.

Every action remains the model's choice. The grammar enforces operand arity,
stack bounds and eventual closure; it neither chooses nor repairs arithmetic.
Transition arithmetic is allocation-free and shared between CPU/device callers.
"""
module RPNPrefixGrammar
import ..TokenPrefixTrie

const ACTIONS=("x0","x_prev_periodic","x_next_periodic","x_prev_clamped","x_next_clamped",
    "zero","one","two","half","neg","square","add","sub","mul","min","max","finish")
const ARITIES=Int32[0,0,0,0,0,0,0,0,0,1,1,2,2,2,2,2,3]
const FINISH=Int32(17)
const EOS_ACTION=Int32(18)

struct Grammar
    offsets::Vector{Int32}
    labels::Vector{Int32}
    targets::Vector{Int32}
    terminal_action::Vector{Int32}
    roots::Vector{Int32}
    arities::Vector{Int32}
    sequences::Vector{Vector{Int32}}
    vocabulary::Int32
    eos::Int32
    max_depth::Int32
    budget::Int32
end

function Grammar(encode,vocabulary::Integer,eos::Integer;max_depth=8,budget=32)
    vocabulary isa Bool && throw(ArgumentError("invalid vocabulary"))
    0<vocabulary<=typemax(Int32)||throw(ArgumentError("invalid vocabulary"))
    eos isa Bool && throw(ArgumentError("invalid EOS"))
    0<=eos<vocabulary||throw(ArgumentError("invalid EOS"))
    max_depth isa Integer && !(max_depth isa Bool) && 2<=max_depth<=64||throw(ArgumentError("invalid stack bound"))
    budget isa Integer && !(budget isa Bool) && 1<=budget<=1024||throw(ArgumentError("invalid operation budget"))
    sequences=Vector{Int32}[]
    for action in ACTIONS
        ids=encode(action*"\n")
        all(t->t isa Integer && !(t isa Bool) && 0<=t<vocabulary,ids)||throw(ArgumentError("invalid encoded action"))
        eos in ids&&throw(ArgumentError("action contains EOS"))
        push!(sequences,Int32.(ids))
    end
    push!(sequences,Int32[eos])
    # Verify codec collisions/prefix ambiguity across ALL actions, even actions
    # which cannot coexist at the same stack depth.
    TokenPrefixTrie.Trie(sequences,vocabulary)
    groups=(findall(==(0),ARITIES),findall(a->a<=1||a==3,ARITIES),
        findall(a->a<=2,ARITIES),findall(a->1<=a<=2,ARITIES),
        Int[FINISH],findall(==(2),ARITIES),Int[EOS_ACTION])
    offsets=Int32[];labels=Int32[];targets=Int32[];terminal=Int32[];roots=Int32[]
    for actions in groups
        localtrie=TokenPrefixTrie.Trie(sequences[actions],vocabulary)
        nodebase=length(terminal);edgebase=length(labels)
        push!(roots,Int32(nodebase+1))
        append!(offsets,localtrie.offsets[1:end-1].+Int32(edgebase))
        append!(labels,localtrie.labels)
        append!(targets,localtrie.targets.+Int32(nodebase))
        localterminal=zeros(Int32,length(localtrie.terminal))
        for action in actions
            node=Int32(1)
            for token in sequences[action]
                node=TokenPrefixTrie.edge_target(localtrie.offsets,localtrie.labels,localtrie.targets,node,token)
            end
            localterminal[node]=Int32(action)
        end
        append!(terminal,localterminal)
    end
    push!(offsets,Int32(length(labels)+1))
    Grammar(offsets,labels,targets,terminal,roots,vcat(ARITIES,Int32(4)),sequences,
        Int32(vocabulary),Int32(eos),Int32(max_depth),Int32(budget))
end

@inline function root_for(roots,depth::Int32,operations::Int32,max_depth::Int32,budget::Int32)
    0<=depth<=max_depth||return Int32(-1)
    if operations>=budget
        depth==0&&return Int32(-1)
        @inbounds return depth==1 ? roots[5] : roots[6]
    end
    @inbounds depth==0&&return roots[1]
    @inbounds depth==1&&return roots[2]
    @inbounds depth==max_depth&&return roots[4]
    @inbounds roots[3]
end

"""Shared action-boundary admission; phase 1 requests EOS, phase 2 is done."""
@inline function action_state(arity::Int32,depth::Int32,operations::Int32,max_depth::Int32,budget::Int32)
    if !(0<=arity<=4) || !(0<=depth<=max_depth) || !(0<=operations<=budget+max_depth-Int32(1))
        return (depth,operations,Int32(0),Int32(1))
    end
    if arity==4 || arity==3
        return (depth,operations,arity-Int32(2),Int32(depth!=1))
    end
    if depth<arity || (arity==0&&depth==max_depth) || (operations>=budget&&arity!=2)
        return (depth,operations,Int32(0),Int32(1))
    end
    (depth-arity+Int32(1),operations+Int32(1),Int32(0),Int32(0))
end

"""Consume ONE sampled token. Tuple is (node, depth, opcount, done, rejected).

Count exhaustion only prohibits new pushes/unaries; the model still chooses
binary reductions until depth one, then finish/EOS. At most budget+max_depth-1
non-finish actions can therefore be emitted by a valid path.
"""
@inline function transition(offsets,labels,targets,terminal,roots,arities,
        node::Int32,depth::Int32,operations::Int32,done::Int32,rejected::Int32,token::Int32,
        max_depth::Int32,budget::Int32)
    if rejected!=0 || !(0<=depth<=max_depth) || !(0<=operations<=budget+max_depth-Int32(1)) ||
            !(done==0||done==1) || (done==1 && (node!=0||depth!=1))
        return (Int32(-1),depth,operations,Int32(0),Int32(1))
    end
    done!=0&&return (Int32(0),depth,operations,done,Int32(0))
    target=TokenPrefixTrie.edge_target(offsets,labels,targets,node,token)
    target<1&&return (Int32(-1),depth,operations,Int32(0),Int32(1))
    @inbounds action=terminal[target]
    action==0&&return (target,depth,operations,Int32(0),Int32(0))
    @inbounds arity=arities[action]
    depth,operations,phase,invalid=action_state(arity,depth,operations,max_depth,budget)
    invalid!=0&&return (Int32(-1),depth,operations,Int32(0),Int32(1))
    if phase==2
        return (Int32(0),depth,operations,Int32(1),Int32(0))
    elseif phase==1
        @inbounds return (roots[7],depth,operations,Int32(0),Int32(0))
    end
    root=root_for(roots,depth,operations,max_depth,budget)
    (root,depth,operations,Int32(0),Int32(root<0))
end
transition(g::Grammar,state,token::Integer)=transition(g.offsets,g.labels,g.targets,g.terminal_action,
    g.roots,g.arities,state[1],state[2],state[3],state[4],state[5],Int32(token),g.max_depth,g.budget)
initial(g::Grammar)=(g.roots[1],Int32(0),Int32(0),Int32(0),Int32(0))
end
