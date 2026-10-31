isdefined(@__MODULE__,:RPNPrefixGrammar)||include("rpn_prefix_grammar.jl")
isdefined(@__MODULE__,:CheckedTypedPointwiseLowering)||include("typed_pointwise_kernel_lowering.jl")
module RPNPointwiseInterface
import ..RPNPrefixGrammar
import ..CheckedTypedPointwiseLowering

const LOOKUP=Dict(name=>Int32(i) for (i,name) in enumerate(RPNPrefixGrammar.ACTIONS))

"""Parse a finite postfix program, never arbitrary Julia or an inferred repair."""
function parse_program(text::AbstractString;budget=32,max_depth=8)
    reject(reason)=(;accepted=false,reason,actions=Int32[])
    ncodeunits(text)<=16384||return reject("program too large")
    budget isa Integer && !(budget isa Bool) && 1<=budget<=1024||return reject("invalid budget")
    max_depth isa Integer && !(max_depth isa Bool) && 2<=max_depth<=64||return reject("invalid stack bound")
    answer=first(split(String(text),r"<\|im_end\|>|<\|endoftext\|>";limit=2))
    lines=filter(!isempty,strip.(split(answer,'\n')))
    isempty(lines)&&return reject("empty program")
    length(lines)<=budget+max_depth||return reject("program exceeds finite bound")
    last(lines)=="finish"||return reject("missing final finish")
    actions=Int32[];depth=Int32(0);operations=Int32(0)
    for (position,line) in enumerate(lines)
        haskey(LOOKUP,line)||return reject("unknown action")
        action=LOOKUP[line]
        action==RPNPrefixGrammar.FINISH&&position!=length(lines)&&return reject("instructions after finish")
        arity=RPNPrefixGrammar.ARITIES[action]
        depth,operations,phase,rejected=RPNPrefixGrammar.action_state(arity,depth,operations,Int32(max_depth),Int32(budget))
        rejected==0||return reject("arity/stack/budget violation")
        push!(actions,action)
    end
    (;accepted=true,reason="finite pure postfix program",actions)
end

function lower(text::AbstractString;budget=32,max_depth=8)
    parsed=parse_program(text;budget,max_depth)
    reject(reason)=(;accepted=false,reason,code="",serial_code="",certificate=nothing)
    parsed.accepted||return reject(parsed.reason)
    stack=Symbol[];body=Any[]
    for (position,action) in enumerate(parsed.actions)
        action==RPNPrefixGrammar.FINISH&&break
        arity=RPNPrefixGrammar.ARITIES[action]
        expression=if arity==0
            if action==1;:(x[i])
            elseif action==2;:(x[mod1(i-1,n)])
            elseif action==3;:(x[mod1(i+1,n)])
            elseif action==4;:(x[max(1,i-1)])
            elseif action==5;:(x[min(n,i+1)])
            elseif action==6;0f0
            elseif action==7;1f0
            elseif action==8;2f0
            else;0.5f0
            end
        elseif arity==1
            value=pop!(stack)
            action==10 ? Expr(:call,:-,value) : Expr(:call,:*,value,value)
        else
            right=pop!(stack);left=pop!(stack)
            operator=action==12 ? :+ : action==13 ? :- : action==14 ? :* : action==15 ? :min : :max
            Expr(:call,operator,left,right)
        end
        variable=Symbol("rpn_value_",position)
        push!(body,Expr(:(=),variable,expression));push!(stack,variable)
    end
    length(stack)==1||return reject("internal stack invariant")
    push!(body,Expr(:(=),Expr(:ref,:out,:i),only(stack)))
    signature=Meta.parse("kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)")
    definition=Expr(:function,signature,Expr(:block,
        Expr(:for,Expr(:(=),:i,Expr(:call,:(:),1,:n)),Expr(:block,body...)),Expr(:return,:nothing)))
    serial=sprint(Base.show_unquoted,definition)
    lowered=CheckedTypedPointwiseLowering.lower(serial)
    lowered.accepted||return reject("checker rejected generated program: "*lowered.reason)
    (;lowered...,serial_code=serial,certificate=(;lowered.certificate...,
        interface="Pure finite postfix program over ONE Float32 input vector",
        operations=length(parsed.actions)-1,
        additional_preconditions=["ordinary one-based arrays","0<=n<=length(x) and n<=length(out)","distinct input/output storage"],
        arithmetic_policy="Exactly the model-selected actions in order; no task matching, arithmetic repair, hidden-answer lookup, or unconstrained Julia evaluation"))
end
end
