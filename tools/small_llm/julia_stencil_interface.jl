isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
module JuliaStencilInterface
import ..RPNPointwiseInterface
using SHA

"""Translate a small explicit Float32 Julia stencil subset to trusted RPN IR.

No model code, macro or import is executed. Only one ordinary serial loop is
admitted. Input accesses must match five safe current/neighbor patterns exactly;
all execution uses newly generated canonical code, never the original source.
This is not unrestricted Julia compilation or a proof of task correctness.
Finite numeric values are compared ignoring signed-zero distinctions.
"""
function lower(code::AbstractString)
    reject(reason)=(;accepted=false,reason,code="",serial_code="",program="",certificate=nothing)
    ncodeunits(code)<=65536||return reject("source too large")
    statements(x)=x isa Expr&&x.head in (:toplevel,:block) ? filter(y->!(y isa LineNumberNode),x.args) : Any[x]
    tree=try Meta.parseall(code) catch;return reject("parse failure") end
    top=statements(tree)
    length(top)==1||return reject("one function only, no imports")
    definition=only(top)
    definition isa Expr&&definition.head==:function&&length(definition.args)==2||return reject("explicit kernel! function required")
    signature,body=definition.args
    signature in (:(kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)),
        :(kernel!(out::AbstractVector{Float32},x::AbstractVector{Float32},n::Int32)))||return reject("canonical typed out/x/n signature required")
    parent=filter(x->x!==:nothing&&!(x isa Expr&&x.head==:return&&x.args==Any[:nothing]),statements(body))
    length(parent)==1||return reject("one loop plus optional return nothing")
    loop=only(parent)
    loop isa Expr&&loop.head==:for&&length(loop.args)==2||return reject("serial for loop required")
    binding,loopbody=loop.args
    binding==:(i=1:n)||return reject("canonical loop i=1:n required")
    known=Dict{Symbol,Vector{String}}();program=String[];stores=0;failure=""
    reserved=Set((:out,:x,:n,:i,:nothing,:Base,:KI,:+,:-,:*,:/,:^,:min,:max,:clamp,:abs,:Float32))
    refs=[:(x[i]),:(x[mod1(i-1,n)]),:(x[mod1(i+1,n)]),:(x[max(1,i-1)]),:(x[min(n,i+1)])]
    pushes=("x0","x_prev_periodic","x_next_periodic","x_prev_clamped","x_next_clamped")
    function expression(x,depth=0)
        depth<=64||begin failure="expression nesting too deep";return String[] end
        if x isa Float32
            x==0f0&&return ["zero"]
            x==1f0&&return ["one"]
            x==2f0&&return ["two"]
            x==0.5f0&&return ["half"]
            failure="only Float32 literals0,1,2,half supported";return String[]
        elseif x isa Symbol&&haskey(known,x)
            return copy(known[x])
        end
        for i in eachindex(refs)
            x==refs[i]&&return [pushes[i]]
        end
        x isa Expr&&x.head==:call||begin failure="unknown expression or unsafe index";return String[] end
        op=first(x.args);args=x.args[2:end]
        parts=String[]
        if op==:-&&length(args)==1
            append!(parts,expression(args[1],depth+1));push!(parts,"neg")
        elseif op in (:+,:-,:*,:min,:max)&&length(args)==2
            append!(parts,expression(args[1],depth+1));append!(parts,expression(args[2],depth+1))
            push!(parts,op==:+ ? "add" : op==:- ? "sub" : op==:* ? "mul" : String(op))
        elseif op==:^&&length(args)==2&&args[2]===2
            append!(parts,expression(args[1],depth+1));push!(parts,"square")
        elseif op==:abs&&length(args)==1
            child=expression(args[1],depth+1);append!(parts,child);append!(parts,child);append!(parts,["neg","max"])
        elseif op==:clamp&&length(args)==3&&args[2] isa Float32&&args[3] isa Float32&&args[2]<=args[3]
            append!(parts,expression(args[1],depth+1));append!(parts,expression(args[2],depth+1));push!(parts,"max")
            append!(parts,expression(args[3],depth+1));push!(parts,"min")
        else
            failure="unsupported call/arity or non-Float32 computation";return String[]
        end
        length(parts)<=32||(failure="expanded expression exceeds32 operations")
        parts
    end
    for statement in statements(loopbody)
        statement===:nothing&&continue
        statement isa Expr&&statement.head==:(=)&&length(statement.args)==2||return reject("only scalar assignments and one output store")
        lhs,rhs=statement.args;value=expression(rhs)
        isempty(failure)||return reject(failure)
        if lhs isa Symbol&&!(lhs in reserved)
            known[lhs]=value
        elseif lhs==:(out[i])
            stores+=1;program=value
        else
            return reject("argument mutation, shadowing or noncanonical store")
        end
    end
    stores==1||return reject("exactly one output store")
    push!(program,"finish");text=join(program,"\n")*"\n"
    result=RPNPointwiseInterface.lower(text)
    result.accepted||return reject("trusted IR rejected: "*result.reason)
    (;result...,program=text,certificate=(;result.certificate...,
        interface="Explicit Float32 Julia single-input stencil translated to finite RPN",
        source_sha256=bytes2hex(sha256(code)),
        scope="Syntactic/pure-subset translation only, no task-ID matching, arithmetic repair or hidden-answer lookup; no general Julia or signed-zero/NaN equivalence claim"))
end
end
