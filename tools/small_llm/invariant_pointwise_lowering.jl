isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")

module InvariantPointwiseLowering
import ..CheckedTypedPointwiseLowering
using SHA

"""Separate posthoc admission expansion for immutable integer scalar aliases.

Only total Base integer +,-,*,min,max expressions over Int/Int32 arguments,
integer literals and earlier aliases are inlined. No array reads, arbitrary
calls, mutation, alias shadowing or floating bounds. The frozen compiler then
checks the entire normalized program. This never supplies task arithmetic.
"""
function lower(code::AbstractString)
    original=CheckedTypedPointwiseLowering.lower(code)
    original.accepted && return original
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    ncodeunits(code)<=65536 || return reject("source too large")
    statements(x)=x isa Expr && x.head in (:toplevel,:block) ? filter(y->!(y isa LineNumberNode),x.args) : Any[x]
    tree=Meta.parseall(code);top=statements(tree)
    definitions=filter(x->!(x isa Expr && x.head in (:import,:using)),top)
    length(definitions)==1 || return original
    definition=only(definitions)
    definition isa Expr && definition.head==:function && length(definition.args)==2 || return original
    signature,body=definition.args
    signature isa Expr && signature.head==:call && first(signature.args)==:kernel! || return original
    known=Set{Symbol}();arguments=Set{Symbol}()
    for arg in signature.args[2:end]
        arg isa Expr && arg.head==:(::) && first(arg.args) isa Symbol || return original
        name,type=arg.args;push!(arguments,name)
        type in (:Int,:Int32) && push!(known,name)
    end
    parts=statements(body);position=findfirst(x->x isa Expr && x.head==:for,parts)
    position===nothing && return original
    position>1 || return original
    aliases=Dict{Symbol,Any}()
    reserved=Set((:KI,:Base,:nothing,:+,:-,:*,:/,:div,:mod,:mod1,:min,:max,:clamp,:muladd,:Float32,:Int32,:Int))
    function substitute(x)
        x isa Symbol && return get(aliases,x,x)
        x isa Expr && return Expr(x.head,map(substitute,x.args)...)
        x
    end
    function integer_pure(x)
        x isa Symbol && return x in known
        x isa Integer && !(x isa Bool) && return true
        x isa Expr && x.head==:call && length(x.args)>=2 || return false
        op=first(x.args);arity=length(x.args)-1
        ((op in (:+,:*,:min,:max) && arity>=1) || (op==:- && arity in (1,2))) && all(integer_pure,x.args[2:end])
    end
    for assignment in parts[1:position-1]
        assignment isa Expr && assignment.head==:(=) && length(assignment.args)==2 || return reject("only scalar invariant aliases before loop")
        name,rhs=assignment.args
        name isa Symbol && !(name in union(arguments,reserved,Set(keys(aliases)))) || return reject("alias shadows an existing binding")
        integer_pure(rhs) || return reject("invariant must use total integer scalar arithmetic")
        aliases[name]=substitute(rhs);push!(known,name)
    end
    function shadows_alias(x)
        x isa Expr || return false
        x.head==:(=) && first(x.args) in keys(aliases) && return true
        any(shadows_alias,x.args)
    end
    any(shadows_alias,parts[position:end]) && return reject("alias mutated or shadowed inside loop")
    normalized=Expr(:function,signature,Expr(:block,map(substitute,parts[position:end])...))
    imports=filter(x->x isa Expr && x.head in (:import,:using),top)
    normalized_source=join(sprint.(Base.show_unquoted,[imports...;normalized]),"\n")
    result=CheckedTypedPointwiseLowering.lower(normalized_source)
    result.accepted || return result
    (;result...,certificate=(;result.certificate...,invariant_aliases=sort!(String.(collect(keys(aliases)))),
        original_source_sha256=bytes2hex(sha256(code)),
        expansion="pure integer aliases over immutable scalar arguments substituted before the unchanged admission checker; no task recognition or arithmetic repair"))
end
end
