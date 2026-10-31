isdefined(@__MODULE__,:ScalarKernelInterface) || include("scalar_kernel_interface.jl")

module ScalarProgramInterface
import ..ScalarKernelInterface, ..CheckedTypedPointwiseLowering

"""Separate posthoc scalar-program adapter: pure local assignments + a value.

The SAME frozen checker admits every assignment and expression; no arbitrary
evaluation, repair, task recognition or extra primitive is added. The public
signature/index/store wrapper is the same as the strict expression adapter.
"""
function lower(code::AbstractString,id::AbstractString)
    strict=ScalarKernelInterface.lower(code,id)
    strict.accepted&&return strict
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    haskey(ScalarKernelInterface.INTERFACES,id)||return reject("unknown public interface")
    ncodeunits(code)<=65536||return reject("source too large")
    nodes=filter(x->!(x isa LineNumberNode),Meta.parseall(code).args)
    while length(nodes)==1&&only(nodes) isa Expr&&only(nodes).head in (:block,:toplevel)
        nodes=filter(x->!(x isa LineNumberNode),only(nodes).args)
    end
    isempty(nodes)&&return reject("missing scalar program")
    value=last(nodes)
    value isa Expr&&value.head==:return&&length(value.args)==1&&(value=only(value.args))
    value isa Expr&&value.head in (:function,:(=),:for,:while,:block,:using,:import,:macrocall,:error,:incomplete)&&return reject("last statement is not a value")
    interface=ScalarKernelInterface.INTERFACES[id];signature=Meta.parse(interface.signature);limit=Meta.parse(interface.limit)
    loopbody=Expr(:block,nodes[1:end-1]...,Expr(:(=),Expr(:ref,:out,:i),value))
    definition=Expr(:function,signature,Expr(:block,Expr(:for,Expr(:(=),:i,Expr(:call,:(:),1,limit)),loopbody),Expr(:return,:nothing)))
    result=CheckedTypedPointwiseLowering.lower(sprint(Base.show_unquoted,definition))
    result.accepted||return result
    (;result...,certificate=(;result.certificate...,interface=String(id),
        adapter="POSTHOC pure scalar locals plus final value, checked by unchanged typed pointwise lowerer; no arithmetic repair"))
end
end
