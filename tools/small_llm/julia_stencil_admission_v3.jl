isdefined(@__MODULE__,:JuliaStencilAdmissionV2)||include("julia_stencil_admission_v2.jl")
module JuliaStencilAdmissionV3
import ..JuliaStencilAdmissionV2

"""Fail-closed source admission refinement, separate from frozen V1 studies.

Require an optional return nothing to be AFTER the loop, never before it.
Also reject all-underscore temporary bindings: Julia prohibits reading those
bindings whereas a naïve AST substitution could mistakenly make them usable.
Only admission changes; admitted programs retain identical shared numeric IR.
"""
function lower(source::AbstractString)
    original=JuliaStencilAdmissionV2.lower(source)
    original.accepted||return original
    reject(reason)=(;accepted=false,reason,code="",serial_code="",program="",certificate=nothing)
    statements(x)=x isa Expr&&x.head in (:toplevel,:block) ?
        filter(y->!(y isa LineNumberNode)&&y!==:nothing,x.args) : Any[x]
    definition=only(statements(Meta.parseall(source)))
    body=statements(definition.args[2])
    (length(body) in (1,2)&&first(body) isa Expr&&first(body).head==:for)||return reject("loop must precede optional return nothing")
    length(body)==1||(last(body) isa Expr&&last(body).head==:return&&last(body).args==Any[:nothing])||return reject("only trailing return nothing")
    function underscore_binding(x)
        x isa Expr||return false
        if x.head==:(=)&&first(x.args) isa Symbol
            name=String(first(x.args))
            !isempty(name)&&all(==('_'),name)&&return true
        end
        any(underscore_binding,x.args)
    end
    underscore_binding(definition)&&return reject("all-underscore temporary binding unsupported")
    original
end
end
