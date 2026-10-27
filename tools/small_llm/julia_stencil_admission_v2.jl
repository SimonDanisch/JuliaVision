isdefined(@__MODULE__,:JuliaStencilInterface)||include("julia_stencil_interface.jl")
module JuliaStencilAdmissionV2
import ..JuliaStencilInterface

"""Conservative admission refinement; numeric CPU/GPU lowering stays identical.

V1 omitted mod1 from its protected helper names. A scalar binding named mod1
could therefore be translated although ordinary Julia neighbor calls would
call a Float32, not Base.mod1. Do not silently retrofit frozen V1 experiments:
report this admission audit separately and use V2 in future registrations.
"""
function lower(source::AbstractString)
    original=JuliaStencilInterface.lower(source)
    original.accepted||return original
    tree=Meta.parseall(source)
    function shadows_mod1(x)
        x isa Expr||return false
        x.head==:(=)&&first(x.args)===:mod1&&return true
        any(shadows_mod1,x.args)
    end
    shadows_mod1(tree)||return original
    (;accepted=false,reason="protected neighbor helper mod1 shadowed",code="",serial_code="",program="",certificate=nothing)
end
end
