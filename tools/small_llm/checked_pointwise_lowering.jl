isdefined(@__MODULE__,:PointwiseKernelLowering) || include("pointwise_kernel_lowering.jl")
module CheckedPointwiseLowering
import ..PointwiseKernelLowering

"""Narrower admission guard for the unchanged, previously exercised lowering.

V1's expression whitelist also admits floating loop limits. Such a range can
have a different induction type, so a native-Int global index is not a general
semantics-preserving replacement. Reject those limits and shadowed API/math
bindings; do not retroactively edit frozen V1 artifacts.
"""
function lower(code::AbstractString)
    result=PointwiseKernelLowering.lower(code)
    result.accepted || return result
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    nodes=filter(x->!(x isa LineNumberNode) && !(x isa Expr && x.head in (:using,:import)),Meta.parseall(code).args)
    signature,body=only(nodes).args
    scalar_names=Set{Symbol}();bindings=Set{Symbol}()
    for arg in signature.args[2:end]
        name,type=arg.args;push!(bindings,name)
        type in (:Int,:Int32) && push!(scalar_names,name)
    end
    function collect_bindings(x)
        x isa Expr || return
        if x.head==:(=) && first(x.args) isa Symbol
            push!(bindings,first(x.args))
        end
        foreach(collect_bindings,x.args)
    end
    collect_bindings(body)
    reserved=Set((:KI,:Base,:nothing,:+,:-,:*,:/,:div,:mod,:mod1,:min,:max,:muladd,:Float32,:Int32,:Int))
    isempty(intersect(bindings,reserved)) || return reject("shadowed runtime or arithmetic binding")
    integer_ops=Set((:+,:-,:*,:div,:mod,:mod1,:min,:max,:Int32,:Int))
    function integer_limit(x)
        x isa Symbol && return x in scalar_names
        x isa Int && return true
        x isa Int32 && return true
        x isa Expr && x.head==:call && first(x.args) in integer_ops && all(integer_limit,x.args[2:end])
    end
    integer_limit(Meta.parse(result.certificate.limit)) || return reject("loop bound is not established Int/Int32 arithmetic")
    (;result...,certificate=(;result.certificate...,
        additional_admission="integer-only loop bound, native-Int induction for 1:N, no shadowed runtime/math bindings"))
end
end
