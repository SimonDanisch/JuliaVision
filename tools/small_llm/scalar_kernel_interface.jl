isdefined(@__MODULE__,:CheckedTypedPointwiseLowering) || include("typed_pointwise_kernel_lowering.jl")

module ScalarKernelInterface
import ..CheckedTypedPointwiseLowering

# Public user interfaces only. No benchmark answers, assertions or formulas.
const INTERFACES=Dict(
    "shared_periodic_laplacian"=>(;signature="kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32)",limit="n"),
    "shared_indexed_interpolation"=>(;signature="kernel!(out::Vector{Float32},x::Vector{Float32},indices::Vector{Int32},t::Vector{Float32},n::Int32,m::Int32)",limit="n"),
    "shared_column_bias_clip"=>(;signature="kernel!(out::Vector{Float32},x::Vector{Float32},bias::Vector{Float32},rows::Int32,cols::Int32)",limit="rows*cols"),
    "shared_block_reverse"=>(;signature="kernel!(out::Vector{Float32},x::Vector{Float32},n::Int32,width::Int32)",limit="n"))

"""Wrap ONE scalar value expression; unchanged checker must certify the result.

This supplies the public signature/index/store/guard scaffolding only. It does
not recognize or fix an expression's arithmetic, indexing or task semantics.
"""
function lower(code::AbstractString,id::AbstractString)
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    haskey(INTERFACES,id)||return reject("unknown public interface")
    ncodeunits(code)<=65536||return reject("source too large")
    expressions=filter(x->!(x isa LineNumberNode),Meta.parseall(code).args)
    length(expressions)==1||return reject("exactly one scalar expression required")
    value=only(expressions)
    value isa Expr && value.head in (:function,:(=),:for,:while,:block,:return,:using,:import,:macrocall,:toplevel,:error,:incomplete) &&
        return reject("not a scalar value expression")
    interface=INTERFACES[id];signature=Meta.parse(interface.signature);limit=Meta.parse(interface.limit)
    definition=Expr(:function,signature,Expr(:block,
        Expr(:for,Expr(:(=),:i,Expr(:call,:(:),1,limit)),Expr(:block,Expr(:(=),Expr(:ref,:out,:i),value))),
        Expr(:return,:nothing)))
    source=sprint(Base.show_unquoted,definition)
    result=CheckedTypedPointwiseLowering.lower(source)
    result.accepted||return result
    (;result...,certificate=(;result.certificate...,interface=String(id),
        adapter="public signature, loop/index and unique output store only; model supplies the complete value expression, with no arithmetic repair"))
end
end
