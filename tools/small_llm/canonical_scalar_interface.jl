isdefined(@__MODULE__,:ScalarProgramInterface) || include("scalar_program_interface.jl")

module CanonicalScalarInterface
import ..ScalarProgramInterface

"""Recognize ONLY the exact one-dimensional global-x index builtin.

Under the public wrapper's contract it equals i. Preserve casts/arithmetic;
reject local rebinding of the builtin or its known module aliases. No local/y/z
indices, arguments, arbitrary calls or guessed names are canonicalized.
"""
function lower(code::AbstractString,id::AbstractString)
    reject(reason)=(;accepted=false,reason,code="",certificate=nothing)
    ncodeunits(code)<=65536||return reject("source too large")
    tree=Meta.parseall(code);reserved=Set((:get_global_id,:KI,:KernelInterface))
    function shadows(x)
        x isa Expr||return false
        x.head==:(=)&&first(x.args) isa Symbol&&first(x.args) in reserved&&return true
        any(shadows,x.args)
    end
    shadows(tree)&&return reject("shadowed global-index builtin/module")
    substitutions=Ref(0)
    function normalize(x)
        x isa Expr||return x
        if x.head==:. && length(x.args)==2 && last(x.args)==QuoteNode(:x)
            call=first(x.args)
            if call isa Expr&&call.head==:call&&length(call.args)==1
                f=only(call.args)
                qualified=f isa Expr && f.head == :. && length(f.args)==2 && first(f.args) in (:KI,:KernelInterface) && last(f.args)==QuoteNode(:get_global_id)
                if f===:get_global_id||qualified
                    substitutions[]+=1;return :i
                end
            end
        end
        Expr(x.head,map(normalize,x.args)...)
    end
    function statements(x)
        x isa LineNumberNode && return Any[]
        x isa Expr && x.head==:toplevel && return reduce(vcat,map(statements,x.args);init=Any[])
        Any[x]
    end
    normalized=join((sprint(Base.show_unquoted,x) for x in statements(normalize(tree))),"\n")
    result=ScalarProgramInterface.lower(normalized,id)
    result.accepted||return result
    (;result...,certificate=(;result.certificate...,global_x_alias_substitutions=substitutions[],
        index_alias_proof="Wrapper i is exactly KI.get_global_id().x for this one-dimensional guarded invocation. Preserve Int32 casts; no arithmetic or unknown API repair."))
end
end
