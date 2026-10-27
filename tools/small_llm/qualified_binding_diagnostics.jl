isdefined(@__MODULE__,:PublicProgramDiagnostics)||include("public_program_diagnostics.jl")
module QualifiedBindingDiagnostics
using JSON3,SHA

"""Inspect literal module.binding references against a trusted name inventory.
No binding is called and no generated code is evaluated. Shadowed module names
are conservatively omitted. This is NOT a type checker or execution proof.
"""
function analyze(code::AbstractString,bindings)
    ncodeunits(code)<=262144||throw(ArgumentError("bounded source"))
    inventory=Dict(Symbol(String(k))=>Set(String.(v)) for (k,v) in pairs(bindings))
    tree=Meta.parseall(code);nodes=Any[];stack=Any[tree];shadows=Set{Symbol}()
    function shadow_lhs(x)
        x isa Symbol&&haskey(inventory,x)&&push!(shadows,x)
        if x isa Expr
            if x.head in (:tuple,:parameters)
                foreach(shadow_lhs,x.args)
            elseif x.head in (Symbol("::"),:kw,Symbol("..."))&&!isempty(x.args)
                shadow_lhs(first(x.args))
            end
        end
        nothing
    end
    while !isempty(stack)
        node=pop!(stack);node isa Expr||continue
        length(nodes)<100000||throw(ArgumentError("bounded AST"))
        node.head in (:quote,:inert)&&continue
        push!(nodes,node)
        node.head===Symbol("=")&&!isempty(node.args)&&shadow_lhs(first(node.args))
        if node.head in (:local,:global)
            foreach(shadow_lhs,node.args)
        elseif node.head===:module&&length(node.args)>=2
            shadow_lhs(node.args[2])
        elseif node.head===Symbol("->")
            shadow_lhs(first(node.args))
        elseif node.head===:function&&!isempty(node.args)
            signature=first(node.args)
            signature isa Expr&&signature.head===:call&&foreach(shadow_lhs,signature.args[2:end])
        end
        append!(stack,node.args)
    end
    missing=Set{String}()
    for node in nodes
        if node.head===Symbol(".")&&length(node.args)==2
            mod,field=node.args
            if mod isa Symbol&&haskey(inventory,mod)&&mod∉shadows&&field isa QuoteNode&&field.value isa Symbol
                name=String(field.value)
                name in inventory[mod]||push!(missing,string(mod,".",name))
            end
        end
    end
    (;references_to_unlisted_bindings=sort!(collect(missing)),conservatively_skipped_modules=sort!(string.(collect(shadows))),
        scope="Literal installed binding-name comparison only; no call/method/semantic/reachability guarantee, no candidate execution or benchmark tests")
end

function render(report)
    isempty(report.references_to_unlisted_bindings)&&return "No unlisted literal module.binding reference detected; this is NOT a runtime/type proof."
    "These literal module.binding names are absent from the installed namespace inventory: "*
        join(report.references_to_unlisted_bindings,", ")*". Check the real installed API; no replacement function or algorithm is inferred by this tool."
end
end
