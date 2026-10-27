isdefined(@__MODULE__,:QualifiedBindingDiagnostics)||include("qualified_binding_diagnostics.jl")
module QualifiedBindingDiagnosticsV2
import ..QualifiedBindingDiagnostics

"""Conservative extra scope handling around the frozen literal-name checker.

No candidate code, imports, macros or bindings are executed. Short-form method
arguments, let locals, explicit import aliases and type parameters can shadow a
module name. Such modules are omitted globally rather than incorrectly judged
against an unrelated installed namespace. Not a complete Julia scope/type model.
"""
function analyze(code::AbstractString,bindings)
    ncodeunits(code)<=262144||throw(ArgumentError("bounded source"))
    inventory=Dict(Symbol(String(k))=>Set(String.(v)) for (k,v) in pairs(bindings))
    shadows=Set{Symbol}();contexts=Set{String}();parse_ok=true
    function bind(x,depth=0)
        depth<=64||throw(ArgumentError("bounded binding depth"))
        if x isa Symbol
            haskey(inventory,x)&&push!(shadows,x)
        elseif x isa Expr
            if x.head in (:tuple,:parameters,:block)
                foreach(y->bind(y,depth+1),x.args)
            elseif x.head in (:(::),:kw,:(...),:(=),:(<:),:(>:))&&!isempty(x.args)
                bind(first(x.args),depth+1)
            end
        end
    end
    function signature(x,depth=0)
        depth<=64||throw(ArgumentError("bounded signature depth"))
        x isa Expr||return
        if x.head==:where
            foreach(bind,x.args[2:end]);signature(first(x.args),depth+1)
        elseif x.head==:(::)
            signature(first(x.args),depth+1) # return type is a reference, not a binding
        elseif x.head==:call
            bind(first(x.args));foreach(bind,x.args[2:end])
        end
    end
    function imported(x)
        x isa Expr||return
        if x.head==:as
            bind(last(x.args));push!(contexts,"explicit_import_alias")
        elseif x.head==:(:)
            for name in x.args[2:end]
                if name isa Expr&&name.head==Symbol(".")
                    foreach(bind,name.args)
                else
                    bind(name)
                end
            end
        elseif x.head==Symbol(".")&&length(x.args)>1
            bind(last(x.args)) # Other.JSON3 is not necessarily installed JSON3
        end
    end
    tree=try Meta.parseall(code) catch;parse_ok=false;nothing end
    stack=Any[(tree,0)];nodes=0
    while !isempty(stack)
        node,depth=pop!(stack);node isa Expr||continue
        nodes+=1
        nodes<=100000&&depth<=128||throw(ArgumentError("bounded AST size/depth"))
        node.head in (:quote,:inert)&&continue
        if node.head in (:error,:incomplete)
            parse_ok=false;push!(contexts,"parse_error");continue
        elseif node.head==:(=)&&!isempty(node.args)
            signature(first(node.args))
        elseif node.head==:function&&!isempty(node.args)
            signature(first(node.args))
        elseif node.head==:let&&!isempty(node.args)
            bind(first(node.args));push!(contexts,"let_scope")
        elseif node.head in (:import,:using)
            foreach(imported,node.args)
        elseif node.head==:struct&&length(node.args)>=2
            bind(node.args[2])
        elseif node.head in (:abstract,:primitive)&&!isempty(node.args)
            bind(first(node.args))
        end
        append!(stack,[(child,depth+1) for child in node.args if child isa Expr])
    end
    if !parse_ok
        return (;references_to_unlisted_bindings=String[],conservatively_skipped_modules=sort!(string.(collect(keys(inventory)))),
            source_parse_ok=false,extra_scope_contexts=sort!(collect(contexts)),
            scope="Unanalysable parse; no clean namespace/type/runtime conclusion")
    end
    reduced=Dict(k=>v for (k,v) in inventory if k∉shadows)
    original=QualifiedBindingDiagnostics.analyze(code,reduced)
    skipped=sort!(unique(vcat(original.conservatively_skipped_modules,string.(collect(shadows)))))
    (;references_to_unlisted_bindings=original.references_to_unlisted_bindings,
        conservatively_skipped_modules=skipped,source_parse_ok=true,extra_scope_contexts=sort!(collect(contexts)),
        scope="Literal installed binding comparison with conservative extra scope exclusions; no candidate execution, macro expansion, exact scope, call/type/algorithm/reachability guarantee")
end
end
