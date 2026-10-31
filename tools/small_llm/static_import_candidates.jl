module StaticImportCandidates

"""Suggest public namespaces for literal unqualified CALL names.

This is deliberately not an UndefVarError predictor: no code/import/macro is
executed, and inventory membership says nothing about method applicability.
Definitions anywhere suppress a name globally (safe over-approximation, not
Julia scope resolution). Unknown imports or opaque metaprogramming suppress all
suggestions. Frozen experiments do not use this development helper.
"""
function analyze(source::AbstractString, corpus)
    ncodeunits(source)<=262144||throw(ArgumentError("bounded source"))
    exports=Dict{String,Set{String}}()
    for row in corpus.rows
        mod,name=split(String(row.qualified),'.';limit=2)
        push!(get!(exports,mod,Set{String}()),name)
    end
    known=Set(String.(corpus.modules));available=Set{String}()
    # Effective Base namespace includes ordinary Base/Core call names, not just
    # doc-captured functions. Do not confuse inherited names with exports of M.
    union!(available,String.(corpus.bindings.Base))
    definitions=Set{String}();calls=Set{String}();unknown=Set{String}()
    opaque=false;parse_ok=true
    function bind(x,depth=0)
        depth<=64||throw(ArgumentError("bounded binding"))
        if x isa Symbol
            push!(definitions,String(x))
        elseif x isa Expr
            if x.head in (:tuple,:parameters,:block)
                foreach(y->bind(y,depth+1),x.args)
            elseif x.head in (:(::),:kw,:(...),:(=),:(<:),:(>:),:curly)&&!isempty(x.args)
                bind(first(x.args),depth+1)
            end
        end
    end
    function signature(x,depth=0)
        depth<=64||throw(ArgumentError("bounded signature"))
        x isa Expr||return
        if x.head==:where
            foreach(bind,x.args[2:end]);signature(first(x.args),depth+1)
        elseif x.head==:(::)
            signature(first(x.args),depth+1)
        elseif x.head==:call
            bind(first(x.args));foreach(bind,x.args[2:end])
        end
    end
    path(x)=x isa Symbol ? [String(x)] : x isa QuoteNode&&x.value isa Symbol ? [String(x.value)] :
        x isa Expr&&x.head==Symbol(".") ? reduce(vcat,(path(y) for y in x.args);init=String[]) : String[]
    function imported(x,isusing)
        if x isa Expr&&x.head==:(:)
            parts=path(first(x.args))
            (length(parts)==1&&only(parts) in known)||push!(unknown,isempty(parts) ? "unresolved_import" : join(parts,'.'))
            for selected in x.args[2:end]
                if selected isa Expr&&selected.head==:as
                    bind(last(selected.args))
                else
                    foreach(name->push!(definitions,name),path(selected))
                end
            end
        elseif x isa Expr&&x.head==:as
            bind(last(x.args))
            parts=path(first(x.args))
            (length(parts)==1&&only(parts) in known)||push!(unknown,isempty(parts) ? "unresolved_import" : join(parts,'.'))
        else
            parts=path(x)
            if length(parts)!=1||only(parts)∉known
                push!(unknown,isempty(parts) ? "unresolved_import" : join(parts,'.'))
            else
                mod=only(parts);push!(definitions,mod)
                isusing&&union!(available,get(exports,mod,Set{String}()))
            end
        end
    end
    tree=try Meta.parseall(source) catch;parse_ok=false;nothing end
    stack=Any[(tree,0)];nodes=0
    while !isempty(stack)
        node,depth=pop!(stack);node isa Expr||continue
        nodes+=1;nodes<=100000&&depth<=128||throw(ArgumentError("bounded AST"))
        if node.head in (:error,:incomplete)
            parse_ok=false;continue
        elseif node.head in (:quote,:inert)
            continue
        elseif node.head in (:macrocall,:module,:baremodule)
            opaque=true;continue
        elseif node.head in (:using,:import)
            foreach(x->imported(x,node.head==:using),node.args);continue
        elseif node.head==:function&&!isempty(node.args)
            signature(first(node.args))
        elseif node.head==:(=)&&!isempty(node.args)
            lhs=first(node.args)
            if lhs isa Expr&&lhs.head in (:call,:where)
                signature(lhs)
            elseif lhs isa Expr&&lhs.head==:(::)&&first(lhs.args) isa Expr
                signature(lhs)
            else
                bind(lhs)
            end
        elseif node.head in (:local,:global)
            foreach(bind,node.args)
        elseif node.head==:try&&length(node.args)>=2&&node.args[2] isa Symbol
            bind(node.args[2])
        elseif node.head in (:let,:for,:generator)&&!isempty(node.args)
            bind(first(node.args))
        elseif node.head==:struct&&length(node.args)>=2
            bind(node.args[2])
        elseif node.head in (:abstract,:primitive)&&!isempty(node.args)
            bind(first(node.args))
        elseif node.head==:(->)&&!isempty(node.args)
            bind(first(node.args))
        end
        if node.head==:call&&!isempty(node.args)&&first(node.args) isa Symbol
            name=String(first(node.args));push!(calls,name)
            name in ("eval","include","include_string")&&(opaque=true)
        elseif node.head==:call&&!isempty(node.args)
            parts=path(first(node.args))
            !isempty(parts)&&last(parts) in ("eval","include","include_string")&&(opaque=true)
        end
        append!(stack,[(x,depth+1) for x in node.args if x isa Expr])
    end
    suggestions=Any[]
    if parse_ok&&!opaque&&isempty(unknown)
        for name in sort!(collect(setdiff(calls,union(available,definitions))))
            candidates=sort!([mod for (mod,names) in exports if name in names&&mod!="Base"])
            isempty(candidates)||push!(suggestions,(;name,possible_modules=candidates))
        end
    end
    (;suggestions,source_parse_ok=parse_ok,opaque_metaprogramming=opaque,
        unknown_imports=sort!(collect(unknown)),
        scope="Development source-only import candidates for literal calls. Not missing-binding proof, import recommendation, runtime/type/algorithm validation or quality evidence.")
end
end
