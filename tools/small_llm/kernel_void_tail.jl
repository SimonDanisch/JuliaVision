module KernelVoidTail

"""Append `nothing` to a kernel entry, discarding its implicit return value.

GPU entry return values have no caller consuming them. Existing statements,
side effects, math and explicit returns are untouched. Explicit valued returns
or macro-generated returns can still fail the backend's void-return check.
This is not a bounds/race-safety certificate; candidates remain sandbox-only.
"""
function normalize(code::AbstractString)
    reject(reason)=(;accepted=false,reason,code=String(code))
    ncodeunits(code)<=65536 || return reject("source too large")
    tree=Meta.parseall(code)
    top=filter(x->!(x isa LineNumberNode),tree.args)
    positions=Int[]
    for (i,definition) in enumerate(top)
        definition isa Expr && definition.head in (:function,:(=)) && length(definition.args)==2 || continue
        signature=first(definition.args)
        signature isa Expr && signature.head==:call && first(signature.args)==:kernel! && push!(positions,i)
    end
    length(positions)==1 || return reject("one top-level kernel! entry required")
    position=only(positions);definition=top[position];signature,body=definition.args
    shadowed=false
    function binding_name(x)
        x isa Symbol && return x
        x isa Expr && x.head in (:(::),:kw,:(...)) && return binding_name(first(x.args))
        nothing
    end
    any(arg->binding_name(arg)===:nothing,signature.args[2:end]) && return reject("shadowed nothing binding")
    function scan(x)
        x isa Expr || return
        if x.head in (:(=),:local,:global) && !isempty(x.args) && binding_name(first(x.args))===:nothing
            shadowed=true
        end
        foreach(scan,x.args)
    end
    scan(body);shadowed && return reject("shadowed nothing binding")
    statements=body isa Expr && body.head==:block ? body.args : Any[body]
    top[position]=Expr(definition.head,signature,Expr(:block,statements...,:nothing))
    output=join((sprint(Base.show_unquoted,statement) for statement in top),"\n")*"\n"
    (;accepted=true,reason="append void tail; preserve existing math/statements/explicit returns",code=output)
end
end
