module KernelBoundsAdvisory

function global_index_calls(code::AbstractString)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    count=0
    function walk(x)
        x isa Expr || return
        if x.head==:call
            f=first(x.args)
            name=f isa Symbol ? f : f isa Expr && f.head==:. && last(f.args) isa QuoteNode ? last(f.args).value : nothing
            name==:get_global_id && (count+=1)
        end
        foreach(walk,x.args)
    end
    walk(Meta.parseall(code));count
end

"""Narrow syntax advisory for direct global-index stores lacking a local guard.

Not a bounds proof: aliases, custom bindings/macros, early-return flow, computed
indices and arbitrary helper functions are unresolved. Do not use as a hard
token constraint, safety certificate, or frozen benchmark score/repair trigger.
"""
function direct_global_stores(code::AbstractString)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    tree=Meta.parseall(code);indices=Set{Symbol}();rows=NamedTuple[]
    name(x)=x isa Symbol ? x : x isa Expr && x.head==:. && last(x.args) isa QuoteNode ? last(x.args).value : nothing
    function global_index(x)
        x isa Expr || return false
        x.head==:call && name(first(x.args))==:get_global_id && return true
        any(global_index,x.args)
    end
    function collect_indices(x)
        x isa Expr || return
        if x.head==:(=) && length(x.args)==2 && first(x.args) isa Symbol && global_index(last(x.args))
            push!(indices,first(x.args))
        end
        foreach(collect_indices,x.args)
    end
    collect_indices(tree)
    function upper_guard(x,index)
        x isa Expr || return false
        x.head==:&& && return any(y->upper_guard(y,index),x.args)
        x.head==:|| && return all(y->upper_guard(y,index),x.args)
        x.head==:comparison && return any(j->upper_guard(Expr(:call,x.args[j+1],x.args[j],x.args[j+2]),index),1:2:length(x.args)-2)
        x.head==:call || return false
        length(x.args)==3 || return false
        op,a,b=x.args
        (op in (:<,:<=) && a==index) || (op in (:>,:>=) && b==index)
    end
    function walk(x,conditions,line)
        x isa Expr || return
        if x.head==:if
            walk(x.args[1],conditions,line)
            walk(x.args[2],(conditions...,x.args[1]),line)
            length(x.args)>2 && walk(x.args[3],conditions,line)
            return
        elseif x.head==:&&
            walk(x.args[1],conditions,line)
            walk(x.args[2],(conditions...,x.args[1]),line)
            return
        elseif x.head==:(=) && length(x.args)==2
            lhs=first(x.args)
            if lhs isa Expr && lhs.head==:ref && length(lhs.args)==2 &&
                    first(lhs.args) in (:out,:output) && last(lhs.args) in indices
                index=last(lhs.args)
                if !any(condition->upper_guard(condition,index),conditions)
                    push!(rows,(;line,kind="unrecognized_global_store_upper_guard",severity="advisory",
                        expression=sprint(Base.show_unquoted,x),
                        message="Direct global-index output store has no recognized enclosing upper-bound guard. A rounded-up dispatch may overwrite padding or access beyond allocation. This syntax-only warning may miss other guards or unsafe computed indices."))
                end
            end
        end
        for child in x.args
            child isa LineNumberNode ? (line=child.line) : walk(child,conditions,line)
        end
    end
    walk(tree,(),0);rows
end
end
