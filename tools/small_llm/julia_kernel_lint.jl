module JuliaKernelLint

"""Advisory AST checks; never expand macros or execute candidate code.

These findings are not formal proof and must not become hard token masks.
Unqualified/custom macro bindings and arbitrary aliasing remain unresolved.
"""
function atomic_extrema_assignments(code::AbstractString)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    tree=Meta.parseall(code);rows=NamedTuple[]
    name(x)=x isa Symbol ? x : x isa Expr && x.head==:. && last(x.args) isa QuoteNode ? last(x.args).value : nothing
    function walk(expression,line)
        expression isa Expr || return
        if expression.head==:macrocall && name(first(expression.args))==Symbol("@atomic")
            location=findfirst(x->x isa LineNumberNode,expression.args)
            location===nothing || (line=expression.args[location].line)
            statement=last(expression.args)
            if statement isa Expr && statement.head==:(=) && length(statement.args)==2
                lhs,rhs=statement.args
                if lhs isa Expr && lhs.head==:ref && rhs isa Expr && rhs.head==:call &&
                        name(first(rhs.args)) in (:min,:max) && any(arg->arg==lhs,rhs.args[2:end])
                    push!(rows,(;line,kind="potential_non_rmw_atomic_extremum",severity="advisory",
                        expression=sprint(Base.show_unquoted,expression),
                        message="For Atomix, assignment-form @atomic stores a value computed using an ordinary self-load; it is not an atomic min/max RMW. Use the documented function-form atomic min/max when a contended reduction is intended. Confirm the macro binding; this AST check does not resolve custom macros or aliasing."))
                end
            end
        end
        for arg in expression.args
            arg isa LineNumberNode ? (line=arg.line) : walk(arg,line)
        end
    end
    walk(tree,0)
    rows
end
end
