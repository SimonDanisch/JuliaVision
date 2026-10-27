"""Conservative AST observations, not a Julia binding/type-checking proof.

Keep separate from frozen functional scores. No candidate code is evaluated.
"""
function api_surface_audit(task_id,code)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    tree=Meta.parseall(code);qualified_launches=String[];parse_problem=false
    function visit(node)
        node isa Expr || return
        node.head in (:error,:incomplete) && (parse_problem=true)
        if node.head==:call && !isempty(node.args)
            callee=first(node.args)
            if callee isa Expr && callee.head==:. && length(callee.args)==2 &&
                    callee.args[1] in (:KernelInterface,:KI) && callee.args[2]==QuoteNode(:launch)
                push!(qualified_launches,sprint(Base.show_unquoted,node))
            end
        end
        foreach(visit,node.args)
    end
    visit(tree)
    (;task=task_id,parse_problem,qualified_lowlevel_launch_calls=qualified_launches,
        requires_review=task_id=="ki_launch"&&!isempty(qualified_launches),
        scope="post-hoc public-spec audit; does NOT change frozen functional scores",
        limitation="Aliases, shadowed names, dead code and indirect calls require manual/type-aware review")
end
