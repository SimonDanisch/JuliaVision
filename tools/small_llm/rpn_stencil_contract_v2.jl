isdefined(@__MODULE__,:rpn_stencil_tasks)||include("rpn_stencil_tasks.jl")

"""Corrected NEW prompt, never substituted into an already frozen experiment.

Generic constant-only illustration, not an answer to any stencil. Reusing the
old tasks with this prompt is posthoc development, NOT independent transfer.
"""
function rpn_stencil_messages_v2(task)
    previous=rpn_stencil_messages(task)
    user=String(last(last(previous)))
    parts=split(user,"\nTask:\n";limit=2);length(parts)==2||error("public task delimiter changed")
    old="maximum32 non-finish operations."
    occursin(old,first(parts))||error("preserve old contract; unexpected text")
    clarified=replace(first(parts),old=>"a normal budget of32 non-finish operations. At that budget you may only use binary operations to reduce the existing stack to one value, then finish. Those final reductions can bring the total to39 non-finish operations. No new pushes or unary operations are allowed during that closure.")
    example="""
The stack starts EMPTY; x0 is not implicitly present and need not be pushed
unless the calculation needs it. The constants zero/one/two/half are pushes,
not multiplications. To multiply a value by half, push half THEN use mul.
Generic constant-only illustration (not the requested stencil):
two
one
sub
half
mul
finish
This computes (2-1)*0.5. Stack sizes are1,2,1,2,1 before finish.
Do not copy this constant calculation as your task answer.
"""
    [first(previous),"user"=>clarified*"\n"*example*"\nTask:\n"*last(parts)]
end
