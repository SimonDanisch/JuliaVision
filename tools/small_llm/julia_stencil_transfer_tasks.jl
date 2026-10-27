using JSON3

"""New authored development transfer tasks, not an external Julia benchmark.
Public examples are deliberately nonmonotonic. No hidden tests/reference code.
"""
function julia_stencil_transfer_tasks()
    x=Float32[2,-1,0.5,-2]
    rows=[
        ("absolute_current","Return the absolute value of the current element.",Float32[2,1,0.5,2],0.75f0),
        ("positive_neighbor_drop","Subtract the current element from the previous clamped neighbor; return zero if that difference is negative.",Float32[0,3,0,2.5],0f0),
        ("periodic_neighbor_midpoint","Add the previous and next periodic neighbors, then multiply by one half.",Float32[-1.5,1.25,-1.5,1.25],-0.75f0),
        ("neighbor_box_upper_gap","Take the larger of the previous and next clamped neighbors and subtract the current element.",Float32[0,3,-1.5,2.5],0f0),
        ("current_neighbor_product","Multiply the current element by the next periodic neighbor.",Float32[-2,-0.5,-1,-4],0.5625f0),
        ("unit_positive_deficit","Clamp the current element to the interval zero through one, then subtract that clamped value from one.",Float32[0,1,0.5,1],1f0),
        ("clamped_threeway_minimum","Return the minimum of the previous clamped neighbor, current element, and next clamped neighbor.",Float32[-1,-1,-2,-2],-0.75f0),
        ("squared_neighbor_maximum","Square each of the previous and next periodic neighbors, then return the larger square.",Float32[4,4,4,4],0.5625f0)]
    tasks=[(;id=r[1],specification=r[2],public_cases=[
        (;input=copy(x),output=copy(r[3])),(;input=Float32[-0.75],output=Float32[r[4]])]) for r in rows]
    push!(tasks[end].public_cases,(;input=Float32[0.25,1.5,-0.5],output=Float32[2.25,0.25,2.25]))
    tasks
end

function julia_stencil_transfer_messages(task)
    contract="""
Return ONLY one Julia function, no fences or prose, with exactly this signature:
function kernel!(out::Vector{Float32}, x::Vector{Float32}, n::Int32)
    for i=1:n
        out[i] = YOUR_EXPRESSION
    end
    return nothing
end
You may introduce scalar temporary assignments inside the loop. No imports,
macros, helper functions, mutation other than out[i], or other loops.
Use only Float32 constants 0f0, 1f0, 2f0, 0.5f0. Arithmetic expressions may use
binary +, -, *, min, max; unary minus, abs, square via ^2, and clamp with constant
bounds. Use nested min/max for three arguments. Do not use division or Float64.
The ONLY input accesses allowed are exactly these five patterns:
x[i], x[mod1(i-1,n)], x[mod1(i+1,n)], x[max(1,i-1)], x[min(n,i+1)].
These denote current, previous/next periodic, and previous/next clamped values.
Do not abbreviate index expressions into temporary index variables.
Empty input has no outputs. Input and output are distinct arrays. Compute the
task for arbitrary finite Float32 values, not just the supplied public examples.
The expression must expand to at most32 postfix operations with stack depth8.
"""
    ["system"=>"Write a small explicit Float32 Julia stencil following the interface exactly.",
        "user"=>contract*"\nTask:\n"*task.specification*"\nPublic examples (JSON):\n"*JSON3.write(task.public_cases)]
end
