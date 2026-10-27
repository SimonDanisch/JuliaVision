"""Public authored related-family tasks; no reference program or test input.

These are a prospective interface ablation, NOT independent public held-out
benchmarks. Both arms must receive identical prompts and arithmetic choices.
"""
function rpn_stencil_tasks()
    [
        (;id="central_periodic_gradient",specification="For each i, return half the difference between the next and previous periodic neighbor (next minus previous)."),
        (;id="half_diffusion_update",specification="For each i, add to the current value half of its periodic second difference: previous plus next minus twice the current value."),
        (;id="clamped_neighbor_energy",specification="For each i, return the sum of two squared differences: previous clamped neighbor minus current, and next clamped neighbor minus current."),
        (;id="positive_forward_increment",specification="For each i, return the positive part of next periodic neighbor minus current: return zero when this difference is negative."),
        (;id="unit_interval_projection",specification="For each i, project the current value onto the closed interval from zero to one, preserving values already in that interval."),
        (;id="clamped_morphological_range",specification="For each i, consider the previous clamped neighbor, current value, and next clamped neighbor. Return their maximum minus their minimum."),
        (;id="nonnegative_limited_slope",specification="For each i, compute two slopes: current minus previous periodic neighbor, and next periodic neighbor minus current. Return the smaller slope, but never a negative value (use zero instead)."),
        (;id="periodic_neighbor_sum_energy",specification="For each i, square the sum of the previous and next periodic neighbors, then subtract the square of the current value.")
    ]
end

function rpn_stencil_messages(task)
    contract="""
Return ONLY a postfix stack program, one opcode per line, ending with finish.
There are no Julia expressions, comments, Markdown fences, explanations or loops.
Push opcodes: x0 (current), x_prev_periodic, x_next_periodic,
x_prev_clamped, x_next_clamped, zero, one, two, half.
Unary opcodes: neg, square. Binary opcodes: add, sub, mul, min, max.
A binary operation pops right operand b, then left operand a, and pushes a OP b.
finish requires exactly one stack value and ends the program. Maximum stack depth8;
maximum32 non-finish operations. Choose all arithmetic and neighbor accesses yourself.
Input x is a one-dimensional Float32 array of length n. Periodic indices wrap;
clamped indices stop at the first/last element. Empty inputs produce no outputs.
The trusted framework supplies iteration, bounds checking and one output store.
"""
    ["system"=>"You write finite postfix programs. Follow the output contract exactly.",
     "user"=>contract*"\nTask:\n"*task.specification]
end
