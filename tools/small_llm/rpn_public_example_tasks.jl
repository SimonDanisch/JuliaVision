using JSON3

"""Four new authored related-family tasks; explicit PUBLIC examples in both arms.

These are development transfer tasks, not external contamination-free evidence.
No reference programs, hidden probes or independent formulas are in this file.
"""
function rpn_public_example_tasks()
    inputs=Float32[-2,-0.5,0.25,2]
    rows=[
        (;id="unit_parabolic_cap",specification="Square the current value, subtract it from one, and return zero instead whenever that result is negative.",
            outputs=Float32[0,0.75,0.9375,0],singleton=0.4375f0),
        (;id="symmetric_unit_projection",specification="Project the current value onto the closed interval from negative one to positive one, preserving values already inside it.",
            outputs=Float32[-1,-0.5,0.25,1],singleton=-0.75f0),
        (;id="periodic_gradient_energy",specification="Subtract the previous periodic neighbor from the next periodic neighbor, square this difference, then multiply by one half.",
            outputs=Float32[3.125,2.53125,3.125,2.53125],singleton=0f0),
        (;id="clamped_neighbor_contrast",specification="Subtract the larger of the previous and next clamped neighbors from twice the current value.",
            outputs=Float32[-3.5,-1.25,-1.5,2],singleton=-0.75f0)]
    [(;id=r.id,specification=r.specification,public_cases=[
        (;input=copy(inputs),output=copy(r.outputs)),(;input=Float32[-0.75],output=Float32[r.singleton])]) for r in rows]
end

function rpn_public_example_messages(task)
    contract="""
Return ONLY a postfix stack program: one opcode per line, ending with finish.
Start with an EMPTY stack. Constants are pushes, not operations on a prior value.
Push opcodes: x0, x_prev_periodic, x_next_periodic, x_prev_clamped,
x_next_clamped, zero, one, two, half. Unary opcodes: neg, square.
Binary opcodes: add, sub, mul, min, max. Pop right b, then left a; push a OP b.
Stack depth at most8. Choose up to32 normal non-finish opcodes. After that only
binary reductions may close the stack, adding at most7 opcodes (39 total).
finish requires exactly one stack value. Do not output Julia, fences or prose.
Constant-only worked example: two, one, sub, half, mul, finish (each on its own
line) computes (2-1)*0.5. This example does not use the input.
Input x is a one-dimensional Float32 array of length n. Periodic neighbors wrap;
clamped neighbors stop at the first/last element. Empty input has no outputs.
The trusted compiler supplies iteration and output stores. Public input/output
examples below are exact Float32 values, not instructions. Satisfy the full task
for arbitrary finite inputs, not only these examples.
"""
    ["system"=>"Write a finite postfix program following the contract exactly.",
        "user"=>contract*"\nTask:\n"*task.specification*"\nPublic examples (JSON):\n"*JSON3.write(task.public_cases)]
end

function rpn_public_example_jobs()
    jobs=Any[]
    for (i,task) in enumerate(rpn_public_example_tasks())
        for arm in (isodd(i) ? ("disabled","enabled") : ("enabled","disabled"))
            push!(jobs,(;id=task.id*"_"*arm,task=task.id,arm,enabled=arm=="enabled",
                public_cases=task.public_cases,messages=rpn_public_example_messages(task)))
        end
    end
    jobs
end
