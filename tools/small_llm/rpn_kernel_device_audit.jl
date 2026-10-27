isdefined(@__MODULE__,:RPNPointwiseInterface)||include("rpn_pointwise_interface.jl")
import Mantle
import KernelInterface as KI
using SHA

"""Trusted grader, never a model-generation input.

Only the already-checked finite RPN lowering is evaluated, NOT candidate Julia.
Call from a fresh grading-only worker after generation is finished. The caller
owns device and cleanup. This does not create a device or load a model.
"""
function audit_rpn_kernel_device(device,program,cases)
    lowered=RPNPointwiseInterface.lower(program)
    lowered.accepted||return (;admitted=false,passed=false,reason=lowered.reason,rows=Any[])
    compiled=Module(gensym(:RPNDeviceAudit))
    Core.eval(compiled,:(import KernelInterface as KI))
    Core.eval(compiled,Meta.parse("begin\n"*lowered.serial_code*"\n"*lowered.code*"\nend"))
    # Julia1.12: the newly defined kernel methods must be visible to reflection.
    kernel=Base.invokelatest(getfield,compiled,:kernel!)
    rows=Base.invokelatest(audit_compiled_rpn_cases,device,kernel,cases)
    (;admitted=true,passed=all(r->r.passed,rows),
        code_sha256=bytes2hex(sha256(lowered.serial_code*"\n"*lowered.code)),rows)
end

function audit_compiled_rpn_cases(device,kernel,cases)
    isempty(cases)&&throw(ArgumentError("nonempty independent test cases required"))
    rows=Any[]
    for case in cases
        n=Int(case.n);x=Float32.(case.x);initial=Float32.(case.initial);expected=Float32.(case.expected)
        0<=n<=length(x)&&n<=length(initial)&&length(initial)==length(expected)||throw(ArgumentError("invalid case extent"))
        initial[n+1:end]==expected[n+1:end]||throw(ArgumentError("expected padding must remain unchanged"))
        cpu=copy(initial);kernel(cpu,x,Int32(n))
        owned=Any[];plan=nothing
        try
            out=Mantle.Buffer(device,initial);push!(owned,out)
            input=Mantle.Buffer(device,x);push!(owned,input)
            graph=Mantle.Graph(device)
            Mantle.dispatch!(graph,kernel,(out,input,Int32(n)),64*cld(max(n,1),64);group=64)
            plan=Mantle.record!(Mantle.Plan(graph;budget=Inf))
            before=Mantle.submissions(device);Mantle.run!(plan)
            submissions=Mantle.submissions(device)-before
            actual=Array(out)
            cpu_match=isapprox(cpu,expected;atol=2f-5,rtol=2f-5)
            gpu_match=isapprox(actual,expected;atol=2f-5,rtol=2f-5)
            padding=cpu[n+1:end]==actual[n+1:end]==initial[n+1:end]
            push!(rows,(;n,passed=cpu_match&&gpu_match&&padding&&submissions==1,cpu_match,gpu_match,padding,submissions))
        finally
            plan===nothing||Mantle.free!(plan)
            foreach(Mantle.free!,reverse(owned))
        end
    end
    rows
end
