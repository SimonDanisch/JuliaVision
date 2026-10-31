module JuliaTaskEval
using JSON3, SHA
include("julia_tasks.jl")

function extract_code(text;thinking=false)
    # Treat the first terminated answer as final; fixed-budget GPU generation may
    # continue after EOS. Do not silently choose a later, more successful answer.
    answer=first(split(text,r"<\|im_end\|>|<\|endoftext\|>";limit=2))
    if thinking
        closing=findlast("</think>",answer)
        closing===nothing && return "" # no final answer within the budget
        answer=answer[nextind(answer,last(closing)):end]
    end
    matches=collect(eachmatch(r"```(?:julia)?\s*\n(.*?)```"s,answer))
    isempty(matches) ? strip(replace(answer,r"^\s*```(?:julia)?\s*\n"=>"")) : join((m.captures[1] for m in matches),"\n")
end

function evaluate(task,code;output=nothing,compiled_modules="existing",depot_order="withsystem")
    compiled_modules in ("existing","yes") || throw(ArgumentError("invalid compiled-modules policy"))
    depot_order in ("privatefirst","cachefirst","withsystem") || throw(ArgumentError("invalid depot order"))
    result=mktempdir() do dir
        write(joinpath(dir,"candidate.jl"),code)
        setup=hasproperty(task,:setup) ? task.setup : ""
        write(joinpath(dir,"runner.jl"),"using Test\n"*setup*"\ninclude(\"/input/candidate.jl\")\n@testset \"contract\" begin\n"*task.tests*"\nend\nprintln(\"CONTRACT_PASSED\")\n")
        started=time()
        # A real log file means the child's RLIMIT_FSIZE also bounds stdout;
        # do not accumulate arbitrarily large generated output in a host pipe.
        logfile=joinpath(dir,"execution.log")
        category=hasproperty(task,:category) ? task.category : "base"
        process=open(logfile,"w") do log
            command=addenv(`bash $(joinpath(@__DIR__,"julia_sandbox.sh")) $dir $category`,
                "JULIA_TASK_COMPILED_MODULES"=>compiled_modules,"JULIA_TASK_DEPOT_ORDER"=>depot_order)
            p=run(pipeline(ignorestatus(command),stdout=log,stderr=log);wait=false)
            try
                wait(p)
            finally
                if process_running(p)
                    kill(p)
                    wait(p)
                end
            end
            p
        end
        diagnostic=open(io->String(read(io,65536)),logfile)
        (;id=task.id,passed=success(process)&&occursin("CONTRACT_PASSED",diagnostic),compiled_modules,depot_order,
          exitcode=process.exitcode,elapsed_seconds=time()-started,diagnostic,
          log_truncated=filesize(logfile)>65536,code_sha256=bytes2hex(sha256(code)))
    end
    output===nothing || open(io->JSON3.pretty(io,result),output,"w")
    result
end

function validate_harness(;tasks=JULIA_TASKS,output=joinpath(@__DIR__,"julia_harness_validation.json"))
    rows=Any[]
    for task in tasks
        result=evaluate(task,task.reference)
        push!(rows,result)
        open(io->JSON3.pretty(io,rows),replace(output,".json"=>"_progress.json"),"w")
        println(task.id," ",result.passed ? "PASS" : "FAIL"); flush(stdout)
    end
    # Verify the harness rejects a candidate rather than passing vacuously.
    rejected=evaluate(first(tasks),"solve(args...)=[]")
    all(r->r.passed,rows) && !rejected.passed || @warn "Harness not yet qualified"
    result=(;scope="local authored diagnostic, not held-out benchmark; Makie construction, not rendered aesthetics",references=rows,mutant_rejected=!rejected.passed)
    open(io->JSON3.pretty(io,result),output,"w")
    result
end

function validate_sandbox_boundaries(;output=joinpath(@__DIR__,"sandbox_system_isolation_validation.json"))
    # Import macros before the outer @testset is expanded, not inside its body.
    task=(;id="sandbox_boundaries",category="base",setup="using Sockets",
        tests=read(joinpath(@__DIR__,"sandbox_checks","runner.jl"),String))
    evaluate(task,"";output)
end

const PUBLIC_PROBES=Dict(
    "stable_groups"=>"println(repr(solve([\"first\",\"second\",\"first\"],[2,4,7])))",
    "missing_windows"=>"println(repr(solve([1.0,missing,5.0],2)))",
    "alias_stencil"=>"x=[1.,4.,9.,16.]; println(repr(solve!(x,x)))",
    "ka_tail"=>"println(repr(solve!(ones(Float32,3),Float32[1,2,3],2f0,1f0)))",
    "makie_link"=>"f,a,b=makeplot(collect(1:3),[1.,-1.,2.]); println(typeof(f)); println(a.title[])",
    "makie_band"=>"f,a=makeplot(collect(1:3),[1.,2.,1.],fill(0.2,3)); println(typeof(f)); println(a.title[])",
)

"""Public execution feedback, deliberately separate from hidden contract tests.

Prints outputs or Julia exceptions, never oracle results/hidden test values.
"""
function diagnose(task,code)
    probe=(;id=task.id,category=task.category,tests=PUBLIC_PROBES[task.id],
        setup=hasproperty(task,:setup) ? task.setup : "")
    evaluate(probe,code)
end
end
