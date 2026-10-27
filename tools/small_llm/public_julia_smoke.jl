# Public-input execution only. This is a separate diagnostic from the frozen
# parser/dispatch/JET policy; adding it must not silently change that policy.
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
isdefined(@__MODULE__,:PublicJuliaSignatures) || include("public_julia_signatures.jl")
using JSON3, SHA

function public_smoke_expressions(prompt)
    ncodeunits(prompt)<=65536 || throw(ArgumentError("public prompt too large"))
    name=Symbol(PublicJuliaSignatures.from_task_prompt(prompt).name)
    literal(x)=x isa Number || x isa String || x isa Char || x in (true,false,nothing) ||
        (x isa Expr && ((x.head in (:vect,:tuple) && all(literal,x.args)) ||
            (x.head==:ref && !isempty(x.args) && first(x.args) in (:String,:Int,:Float64) && all(literal,x.args[2:end])) ||
            (x.head==:call && length(x.args)==2 && first(x.args) in (:+,:-) && x.args[2] isa Number)))
    call(x)=x isa Expr && x.head==:call && !isempty(x.args) && first(x.args)==name && all(literal,x.args[2:end])
    rows=String[]
    for line in split(prompt,'\n')
        startswith(strip(line),"julia> ") || continue
        raw=strip(strip(line)[8:end]); expression=Meta.parse(raw)
        safe=call(expression) || (expression isa Expr && expression.head==:call &&
            length(expression.args)==3 && first(expression.args)==:(==) &&
            call(expression.args[2]) && literal(expression.args[3]))
        safe || throw(ArgumentError("unsupported public example; never execute arbitrary prompt code"))
        push!(rows,sprint(Base.show_unquoted,expression))
    end
    isempty(rows) && throw(ArgumentError("no public single-line examples"))
    rows
end

function diagnose_public_smoke(task,code)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    expressions=public_smoke_expressions(task.prompt)
    probes=join(["try\n value=begin $expression end\n push!(rows,(example=$(repr(expression)),completed=true,value=first(repr(value),1000)))\ncatch err\n push!(rows,(example=$(repr(expression)),completed=false,error=first(sprint(showerror,err),2000)))\nend" for expression in expressions],"\n")
    checks="rows=Any[]\n"*probes*"\nprintln(\"PUBLIC_SMOKE=\",JSON3.write(rows))"
    diagnostic_task=(;id=task.id,category="base",setup="using JSON3",tests=checks)
    result=JuliaTaskEval.evaluate(diagnostic_task,code)
    prefix="PUBLIC_SMOKE="
    lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    payload=length(lines)==1 ? JSON3.read(only(lines)[length(prefix)+1:end]) : nothing
    (;task=task.id,scope="public prompt examples only; runtime completion/observed values, NOT hidden correctness tests or the frozen JET policy",
        analysis_completed=result.passed&&payload!==nothing,payload,
        loading_or_harness_error=payload===nothing ? result.diagnostic : nothing,
        elapsed_seconds=result.elapsed_seconds,code_sha256=bytes2hex(sha256(code)),
        public_examples_sha256=bytes2hex(sha256(join(expressions,"\n"))))
end

function prepare_external_public_smoke()
    isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || error("load the frozen task adapter first")
    directory=joinpath(@__DIR__,"humaneval_subset_results");rows=Any[]
    for task in load_frozen_humaneval_tasks()
        source=joinpath(directory,task.id*"_baseline.json")
        isfile(source) || continue
        candidate=JSON3.read(read(source,String));path=joinpath(directory,task.id*"_public_smoke.json")
        if isfile(path)
            result=JSON3.read(read(path,String))
            result.code_sha256==bytes2hex(sha256(String(candidate.code))) || error("public smoke source drift")
            result.public_examples_sha256==bytes2hex(sha256(join(public_smoke_expressions(task.prompt),"\n"))) || error("public example drift")
        else
            result=diagnose_public_smoke(task,String(candidate.code))
            open(io->JSON3.pretty(io,result),path,"w")
        end
        push!(rows,(;task=task.id,analysis_completed=result.analysis_completed,
            runtime_errors=result.payload===nothing ? nothing : count(row->!row.completed,result.payload)))
    end
    report=(;scope="public-input runtime diagnostic preparation only; no repairs selected and no change to frozen JET experiment",rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"public_smoke_summary.json"),"w")
    report
end
