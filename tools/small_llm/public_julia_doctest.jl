# Semantic feedback from examples ALREADY in the public prompt. Expected text
# is compared as text, never parsed/evaluated as Julia. No hidden test access.
isdefined(@__MODULE__,:public_smoke_expressions) || include("public_julia_smoke.jl")

function public_doctest_examples(prompt)
    expressions=public_smoke_expressions(prompt)
    matches=collect(eachmatch(r"(?ms)^julia> ([^\n]+)\n(.*?)(?=^julia> |^```|\z)",prompt))
    length(matches)==length(expressions) || throw(ArgumentError("unsupported public doctest layout"))
    [(;expression,expected=strip(String(m.captures[2]))) for (expression,m) in zip(expressions,matches)]
end

function prepare_external_public_doctests()
    isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || error("load frozen task adapter first")
    directory=joinpath(@__DIR__,"humaneval_subset_results");rows=Any[]
    for task in load_frozen_humaneval_tasks()
        source=joinpath(directory,task.id*"_baseline.json");isfile(source) || continue
        candidate=JSON3.read(read(source,String));path=joinpath(directory,task.id*"_public_doctest.json")
        if isfile(path)
            result=JSON3.read(read(path,String))
            result.code_sha256==bytes2hex(sha256(String(candidate.code))) || error("doctest source drift")
            result.public_examples_sha256==bytes2hex(sha256(JSON3.write(public_doctest_examples(task.prompt)))) || error("public example drift")
        else
            result=diagnose_public_doctest(task,String(candidate.code))
            open(io->JSON3.pretty(io,result),path,"w")
        end
        push!(rows,(;task=task.id,analysis_completed=result.analysis_completed,
            failed_public_examples=result.payload===nothing ? nothing : count(row->!row.matches_public_example,result.payload)))
    end
    report=(;scope="public prompt doctest diagnostics, no hidden scoring or model revisions",rows)
    open(io->JSON3.pretty(io,report),joinpath(directory,"public_doctest_summary.json"),"w")
    report
end

function qualify_public_doctests()
    isdefined(@__MODULE__,:load_frozen_humaneval_tasks) || error("load frozen task adapter first")
    path=joinpath(@__DIR__,"humaneval_subset_results","public_doctest_qualification.json")
    isfile(path) && error("public doctest qualification already saved")
    rows=Any[]
    for task in load_frozen_humaneval_tasks()
        result=diagnose_public_doctest(task,task.reference)
        push!(rows,(;task=task.id,passed=result.analysis_completed&&all(row->row.completed&&row.matches_public_example,result.payload),
            elapsed_seconds=result.elapsed_seconds,public_examples_sha256=result.public_examples_sha256))
    end
    report=(;scope="public-example diagnostic qualification only, NOT model intelligence",
        qualified=count(row->row.passed,rows),selected=length(rows),rows,
        source_sha256=bytes2hex(sha256(read(@__FILE__))))
    open(io->JSON3.pretty(io,report),path,"w")
    report
end

function diagnose_public_doctest(task,code)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    examples=public_doctest_examples(task.prompt)
    probes=join(["""
        try
            value=begin $(example.expression) end
            actual=sprint(show,MIME(\"text/plain\"),value;context=(:limit=>false,))
            expected=$(repr(example.expected))
            normalize(s)=join(strip.(split(strip(s),'\\n')),\"\\n\")
            push!(rows,(example=$(repr(example.expression)),completed=true,
                matches_public_example=normalize(actual)==normalize(expected),
                actual=first(actual,3000),expected=expected))
        catch err
            push!(rows,(example=$(repr(example.expression)),completed=false,
                matches_public_example=false,error=first(sprint(showerror,err),2000)))
        end
        """ for example in examples],"\n")
    checks="rows=Any[]\n"*probes*"\nprintln(\"PUBLIC_DOCTEST=\",JSON3.write(rows))"
    result=JuliaTaskEval.evaluate((;id=task.id,category="base",setup="using JSON3",tests=checks),code)
    prefix="PUBLIC_DOCTEST=";lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    payload=length(lines)==1 ? JSON3.read(only(lines)[length(prefix)+1:end]) : nothing
    (;task=task.id,scope="public prompt doctests only; whitespace-normalized display comparison, NOT private correctness tests",
        analysis_completed=result.passed&&payload!==nothing,payload,
        loading_or_harness_error=payload===nothing ? result.diagnostic : nothing,
        elapsed_seconds=result.elapsed_seconds,code_sha256=bytes2hex(sha256(code)),
        public_examples_sha256=bytes2hex(sha256(JSON3.write(examples))))
end
