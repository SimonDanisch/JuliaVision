# JET may execute generated-function inference hooks. Candidate definitions and
# all inference therefore remain inside the restricted subprocess, not Main.
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3, SHA
const JET_PUBLIC_SIGNATURES=Dict(
    "stable_groups"=>(name="solve",types="[Tuple{Vector{String},Vector{Int}}]"),
    "missing_windows"=>(name="solve",types="[Tuple{Vector{Union{Missing,Float64}},Int},Tuple{Vector{Float64},Int}]"),
    "alias_stencil"=>(name="solve!",types="let V=typeof(view(zeros(3),1:2));[Tuple{Vector{Float64},Vector{Float64}},Tuple{V,V}];end"))

function diagnose_jet(task,code)
    ncodeunits(code)<=65536 || throw(ArgumentError("candidate too large"))
    started=time()
    function parse_failure(e)
        e isa Expr || return nothing
        if e.head in (:error,:incomplete)
            detail=isempty(e.args) ? e : first(e.args)
            return first(detail isa Exception ? sprint(showerror,detail) : sprint(show,e),3000)
        end
        for a in e.args
            result=parse_failure(a)
            result===nothing || return result
        end
        nothing
    end
    syntax_error=parse_failure(Meta.parseall(code))
    if syntax_error!==nothing
        payload=(;version=nothing,completed=true,rows=[(;signature="candidate definition",report_count=1,
            kind="syntax",text=syntax_error)])
        return (;task=task.id,scope="Julia parser rejection; JET not run on invalid code, no hidden assertions/values",
            analysis_completed=true,payload,elapsed_seconds=time()-started,diagnostic=syntax_error,
            code_sha256=bytes2hex(sha256(code)))
    end
    spec=JET_PUBLIC_SIGNATURES[task.id]
    tests="""
    @assert JET.JET_AVAILABLE
    rows=Any[]
    for signature in $(spec.types)
        if !isdefined(Main,$(repr(Symbol(spec.name)))) || !hasmethod($(spec.name),signature)
            push!(rows,(signature=string(signature),report_count=1,kind="dispatch",
                text="No applicable method for $(spec.name) at public signature "*string(signature)))
        else
            try
                report=JET.report_call($(spec.name),signature;target_modules=(Main,),mode=:basic)
                push!(rows,(signature=string(signature),report_count=length(JET.get_reports(report)),kind="JET",text=sprint(show,report)))
            catch err
                err isa InterruptException && rethrow()
                push!(rows,(signature=string(signature),report_count=0,kind="analysis_error",text=sprint(showerror,err)))
            end
        end
    end
    println("JET_DIAGNOSTICS=",JSON3.write((version=string(pkgversion(JET)),completed=all(r->r.kind!="analysis_error",rows),rows=rows)))
    """
    analysis_task=(;id=task.id,category="base",setup="using JET,JSON3",tests)
    result=JuliaTaskEval.evaluate(analysis_task,code)
    prefix="JET_DIAGNOSTICS="
    lines=filter(line->startswith(line,prefix),split(result.diagnostic,'\n'))
    payload=length(lines)==1 ? JSON3.read(only(lines)[length(prefix)+1:end]) : nothing
    (;task=task.id,scope="advisory type analysis at public input signatures; no hidden assertions/values, not proof of correctness",
        analysis_completed=result.passed&&payload!==nothing&&payload.completed,payload,
        elapsed_seconds=result.elapsed_seconds,diagnostic=result.diagnostic,
        code_sha256=bytes2hex(sha256(code)))
end

function qualify_jet_diagnostics(;version="v4")
    occursin(r"^v[0-9]+$",version) || throw(ArgumentError("version"))
    rows=Any[];outdir=joinpath(@__DIR__,"julia_jet_"*version*"_results");mkpath(outdir)
    for task in JuliaTaskEval.JULIA_TASKS
        haskey(JET_PUBLIC_SIGNATURES,task.id) || continue
        for arm in ("reference","baseline")
            source=joinpath(@__DIR__,"julia_tasks_results",task.id*"_baseline.json")
            code=arm=="reference" ? task.reference : String(JSON3.read(read(source,String)).code)
            row=diagnose_jet(task,code)
            open(io->JSON3.pretty(io,row),joinpath(outdir,task.id*"_"*arm*"_diagnostics.json"),"w")
            push!(rows,(;arm,row...))
            println(task.id," ",arm," JET completed=",row.analysis_completed,
                " reports=",row.payload===nothing ? "unknown" : sum(r.report_count for r in row.payload.rows));flush(stdout)
        end
    end
    report=(;scope="JET diagnostic qualification, not model repair accuracy",rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"julia_jet_"*version*"_qualification.json"),"w")
    report
end
