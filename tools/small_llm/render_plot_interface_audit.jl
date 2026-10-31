isdefined(@__MODULE__,:admit_plot_interface)||include("plot_interface_admission.jl")
isdefined(@__MODULE__,:render_julia_candidate)||include("julia_render.jl")
using JSON3,SHA,Dates

function plot_render_packet(spec)
    axes=[(;title=a.title,xlabel=a.xlabel,ylabel=a.ylabel,xscale=a.xscale,yscale=a.yscale,legend_position=String(a.legend_position)) for a in spec.axes]
    series=map(spec.series) do s
        common=(;kind=s.kind,axis=s.axis,x=s.x,label=s.label,color=s.color)
        s.kind=="band" ? (;common...,low=s.low,high=s.high) : (;common...,y=s.y)
    end
    (;size=collect(spec.size),axes,series,legend=spec.legend,linkx=spec.linkx)
end

function plot_render_probe()
    x=collect(range(0.1,3.0;length=64));a=2 .+sin.(x);b=0.5 .+exp.(-x./3)
    (;x,a,b,lo=a.-0.15,hi=a.+0.15)
end

"""Uniform PRE-GRADE engineering audit registered after generation started.
NOT a retrospective change to the primary drawing-plan score or a preference
score. All16 outcomes retained, including unsupported/failed/interrupted ones.
"""
function freeze_plot_render_audit(;directory=joinpath(@__DIR__,"plot_interface_results"))
    path=joinpath(directory,"render_audit_conditions.json");isfile(path)&&error("Preserve render registration")
    isfile(joinpath(directory,"summary.json"))&&error("Register before ANY private plot grading")
    digest(path)=bytes2hex(sha256(read(path)))
    primary=joinpath(directory,"conditions.json");p=JSON3.read(read(primary,String))
    existing=Dict(String(j.id)=>digest(joinpath(directory,String(j.id)*".json")) for j in p.jobs if isfile(joinpath(directory,String(j.id)*".json")))
    previous=joinpath(directory,"render_audit_conditions_before_incomplete_fix.json")
    result=(;ready=true,registered_utc=string(now(UTC)),primary_conditions_sha256=digest(primary),
        pre_grade_amendment=isfile(previous) ? (;previous_sha256=digest(previous),reason="Before any private grades/render attempts: interrupted render outcomes remain INCOMPLETE rather than merely resolved. Original registration retained unchanged.") : nothing,
        scope="Separate uniform PRE-GRADE render/layout development audit. Implementation registered AFTER plot generation began, not part of the original prospective primary score; no hidden grade seen/used. Not an aesthetic preference score.",
        policy="Wait for ALL16 primary grades after model-worker retirement. Attempt every admitted nontruncated plan on SAME64-point private positive data, whether specification-correct or wrong. Generated Julia NEVER executes: render only trusted builder+serialized checked IR in existing CPU-only networkless Cairo jail. Keep first render attempt, no silent rerenders. Preserve unsupported, failure, clipping and interrupted outcomes separately. Unchanged180s wall/150CPU/12GiB jail bounds.",
        jobs=String[j.id for j in p.jobs],existing_generation_sha256=existing,probe_sha256=bytes2hex(sha256(JSON3.write(plot_render_probe()))),
        sources=Dict(file=>digest(joinpath(@__DIR__,file)) for file in
            ("render_plot_interface_audit.jl","plot_interface_admission.jl","safe_plot_spec.jl","safe_plot_julia_source.jl","safe_plot_builder.jl","julia_render.jl","julia_sandbox.sh")))
    open(io->JSON3.pretty(io,result),path,"w");result
end

function render_plot_interface_audit(;directory=joinpath(@__DIR__,"plot_interface_results"),max_cases=16)
    max_cases isa Integer&&!(max_cases isa Bool)&&1<=max_cases<=16||throw(ArgumentError("Bounded render count"))
    digest(path)=bytes2hex(sha256(read(path)))
    registration=joinpath(directory,"render_audit_conditions.json");p=JSON3.read(read(registration,String))
    conditionpath=joinpath(directory,"conditions.json");p.primary_conditions_sha256==digest(conditionpath)||error("Primary drift")
    for (file,hash) in pairs(p.sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Renderer/source drift")
    end
    p.probe_sha256==bytes2hex(sha256(JSON3.write(plot_render_probe())))||error("Probe drift")
    summarypath=joinpath(directory,"summary.json");summary=JSON3.read(read(summarypath,String))
    summary.complete&&summary.conditions_sha256==digest(conditionpath)||error("Complete ALL16 primary grades first")
    conditions=JSON3.read(read(conditionpath,String));artifacts=joinpath(directory,"render_artifacts");mkpath(artifacts)
    rows=Any[];new_attempts=0
    for job in conditions.jobs
        id=String(job.id);basename(id)==id||error("Unsafe public ID")
        output=joinpath(directory,id*".json");generation_hash=digest(output)
        score=only(filter(r->r.id==id,summary.rows));score.generation_sha256==generation_hash||error("Primary output drift")
        resultpath=joinpath(artifacts,id*"_render.json")
        if isfile(resultpath)
            prior=JSON3.read(read(resultpath,String))
            prior.generation_sha256==generation_hash&&prior.render_conditions_sha256==digest(registration)||error("Cached render drift")
            push!(rows,prior);continue
        end
        receipt=JSON3.read(read(output,String))
        attempt=joinpath(artifacts,id*"_attempt.json");image=joinpath(artifacts,id*".png")
        row=if !score.admitted
            (;id,arm=job.arm,status="unsupported_or_truncated_output",attempted=false,rendered=false,within_canvas=nothing)
        elseif ispath(attempt)||ispath(image)||ispath(image*".json")
            (;id,arm=job.arm,status="interrupted_first_render",attempted=true,rendered=false,within_canvas=nothing)
        elseif new_attempts>=max_cases
            (;id,arm=job.arm,status="pending",attempted=false,rendered=false,within_canvas=nothing)
        else
            plan=admit_plot_interface(String(receipt.answer),String(job.arm));plan.accepted||error("Admission drift")
            payload=JSON3.write(plot_render_packet(plan.spec))
            source="import Makie\n"*read(joinpath(@__DIR__,"safe_plot_spec.jl"),String)*"\n"*read(joinpath(@__DIR__,"safe_plot_builder.jl"),String)*"\n"*"""
            function render_plan()
                x=collect(range(0.1,3.0;length=64));a=2 .+sin.(x);b=0.5 .+exp.(-x./3)
                data=Dict("x"=>x,"a"=>a,"b"=>b,"lo"=>a.-0.15,"hi"=>a.+0.15)
                spec=SafePlotSpec.parse_spec($(repr(payload)))
                SafePlotBuilder.build(spec,data).figure
            end
            """
            open(io->JSON3.pretty(io,(;id,generation_sha256=generation_hash,render_conditions_sha256=digest(registration),trusted_source_sha256=bytes2hex(sha256(source)),started_utc=string(now(UTC)))),attempt,"w")
            new_attempts+=1
            rendered=render_julia_candidate(source,"render_plan()";artifact_path=image)
            inside=rendered.rendered&&rendered.layout!==nothing&&all(a->a.within_canvas,rendered.layout.axes)
            (;id,arm=job.arm,status=rendered.rendered ? inside ? "rendered_unclipped" : "rendered_with_clipping" : "render_failed",
                attempted=true,rendered=rendered.rendered,within_canvas=rendered.rendered ? inside : nothing,render=rendered)
        end
        row=(;row...,generation_sha256=generation_hash,render_conditions_sha256=digest(registration),aesthetic_score=nothing)
        row.status=="pending"||open(io->JSON3.pretty(io,row),resultpath,"w")
        push!(rows,row)
        println("Plot render audit ",id," ",row.status);flush(stdout)
    end
    result=(;complete=all(r->r.status∉("pending","interrupted_first_render"),rows),scope=p.scope,policy=p.policy,rows,
        primary_summary_sha256=digest(summarypath),render_conditions_sha256=digest(registration),
        caveat="Actual images/layout only. This post-registration stage does not alter primary specification grades. Renderability/unclipped labels do NOT establish aesthetic quality or scientific validity.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"render_audit_summary.json"),"w");result
end
