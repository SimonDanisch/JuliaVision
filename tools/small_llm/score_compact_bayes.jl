isdefined(@__MODULE__,:bayes_extraction_tasks)||include("bayes_extraction_tasks.jl")
include("bayes_extraction_references.jl")
include("compact_bayes_admission.jl")
include("portable_evidence.jl")
isdefined(@__MODULE__,:validate_julia_stencil_receipt)||include("julia_stencil_transfer_study.jl")
import Mantle

function score_compact_bayes(device;directory=joinpath(@__DIR__,"compact_bayes_results"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Explicit software grading device")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (f,h) in pairs(p.private_source_sha256)
        digest(joinpath(@__DIR__,String(f)))==h||error("Private source drift")
    end
    for (f,h) in pairs(p.source_sha256)
        digest(joinpath(@__DIR__,String(f)))==h||error("Public source drift")
    end
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
    digest(parentpath)==p.parent_conditions_sha256&&digest(joinpath(dirname(parentpath),"summary.json"))==p.parent_summary_sha256||error("Original study drift")
    length(p.jobs)==8&&all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("Finish ALL8 before grading")
    references=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()));tasks=Dict(t.id=>t for t in bayes_extraction_tasks())
    rows=map(p.jobs) do job
        output=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(output,String))
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=digest(path),attempt_sha256=digest(joinpath(directory,job.id*"_attempt.json")))
        digest(joinpath(dirname(parentpath),String(job.task)*"_bayes_network.json"))==job.original_generation_sha256||error("Original generation drift")
        failed=hasproperty(m,:inference_error)
        a=failed||m.truncated ? (;accepted=false,reason=failed ? "inference_error" : "truncated",network=nothing,packed=nothing) : admit_compact_bayes(String(m.answer))
        r=references[job.task];task=tasks[job.task];faithful=nothing;probability=nothing;solver_status=nothing;submissions=0;solver_seconds=0.
        if a.accepted
            faithful=bayes_extraction_faithful(a.network,JSON3.read(JSON3.write(r.network)))
            target=findfirst(==(task.target),a.packed.query_states)
            if a.packed.query==Symbol(task.query)&&target!==nothing
                started=time();e=PortableEvidence.prepare_bayes(device,a.packed.factors,a.packed.query,a.packed.observed)
                try
                    before=Mantle.submissions(device);PortableEvidence.run!(e);Mantle.waitidle(device)
                    submissions=Mantle.submissions(device)-before;solver_status=only(Array(e.outputs.status))
                    probability=solver_status==0 ? Float64(Array(e.outputs.posterior)[target]) : nothing
                finally close(e) end
                solver_seconds=time()-started
            end
        end
        numeric_correct=probability!==nothing&&abs(probability-Float64(r.answer))<=1e-5
        correct=faithful===true&&numeric_correct
        (;id=job.id,task=job.task,arm=job.arm,status=failed ? "inference_error" : correct ? "passed" : "failed",
            admitted=a.accepted,reason=a.reason,network_faithful=faithful,numeric_correct,probability,expected_probability=Float64(r.answer),
            solver_status,solver_submissions=submissions,solver_seconds,truncated=failed ? nothing : m.truncated,generation_sha256=digest(output))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,rows,total=8,passed=count(r->r.status=="passed",rows),admitted=count(r->r.admitted,rows),conditions_sha256=digest(path))
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
