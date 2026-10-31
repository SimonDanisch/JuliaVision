isdefined(@__MODULE__,:bayes_extraction_tasks)||include("bayes_extraction_tasks.jl")
include("bayes_extraction_references.jl")
include("bayes_thinking_admission.jl")
include("portable_evidence.jl")
isdefined(@__MODULE__,:validate_julia_stencil_receipt)||include("julia_stencil_transfer_study.jl")
import Mantle

function score_bayes_thinking(device;directory=joinpath(@__DIR__,"bayes_thinking_results"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Explicit software grading device required")
    digest(path)=bytes2hex(sha256(read(path)))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (f,h) in pairs(p.private_source_sha256)
        digest(joinpath(@__DIR__,String(f)))==h||error("Private scorer drift")
    end
    parentpath=joinpath(@__DIR__,"bayes_extraction_results/conditions.json")
    digest(parentpath)==p.parent_conditions_sha256||error("Parent drift")
    digest(joinpath(dirname(parentpath),"summary.json"))==p.parent_summary_sha256||error("Parent score drift")
    length(p.jobs)==16&&all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("Finish ALL16 before grades")
    references=Dict(t.id=>r for (t,r) in zip(bayes_extraction_tasks(),bayes_extraction_references()))
    tasks=Dict(t.id=>t for t in bayes_extraction_tasks())
    rows=map(p.jobs) do job
        output=joinpath(directory,String(job.id)*".json");m=JSON3.read(read(output,String))
        validate_julia_stencil_receipt(m,job,p;conditions_sha256=digest(path),attempt_sha256=digest(joinpath(directory,job.id*"_attempt.json")))
        digest(joinpath(dirname(parentpath),String(job.id)*".json"))==job.original_generation_sha256||error("Original output drift")
        failed=hasproperty(m,:inference_error)
        failed||(m.thinking_budget==p.decoding.thinking_budget&&m.forced_closures isa Integer&&!(m.forced_closures isa Bool)&&m.forced_closures in (0,1))||error("Controller receipt mismatch")
        a=admit_bayes_thinking(m,String(job.arm));r=references[job.task];task=tasks[job.task]
        probability=a.probability;faithful=nothing;solver_status=nothing;solver_seconds=0.;submissions=0
        if a.accepted&&job.arm=="bayes_network"
            faithful=bayes_extraction_faithful(a.network,JSON3.read(JSON3.write(r.network)))
            target=findfirst(==(task.target),a.packed.query_states)
            if a.packed.query==Symbol(task.query)&&target!==nothing
                started=time();execution=PortableEvidence.prepare_bayes(device,a.packed.factors,a.packed.query,a.packed.observed)
                try
                    before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
                    submissions=Mantle.submissions(device)-before;solver_status=only(Array(execution.outputs.status))
                    probability=solver_status==0 ? Float64(Array(execution.outputs.posterior)[target]) : nothing
                finally close(execution) end
                solver_seconds=time()-started
            end
        end
        numeric_correct=probability!==nothing&&abs(probability-Float64(r.answer))<=1e-5
        correct=numeric_correct&&(job.arm=="direct_probability"||faithful===true)
        (;id=job.id,task=job.task,arm=job.arm,status=failed ? "inference_error" : correct ? "passed" : "failed",
            admitted=a.accepted,reason=a.reason,numeric_correct,network_faithful=faithful,probability,expected_probability=Float64(r.answer),
            forced_closures=failed ? nothing : m.forced_closures,truncated=failed ? nothing : m.truncated,
            generation_seconds=failed ? nothing : m.elapsed_seconds,solver_status,solver_seconds,solver_submissions=submissions,generation_sha256=digest(output))
    end
    arms=map(("direct_probability","bayes_network")) do arm
        selected=filter(r->r.arm==arm,rows)
        (;arm,total=length(selected),passed=count(r->r.status=="passed",selected),admitted=count(r->r.admitted,selected),
            numeric_correct=count(r->r.numeric_correct,selected),faithful_networks=count(r->r.network_faithful===true,selected))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,rows,arms,conditions_sha256=digest(path))
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
