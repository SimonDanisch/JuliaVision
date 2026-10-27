isdefined(@__MODULE__,:bayes_extraction_tasks)||include("bayes_extraction_tasks.jl")
include("bayes_extraction_references.jl")
include("bayes_extraction_admission.jl")
include("portable_evidence.jl")
isdefined(@__MODULE__,:validate_pubmedqa_receipt)||include("score_pubmedqa_reading.jl")
import Mantle

function score_bayes_extraction(device;directory=joinpath(@__DIR__,"bayes_extraction_results"))
    occursin("llvmpipe",lowercase(sprint(show,device)))||error("Use explicitly SOFTWARE grading device")
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    digest(@__FILE__)==p.private_scorer_sha256||error("Scorer drift")
    digest(joinpath(@__DIR__,"bayes_extraction_tasks.jl"))==p.public_task_sha256||error("Task/target drift")
    for (file,hash) in pairs(p.private_sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Private source drift")
    end
    length(p.jobs)==16&&all(isfile(joinpath(directory,String(j.id)*".json")) for j in p.jobs)||error("Finish ALL16 before grading")
    tasks=bayes_extraction_tasks();references=bayes_extraction_references()
    lookup=Dict(t.id=>(t,r) for (t,r) in zip(tasks,references))
    rows=map(p.jobs) do job
        path=joinpath(directory,job.id*".json");receipt=JSON3.read(read(path,String))
        validate_pubmedqa_receipt(receipt,job,p;condition_hash=digest(conditionpath),attempt_hash=digest(joinpath(directory,job.id*"_attempt.json")))
        task,reference=lookup[job.task];inference_failed=hasproperty(receipt,:inference_error)
        admitted=inference_failed||receipt.truncated ? (;accepted=false,reason=inference_failed ? "inference_error" : "truncated",probability=nothing,network=nothing,packed=nothing) : admit_bayes_extraction(String(receipt.answer),String(job.arm))
        faithful=nothing;probability=admitted.probability;solver_status=nothing;solver_seconds=0.;submissions=0
        if admitted.accepted&&job.arm=="bayes_network"
            faithful=bayes_extraction_faithful(admitted.network,JSON3.read(JSON3.write(reference.network)))
            packed=admitted.packed;target=findfirst(==(task.target),packed.query_states)
            if packed.query==Symbol(task.query)&&target!==nothing
                started=time();execution=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
                try
                    before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
                    submissions=Mantle.submissions(device)-before
                    solver_status=only(Array(execution.outputs.status))
                    probability=solver_status==0 ? Float64(Array(execution.outputs.posterior)[target]) : nothing
                finally
                    close(execution)
                end
                solver_seconds=time()-started
            end
        end
        numeric_correct=probability!==nothing&&abs(probability-Float64(reference.answer))<=1e-5
        correct=numeric_correct&&(job.arm=="direct_probability"||faithful===true)
        (;id=job.id,task=job.task,arm=job.arm,status=inference_failed ? "inference_error" : correct ? "passed" : "failed",
            admitted=admitted.accepted,admission_reason=admitted.reason,numeric_correct,network_faithful=faithful,
            probability,expected_probability=Float64(reference.answer),solver_status,solver_seconds,solver_submissions=submissions,
            truncated=inference_failed ? nothing : receipt.truncated,generation_sha256=digest(path))
    end
    arms=map(("direct_probability","bayes_network")) do arm
        selected=filter(r->r.arm==arm,rows)
        (;arm,total=length(selected),passed=count(r->r.status=="passed",selected),admitted=count(r->r.admitted,selected),
            numeric_correct=count(r->r.numeric_correct,selected),faithful_networks=count(r->r.network_faithful===true,selected),solver_seconds=sum(r.solver_seconds for r in selected))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,rows,arms,conditions_sha256=digest(conditionpath),scorer_sha256=digest(@__FILE__),
        caveat="Authored8-task development cohort. Extra solver compute and different output formats; no trained model, autonomous tool selection, clinical or frontier-level gain is established by this arithmetic alone.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
