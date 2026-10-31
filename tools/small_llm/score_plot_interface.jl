include("plot_interface_references.jl")
include("plot_interface_admission.jl")
isdefined(@__MODULE__,:validate_pubmedqa_receipt)||include("score_pubmedqa_reading.jl")

function score_plot_interface(;directory=joinpath(@__DIR__,"plot_interface_results"))
    digest(path)=bytes2hex(sha256(read(path)))
    conditionpath=joinpath(directory,"conditions.json");p=JSON3.read(read(conditionpath,String))
    digest(@__FILE__)==p.private_scorer_sha256||error("Private scorer drift")
    for (file,hash) in pairs(p.private_sources)
        digest(joinpath(@__DIR__,String(file)))==hash||error("Private frontend/reference drift")
    end
    # The initial grader has no partial-score mode: no hidden correctness
    # feedback while even one immutable model answer remains unfinished.
    all(isfile(joinpath(directory,String(job.id)*".json")) for job in p.jobs)||error("Finish ALL16 plot answers before private grading")
    tasks=unique(String(j.task) for j in p.jobs);references=plot_interface_references()
    length(tasks)==length(references)==8||error("Complete cohort required")
    expected=Dict(tasks[i]=>SafePlotSpec.parse_spec(JSON3.write(references[i])) for i in eachindex(tasks))
    rows=map(p.jobs) do job
        path=joinpath(directory,String(job.id)*".json");attempt=joinpath(directory,String(job.id)*"_attempt.json")
        receipt=JSON3.read(read(path,String))
        validate_pubmedqa_receipt(receipt,job,p;condition_hash=digest(conditionpath),attempt_hash=digest(attempt))
        inference_failed=hasproperty(receipt,:inference_error)
        admitted=inference_failed||receipt.truncated ? (;accepted=false,reason=inference_failed ? "inference_error" : "truncated",spec=nothing) :
            admit_plot_interface(String(receipt.answer),String(job.arm))
        correct=admitted.accepted&&plot_specs_equivalent(admitted.spec,expected[String(job.task)])
        (;id=job.id,task=job.task,arm=job.arm,status=inference_failed ? "inference_error" : correct ? "passed" : "failed",
            admitted=admitted.accepted,admission_reason=admitted.reason,specification_correct=correct,
            truncated=inference_failed ? nothing : receipt.truncated,generation_sha256=digest(path),
            actual_render="not_yet_attempted",aesthetic_score=nothing)
    end
    arms=map(["julia_source","plot_spec"]) do arm
        selected=filter(r->r.arm==arm,rows)
        (;arm,total=length(selected),admitted=count(r->r.admitted,selected),specification_correct=count(r->r.specification_correct,selected))
    end
    result=(;complete=true,scope=p.scope,policy=p.policy,rows,arms,conditions_sha256=digest(conditionpath),scorer_sha256=digest(@__FILE__),
        caveat="This grades independent drawing-plan requirements, NOT arbitrary Julia execution or aesthetic preference. Actual image/layout qualification of model plans is a separate pending stage; reference image success is not model performance.")
    open(io->JSON3.pretty(io,result),joinpath(directory,"summary.json"),"w");result
end
