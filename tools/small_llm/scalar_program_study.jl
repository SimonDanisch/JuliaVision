include("laptop_scalar_kernel_study.jl")
include("scalar_program_interface.jl")

function score_scalar_program_study()
    validation=joinpath(@__DIR__,"scalar_program_interface_validation.json")
    q=JSON3.read(read(validation,String));q.complete||error("adapter unqualified")
    digest(path)=bytes2hex(open(sha256,path))
    for (file,hash) in pairs(q.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash||error("adapter validation drift")
    end
    original=joinpath(@__DIR__,"laptop_scalar_kernel_results");s=laptop_shared_source_state(original)
    summary=JSON3.read(read(joinpath(original,"summary.json"),String));summary.complete||error("finish frozen scoring first")
    directory=joinpath(@__DIR__,"scalar_program_study_results");mkpath(directory);rows=Any[]
    tasks=shared_source_kernel_tasks()
    for job in s.packet.jobs
        path=joinpath(original,String(job.id)*".json");candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(s.path)||error("generation drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        lowered=ScalarProgramInterface.lower(String(candidate.code),String(job.task))
        if !lowered.accepted
            push!(rows,(;id=job.id,status="failed",stage="compiler_admission",reason=lowered.reason,generation_sha256=digest(path)));continue
        end
        task=only(filter(t->t.id==job.task,tasks));scorepath=joinpath(directory,String(job.id)*"_score.json")
        result=if isfile(scorepath)
            prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(lowered.code))||error("score drift");prior
        else
            JuliaTaskEval.evaluate(task,lowered.code;output=scorepath)
        end
        push!(rows,(;id=job.id,status=result.passed ? "passed" : "failed",stage="execution",certificate=lowered.certificate,generation_sha256=digest(path)))
    end
    packet=(;scope="POSTHOC uniform pure-scalar-program admission expansion after one expression-only output used local assignments. ALL four frozen outputs retained; no neural regeneration, arithmetic repair or replacement of frozen strict-interface scores. Not held-out/model-only evidence.",
        rows,complete=all(r->r.status in ("passed","failed","inference_error"),rows),conditions_sha256=digest(s.path),baseline_summary_sha256=digest(joinpath(original,"summary.json")),
        validation_sha256=digest(validation),source_sha256=Dict(f=>digest(joinpath(@__DIR__,f)) for f in
            ("scalar_program_study.jl","scalar_program_interface.jl","scalar_kernel_interface.jl","typed_pointwise_kernel_lowering.jl")))
    open(io->JSON3.pretty(io,packet),joinpath(directory,"summary.json"),"w");packet
end
