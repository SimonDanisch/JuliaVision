include("laptop_scalar_kernel_study.jl")
isdefined(@__MODULE__,:verify_transfer2_frozen_compiler) || include("qualify_compiler_transfer2.jl")

function score_canonical_scalar_development_study()
    frozen=verify_transfer2_frozen_compiler()
    original=joinpath(@__DIR__,"laptop_scalar_kernel_results");s=laptop_shared_source_state(original)
    baseline=joinpath(original,"summary.json");JSON3.read(read(baseline,String)).complete||error("finish original scores")
    directory=joinpath(@__DIR__,"canonical_scalar_development_results");mkpath(directory);rows=Any[]
    digest(path)=bytes2hex(open(sha256,path));tasks=shared_source_kernel_tasks()
    for job in s.packet.jobs
        path=joinpath(original,String(job.id)*".json");candidate=JSON3.read(read(path,String))
        candidate.conditions_sha256==digest(s.path)||error("generation drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error"));continue
        end
        lowered=CanonicalScalarInterface.lower(String(candidate.code),String(job.task))
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
    report=(;scope="POSTHOC exact global-x alias normalization and pure locals over ALL four frozen scalar outputs; no model regeneration, arithmetic/type repair or changed original scores. Not held-out evidence. Compiler separately frozen for prospective transfer2.",
        rows,complete=true,conditions_sha256=digest(s.path),baseline_summary_sha256=digest(baseline),frozen_compiler_sha256=digest(frozen),source_sha256=digest(@__FILE__))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
