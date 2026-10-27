isdefined(@__MODULE__,:kernel_transfer_tasks) || include("kernel_transfer_tasks.jl")
isdefined(@__MODULE__,:CheckedPointwiseLowering) || include("checked_pointwise_lowering.jl")
isdefined(@__MODULE__,:KernelVoidTail) || include("kernel_void_tail.jl")
isdefined(@__MODULE__,:JuliaTaskEval) || include("julia_eval.jl")
using JSON3,SHA

"""Offline scoring only. Never creates a hardware device in this process."""
function score_laptop_pending_study(;directory=joinpath(@__DIR__,"laptop_pending_results"))
    digest(path)=bytes2hex(open(sha256,path))
    path=joinpath(directory,"conditions.json");p=JSON3.read(read(path,String))
    for (file,hash) in pairs(p.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash || error("migration source drift")
    end
    gatepath=joinpath(@__DIR__,"laptop_portable_identity_gate.json")
    digest(gatepath)==p.gate_sha256 || error("Laptop gate receipt drift")
    gate=JSON3.read(read(gatepath,String))
    gate.complete && gate.normal_final_waitidle && gate.full_model_passed==10 && gate.hardware_fixture_passed==156 || error("gate incomplete")
    gate.portable_manifest_sha256==p.portable_manifest_sha256 || error("portable identity drift")
    extension=joinpath(@__DIR__,"safe_budget_transfer_extension_results/conditions.json")
    digest(extension)==p.extension_sha256 || error("original extension drift")
    original_protocol=JSON3.read(read(joinpath(@__DIR__,"kernel_transfer_results/conditions.json"),String))
    for (file,hash) in pairs(original_protocol.source_sha256)
        digest(joinpath(@__DIR__,String(file)))==hash || error("original evaluator/source drift")
    end
    rows=Any[];tasks=kernel_transfer_tasks()
    for job in p.jobs
        output=joinpath(directory,String(job.id)*".json");attempt=replace(output,".json"=>"_attempt.json")
        if !isfile(output)
            push!(rows,(;id=job.id,status=isfile(attempt) ? "interrupted_attempt" : "pending"));continue
        end
        candidate=JSON3.read(read(output,String))
        candidate.conditions_sha256==digest(path) || error("generation protocol drift")
        candidate.attempt_sha256==digest(attempt) || error("immutable attempt receipt drift")
        if hasproperty(candidate,:inference_error)
            push!(rows,(;id=job.id,status="inference_error",error=String(candidate.inference_error)));continue
        end
        candidate.messages==job.messages && candidate.original_sha256==job.original_sha256 || error("prompt drift")
        candidate.budget_tokens==p.decoding.max_tokens && candidate.thinking_budget==p.thinking_budget || error("budget drift")
        for field in (:seed,:topk,:temperature,:top_p,:presence)
            # Float32 caller parameters serialize at Float32 precision.
            a=getproperty(candidate.sampling,field);b=getproperty(p.decoding,field)
            field in (:temperature,:top_p,:presence) ? Float32(a)==Float32(b) || error("sampling drift") : a==b || error("sampling drift")
        end
        candidate.decode_submissions==candidate.decode_steps_executed || error("decode submission invariant")
        originalpath=joinpath(@__DIR__,"kernel_transfer_results",String(job.id)*".json")
        digest(originalpath)==job.original_sha256 || error("original record drift")
        original=JSON3.read(read(originalpath,String))
        candidate.input_sha256==original.input_sha256 || error("tokenized prompt drift")
        prefix=first(split(String(candidate.answer),"</think>";limit=2))
        matches=startswith(String(original.answer),prefix)
        task=only(filter(t->t.id==job.task,tasks));code=String(candidate.code)
        lowered=CheckedPointwiseLowering.lower(code);intermediate=lowered.accepted ? lowered.code : code
        tail=KernelVoidTail.normalize(intermediate);compiled=tail.accepted ? tail.code : intermediate
        for (variant,selected) in (("raw",code),("compiler",compiled))
            scorepath=replace(output,".json"=>"_"*variant*"_score.json")
            result=if isfile(scorepath)
                prior=JSON3.read(read(scorepath,String));prior.code_sha256==bytes2hex(sha256(selected)) || error("score drift");prior
            else
                JuliaTaskEval.evaluate(task,selected;output=scorepath)
            end
            push!(rows,(;id=job.id,task=job.task,arm=job.arm,variant,status=result.passed ? "passed" : "failed",
                truncated=candidate.truncated,finished_thinking=candidate.finished_thinking,forced_closures=candidate.forced_closures,
                pre_closure_decoded_prefix_matches_original=matches,generation_sha256=digest(output)))
        end
    end
    report=(;scope=p.scope,policy=p.policy,rows,complete=all(row->row.status in ("passed","failed","inference_error"),rows),
        conditions_sha256=digest(path),scorer_sha256=digest(@__FILE__),
        execution="Candidate code executed only in the original sandboxed software-Vulkan evaluator; no privileged/Radeon execution")
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w");report
end
