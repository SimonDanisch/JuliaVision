isdefined(@__MODULE__,:diagnose_jet) || include("julia_jet_diagnostics.jl")
isdefined(@__MODULE__,:PublicJuliaSignatures) || include("public_julia_signatures.jl")
isdefined(@__MODULE__,:run_humaneval_baselines) || include("run_humaneval_subset.jl")
isdefined(@__MODULE__,:inference_runtime_provenance) || include("runtime_provenance.jl")

function validate_external_candidate(task,candidate)
    directory=joinpath(@__DIR__,"humaneval_subset_results")
    candidate.task==task.id || error("candidate task drift")
    candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("candidate contract drift")
    candidate.conditions_sha256==bytes2hex(sha256(read(joinpath(directory,"conditions.json")))) || error("candidate conditions drift")
    nothing
end

function prepare_humaneval_diagnostics()
    directory=joinpath(@__DIR__,"humaneval_subset_results");rows=Any[]
    for task in load_frozen_humaneval_tasks()
        source=joinpath(directory,task.id*"_baseline.json")
        if !isfile(source)
            push!(rows,(;task=task.id,status="baseline_pending"));continue
        end
        spec=PublicJuliaSignatures.from_task_prompt(task.prompt)
        JET_PUBLIC_SIGNATURES[task.id]=spec
        candidate=JSON3.read(read(source,String));path=joinpath(directory,task.id*"_diagnostics.json")
        validate_external_candidate(task,candidate)
        if isfile(path)
            diagnosis=JSON3.read(read(path,String))
            diagnosis.code_sha256==bytes2hex(sha256(String(candidate.code))) || error("diagnostic source drift")
        else
            diagnosis=diagnose_jet(task,String(candidate.code))
            open(io->JSON3.pretty(io,diagnosis),path,"w")
        end
        positive=diagnosis.analysis_completed && sum(r.report_count for r in diagnosis.payload.rows)>0
        push!(rows,(;task=task.id,status=diagnosis.analysis_completed ? "completed" : "tool_unavailable",
            positive,public_signature=spec,elapsed_seconds=diagnosis.elapsed_seconds))
    end
    report=(;scope="static-positive policy selection uses public signatures ONLY, never hidden pass/fail scores; tool unavailable retains baseline",
        rows,positive=count(r->hasproperty(r,:positive)&&r.positive,rows))
    open(io->JSON3.pretty(io,report),joinpath(directory,"diagnostic_summary.json"),"w")
    report
end

function external_repair_jobs()
    directory=joinpath(@__DIR__,"humaneval_subset_results");jobs=Any[]
    for (i,task) in enumerate(load_frozen_humaneval_tasks())
        source=joinpath(directory,task.id*"_baseline.json")
        diagnostic_path=joinpath(directory,task.id*"_diagnostics.json")
        isfile(source)&&isfile(diagnostic_path) || continue
        baseline=JSON3.read(read(source,String));diagnosis=JSON3.read(read(diagnostic_path,String))
        validate_external_candidate(task,baseline)
        diagnosis.code_sha256==bytes2hex(sha256(String(baseline.code))) || error("diagnostic source drift")
        diagnosis.analysis_completed && sum(r.report_count for r in diagnosis.payload.rows)>0 || continue
        evidence=join(("$(r.kind), public signature $(r.signature):\n$(r.text)" for r in diagnosis.payload.rows),"\n\n")
        for arm in (isodd(i) ? ("self_review","jet_feedback") : ("jet_feedback","self_review"))
            output=joinpath(directory,task.id*"_"*arm*".json")
            isfile(output) && continue
            messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
                "user"=>task.prompt,"assistant"=>String(baseline.answer)]
            request="Review the complete prior solution for compliance with the original request. Fix any problems and return the complete executable Julia solution."
            arm=="jet_feedback" && (request*="\n\nPublic parser/dispatch/JET diagnostics follow. They are advisory, contain no private assertions or expected values, and are not a proof of correctness:\n"*evidence)
            push!(messages,"user"=>request)
            push!(jobs,(;task,arm,output,messages,diagnostic_path))
        end
    end
    jobs
end

function run_humaneval_repairs(;seconds=240,model_dir="gen/qwen35-4b",jobs=external_repair_jobs(),
        conditions_path=joinpath(@__DIR__,"humaneval_subset_results","conditions.json"),extra_source_files=())
    seconds>0 || throw(ArgumentError("time budget"))
    isempty(jobs) && return (;complete=true,completed=String[])
    cache=qualified_qwen_cache();directory=joinpath(@__DIR__,"humaneval_subset_results")
    # Pure CPU tokenization checks before reserving/creating a Radeon device.
    tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
    all(job->length(SmallLLMChecks.QwenText.chat_ids(tokenizer,job.messages;thinking=true))+2080<=8192,jobs) ||
        error("repair prompt exceeds experiment capacity; do not silently truncate evidence")
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            runtime=inference_runtime_provenance()
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device,quantcache=cache)
            for job in jobs
                time()<deadline-75 || return (;complete=false,completed)
                answer=generate_julia_answer(model,tokenizer,job.messages;max_tokens=2048,thinking=true,output_chunk=128)
                job_conditions=hasproperty(job,:conditions_path) ? job.conditions_path : conditions_path
                row=(;task=job.task.id,arm=job.arm,messages=job.messages,runtime_provenance=runtime,
                    contract_sha256=bytes2hex(sha256(job.task.tests)),
                    diagnostic_sha256=bytes2hex(sha256(read(job.diagnostic_path))),
                    conditions_sha256=bytes2hex(sha256(read(job_conditions))),
                    generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                        ("humaneval_jet_policy.jl","run_humaneval_subset.jl","public_julia_signatures.jl",
                         "julia_jet_diagnostics.jl","qwen35.jl","qwen_weight_cache.jl","qwen_text.jl","device_finish.jl",extra_source_files...)),answer...)
                open(io->JSON3.pretty(io,row),job.output,"w");push!(completed,job.output)
                println(job.task.id," ",job.arm," external repair saved");flush(stdout)
            end
            (;complete=true,completed,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                finish_device!(device)
            end
        end
    end
end

function score_humaneval_workflows()
    directory=joinpath(@__DIR__,"humaneval_subset_results");rows=Any[]
    for task in load_frozen_humaneval_tasks()
        baseline_path=joinpath(directory,task.id*"_baseline_score.json")
        diagnostic_path=joinpath(directory,task.id*"_diagnostics.json")
        for arm in ("self_review","jet_feedback")
            if !isfile(baseline_path)||!isfile(diagnostic_path)
                push!(rows,(;task=task.id,arm,status="pending"));continue
            end
            diagnosis=JSON3.read(read(diagnostic_path,String))
            source=JSON3.read(read(joinpath(directory,task.id*"_baseline.json"),String))
            validate_external_candidate(task,source)
            diagnosis.code_sha256==bytes2hex(sha256(String(source.code))) || error("diagnostic source drift")
            eligible=diagnosis.analysis_completed && sum(r.report_count for r in diagnosis.payload.rows)>0
            if !eligible
                baseline=JSON3.read(read(baseline_path,String))
                push!(rows,(;task=task.id,arm,status=baseline.passed ? "passed" : "failed",
                    action=diagnosis.analysis_completed ? "no_static_findings_retain_baseline" : "tool_unavailable_retain_baseline"));continue
            end
            path=joinpath(directory,task.id*"_"*arm*".json")
            if !isfile(path)
                push!(rows,(;task=task.id,arm,status="pending"));continue
            end
            candidate=JSON3.read(read(path,String))
            validate_external_candidate(task,candidate)
            candidate.diagnostic_sha256==bytes2hex(sha256(read(diagnostic_path))) || error("revision diagnostic drift")
            score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(path,".json"=>"_score.json"))
            push!(rows,(;task=task.id,arm,status=score.passed ? "passed" : "failed",action="single_revision",
                truncated=candidate.truncated,generation_seconds=candidate.elapsed_seconds))
        end
    end
    report=(;scope="conditional static-positive self-review vs static-positive JET feedback; same frozen12public basic-test tasks",
        policy="no hidden baseline score used to choose revisions, including already-correct candidates; no manual best-of selection",
        rows,totals=Dict(arm=>(;passed=count(r->r.arm==arm&&r.status=="passed",rows),
            evaluated=count(r->r.arm==arm&&r.status!="pending",rows),pending=count(r->r.arm==arm&&r.status=="pending",rows)) for arm in ("self_review","jet_feedback")))
    open(io->JSON3.pretty(io,report),joinpath(directory,"workflow_summary.json"),"w")
    report
end
