isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")

function run_julia_jet_repair(;seconds=240,model_dir="gen/qwen35-4b",version="v4",after_jobs=nothing)
    seconds>0 || throw(ArgumentError("time budget"))
    occursin(r"^v[0-9]+$",version) || throw(ArgumentError("version"))
    qualification=JSON3.read(read(joinpath(@__DIR__,"julia_jet_"*version*"_qualification.json"),String))
    refs=filter(r->r.arm=="reference",qualification.rows)
    length(refs)==3 && all(r->r.analysis_completed&&sum(x.report_count for x in r.payload.rows)==0,refs) || error("reference diagnostic qualification failed")
    outdir=joinpath(@__DIR__,"julia_jet_"*version*"_results");jobs=Any[]
    for task in JuliaTaskEval.JULIA_TASKS
        task.category=="base" || continue
        path=joinpath(outdir,task.id*"_repair.json");isfile(path) && continue
        diagnosis=JSON3.read(read(joinpath(outdir,task.id*"_baseline_diagnostics.json"),String))
        diagnosis.analysis_completed || error("diagnostic pipeline did not complete")
        # No free extra attempt when this tool has supplied no new evidence.
        sum(r.report_count for r in diagnosis.payload.rows)>0 || continue
        baseline=JSON3.read(read(joinpath(@__DIR__,"julia_tasks_results",task.id*"_baseline.json"),String))
        evidence=join(("$(r.kind), public signature $(r.signature):\n$(r.text)" for r in diagnosis.payload.rows),"\n\n")
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
            "user"=>task.prompt,"assistant"=>String(baseline.answer),
            "user"=>"Julia parser/dispatch/type analysis at PUBLIC input signatures produced these advisory diagnostics. No private test assertions or expected values are included. Fix the complete function while preserving all requested behavior. Return complete executable Julia code only.\n\n"*evidence]
        push!(jobs,(;task,path,messages,diagnosis))
    end
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        sources=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("run_julia_jet_repair.jl","julia_jet_diagnostics.jl","run_julia_tasks.jl","qwen35.jl","prefill_qwen35.jl","qwen_text.jl","models.toml"))
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            for job in jobs
                time()<deadline-95 || return (;complete=false,completed)
                answer=generate_julia_answer(model,tokenizer,job.messages;thinking=false,max_tokens=512,output_chunk=128)
                row=(;task=job.task.id,category=job.task.category,model=basename(model_dir),arm="parser_dispatch_jet_repair",thinking=false,
                    scope="one 512-token revision only when new static diagnostics exist; original self-review and public-execution revision controls retained",
                    contract_sha256=bytes2hex(sha256(job.task.tests)),messages=job.messages,diagnostics=job.diagnosis,
                    answer...,generation_source_sha256=sources)
                open(io->JSON3.pretty(io,row),job.path,"w");push!(completed,job.path)
                println(job.task.id," static-diagnostic repair saved");flush(stdout)
            end
            extra=after_jobs===nothing ? nothing : after_jobs(model,tokenizer,deadline)
            (;complete=true,completed,extra,elapsed_seconds=time()-started)
        finally
            try
                model===nothing || close(model)
            finally
                Mantle.waitidle(device)
            end
        end
    end
end

function score_julia_jet_repair(;version="v4")
    dir=joinpath(@__DIR__,"julia_jet_"*version*"_results");rows=Any[]
    for task in JuliaTaskEval.JULIA_TASKS
        task.category=="base" || continue
        source=joinpath(dir,task.id*"_repair.json")
        if !isfile(source)
            push!(rows,(;task=task.id,status="no static-diagnostic revision generated; retain baseline failure"));continue
        end
        candidate=JSON3.read(read(source,String))
        candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(source,".json"=>"_score.json"))
        push!(rows,(;task=task.id,status=score.passed ? "passed" : "failed",generation_seconds=candidate.elapsed_seconds,
            diagnostics_seconds=candidate.diagnostics.elapsed_seconds,budget_tokens=candidate.budget_tokens,truncated=candidate.truncated))
        println(task.id," static-diagnostic repair ",score.passed ? "PASS" : "FAIL");flush(stdout)
    end
    report=(;scope="authored Base Julia development diagnostic, not held-out; only static-positive candidates revised",
        comparison="baseline, self-review and public-execution feedback all 0/3 on these Base tasks",
        budget="one extra 512-token revision on two static-positive candidates, separate CPU diagnostic costs; no reference solution in prompts",
        passed=count(r->r.status=="passed",rows),total=length(rows),rows)
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"julia_jet_"*version*"_summary.json"),"w")
    report
end
