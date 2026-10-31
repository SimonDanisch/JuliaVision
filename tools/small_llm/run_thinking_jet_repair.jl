isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
isdefined(@__MODULE__,:diagnose_jet) || include("julia_jet_diagnostics.jl")
isdefined(@__MODULE__,:check_tool_prompt) || include("check_tool_prompt.jl")

function prepare_thinking_jet_repair()
    directory=joinpath(@__DIR__,"thinking_jet_results");mkpath(directory)
    path=joinpath(directory,"registration.json")
    isfile(path) && error("registration already frozen")
    rows=Any[]
    for id in ("missing_windows","alias_stencil")
        task=only(filter(t->t.id==id,JuliaTaskEval.JULIA_TASKS))
        source=joinpath(@__DIR__,"julia_thinking_results",id*"_baseline.json")
        baseline=JSON3.read(read(source,String))
        baseline.finished_thinking && !baseline.truncated || error("baseline incomplete")
        diagnosis=diagnose_jet(task,String(baseline.code))
        diagnosis.analysis_completed || error("diagnostic infrastructure failed")
        evidence=join(("$(r.kind), public signature $(r.signature):\n$(r.text)" for r in diagnosis.payload.rows),"\n\n")
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
            "user"=>task.prompt,"assistant"=>String(baseline.answer)]
        push!(rows,(;id,messages,evidence,diagnosis,contract_sha256=bytes2hex(sha256(task.tests)),
            baseline_source_sha256=bytes2hex(sha256(read(source)))))
    end
    packet=(;scope="two authored Base Julia development cases; thinking baseline followed by thinking revision, paired self-review vs public JET/dispatch evidence",
        frozen_before_generation=true,budget_tokens=2048,thinking=true,rows,
        limitation="more total inference than one-shot baseline; only matched revision arms isolate diagnostic value; not held-out",
        source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("run_thinking_jet_repair.jl","julia_jet_diagnostics.jl","qwen_text.jl")))
    open(io->JSON3.pretty(io,packet),path,"w")
    packet
end

function run_thinking_jet_repair(;seconds=240,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    directory=joinpath(@__DIR__,"thinking_jet_results");registration=joinpath(directory,"registration.json")
    packet=JSON3.read(read(registration,String))
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            for (i,task) in enumerate(packet.rows), arm in (isodd(i) ? ("self_review","jet_feedback") : ("jet_feedback","self_review"))
                path=joinpath(directory,String(task.id)*"_"*arm*".json")
                isfile(path) && continue
                # Measured completed2048-cap revisions take about30s here.
                # Keep75s for a fresh session/prompt plus the whole revision,
                # rather than stopping with95s unused after every first job.
                time()<deadline-75 || return (;complete=false,completed)
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in task.messages]
                request="Review the complete prior solution for compliance with the original request. Fix any problems and return the complete executable Julia solution."
                arm=="jet_feedback" && (request*="\n\nPublic parser/dispatch/JET diagnostics follow. They are advisory, contain no private assertions or expected values, and are not a proof of correctness:\n"*String(task.evidence))
                push!(messages,"user"=>request)
                check_tool_prompt(String(task.id)*"_"*arm,messages,true,tokenizer)
                answer=generate_julia_answer(model,tokenizer,messages;max_tokens=2048,thinking=true,output_chunk=128)
                result=(;task=task.id,arm,messages,contract_sha256=task.contract_sha256,
                    registration_sha256=bytes2hex(sha256(read(registration))),
                    generation_source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
                        ("run_thinking_jet_repair.jl","run_julia_tasks.jl","qwen35.jl","prefill_qwen35.jl",
                         "qwen_text.jl","device_finish.jl","check_tool_prompt.jl","models.toml")),answer...)
                open(io->JSON3.pretty(io,result),path,"w");push!(completed,path)
                println(task.id," ",arm," saved");flush(stdout)
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

function score_thinking_jet_repair()
    directory=joinpath(@__DIR__,"thinking_jet_results");rows=Any[]
    for task in JuliaTaskEval.JULIA_TASKS, arm in ("self_review","jet_feedback")
        path=joinpath(directory,task.id*"_"*arm*".json")
        isfile(path) || continue
        candidate=JSON3.read(read(path,String))
        candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(path,".json"=>"_score.json"))
        push!(rows,(;task=task.id,arm,passed=score.passed,truncated=candidate.truncated,
            finished_thinking=candidate.finished_thinking,generation_seconds=candidate.elapsed_seconds))
    end
    report=(;scope="matched2048-token thinking revision arms on2 authored development tasks; original thinking baselines0/2",
        rows,totals=Dict(arm=>count(r->r.arm==arm&&r.passed,rows) for arm in ("self_review","jet_feedback")))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w")
    report
end
