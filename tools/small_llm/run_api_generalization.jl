isdefined(@__MODULE__,:generate_julia_answer) || include("run_julia_tasks.jl")
isdefined(@__MODULE__,:finish_device!) || include("device_finish.jl")
isdefined(@__MODULE__,:JULIA_ADVANCED_TASKS) || include("julia_advanced_tasks.jl")
isdefined(@__MODULE__,:JuliaAPICards) || include("julia_api_cards.jl")
isdefined(@__MODULE__,:check_tool_prompt) || include("check_tool_prompt.jl")
import KernelInterface, Makie

function prepare_api_generalization()
    directory=joinpath(@__DIR__,"api_generalization_results");mkpath(directory)
    qualification=JuliaTaskEval.validate_harness(;tasks=JULIA_ADVANCED_TASKS,
        output=joinpath(directory,"harness_validation.json"))
    all(r->r.passed,qualification.references) && qualification.mutant_rejected || error("unqualified harness")
    tasks=filter(t->t.id in ("ki_launch","makie_update"),JULIA_ADVANCED_TASKS)
    rows=map(tasks) do task
        # Explicit trusted API names selected from the public task, not private
        # assertions, gold bodies, generated code, or previous candidate scores.
        cards=task.id=="ki_launch" ? JuliaAPICards.cards(KernelInterface,[:Kernel,:launch,:synchronize];chars=3000) :
            JuliaAPICards.cards(Makie,[:update!];chars=5000)
        all(c->c.defined,cards) || error("missing installed API")
        messages=["system"=>"You are a Julia programmer. Return only executable Julia code. Follow the requested API and handle the specified edge cases.",
            "user"=>task.prompt]
        evidence=JuliaAPICards.format_cards(cards)
        (;id=task.id,category=task.category,messages,cards,evidence,
            contract_sha256=bytes2hex(sha256(task.tests)),prompt_sha256=bytes2hex(sha256(task.prompt)))
    end
    packet=(;scope="authored development tasks; paired single-shot direct vs installed API docs; not held-out or rendered aesthetics",
        frozen_before_generation=true,budget_tokens=512,thinking=false,rows,
        source_sha256=Dict(f=>bytes2hex(sha256(read(joinpath(@__DIR__,f)))) for f in
            ("run_api_generalization.jl","julia_advanced_tasks.jl","julia_api_cards.jl","qwen_text.jl")))
    path=joinpath(directory,"registration.json")
    isfile(path) && error("registration already exists; do not silently refreeze")
    open(io->JSON3.pretty(io,packet),path,"w")
    packet
end

function run_api_generalization(;seconds=240,model_dir="gen/qwen35-4b")
    seconds>0 || throw(ArgumentError("time budget"))
    directory=joinpath(@__DIR__,"api_generalization_results")
    registration=joinpath(directory,"registration.json")
    packet=JSON3.read(read(registration,String))
    let
        started=time();deadline=started+seconds;device=Mantle.Device("radeon");model=nothing;completed=String[]
        try
            model=SmallLLMChecks.Qwen35Experiment.loadmodel(model_dir;quantize=:q4k,device)
            tokenizer=SmallLLMChecks.QwenText.Tokenizer(model_dir)
            for (i,task) in enumerate(packet.rows), arm in (isodd(i) ? ("direct","api_docs") : ("api_docs","direct"))
                path=joinpath(directory,String(task.id)*"_"*arm*".json")
                isfile(path) && continue
                time()<deadline-75 || return (;complete=false,completed)
                messages=[String(first(keys(m)))=>String(first(values(m))) for m in task.messages]
                arm=="api_docs" && (messages[end]="user"=>(last(messages[end])*"\n\nInstalled package API documentation (may be incomplete; use only as relevant):\n"*String(task.evidence)))
                check_tool_prompt(String(task.id)*"_"*arm,messages,false,tokenizer)
                answer=generate_julia_answer(model,tokenizer,messages;max_tokens=512,thinking=false,output_chunk=128)
                result=(;task=task.id,arm,messages,contract_sha256=task.contract_sha256,
                    registration_sha256=bytes2hex(sha256(read(registration))),
                    generation_source_sha256=bytes2hex(sha256(read(@__FILE__))),answer...)
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

function score_api_generalization()
    directory=joinpath(@__DIR__,"api_generalization_results");rows=Any[]
    for task in JULIA_ADVANCED_TASKS, arm in ("direct","api_docs")
        path=joinpath(directory,task.id*"_"*arm*".json")
        isfile(path) || continue
        candidate=JSON3.read(read(path,String))
        candidate.contract_sha256==bytes2hex(sha256(task.tests)) || error("contract drift")
        score=JuliaTaskEval.evaluate(task,String(candidate.code);output=replace(path,".json"=>"_score.json"))
        push!(rows,(;task=task.id,arm,passed=score.passed,truncated=candidate.truncated,
            input_tokens=candidate.input_tokens,generation_seconds=candidate.elapsed_seconds))
    end
    report=(;scope="two authored API tasks, equal output budgets, different input lengths; development evidence only",
        rows,totals=Dict(arm=>count(r->r.arm==arm&&r.passed,rows) for arm in ("direct","api_docs")))
    open(io->JSON3.pretty(io,report),joinpath(directory,"summary.json"),"w")
    report
end
